# SPDX-License-Identifier: Apache-2.0
"""Tests for MLX-native temperature/top-k/top-p sampling."""

import mlx.core as mx
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

from vllm_metal.v1.sampling_batch import (
    TOP_K_CHUNK_SIZE,
    SamplingBatch,
    chunked_categorical,
)

VOCAB_SIZE = 512
BATCH_SIZE = 4
# Large enough that top-k up to 16 takes the chunked candidates.
CHUNKED_VOCAB_SIZE = 64 * TOP_K_CHUNK_SIZE


def _masked_in_vocab_order(
    logits: mx.array,
    top_k: list[int],
    top_p: list[float],
    min_p: list[float] | None = None,
) -> mx.array:
    """The native path's candidate mask, scattered back into vocab order.

    Calls the production masking directly so these tests exercise what
    sampling runs, and only rearranges its output for comparison with vLLM.
    """
    sorted_desc, sorted_idx = SamplingBatch._sorted_candidate_logits(
        logits, top_k, top_p, min_p if min_p is not None else [0.0] * len(top_k)
    )
    return mx.put_along_axis(
        mx.full(logits.shape, -mx.inf, dtype=logits.dtype),
        sorted_idx,
        sorted_desc,
        axis=-1,
    )


def _sorted_probs(
    logits: mx.array, top_k: list[int], top_p: list[float], min_p: list[float]
) -> mx.array:
    """Vocab-order probabilities of the full-sort candidates."""
    sorted_desc, sorted_idx = SamplingBatch._sorted_candidate_logits(
        logits, top_k, top_p, min_p
    )
    return mx.put_along_axis(
        mx.zeros(logits.shape), sorted_idx, mx.softmax(sorted_desc, axis=-1), axis=-1
    )


def _chunked_probs(
    logits: mx.array, top_k: list[int], top_p: list[float], min_p: list[float]
) -> mx.array:
    """Vocab-order probabilities of the chunked top-k candidates; the tied
    group's mass is spread over its first ``tie_count`` tokens."""
    assert SamplingBatch._uses_top_k_candidates(top_k, logits.shape[-1])
    cand_logits, token_ids, tie_value, tie_count = SamplingBatch._top_k_candidates(
        logits, top_k, top_p, min_p
    )
    probs = mx.softmax(cand_logits, axis=-1)
    num_ids = token_ids.shape[-1]
    out = mx.put_along_axis(
        mx.zeros(logits.shape), token_ids, probs[:, :num_ids], axis=-1
    )
    is_tied = logits == tie_value
    in_group = is_tied & (mx.cumsum(is_tied.astype(mx.int32), axis=-1) <= tie_count)
    group_share = probs[:, num_ids:] / mx.maximum(tie_count, 1.0)
    return out + mx.where(in_group, group_share, 0.0)


def _spread_ties(value: float, stride: int) -> mx.array:
    """One row of small logits with ``value`` at every ``stride``-th token, so
    the tied tokens fall in many different chunks."""
    logits = mx.random.uniform(
        low=-1.0, high=0.0, shape=(1, CHUNKED_VOCAB_SIZE), key=mx.random.key(4)
    )
    return mx.where(mx.arange(CHUNKED_VOCAB_SIZE) % stride == 0, value, logits)


def _random_params(**overrides) -> SamplingParams:
    defaults = {"temperature": 0.7, "top_k": 20, "top_p": 0.95}
    defaults.update(overrides)
    return SamplingParams(**defaults)


class TestNativeRandomEligibility:
    def test_plain_temperature_top_k_top_p_batch_is_eligible(self) -> None:
        params = [
            _random_params(temperature=0.7),
            _random_params(temperature=1.0),
        ]

        assert SamplingBatch.params_allow_native_random(params)

    def test_empty_specific_token_list_is_eligible(self) -> None:
        assert SamplingBatch.params_allow_native_random(
            [_random_params(logprob_token_ids=[])]
        )

    @pytest.mark.parametrize(
        ("label", "params_list"),
        [
            ("greedy request", [_random_params(temperature=0.0)]),
            (
                "mixed greedy and random",
                [_random_params(), _random_params(temperature=0.0)],
            ),
            (
                "mixed top_k",
                [_random_params(top_k=20), _random_params(top_k=40)],
            ),
            (
                "mixed top_p",
                [_random_params(top_p=0.95), _random_params(top_p=0.8)],
            ),
            (
                "mixed min_p",
                [_random_params(min_p=0.1), _random_params(min_p=0.0)],
            ),
        ],
    )
    def test_per_row_masks_keep_mixed_batches_eligible(
        self, label, params_list
    ) -> None:
        """Masks are per row: one request must never push the whole batch
        onto the torch sampler (that serialises every decode step)."""
        assert SamplingBatch.params_allow_native_random(params_list)

    @pytest.mark.parametrize(
        ("label", "params_list"),
        [
            ("empty batch", []),
            ("seeded request", [_random_params(seed=7)]),
            ("frequency penalty", [_random_params(frequency_penalty=0.5)]),
            ("presence penalty", [_random_params(presence_penalty=0.5)]),
            ("repetition penalty", [_random_params(repetition_penalty=1.2)]),
            ("sample logprobs", [_random_params(logprobs=1)]),
            (
                "logprob token ids",
                [_random_params(logprob_token_ids=[1, 2])],
            ),
            (
                "allowed token ids",
                [_random_params(allowed_token_ids=[1, 2])],
            ),
        ],
    )
    def test_ineligible_batches_fall_back(self, label, params_list) -> None:
        assert not SamplingBatch.params_allow_native_random(params_list)

    def test_bad_words_fall_back(self) -> None:
        params = _random_params()
        params._bad_words_token_ids = [[99]]

        assert not SamplingBatch.params_allow_native_random([params])


class TestTopKTopPMaskParity:
    """The MLX mask must keep exactly the candidates vLLM's mask keeps."""

    @pytest.mark.parametrize(
        ("top_k", "top_p"),
        [
            (20, 1.0),
            (0, 0.95),
            (20, 0.95),
            (1, 1.0),
            (0, 1.0),
            (VOCAB_SIZE, 0.5),
            (0, 0.999999),
        ],
    )
    def test_mask_matches_vllm_reference(self, top_k: int, top_p: float) -> None:
        logits = mx.random.normal((BATCH_SIZE, VOCAB_SIZE), key=mx.random.key(42))
        mx.eval(logits)
        logits_torch = torch.tensor(logits.tolist())
        k_arg = (
            None
            if top_k <= 0 or top_k >= VOCAB_SIZE
            else torch.full((BATCH_SIZE,), top_k)
        )
        p_arg = None if top_p == 1.0 else torch.full((BATCH_SIZE,), top_p)

        masked_mlx = _masked_in_vocab_order(
            logits, [top_k] * BATCH_SIZE, [top_p] * BATCH_SIZE
        )
        mx.eval(masked_mlx)
        masked_ref = apply_top_k_top_p(logits_torch.clone(), k_arg, p_arg)

        kept_mlx = [[v != float("-inf") for v in row] for row in masked_mlx.tolist()]
        kept_ref = (masked_ref != float("-inf")).tolist()
        assert kept_mlx == kept_ref

    def test_top_p_boundary_ties_are_masked_positionally(self) -> None:
        """Boundary ties must not all survive (vLLM masks sorted positions).

        Four equal logits at top_p=0.25: each holds 0.25 mass, so exactly
        one sorted position has leading mass < 0.25 and only one survives.
        """
        logits = mx.array([[2.0, 2.0, 2.0, 2.0]])
        logits_torch = torch.full((1, 4), 2.0)

        masked = _masked_in_vocab_order(logits, [0], [0.25])
        mx.eval(masked)
        masked_ref = apply_top_k_top_p(logits_torch, None, torch.full((1,), 0.25))

        kept_count = sum(v != float("-inf") for v in masked.tolist()[0])
        kept_count_ref = int((masked_ref != float("-inf")).sum().item())
        assert kept_count == kept_count_ref == 1

    def test_per_row_masks_match_vllm_reference(self) -> None:
        """Each row keeps exactly the candidates vLLM keeps for its own k/p."""
        logits = mx.random.normal((4, VOCAB_SIZE), key=mx.random.key(9))
        mx.eval(logits)
        top_k = [20, 0, 7, VOCAB_SIZE]
        top_p = [1.0, 0.9, 0.5, 0.75]
        masked_mlx = _masked_in_vocab_order(logits, top_k, top_p)
        mx.eval(masked_mlx)
        logits_torch = torch.tensor(logits.tolist())
        k_arg = torch.tensor([k if 0 < k < VOCAB_SIZE else VOCAB_SIZE for k in top_k])
        masked_ref = apply_top_k_top_p(logits_torch.clone(), k_arg, torch.tensor(top_p))

        kept_mlx = [[v != float("-inf") for v in row] for row in masked_mlx.tolist()]
        kept_ref = (masked_ref != float("-inf")).tolist()
        assert kept_mlx == kept_ref

    def test_mixed_top_k_rows_keep_ties_at_their_own_kth_value(self) -> None:
        """Each row's ties at its own k-th largest survive, whatever the
        other rows in the batch ask for."""
        logits = mx.array(
            [[3.0, 2.0, 2.0, 2.0, 1.0, 0.0], [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]]
        )
        sorted_desc, sorted_idx = SamplingBatch._sorted_candidate_logits(
            logits, [2, 4], [1.0, 1.0], [0.0, 0.0]
        )
        mx.eval(sorted_desc, sorted_idx)
        kept_rows = [
            {
                int(t)
                for t, v in zip(idx_row, val_row, strict=True)
                if v != float("-inf")
            }
            for idx_row, val_row in zip(
                sorted_idx.tolist(), sorted_desc.tolist(), strict=True
            )
        ]
        assert kept_rows == [{0, 1, 2, 3}, {0, 1, 2, 3}]

    def test_single_row_top_k_keeps_ties_at_the_kth_value(self) -> None:
        """One row on its own must keep the same boundary ties vLLM keeps.

        Top-k masks by value, so every logit equal to the k-th largest
        survives: ``top_k=2`` over ``[3, 2, 2, 2, 1, 0]`` keeps four tokens.
        A candidate set capped at the row's own ``top_k`` would keep two.
        """
        logits = mx.array([[3.0, 2.0, 2.0, 2.0, 1.0, 0.0]])
        sorted_desc, sorted_idx = SamplingBatch._sorted_candidate_logits(
            logits, [2], [1.0], [0.0]
        )
        mx.eval(sorted_desc, sorted_idx)
        kept = {
            int(t)
            for t, v in zip(
                sorted_idx.tolist()[0], sorted_desc.tolist()[0], strict=True
            )
            if v != float("-inf")
        }

        masked_ref = apply_top_k_top_p(
            torch.tensor([[3.0, 2.0, 2.0, 2.0, 1.0, 0.0]]),
            torch.tensor([2]),
            None,
        )
        kept_ref = {
            i for i, v in enumerate(masked_ref.tolist()[0]) if v != float("-inf")
        }
        assert kept == kept_ref == {0, 1, 2, 3}


class TestTopKCandidates:
    @pytest.mark.parametrize(
        ("top_k", "top_p", "min_p"),
        [
            ([16] * 4, [0.95] * 4, [0.0] * 4),
            ([16, 8, 3, 1], [0.9, 1.0, 0.5, 1.0], [0.0, 0.05, 0.0, 0.0]),
            ([5] * 4, [1.0] * 4, [0.1] * 4),
        ],
    )
    def test_probabilities_match_the_full_sort(
        self, top_k: list[int], top_p: list[float], min_p: list[float]
    ) -> None:
        logits = mx.random.normal((4, CHUNKED_VOCAB_SIZE), key=mx.random.key(17)) * 3
        chunked = _chunked_probs(logits, top_k, top_p, min_p)
        reference = _sorted_probs(logits, top_k, top_p, min_p)
        mx.eval(chunked, reference)

        assert mx.allclose(chunked, reference, atol=1e-6).item()

    @pytest.mark.parametrize(("top_k", "top_p"), [(16, 1.0), (16, 0.9), (4, 0.5)])
    def test_kept_tokens_match_vllm_reference(self, top_k: int, top_p: float) -> None:
        logits = mx.random.normal((4, CHUNKED_VOCAB_SIZE), key=mx.random.key(42))
        chunked = _chunked_probs(logits, [top_k] * 4, [top_p] * 4, [0.0] * 4)
        mx.eval(chunked)
        masked_ref = apply_top_k_top_p(
            torch.tensor(logits.tolist()),
            torch.full((4,), top_k),
            None if top_p == 1.0 else torch.full((4,), top_p),
        )

        kept = [[p > 0.0 for p in row] for row in chunked.tolist()]
        assert kept == (masked_ref != float("-inf")).tolist()

    def test_kth_value_ties_outside_the_sorted_chunks_are_kept(self) -> None:
        """Top-k keeps every token tied at the k-th value, including ties in
        chunks whose maxima were not sorted. 41 tokens tie at the top, one
        per 100 tokens, and ``top_k=2`` sorts only two chunks."""
        logits = _spread_ties(3.0, stride=100)
        chunked = _chunked_probs(logits, [2], [1.0], [0.0])
        mx.eval(chunked)
        masked_ref = apply_top_k_top_p(
            torch.tensor(logits.tolist()), torch.tensor([2]), None
        )

        kept = {i for i, p in enumerate(chunked.tolist()[0]) if p > 0.0}
        kept_ref = {
            i for i, v in enumerate(masked_ref.tolist()[0]) if v != float("-inf")
        }
        assert kept == kept_ref == set(range(0, CHUNKED_VOCAB_SIZE, 100))
        assert mx.allclose(chunked[0, ::100], mx.array(1 / 41)).item()

    def test_top_p_through_kth_value_ties_keeps_the_sorted_count(self) -> None:
        """When top-p cuts through the ties at the k-th value, as many tied
        tokens survive as in the full sort and in vLLM."""
        logits = _spread_ties(3.0, stride=100)
        logits[0, 7] = 6.0
        top_k, top_p = [3], [0.5]
        chunked = _chunked_probs(logits, top_k, top_p, [0.0])
        reference = _sorted_probs(logits, top_k, top_p, [0.0])
        mx.eval(chunked, reference)
        masked_ref = apply_top_k_top_p(
            torch.tensor(logits.tolist()), torch.tensor(top_k), torch.tensor(top_p)
        )

        kept_count = sum(p > 0.0 for p in chunked.tolist()[0])
        assert kept_count == int((masked_ref != float("-inf")).sum().item())
        assert mx.allclose(
            mx.sort(chunked, axis=-1), mx.sort(reference, axis=-1), atol=1e-6
        ).item()

    def test_tied_group_draws_cover_every_tied_token(self) -> None:
        """A draw that lands on the tied group picks among all its tokens,
        not just the ones in the sorted chunks."""
        logits = mx.broadcast_to(
            _spread_ties(3.0, stride=100), (64, CHUNKED_VOCAB_SIZE)
        )
        params = [_random_params(temperature=1.0, top_k=2, top_p=1.0)] * 64

        drawn = set()
        key = mx.random.key(8)
        for _ in range(50):
            key, subkey = mx.random.split(key)
            tokens = SamplingBatch._native_random_tokens(logits, params, subkey)
            mx.eval(tokens)
            drawn.update(tokens.tolist())

        assert drawn == set(range(0, CHUNKED_VOCAB_SIZE, 100))


def _draw_frequencies(
    logits: mx.array, params: SamplingParams, draws: int, seed: int
) -> mx.array:
    """Empirical token frequencies of ``draws`` native draws of one row."""
    rows = 512
    batch = mx.broadcast_to(logits, (rows, logits.shape[-1]))
    counts = mx.zeros((logits.shape[-1],), dtype=mx.int32)
    key = mx.random.key(seed)
    for _ in range(draws // rows):
        key, subkey = mx.random.split(key)
        tokens = SamplingBatch._native_random_tokens(batch, [params] * rows, subkey)
        counts = counts.at[tokens].add(1)
    return counts.astype(mx.float32) / (draws // rows * rows)


class TestChunkedCategorical:
    def test_never_draws_masked_tokens(self) -> None:
        """Masked tokens, whole masked chunks and the padding of a vocab that
        is not a multiple of the chunk size are never drawn."""
        vocab_size = 200
        allowed = [3, 64, 65, 199]
        logits = mx.full((256, vocab_size), -mx.inf)
        logits[:, allowed] = mx.array([0.0, 1.0, -2.0, 0.5])

        drawn = set()
        key = mx.random.key(2)
        for _ in range(20):
            key, subkey = mx.random.split(key)
            tokens = chunked_categorical(logits, subkey)
            mx.eval(tokens)
            drawn.update(tokens.tolist())

        assert drawn == set(allowed)

    @pytest.mark.parametrize(
        "params",
        [
            _random_params(temperature=0.8, top_k=0, top_p=1.0),
            _random_params(temperature=1.0, top_k=0, top_p=1.0, min_p=0.05),
            _random_params(temperature=1.5, top_k=0, top_p=0.95),
        ],
        ids=["temperature", "min_p", "top_p"],
    )
    def test_draws_follow_the_candidate_distribution(
        self, params: SamplingParams
    ) -> None:
        """Draws match the softmax over the full-sort candidates, for batches
        that skip the sort and for top-p batches that keep it."""
        vocab_size = 200
        logits = mx.random.normal((1, vocab_size), key=mx.random.key(9)) * 2
        expected = _sorted_probs(
            logits / params.temperature, [0], [params.top_p], [params.min_p]
        )[0]

        observed = _draw_frequencies(logits, params, draws=81_920, seed=13)

        total_variation = 0.5 * mx.abs(observed - expected).sum().item()
        assert total_variation < 0.03
        assert mx.all(mx.where(expected == 0.0, observed == 0.0, True)).item()


class TestMlxRandomTokens:
    def test_greedy_row_in_temperature_only_batch_stays_on_the_argmax(
        self,
    ) -> None:
        logits = mx.array([[1.0, 5.0, 5.0, 0.0], [1.0, 2.0, 3.0, 4.0]])
        params = [
            _random_params(temperature=0.0),
            _random_params(temperature=1.0, top_k=0, top_p=1.0),
        ]
        expected = int(mx.argmax(logits[0]).item())

        greedy_row, other_row = set(), set()
        for seed in range(50):
            tokens = SamplingBatch._native_random_tokens(
                logits, params, mx.random.key(seed)
            )
            mx.eval(tokens)
            greedy_row.add(int(tokens.tolist()[0]))
            other_row.add(int(tokens.tolist()[1]))

        assert greedy_row == {expected}
        assert len(other_row) > 1

    def test_greedy_row_with_tied_maximum_stays_on_the_argmax(self) -> None:
        """A greedy row must resolve to the argmax even when the top logits
        tie. Masking it by value keeps every tied token, and the categorical
        draw then picks between them; vLLM's greedy path takes the argmax.
        """
        logits = mx.array([[1.0, 5.0, 5.0, 0.0], [1.0, 2.0, 3.0, 4.0]])
        params = [
            _random_params(temperature=0.0),
            _random_params(temperature=1.0, top_k=0, top_p=0.9),
        ]
        expected = int(mx.argmax(logits[0]).item())

        drawn = set()
        for seed in range(50):
            tokens = SamplingBatch._native_random_tokens(
                logits, params, mx.random.key(seed)
            )
            mx.eval(tokens)
            drawn.add(int(tokens.tolist()[0]))

        assert drawn == {expected}

    def test_mixed_batch_rows_follow_their_own_params(self) -> None:
        """A greedy row sticks to its argmax and a narrow top-k row stays inside
        its candidates while a wide row keeps exploring — all in one draw."""
        logits = mx.random.normal((3, VOCAB_SIZE), key=mx.random.key(21))
        mx.eval(logits)
        argmax = int(mx.argmax(logits[0]).item())
        narrow_candidates = {int(i) for i in mx.argsort(logits[1])[::-1][:4].tolist()}
        params = [
            _random_params(temperature=0.0),
            _random_params(temperature=1.5, top_k=4, top_p=1.0),
            _random_params(temperature=2.0, top_k=0, top_p=1.0),
        ]
        assert SamplingBatch.params_allow_native_random(params)

        greedy_row, narrow_row, wide_row = set(), set(), set()
        key = mx.random.key(1)
        for _ in range(200):
            key, subkey = mx.random.split(key)
            tokens = SamplingBatch._native_random_tokens(logits, params, subkey)
            mx.eval(tokens)
            g, n, w = tokens.tolist()
            greedy_row.add(g)
            narrow_row.add(n)
            wide_row.add(w)

        assert greedy_row == {argmax}
        assert narrow_row <= narrow_candidates and len(narrow_row) > 1
        assert len(wide_row) > len(narrow_candidates)

    def test_samples_stay_inside_candidate_set(self) -> None:
        candidate_count = 8
        logits = mx.full((BATCH_SIZE, VOCAB_SIZE), -100.0)
        logits[:, :candidate_count] = mx.linspace(5.0, 1.0, candidate_count)[None, :]
        mx.eval(logits)
        params = [_random_params(top_k=candidate_count)] * BATCH_SIZE

        sampled = []
        key = mx.random.key(0)
        for _ in range(200):
            key, subkey = mx.random.split(key)
            tokens = SamplingBatch._native_random_tokens(logits, params, subkey)
            mx.eval(tokens)
            sampled.extend(tokens.tolist())

        assert all(0 <= token < candidate_count for token in sampled)
        assert len(set(sampled)) > 1

    def test_same_key_is_deterministic(self) -> None:
        logits = mx.random.normal((BATCH_SIZE, VOCAB_SIZE), key=mx.random.key(3))
        mx.eval(logits)
        params = [_random_params()] * BATCH_SIZE
        key = mx.random.key(11)

        first = SamplingBatch._native_random_tokens(logits, params, key)
        second = SamplingBatch._native_random_tokens(logits, params, key)
        mx.eval(first, second)

        assert first.tolist() == second.tolist()

    def test_per_row_temperatures_shape_distinct_distributions(self) -> None:
        """Row temperatures are per-row facts: a near-zero-temperature row
        must stick to the argmax while a hot row keeps exploring."""
        logits = mx.array([[5.0, 4.5, 4.0, 3.5]] * 2)
        mx.eval(logits)
        params = [
            _random_params(temperature=0.01, top_k=0, top_p=1.0),
            _random_params(temperature=2.0, top_k=0, top_p=1.0),
        ]

        cold_row, hot_row = set(), set()
        key = mx.random.key(7)
        for _ in range(100):
            key, subkey = mx.random.split(key)
            tokens = SamplingBatch._native_random_tokens(logits, params, subkey)
            mx.eval(tokens)
            cold_row.add(tokens.tolist()[0])
            hot_row.add(tokens.tolist()[1])

        assert cold_row == {0}
        assert len(hot_row) > 1

    def test_distribution_tracks_probabilities(self) -> None:
        probs = [0.5, 0.3, 0.2]
        logits = mx.log(mx.array([probs] * 64))
        mx.eval(logits)
        params = [_random_params(temperature=1.0, top_k=0, top_p=1.0)] * 64

        counts = [0, 0, 0]
        key = mx.random.key(5)
        draws = 0
        for _ in range(400):
            key, subkey = mx.random.split(key)
            tokens = SamplingBatch._native_random_tokens(logits, params, subkey)
            mx.eval(tokens)
            for token in tokens.tolist():
                counts[token] += 1
                draws += 1

        for index, expected in enumerate(probs):
            assert abs(counts[index] / draws - expected) < 0.02
