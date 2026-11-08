// SPDX-License-Identifier: Apache-2.0
// Matches vLLM MRv2's FP32 Murmur3 Gumbel-max stream (seed, position, token).
// https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu/sample/gumbel.py#L74-L206
#include <metal_stdlib>
using namespace metal;

inline uint rotate32(uint x, uint n) { return (x << n) | (x >> (32 - n)); }
inline uint mix32(uint h, uint k) {
    k = rotate32(k * 0xCC9E2D51u, 15) * 0x1B873593u;
    return rotate32(h ^ k, 13) * 5u + 0xE6546B64u;
}
inline float uniform32(long seed, long position, uint token) {
    uint h = mix32(0, uint(seed));
    h = mix32(h, uint(ulong(seed) >> 32));
    h = mix32(h, uint(position));
    h = mix32(h, token) ^ 16u;
    h = (h ^ (h >> 16)) * 0x85EBCA6Bu;
    h = (h ^ (h >> 13)) * 0xC2B2AE35u;
    h ^= h >> 16;
    return float(h >> 16) * 1.52587890625e-05f
         + (float(h & 0xFFFF) + 0.5f) * 2.3283064365386963e-10f;
}
inline float log1p_neg(float u) {
    float p = 1.0f / 8;
    for (int i = 7; i >= 1; --i) p = 1.0f / i + u * p;
    return u < 0.25f ? -u * p : log(max(1.0f - u, 5.960464477539063e-08f));
}

kernel void gumbel_sample(
    const device float* logits [[buffer(0)]],
    const device long* mapping [[buffer(1)]],
    const device long* seeds [[buffer(2)]],
    const device long* positions [[buffer(3)]],
    const device float* temperatures [[buffer(4)]],
    device long* argmax [[buffer(5)]],
    device float* maximum [[buffer(6)]],
    constant long* logits_layout [[buffer(7)]],
    constant long& block_size [[buffer(8)]],
    constant long& blocks [[buffer(9)]],
    constant long& apply_temperature [[buffer(10)]],
    const device uint* token_ids [[buffer(11)]],
    constant long& indexed [[buffer(12)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    long slot = mapping[group.y];
    float temp = slot < 0 ? 0.0f : temperatures[slot];
    long seed = slot < 0 ? 0 : seeds[slot];
    long position = positions[group.y];
    // Read padded or broadcast logits directly, without packing the rows.
    const long vocab = logits_layout[0];
    const device float* row = logits + long(group.y) * logits_layout[1];
    float best = -INFINITY;
    uint index = UINT_MAX;
    for (long col = long(group.x) * block_size + tid;
         col < min(vocab, (long(group.x) + 1) * block_size); col += 256) {
        float value = row[col * logits_layout[2]];
        uint token = indexed ? token_ids[long(group.y) * vocab + col] : uint(col);
        if (temp != 0.0f && value != -INFINITY) {
            if (apply_temperature && temp != 1.0f) value /= temp;
            value -= log(-log1p_neg(uniform32(seed, position, token)));
        }
        // Mask NaNs so an all-invalid row still selects its lowest candidate ID.
        if (isnan(value)) value = -INFINITY;
        if (value > best || (value == best && token < index)) {
            best = value;
            index = token;
        }
    }
    threadgroup float maxima[8];
    threadgroup uint indices[8];
    float simd_best = simd_max(best);
    uint simd_index = simd_min(best == simd_best ? index : UINT_MAX);
    if (lane == 0) {
        maxima[simd_group] = simd_best;
        indices[simd_group] = simd_index;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        best = lane < 8 ? maxima[lane] : -INFINITY;
        index = lane < 8 ? indices[lane] : UINT_MAX;
        simd_best = simd_max(best);
        simd_index = simd_min(best == simd_best ? index : UINT_MAX);
        if (lane == 0) {
            long out = long(group.y) * blocks + group.x;
            maximum[out] = simd_best;
            argmax[out] = long(simd_index);
        }
    }
}

// Sort-free top-k, min-p and top-p filter shared by MLX and PyTorch MPS.
// Each row takes a max/mass pass and a histogram pass over fixed slices, then
// one threadgroup solves the cutoff exactly from the histogram's boundary bin
// and the row is masked in vocabulary order. Slice sizes depend only on the
// vocabulary and every merge is fixed-order or integer, so a row's cutoff does
// not depend on the batch it runs in.
// TODO: fold the mask into gumbel_sample and read BF16/FP16 logits directly.
constant uint kFilterSimds = 32;
constant uint kFilterThreads = 1024;
constant int kFilterSlice = 16384;
constant int kFilterBlock = 1024;
constant uint kFilterBins = 1024;
constant float kFilterBinRange = 20.0f;
constant float kFilterBinsPerNat = 51.2f;
constant uint kFilterGather = 3072;
constant uint kFilterKeepTies = UINT_MAX;

inline float filter_clean(float v) { return isnan(v) ? -INFINITY : (v == 0.0f ? 0.0f : v); }
// Order-preserving unsigned key of a float (and back).
inline uint filter_key(float v) {
    uint u = as_type<uint>(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
inline float filter_value(uint k) {
    uint u = (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k;
    return as_type<float>(u);
}
inline uint filter_bin(float v, float m) {
    float d = m - v;
    if (!(d < kFilterBinRange)) return kFilterBins - 1;
    return min(uint(d * kFilterBinsPerNat), kFilterBins - 2);
}
// Fixed-point mass in units of Z / 2^31, so merges are integer additions.
inline uint filter_mass(float v, float m, float scale) { return uint(rint(exp(v - m) * scale)); }
inline bool filter_top_k_on(int k, int vocab) { return k > 0 && k < vocab; }

// Row max and sum of exp(x - max) from the per-slice partials.
inline float2 filter_row_stats(
    const device float* slice_max, const device float* slice_sum, uint row, int slices) {
    float m = -INFINITY;
    for (int s = 0; s < slices; ++s) m = max(m, slice_max[row * slices + s]);
    float z = 0.0f;
    for (int s = 0; s < slices; ++s) {
        float partial = slice_max[row * slices + s];
        z += partial == -INFINITY ? 0.0f : slice_sum[row * slices + s] * exp(partial - m);
    }
    return float2(m, z);
}

inline void filter_reduce16(thread uint* h, threadgroup uint* simd_part, threadgroup uint* merged,
                            uint tid, uint lane, uint simd_group) {
    for (uint b = 0; b < 16; ++b) {
        uint s = simd_sum(h[b]);
        if (lane == 0) simd_part[simd_group * 16 + b] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16) {
        uint s = 0;
        for (uint g = 0; g < kFilterSimds; ++g) s += simd_part[g * 16 + tid];
        merged[tid] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline uint filter_reduce1(uint v, threadgroup uint* simd_part, uint lane, uint simd_group) {
    v = simd_sum(v);
    if (lane == 0) simd_part[simd_group] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint s = 0;
    for (uint g = 0; g < kFilterSimds; ++g) s += simd_part[g];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s;
}

// Key of the rank-th largest value (1-based) among src[0, n).
template <typename P>
uint filter_radix_kth(P src, int n, uint rank, threadgroup uint* simd_part, threadgroup uint* merged,
                      uint tid, uint lane, uint simd_group) {
    uint prefix = 0, prefix_mask = 0;
    for (int shift = 28; shift >= 0; shift -= 4) {
        uint h[16];
        for (uint b = 0; b < 16; ++b) h[b] = 0;
        for (int i = int(tid); i < n; i += int(kFilterThreads)) {
            uint key = filter_key(filter_clean(src[i]));
            if ((key & prefix_mask) == prefix) {
                uint d = (key >> uint(shift)) & 15u;
                for (uint b = 0; b < 16; ++b) h[b] += d == b ? 1u : 0u;
            }
        }
        filter_reduce16(h, simd_part, merged, tid, lane, simd_group);
        uint d = 0;
        for (int b = 15; b >= 0; --b) {
            uint c = merged[b];
            if (rank <= c) { d = uint(b); break; }
            rank -= c;
        }
        prefix |= d << uint(shift);
        prefix_mask |= 15u << uint(shift);
    }
    return prefix;
}

// Key where the descending inclusive mass first reaches target (scaled by
// 2^24) over src values with key >= floor_key. above enters as the mass ranked
// before src and leaves as the mass strictly above the key.
template <typename P>
uint filter_radix_mass(P src, int n, uint floor_key, float m, float scale, ulong target,
                       thread ulong& above, thread uint& ties,
                       threadgroup uint* mass_part, threadgroup uint* mass,
                       threadgroup uint* count_part, threadgroup uint* count,
                       uint tid, uint lane, uint simd_group) {
    uint prefix = 0, prefix_mask = 0;
    for (int shift = 28; shift >= 0; shift -= 4) {
        uint h[16], hc[16];
        for (uint b = 0; b < 16; ++b) { h[b] = 0; hc[b] = 0; }
        for (int i = int(tid); i < n; i += int(kFilterThreads)) {
            float v = filter_clean(src[i]);
            uint key = filter_key(v);
            if (key >= floor_key && (key & prefix_mask) == prefix) {
                uint d = (key >> uint(shift)) & 15u;
                uint q = filter_mass(v, m, scale);
                for (uint b = 0; b < 16; ++b) {
                    h[b] += d == b ? q : 0u;
                    hc[b] += d == b ? 1u : 0u;
                }
            }
        }
        filter_reduce16(h, mass_part, mass, tid, lane, simd_group);
        filter_reduce16(hc, count_part, count, tid, lane, simd_group);
        int digit = -1, lowest = -1;
        ulong run = above, run_lowest = above;
        for (int b = 15; b >= 0; --b) {
            if (count[b] == 0) continue;
            if (((run + ulong(mass[b])) << 24) >= target) { digit = b; break; }
            lowest = b;
            run_lowest = run;
            run += ulong(mass[b]);
        }
        // If rounding leaves the target unreached, take the lowest populated digit.
        if (digit < 0) { digit = lowest; run = run_lowest; }
        above = run;
        prefix |= uint(digit) << uint(shift);
        prefix_mask |= 15u << uint(shift);
        if (shift == 0) ties = count[digit];
    }
    return prefix;
}

kernel void filter_stats(
    const device float* logits [[buffer(0)]],
    const device int* top_k [[buffer(1)]],
    const device float* top_p [[buffer(2)]],
    const device float* min_p [[buffer(3)]],
    device float* slice_max [[buffer(4)]],
    device float* slice_sum [[buffer(5)]],
    constant long& vocab_size [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const int vocab = int(vocab_size);
    const int slices = (vocab + kFilterSlice - 1) / kFilterSlice;
    const uint row = group.y;
    if (!filter_top_k_on(top_k[row], vocab) && top_p[row] >= 1.0f && min_p[row] <= 0.0f) return;
    const device float* x = logits + ulong(row) * ulong(vocab);
    threadgroup float partial[kFilterSimds * 2];
    const int start = int(group.x) * kFilterSlice;
    const int end = min(vocab, start + kFilterSlice);
    float thread_max = -INFINITY, thread_sum = 0.0f;
    for (int i = start + int(tid); i < end; i += int(kFilterThreads)) {
        float v = filter_clean(x[i]);
        if (v == -INFINITY) continue;
        if (v > thread_max) {
            thread_sum = thread_sum * exp(thread_max - v) + 1.0f;
            thread_max = v;
        } else {
            thread_sum += exp(v - thread_max);
        }
    }
    float simd_m = simd_max(thread_max);
    float simd_z = simd_sum(thread_max == -INFINITY ? 0.0f : thread_sum * exp(thread_max - simd_m));
    if (lane == 0) {
        partial[simd_group] = simd_m;
        partial[kFilterSimds + simd_group] = simd_z;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float m = -INFINITY;
        for (uint g = 0; g < kFilterSimds; ++g) m = max(m, partial[g]);
        float z = 0.0f;
        for (uint g = 0; g < kFilterSimds; ++g)
            z += partial[g] == -INFINITY ? 0.0f : partial[kFilterSimds + g] * exp(partial[g] - m);
        slice_max[row * slices + group.x] = m;
        slice_sum[row * slices + group.x] = z;
    }
}

// Count and fixed-point mass per 1/51.2-nat bin of distance below the row max.
kernel void filter_histogram(
    const device float* logits [[buffer(0)]],
    const device int* top_k [[buffer(1)]],
    const device float* top_p [[buffer(2)]],
    const device float* slice_max [[buffer(3)]],
    const device float* slice_sum [[buffer(4)]],
    device atomic_uint* histogram [[buffer(5)]],
    constant long& vocab_size [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const int vocab = int(vocab_size);
    const int slices = (vocab + kFilterSlice - 1) / kFilterSlice;
    const uint row = group.y;
    const bool use_top_p = top_p[row] < 1.0f;
    if (!filter_top_k_on(top_k[row], vocab) && !use_top_p) return;
    const float2 stats = filter_row_stats(slice_max, slice_sum, row, slices);
    const float m = stats.x;
    if (m == -INFINITY) return;
    const float scale = 2147483648.0f / stats.y;
    const device float* x = logits + ulong(row) * ulong(vocab);
    threadgroup atomic_uint local_count[kFilterBins];
    threadgroup atomic_uint local_mass[kFilterBins];
    for (uint b = tid; b < kFilterBins; b += kFilterThreads) {
        atomic_store_explicit(&local_count[b], 0u, memory_order_relaxed);
        atomic_store_explicit(&local_mass[b], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int start = int(group.x) * kFilterSlice;
    const int end = min(vocab, start + kFilterSlice);
    uint tail_count = 0, tail_mass = 0;
    for (int i = start + int(tid); i < end; i += int(kFilterThreads)) {
        float v = filter_clean(x[i]);
        uint b = filter_bin(v, m);
        uint q = use_top_p ? filter_mass(v, m, scale) : 0u;
        // Most tokens land in the last bin. Sum them outside the atomics.
        if (b == kFilterBins - 1) {
            tail_count += 1;
            tail_mass += q;
        } else {
            atomic_fetch_add_explicit(&local_count[b], 1u, memory_order_relaxed);
            if (q) atomic_fetch_add_explicit(&local_mass[b], q, memory_order_relaxed);
        }
    }
    tail_count = simd_sum(tail_count);
    tail_mass = simd_sum(tail_mass);
    if (lane == 0) {
        atomic_fetch_add_explicit(&local_count[kFilterBins - 1], tail_count, memory_order_relaxed);
        atomic_fetch_add_explicit(&local_mass[kFilterBins - 1], tail_mass, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device atomic_uint* out = histogram + ulong(row) * ulong(2 * kFilterBins);
    for (uint b = tid; b < kFilterBins; b += kFilterThreads) {
        uint c = atomic_load_explicit(&local_count[b], memory_order_relaxed);
        uint q = atomic_load_explicit(&local_mass[b], memory_order_relaxed);
        if (c) atomic_fetch_add_explicit(&out[b], c, memory_order_relaxed);
        if (q) atomic_fetch_add_explicit(&out[kFilterBins + b], q, memory_order_relaxed);
    }
}

// Solve each row's cutoff and how many tokens tied at it top-p drops.
// A token is kept when it is above the cutoff, or tied with it and past the
// first drop_ties tied tokens in vocabulary order. tie_prefix counts the tied
// tokens before each block, so blocks can be masked independently.
kernel void filter_select(
    const device float* logits [[buffer(0)]],
    const device int* top_k [[buffer(1)]],
    const device float* top_p [[buffer(2)]],
    const device float* min_p [[buffer(3)]],
    const device float* slice_max [[buffer(4)]],
    const device float* slice_sum [[buffer(5)]],
    const device uint* histogram [[buffer(6)]],
    device float* cutoff [[buffer(7)]],
    device uint* drop_ties [[buffer(8)]],
    device uint* tie_prefix [[buffer(9)]],
    constant long& vocab_size [[buffer(10)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const int vocab = int(vocab_size);
    const int slices = (vocab + kFilterSlice - 1) / kFilterSlice;
    const int blocks = (vocab + kFilterBlock - 1) / kFilterBlock;
    const uint row = group.y;
    const int k = top_k[row];
    const float p = top_p[row];
    const float mp = min_p[row];
    const bool use_k = filter_top_k_on(k, vocab);
    const bool use_p = p < 1.0f;
    const bool use_min_p = mp > 0.0f;
    if (!use_k && !use_p && !use_min_p) {
        if (tid == 0) { cutoff[row] = -INFINITY; drop_ties[row] = kFilterKeepTies; }
        return;
    }
    const float2 stats = filter_row_stats(slice_max, slice_sum, row, slices);
    const float m = stats.x;
    const float scale = 2147483648.0f / stats.y;
    float floor_value = -INFINITY;
    if (m == -INFINITY || (!use_k && !use_p)) {
        if (m != -INFINITY && use_min_p) floor_value = m + log(mp);
        if (tid == 0) { cutoff[row] = floor_value; drop_ties[row] = kFilterKeepTies; }
        return;
    }

    const device float* x = logits + ulong(row) * ulong(vocab);
    const device uint* counts = histogram + ulong(row) * ulong(2 * kFilterBins);
    const device uint* masses = counts + kFilterBins;
    threadgroup uint part_a[kFilterSimds * 16];
    threadgroup uint part_b[kFilterSimds * 16];
    threadgroup uint merged_a[16];
    threadgroup uint merged_b[16];
    threadgroup float gathered[kFilterGather];
    threadgroup atomic_uint gather_count;
    threadgroup uint shared_uint[4];
    threadgroup ulong shared_ulong[2];

    // Top-k: the bin holding the k-th largest, then the exact rank inside it.
    uint k_bin = kFilterBins;
    bool k_gathered = false;
    uint k_count = 0;
    if (use_k) {
        if (tid == 0) {
            uint cum = 0, b = 0;
            for (; b < kFilterBins - 1; ++b) {
                if (cum + counts[b] >= uint(k)) break;
                cum += counts[b];
            }
            shared_uint[0] = b;
            shared_uint[1] = uint(k) - cum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k_bin = shared_uint[0];
        const uint rank = shared_uint[1];
        k_count = counts[k_bin];
        uint key;
        if (k_count <= kFilterGather) {
            if (tid == 0) atomic_store_explicit(&gather_count, 0u, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int i = int(tid); i < vocab; i += int(kFilterThreads)) {
                float v = filter_clean(x[i]);
                if (filter_bin(v, m) == k_bin)
                    gathered[atomic_fetch_add_explicit(&gather_count, 1u, memory_order_relaxed)] = v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            key = filter_radix_kth(gathered, int(k_count), rank, part_a, merged_a, tid, lane, simd_group);
            k_gathered = true;
        } else {
            key = filter_radix_kth(x, vocab, uint(k), part_a, merged_a, tid, lane, simd_group);
        }
        floor_value = filter_value(key);
    }
    if (use_min_p) floor_value = max(floor_value, m + log(mp));
    if (!use_p) {
        if (tid == 0) { cutoff[row] = floor_value; drop_ties[row] = kFilterKeepTies; }
        return;
    }

    // Top-p over the survivors. Bins above the floor's bin survive whole.
    const uint floor_key = filter_key(floor_value);
    const uint floor_bin = floor_value > -INFINITY ? filter_bin(floor_value, m) : kFilterBins;
    uint partial_mass = 0;
    if (floor_bin < kFilterBins) {
        uint s = 0;
        if (k_gathered && floor_bin == k_bin) {
            for (int i = int(tid); i < int(k_count); i += int(kFilterThreads)) {
                float v = gathered[i];
                if (v >= floor_value) s += filter_mass(v, m, scale);
            }
        } else {
            for (int i = int(tid); i < vocab; i += int(kFilterThreads)) {
                float v = filter_clean(x[i]);
                if (filter_bin(v, m) == floor_bin && v >= floor_value) s += filter_mass(v, m, scale);
            }
        }
        partial_mass = filter_reduce1(s, part_a, lane, simd_group);
    }
    const uint p_fixed = uint(rint(p * 16777216.0f));
    if (tid == 0) {
        ulong survivors = floor_bin < kFilterBins ? ulong(partial_mass) : 0ul;
        const uint last = floor_bin < kFilterBins ? floor_bin : kFilterBins - 1;
        for (uint b = 0; b < last; ++b) survivors += masses[b];
        if (floor_bin >= kFilterBins) survivors += masses[kFilterBins - 1];
        const ulong target = survivors * ulong(p_fixed);
        ulong run = 0;
        uint b = 0;
        for (; b < last; ++b) {
            if (((run + ulong(masses[b])) << 24) >= target) break;
            run += masses[b];
        }
        shared_uint[2] = b;
        shared_ulong[0] = run;
        shared_ulong[1] = target;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint p_bin = shared_uint[2];
    ulong above = shared_ulong[0];
    const ulong target = shared_ulong[1];
    uint ties = 1, key;
    if (counts[p_bin] <= kFilterGather) {
        if (tid == 0) atomic_store_explicit(&gather_count, 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = int(tid); i < vocab; i += int(kFilterThreads)) {
            float v = filter_clean(x[i]);
            if (filter_bin(v, m) == p_bin && v >= floor_value)
                gathered[atomic_fetch_add_explicit(&gather_count, 1u, memory_order_relaxed)] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int n = int(atomic_load_explicit(&gather_count, memory_order_relaxed));
        key = filter_radix_mass(gathered, n, floor_key, m, scale, target, above, ties,
                                part_a, merged_a, part_b, merged_b, tid, lane, simd_group);
    } else {
        above = 0;
        key = filter_radix_mass(x, vocab, floor_key, m, scale, target, above, ties,
                                part_a, merged_a, part_b, merged_b, tid, lane, simd_group);
    }
    const float cut = filter_value(key);
    const ulong tie_mass = ulong(filter_mass(cut, m, scale)) << 24;
    uint drop = kFilterKeepTies;
    if (tie_mass > 0) {
        const ulong kept_ties = (target - (above << 24) + tie_mass - 1) / tie_mass;
        if (kept_ties < ulong(ties)) drop = ties - uint(max(kept_ties, 1ul));
    }
    if (tid == 0) { cutoff[row] = cut; drop_ties[row] = drop; }
    if (drop != kFilterKeepTies) {
        device uint* prefix = tie_prefix + ulong(row) * ulong(blocks);
        for (int b = int(simd_group); b < blocks; b += int(kFilterSimds)) {
            uint count = 0;
            for (int j = int(lane); j < kFilterBlock; j += int(kFilterSimds)) {
                int i = b * kFilterBlock + j;
                if (i < vocab && filter_clean(x[i]) == cut) count += 1;
            }
            count = simd_sum(count);
            if (lane == 0) prefix[b] = count;
        }
        threadgroup_barrier(mem_flags::mem_device);
        if (tid == 0) {
            uint run = 0;
            for (int b = 0; b < blocks; ++b) {
                uint count = prefix[b];
                prefix[b] = run;
                run += count;
            }
        }
    }
}

kernel void filter_mask(
    const device float* logits [[buffer(0)]],
    const device float* cutoff [[buffer(1)]],
    const device uint* drop_ties [[buffer(2)]],
    const device uint* tie_prefix [[buffer(3)]],
    device float* masked [[buffer(4)]],
    constant long& vocab_size [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const int vocab = int(vocab_size);
    const int blocks = (vocab + kFilterBlock - 1) / kFilterBlock;
    const uint row = group.y;
    const float cut = cutoff[row];
    const uint drop = drop_ties[row];
    const int col = int(group.x) * kFilterBlock + int(tid);
    const bool in_vocab = col < vocab;
    const float raw = in_vocab ? logits[ulong(row) * ulong(vocab) + col] : -INFINITY;
    const float v = filter_clean(raw);
    threadgroup uint simd_ties[kFilterSimds];
    bool keep = v >= cut;
    if (drop != kFilterKeepTies) {
        const uint tie = (in_vocab && v == cut) ? 1u : 0u;
        const uint before = simd_prefix_exclusive_sum(tie);
        const uint total = simd_sum(tie);
        if (lane == 0) simd_ties[simd_group] = total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint offset = 0;
        for (uint g = 0; g < simd_group; ++g) offset += simd_ties[g];
        const uint base = tie_prefix[ulong(row) * ulong(blocks) + group.x];
        keep = v > cut || (tie == 1u && base + offset + before >= drop);
    }
    if (in_vocab) masked[ulong(row) * ulong(vocab) + col] = keep ? raw : -INFINITY;
}
