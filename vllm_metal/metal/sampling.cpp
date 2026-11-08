// SPDX-License-Identifier: Apache-2.0
// MLX graph adapter for the same Metal sampling shaders used by PyTorch MPS.
#include <cstring>

#include "ops_internal.h"

constexpr int64_t kSamplingBlockSize = 1024;
constexpr int kFilterSliceSize = 16384;
constexpr int kFilterBins = 1024;
constexpr int kFilterThreads = 1024;

class GumbelSamplePrimitive : public Primitive {
 public:
  GumbelSamplePrimitive(Stream stream, bool indexed)
      : Primitive(stream), indexed_(indexed) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("GumbelSamplePrimitive only supports GPU");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    for (auto& out : outputs) out.set_data(allocator::malloc(out.nbytes()));
    auto s = stream();
    auto& device = metal::device(s.device);
    auto* lib = device.get_library("sampling_kern");
    auto* kernel = device.get_kernel("gumbel_sample", lib, "gumbel_sample", {});
    auto& enc = metal::get_command_encoder(s);
    enc.set_compute_pipeline_state(kernel);
    for (int i = 0; i < 5; ++i) enc.set_input_array(inputs[i], i);
    enc.set_output_array(outputs[0], 5);
    enc.set_output_array(outputs[1], 6);
    const auto& logits = inputs[0];
    int64_t layout[] = {logits.shape(1), logits.strides()[0], logits.strides()[1]};
    int64_t blocks = outputs[0].shape(1);
    int64_t apply_temperature = 1, indexed = indexed_;
    enc.set_bytes(layout, 3, 7);
    enc.set_bytes(kSamplingBlockSize, 8);
    enc.set_bytes(blocks, 9);
    enc.set_bytes(apply_temperature, 10);
    enc.set_input_array(inputs[5], 11);
    enc.set_bytes(indexed, 12);
    enc.dispatch_threadgroups(
        MTL::Size::Make(blocks, inputs[0].shape(0), 1), MTL::Size::Make(256, 1, 1));
  }

  const char* name() const override { return "GumbelSamplePrimitive"; }
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const GumbelSamplePrimitive&>(other);
    return indexed_ == rhs.indexed_;
  }

 private:
  bool indexed_;
};

array gumbel_sample_primitive_fn(
    const array& logits, const array& seeds, const array& positions,
    const array& temperatures, const std::optional<array>& token_ids) {
  if (logits.ndim() != 2 || logits.dtype() != float32 || logits.shape(0) == 0 || logits.shape(1) == 0)
    throw std::invalid_argument("gumbel_sample: expected nonempty float32 [rows, vocab] logits");
  int rows = logits.shape(0);
  int blocks = (logits.shape(1) + kSamplingBlockSize - 1) / kSamplingBlockSize;
  for (const auto* value : {&seeds, &positions, &temperatures})
    if (value->ndim() != 1 || value->shape(0) != rows)
      throw std::invalid_argument("gumbel_sample: one seed, position and temperature per row required");
  if (seeds.dtype() != int64 || positions.dtype() != int64 || temperatures.dtype() != float32)
    throw std::invalid_argument("gumbel_sample: expected int64 seeds/positions and float32 temperatures");
  if (token_ids && (token_ids->shape() != logits.shape() || token_ids->dtype() != uint32))
    throw std::invalid_argument("gumbel_sample: candidate IDs must be uint32 with the logits shape");
  auto s = default_stream(Device::gpu);
  auto primitive = std::make_shared<GumbelSamplePrimitive>(s, token_ids.has_value());
  auto outputs = array::make_arrays(
      {{rows, blocks}, {rows, blocks}}, {int64, float32}, primitive,
      {logits, arange(rows, int64), contiguous(seeds),
       contiguous(positions), contiguous(temperatures),
       token_ids ? contiguous(*token_ids) : zeros({1}, uint32)});
  auto& ids = outputs[0];
  auto& scores = outputs[1];
  auto best = max(scores, 1, true, s);
  // Candidate order may differ from vocabulary order. Break ties by token ID.
  auto winning_ids = where(equal(scores, best, s), ids, array(INT64_MAX), s);
  return min(winning_ids, 1, false, s);
}

namespace {

enum class FilterStage { stats, histogram, select, mask };

class SamplingFilterPrimitive : public Primitive {
 public:
  SamplingFilterPrimitive(Stream stream, FilterStage stage)
      : Primitive(stream), stage_(stage) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("SamplingFilterPrimitive only supports GPU");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    for (auto& out : outputs) out.set_data(allocator::malloc(out.nbytes()));
    // Histogram bins merge through atomic adds, so they must start at zero.
    if (stage_ == FilterStage::histogram) std::memset(outputs[0].data<void>(), 0, outputs[0].nbytes());
    auto s = stream();
    auto& device = metal::device(s.device);
    auto* lib = device.get_library("sampling_kern");
    auto* kernel = device.get_kernel(kernel_name(), lib, kernel_name(), {});
    auto& enc = metal::get_command_encoder(s);
    enc.set_compute_pipeline_state(kernel);
    int slot = 0;
    for (const auto& in : inputs) enc.set_input_array(in, slot++);
    for (auto& out : outputs) enc.set_output_array(out, slot++);
    int64_t vocab = inputs[0].shape(1);
    enc.set_bytes(vocab, slot);
    int columns = stage_ == FilterStage::select ? 1
                  : stage_ == FilterStage::mask ? (vocab + kSamplingBlockSize - 1) / kSamplingBlockSize
                                                : (vocab + kFilterSliceSize - 1) / kFilterSliceSize;
    enc.dispatch_threadgroups(
        MTL::Size::Make(columns, inputs[0].shape(0), 1), MTL::Size::Make(kFilterThreads, 1, 1));
  }

  const char* name() const override { return "SamplingFilterPrimitive"; }
  bool is_equivalent(const Primitive& other) const override {
    return stage_ == static_cast<const SamplingFilterPrimitive&>(other).stage_;
  }

 private:
  std::string kernel_name() const {
    switch (stage_) {
      case FilterStage::stats: return "filter_stats";
      case FilterStage::histogram: return "filter_histogram";
      case FilterStage::select: return "filter_select";
      case FilterStage::mask: return "filter_mask";
    }
    return "";
  }

  FilterStage stage_;
};

}  // namespace

array filter_top_k_top_p_primitive_fn(
    const array& logits, const array& top_k, const array& top_p, const array& min_p) {
  if (logits.ndim() != 2 || logits.dtype() != float32 || logits.shape(0) == 0 || logits.shape(1) == 0)
    throw std::invalid_argument("filter_top_k_top_p: expected nonempty float32 [rows, vocab] logits");
  int rows = logits.shape(0);
  int vocab = logits.shape(1);
  for (const auto* value : {&top_k, &top_p, &min_p})
    if (value->ndim() != 1 || value->shape(0) != rows)
      throw std::invalid_argument("filter_top_k_top_p: one top_k, top_p and min_p per row required");
  if (top_k.dtype() != int32 || top_p.dtype() != float32 || min_p.dtype() != float32)
    throw std::invalid_argument("filter_top_k_top_p: expected int32 top_k and float32 top_p/min_p");
  auto s = default_stream(Device::gpu);
  int slices = (vocab + kFilterSliceSize - 1) / kFilterSliceSize;
  int blocks = (vocab + kSamplingBlockSize - 1) / kSamplingBlockSize;
  auto stage = [&](FilterStage which, std::vector<Shape> shapes, std::vector<Dtype> dtypes,
                   std::vector<array> inputs) {
    return array::make_arrays(
        std::move(shapes), std::move(dtypes),
        std::make_shared<SamplingFilterPrimitive>(s, which), std::move(inputs));
  };
  auto x = contiguous(logits);
  auto k = contiguous(top_k);
  auto p = contiguous(top_p);
  auto mp = contiguous(min_p);
  auto stats = stage(FilterStage::stats, {{rows, slices}, {rows, slices}}, {float32, float32},
                     {x, k, p, mp});
  auto histogram = stage(FilterStage::histogram, {{rows, 2 * kFilterBins}}, {uint32},
                         {x, k, p, stats[0], stats[1]});
  auto selected = stage(FilterStage::select, {{rows}, {rows}, {rows, blocks}},
                        {float32, uint32, uint32},
                        {x, k, p, mp, stats[0], stats[1], histogram[0]});
  return stage(FilterStage::mask, {{rows, vocab}}, {float32},
               {x, selected[0], selected[1], selected[2]})[0];
}
