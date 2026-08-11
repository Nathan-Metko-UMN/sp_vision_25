#include "trt_nms_allocator.hpp"

#ifdef HAVE_TENSORRT
#include "tasks/auto_aim/yolos/trt_engine.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{

TrtNmsOutputAllocator::~TrtNmsOutputAllocator()
{
  if (device_buffer_) cudaFree(device_buffer_);
}

void TrtNmsOutputAllocator::allocate()
{
  uint64_t size = static_cast<uint64_t>(kMaxNmsOutputBoxes) * 3 * sizeof(int64_t);
  if (cudaMalloc(&device_buffer_, size) != cudaSuccess)
    throw std::runtime_error("TrtNmsOutputAllocator: cudaMalloc failed");
  device_buffer_capacity_ = size;
}

void TrtNmsOutputAllocator::reset_shape() { shape_known_ = false; }

int TrtNmsOutputAllocator::num_selected() const
{
  if (!shape_known_) {
    throw std::runtime_error(
      "TrtNmsOutputAllocator: notifyShape() was never called since the last reset_shape() -- "
      "the DDS output mechanism didn't behave as validated on real hardware; reading a count "
      "now would be reading stale/undefined state");
  }
  return static_cast<int>(final_shape_.d[0]);
}

void * TrtNmsOutputAllocator::reallocateOutputAsync(
  char const * tensorName, void * currentMemory, uint64_t size, uint64_t /*alignment*/,
  cudaStream_t /*stream*/) noexcept
{
  // Expected to always take the fast path: the fused graph caps
  // selected_indices at kMaxNmsOutputBoxes rows via max_output_boxes_per_class
  // (scripts/onnx/fuse_nms.py), and allocate() already pre-sized
  // device_buffer_ to that worst case, so `size` should never exceed
  // device_buffer_capacity_ in practice. Handled anyway since the interface
  // requires it. noexcept per IOutputAllocator's contract -- can't throw
  // here, so failures are logged (visible, loud) and signaled by returning
  // nullptr (TensorRT's documented failure signal), not by exception.
  if (size <= device_buffer_capacity_) return device_buffer_;

  tools::logger()->error(
    "TrtNmsOutputAllocator: '{}' requested {} bytes, larger than the pre-sized {} -- "
    "growing (this should not happen given max_output_boxes_per_class={})",
    tensorName, size, device_buffer_capacity_, kMaxNmsOutputBoxes);
  void * new_buffer = nullptr;
  if (cudaMalloc(&new_buffer, size) != cudaSuccess) {
    tools::logger()->error("TrtNmsOutputAllocator: cudaMalloc failed while growing '{}'", tensorName);
    return nullptr;
  }
  if (currentMemory) cudaFree(currentMemory);
  device_buffer_ = new_buffer;
  device_buffer_capacity_ = size;
  return device_buffer_;
}

void TrtNmsOutputAllocator::notifyShape(char const * /*tensorName*/, nvinfer1::Dims const & dims) noexcept
{
  final_shape_ = dims;
  shape_known_ = true;
}

}  // namespace auto_aim

#endif  // HAVE_TENSORRT
