#ifndef AUTO_AIM__TRT_NMS_ALLOCATOR_HPP
#define AUTO_AIM__TRT_NMS_ALLOCATOR_HPP

#ifdef HAVE_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>

namespace auto_aim
{

// Implements IOutputAllocator for the fused-NMS engine's selected_indices
// output (see trt_engine.hpp / scripts/onnx/fuse_nms.py), which has a
// data-dependent shape (DDS): its row count (0..kMaxNmsOutputBoxes) is only
// known once NMS actually executes, not from input shapes alone --
// context->getTensorShape() after enqueueV3+sync does NOT report it
// (returns -1); TensorRT requires this callback interface for any DDS
// output instead.
//
// allocate() pre-sizes the device buffer to the worst case
// (kMaxNmsOutputBoxes rows) up front, at owner-controlled setup time --
// NOT lazily inside reallocateOutputAsync(). Because the fused graph caps
// the real output at kMaxNmsOutputBoxes via max_output_boxes_per_class,
// reallocateOutputAsync should therefore see a requested size it already
// satisfies on every call, and in steady state never actually reallocates.
//
// One instance per concurrently-in-flight inference: YOLOV5 uses one
// instance (single-buffered/synchronous); MultiThreadDetector uses one per
// TrtSlot (see mt_detector.hpp/.cpp) -- each slot's context->
// setOutputAllocator() call is re-pointed to that slot's own instance
// immediately before that slot's enqueueV3(), mirroring how
// setTensorAddress() is already re-pointed per-slot.
class TrtNmsOutputAllocator : public nvinfer1::IOutputAllocator
{
public:
  ~TrtNmsOutputAllocator() override;

  // Pre-sizes the device buffer. Call once, after construction, before the
  // first enqueueV3 this allocator will be registered for.
  void allocate();

  // Call immediately before each enqueueV3 this allocator is registered
  // for, so num_selected() can detect (and refuse to return) a stale count
  // left over from a previous frame if notifyShape unexpectedly doesn't
  // fire this time.
  void reset_shape();

  // Row count from the most recent notifyShape() call. Only valid to read
  // after the enqueueV3 that triggered it has been confirmed complete on
  // its stream (cudaStreamSynchronize / cudaEventSynchronize on an event
  // recorded after that enqueueV3) -- mirrors how the output_host buffers
  // elsewhere in this codebase are only read after their own D2H-copy sync.
  // Throws if notifyShape never fired since the last reset_shape() -- a
  // real invariant violation (the DDS mechanism didn't behave as
  // validated), not defensive boilerplate.
  int num_selected() const;

  void * device_buffer() const { return device_buffer_; }

  void * reallocateOutputAsync(
    char const * tensorName, void * currentMemory, uint64_t size, uint64_t alignment,
    cudaStream_t stream) noexcept override;
  void notifyShape(char const * tensorName, nvinfer1::Dims const & dims) noexcept override;

private:
  void * device_buffer_ = nullptr;
  uint64_t device_buffer_capacity_ = 0;
  nvinfer1::Dims final_shape_{};
  bool shape_known_ = false;
};

}  // namespace auto_aim

#endif  // HAVE_TENSORRT
#endif  // AUTO_AIM__TRT_NMS_ALLOCATOR_HPP
