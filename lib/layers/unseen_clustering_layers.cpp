#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda())
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous())
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


/************************************************************
 hard label layer
*************************************************************/
std::vector<at::Tensor> hard_label_cuda_forward(
    float threshold,
    float sample_percentage,
    at::Tensor bottom_prob,
    at::Tensor bottom_label,
    at::Tensor bottom_rand);

std::vector<at::Tensor> hard_label_cuda_backward(
    at::Tensor top_diff);

std::vector<at::Tensor> hard_label_forward(
    float threshold,
    float sample_percentage,
    at::Tensor bottom_prob,
    at::Tensor bottom_label,
    at::Tensor bottom_rand)
{
  CHECK_INPUT(bottom_prob);
  CHECK_INPUT(bottom_label);
  CHECK_INPUT(bottom_rand);

  return hard_label_cuda_forward(threshold, sample_percentage, bottom_prob, bottom_label, bottom_rand);
}

std::vector<at::Tensor> hard_label_backward(
    at::Tensor top_diff) {
  CHECK_INPUT(top_diff);

  return hard_label_cuda_backward(top_diff);
}

/********* python interface ***********/

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hard_label_forward", &hard_label_forward, "hard_label forward (CUDA)");
  m.def("hard_label_backward", &hard_label_backward, "hard_label backward (CUDA)");
}
