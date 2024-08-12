#include <torch/extension.h>

namespace py = pybind11;

std::vector<torch::Tensor> meshsdf_loss_cuda_forward(torch::Tensor verts,
                                                     torch::Tensor faces,
                                                     torch::Tensor points);

std::vector<torch::Tensor> meshsdf_loss_cuda_backward(
    torch::Tensor grad_loss, torch::Tensor verts, torch::Tensor faces,
    torch::Tensor points, torch::Tensor assoc, torch::Tensor baryc);

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> meshsdf_loss_forward(torch::Tensor verts,
                                                torch::Tensor faces,
                                                torch::Tensor points) {
  CHECK_INPUT(verts);
  CHECK_INPUT(faces);
  CHECK_INPUT(points);

  return meshsdf_loss_cuda_forward(verts, faces, points);
}

std::vector<torch::Tensor> meshsdf_loss_backward(
    torch::Tensor grad_loss, torch::Tensor verts, torch::Tensor faces,
    torch::Tensor points, torch::Tensor assoc, torch::Tensor baryc) {
  CHECK_INPUT(grad_loss);
  CHECK_INPUT(verts);
  CHECK_INPUT(faces);
  CHECK_INPUT(points);
  CHECK_INPUT(assoc);
  CHECK_INPUT(baryc);

  return meshsdf_loss_cuda_backward(grad_loss, verts, faces, points, assoc,
                                    baryc);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &meshsdf_loss_forward, "MeshSDFLoss foward (CUDA)");
  m.def("backward", &meshsdf_loss_backward, "MeshSDFLoss backward (CUDA)");
}
