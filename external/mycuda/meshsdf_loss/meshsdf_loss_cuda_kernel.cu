#include <torch/extension.h>
#include "rbd/bvh.h"

struct DistancePointQuery {
  __device__ inline DistancePointQuery(const float* verts, const int64_t* faces, const Vec3& p)
      : verts(verts), faces(faces), query_point(p) {}

  __device__ inline void operator()(int leaf, float& min_dist) {
    int i = faces[leaf * 3 + 0];
    int j = faces[leaf * 3 + 1];
    int k = faces[leaf * 3 + 2];

    Vec3 p = Vec3(verts + 3 * i);
    Vec3 q = Vec3(verts + 3 * j);
    Vec3 r = Vec3(verts + 3 * k);

    float v, w;
    Vec3 c = ClosestPointOnTriangle(p, q, r, query_point, v, w);

    float dist = Length(c - query_point);

    if (dist < min_dist) {
      min_dist = dist;
      best_dist = dist;
      best_v = v;
      best_w = w;
      best_assoc = leaf;
    }
  }

  const float* verts;
  const int64_t* faces;

  Vec3 query_point;
  float best_dist;
  float best_v;
  float best_w;
  int best_assoc;
};

__global__ void meshsdf_loss_cuda_forward_kernel(
    BVH bvh, const float* verts, const int64_t* faces, const float* points,
    size_t num_p, float* loss, float* dist, int64_t* assoc, float* baryc) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_p) {
    Vec3 p = Vec3(points + 3 * i);

    DistancePointQuery point_query(verts, faces, p);

    int stack[32];
    QueryBVH(bvh, p, FLT_MAX, point_query, stack);

    dist[i] = point_query.best_dist;
    assoc[i] = point_query.best_assoc;

    float v = point_query.best_v;
    float w = point_query.best_w;
    float u = 1 - v - w;

    baryc[3 * i + 0] = u;
    baryc[3 * i + 1] = v;
    baryc[3 * i + 2] = w;

    const float* a = verts + 3 * faces[3 * assoc[i] + 0];
    const float* b = verts + 3 * faces[3 * assoc[i] + 1];
    const float* c = verts + 3 * faces[3 * assoc[i] + 2];

    float dx = points[3 * i + 0] - (u * a[0] + v * b[0] + w * c[0]);
    float dy = points[3 * i + 1] - (u * a[1] + v * b[1] + w * c[1]);
    float dz = points[3 * i + 2] - (u * a[2] + v * b[2] + w * c[2]);
    float d = dx * dx + dy * dy + dz * dz;

    atomicAdd(loss, d);
  }
}

__global__ void compute_bounds_cuda_kernel(const float* verts,
                                           const int64_t* faces, size_t num_f,
                                           Bounds* bounds) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_f) {
    // Set the value of lower/upper in place of the constructor.
    bounds[i].lower = +FLT_MAX;
    bounds[i].upper = -FLT_MAX;

    Vec3 a = Vec3(verts + 3 * faces[i * 3 + 0]);
    Vec3 b = Vec3(verts + 3 * faces[i * 3 + 1]);
    Vec3 c = Vec3(verts + 3 * faces[i * 3 + 2]);
    bounds[i] = Union(bounds[i], a);
    bounds[i] = Union(bounds[i], b);
    bounds[i] = Union(bounds[i], c);
  }
}

void compute_bounds_cuda(torch::Tensor verts, torch::Tensor faces,
                         Bounds* h_bounds) {
  const auto num_f = faces.size(0);

  // cudaMalloc() does not run the constructor. lower/upper will be 0 initially.
  Bounds* d_bounds;
  cudaMalloc((void**)&d_bounds, sizeof(Bounds) * num_f);

  const int threads = 512;
  const int blocks = (num_f + threads - 1) / threads;

  compute_bounds_cuda_kernel<<<blocks, threads>>>(
      verts.data_ptr<float>(), faces.data_ptr<int64_t>(), num_f, d_bounds);

  cudaMemcpy(h_bounds, d_bounds, sizeof(Bounds) * num_f,
             cudaMemcpyDeviceToHost);
  cudaFree(d_bounds);
}

std::vector<torch::Tensor> meshsdf_loss_cuda_forward(torch::Tensor verts,
                                                     torch::Tensor faces,
                                                     torch::Tensor points) {
  const auto num_f = faces.size(0);
  const auto num_p = points.size(0);

  // Compute bounds on CUDA.
  Bounds* bounds = new Bounds[num_f];
  compute_bounds_cuda(verts, faces, bounds);

  // Build BVH on CPU.
  BVH h_bvh;
  MedianBVHBuilder builder;
  builder.Build(h_bvh, bounds, num_f);
  delete bounds;

  // Copy BVH to CUDA.
  BVH d_bvh;
  InitBVH(d_bvh);
  CloneBVH(h_bvh, d_bvh);
  FreeBVHHost(h_bvh);

  auto loss = torch::zeros({}, points.options());
  auto dist = torch::zeros({num_p}, points.options());
  auto assoc = torch::zeros({num_p}, faces.options());
  auto baryc = torch::zeros({num_p, 3}, verts.options());

  const int threads = 512;
  const int blocks = (num_p + threads - 1) / threads;

  meshsdf_loss_cuda_forward_kernel<<<blocks, threads>>>(
      d_bvh, verts.data_ptr<float>(), faces.data_ptr<int64_t>(),
      points.data_ptr<float>(), num_p, loss.data_ptr<float>(),
      dist.data_ptr<float>(), assoc.data_ptr<int64_t>(),
      baryc.data_ptr<float>());

  FreeBVH(d_bvh);

  return {loss, dist, assoc, baryc};
}

__global__ void meshsdf_loss_cuda_backward_kernel(
    const float* grad_loss, const float* verts, const int64_t* faces,
    const float* points, const int64_t* assoc, const float* baryc, size_t num_p,
    float* d_verts) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_p) {
    float u = baryc[3 * i + 0];
    float v = baryc[3 * i + 1];
    float w = baryc[3 * i + 2];

    const float* a = verts + 3 * faces[3 * assoc[i] + 0];
    const float* b = verts + 3 * faces[3 * assoc[i] + 1];
    const float* c = verts + 3 * faces[3 * assoc[i] + 2];

    float dx = points[3 * i + 0] - (u * a[0] + v * b[0] + w * c[0]);
    float dy = points[3 * i + 1] - (u * a[1] + v * b[1] + w * c[1]);
    float dz = points[3 * i + 2] - (u * a[2] + v * b[2] + w * c[2]);

    atomicAdd(d_verts + 3 * faces[3 * assoc[i] + 0] + 0,
              -2 * u * dx * grad_loss[0]);
    atomicAdd(d_verts + 3 * faces[3 * assoc[i] + 0] + 1,
              -2 * u * dy * grad_loss[0]);
    atomicAdd(d_verts + 3 * faces[3 * assoc[i] + 0] + 2,
              -2 * u * dz * grad_loss[0]);
    atomicAdd(d_verts + 3 * faces[3 * assoc[i] + 1] + 0,
              -2 * v * dx * grad_loss[0]);
    atomicAdd(d_verts + 3 * faces[3 * assoc[i] + 1] + 1,
              -2 * v * dy * grad_loss[0]);
    atomicAdd(d_verts + 3 * faces[3 * assoc[i] + 1] + 2,
              -2 * v * dz * grad_loss[0]);
    atomicAdd(d_verts + 3 * faces[3 * assoc[i] + 2] + 0,
              -2 * w * dx * grad_loss[0]);
    atomicAdd(d_verts + 3 * faces[3 * assoc[i] + 2] + 1,
              -2 * w * dy * grad_loss[0]);
    atomicAdd(d_verts + 3 * faces[3 * assoc[i] + 2] + 2,
              -2 * w * dz * grad_loss[0]);
  }
}

std::vector<torch::Tensor> meshsdf_loss_cuda_backward(
    torch::Tensor grad_loss, torch::Tensor verts, torch::Tensor faces,
    torch::Tensor points, torch::Tensor assoc, torch::Tensor baryc) {
  const auto num_p = points.size(0);

  const auto d_verts = torch::zeros_like(verts);

  const int threads = 512;
  const int blocks = (num_p + threads - 1) / threads;

  meshsdf_loss_cuda_backward_kernel<<<blocks, threads>>>(
      grad_loss.data_ptr<float>(), verts.data_ptr<float>(),
      faces.data_ptr<int64_t>(), points.data_ptr<float>(),
      assoc.data_ptr<int64_t>(), baryc.data_ptr<float>(), num_p,
      d_verts.data_ptr<float>());

  return {d_verts};
}
