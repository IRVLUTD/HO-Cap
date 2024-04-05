import torch
import math
import numpy as np
from sklearn.neighbors import KDTree


class PointSDFFunction(torch.autograd.Function):
    @staticmethod
    def forward(query_points, points, normals):

        sample_count = 11
        device = query_points.device
        numpy_points = points.detach().numpy()
        numpy_normals = normals.detach().numpy()
        kd_tree = KDTree(numpy_points)

        numpy_query_points = query_points.detach().numpy()
        distances, indices = kd_tree.query(numpy_query_points, k=sample_count)
        distances = distances.astype(np.float32)
        closest_points = numpy_points[indices]
        direction_from_surface = numpy_query_points[:, np.newaxis, :] - closest_points
        inside = np.einsum('ijk,ijk->ij', direction_from_surface, numpy_normals[indices]) < 0
        inside = np.sum(inside, axis=1) > sample_count * 0.5
        distances = distances[:, 0]
        distances[inside] *= -1

        # Any intermediates to be saved in backward must be returned as
        # outputs.
        return (
            # The desired output
            torch.tensor(distances, device=device),
            # intermediate to save for backward
            torch.tensor(indices, device=device),
            # intermediate to save for backward
            torch.tensor(direction_from_surface, device=device),
            torch.tensor(inside, device=device),
        )
    

    # setup_context is responsible for calling methods and/or assigning to
    # the ctx object. Please do not do additional compute (e.g. add
    # Tensors together) in setup_context.
    @staticmethod
    def setup_context(ctx, inputs, output):
        query_points, points, normals = inputs
        # Note that output is whatever you returned from forward.
        # If you returned multiple values, then output is a Tuple of multiple values.
        # If you returned a single Tensor, then output is a Tensor.
        # If you returned a Tuple with a single Tensor, then output is a
        # Tuple with a single Tensor.
        distances, indices, direction_from_surface, inside = output
        # ctx.mark_non_differentiable(indices, direction_from_surface, inside)
        # Tensors must be saved via ctx.save_for_backward. Please do not
        # assign them directly onto the ctx object.
        ctx.save_for_backward(distances, indices, direction_from_surface, inside, normals)


    # @staticmethod
    # def backward(ctx, grad_out, _0, _1, _2):
    #     distances, indices, direction_from_surface, inside, normals = ctx.saved_tensors
    #     gradients = direction_from_surface[:, 0].clone()
    #     gradients[inside] *= -1
    #     near_surface = torch.abs(distances) < math.sqrt(0.0025**2 * 3) * 3 # 3D 2-norm stdev * 3
    #     gradients = torch.where(near_surface[:, None], normals[indices[:, 0]], gradients)
    #     gradients /= torch.linalg.norm(gradients, axis=1)[:, None]
    #     gradients = torch.mul(grad_out[:, None], gradients)
    #     return gradients, None, None
    


    @staticmethod
    def backward(ctx, grad_out, _0, _1, _2):
        distances, indices, direction_from_surface, inside, normals = ctx.saved_tensors
        return PointSDFFunctionBackward.apply(grad_out, distances, indices, direction_from_surface, inside, normals)


class PointSDFFunctionBackward(torch.autograd.Function):
    @staticmethod
    def forward(grad_out, distances, indices, direction_from_surface, inside, normals):
        gradients = direction_from_surface[:, 0].clone()
        gradients[inside] *= -1
        near_surface = torch.abs(distances) < math.sqrt(0.0025**2 * 3) * 3 # 3D 2-norm stdev * 3
        gradients = torch.where(near_surface[:, None], normals[indices[:, 0]], gradients)
        gradients /= torch.linalg.norm(gradients, axis=1)[:, None]
        gradients = torch.mul(grad_out[:, None], gradients)
        return gradients, None, None
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        grad_out, distances, indices, direction_from_surface, inside, normals = inputs
        gradients, _, _ = output
        ctx.save_for_backward(distances, indices, direction_from_surface, inside, normals)    
    
    @staticmethod
    def backward(ctx, grad_out, _0, _1):    
        distances, indices, direction_from_surface, inside, normals = ctx.saved_tensors
        return None, None, None, None, None, None
    

    # # The signature of the vmap staticmethod is:
    # # vmap(info, in_dims: Tuple[Optional[int]], *args)
    # # where *args is the same as the arguments to `forward`.
    # @staticmethod
    # def vmap(info, in_dims, query_points, points, normals):
    #     # For every input (x and dim), in_dims stores an Optional[int]
    #     # that is:
    #     # - None if the input is not being vmapped over or if the input
    #     #   is not a Tensor
    #     # - an integer if the input is being vmapped over that represents
    #     #   the index of the dimension being vmapped over.
    #     q_bdim, p_bdim, n_bdim = in_dims

    #     # A "vmap rule" is the logic of how to perform the operation given
    #     # inputs with one additional dimension. In NumpySort, x has an
    #     # additional dimension (x_bdim). The vmap rule is simply
    #     # to call NumpySort again but pass it a different `dim`.
    #     # x = x.movedim(x_bdim, 0)
    #     # Handle negative dims correctly
    #     # dim = dim if dim >= 0 else dim + x.dim() - 1
    #     # result = NumpySort.apply(x, dim + 1)

    #     query_points = query_points.movedim(q_bdim, 0)
    #     a = []
    #     b = []
    #     c = []
    #     d = []
    #     for i in range(query_points.shape[0]):
    #         distances, indices, direction_from_surface, inside = PointSDFFunction.apply(query_points[i], points, normals)
    #         a.append(distances)
    #         b.append(indices)
    #         c.append(direction_from_surface)
    #         d.append(inside)

    #     output = (torch.stack(a), torch.stack(b), torch.stack(c), torch.stack(d))

    #     # The vmap rule must return a tuple of two things
    #     # 1. the output. Should be the same amount of things
    #     #    as returned by the forward().
    #     # 2. one Optional[int] for each output specifying if each output
    #     # is being vmapped over, and if so, the index of the
    #     # dimension being vmapped over.
    #     #
    #     # NumpySort.forward returns a Tuple of 3 Tensors. Since we moved the
    #     # dimension being vmapped over to the front of `x`, that appears at
    #     # dimension 0 of all outputs.
    #     # The return is (output, out_dims) -- output is a tuple of 3 Tensors
    #     # and out_dims is a Tuple of 3 Optional[int]
    #     return output, (0, 0, 0, 0)


class PointSDFLayer(torch.nn.Module):
    def __init__(self, points, normals):
        super(PointSDFLayer, self).__init__()

        p = torch.from_numpy(points.astype(np.float32))
        n = torch.from_numpy(normals.astype(np.float32))

        # register buffer for points and normals
        self.register_buffer("p", p)
        self.register_buffer("n", n)

    def forward(self, query_points, sample_count=11):
        distances, _, _, _ = PointSDFFunction.apply(query_points, self.p, self.n)
        return distances