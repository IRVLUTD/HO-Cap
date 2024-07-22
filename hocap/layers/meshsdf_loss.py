# import torch
# import meshsdf_loss_cuda


# class MeshSDFLossFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, verts, faces, points):
#         outputs = meshsdf_loss_cuda.forward(verts, faces, points)
#         loss = outputs[0]
#         dist = outputs[1]
#         assoc = outputs[2]
#         baryc = outputs[3]
#         variables = [verts, faces, points, assoc, baryc]
#         ctx.save_for_backward(*variables)
#         return loss, dist, assoc

#     @staticmethod
#     def backward(ctx, grad_loss, grad_dist, grad_assoc):
#         outputs = meshsdf_loss_cuda.backward(grad_loss, *ctx.saved_variables)
#         d_verts = outputs[0]
#         return d_verts, None, None


# class MeshSDFLoss(torch.nn.Module):
#     def __init__(self):
#         super(MeshSDFLoss, self).__init__()

#     def forward(self, verts, faces, points):
#         return MeshSDFLossFunction.apply(verts, faces, points)

import torch
import meshsdf_loss_cuda


class MeshSDFLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, verts, faces, points):
        """
        Forward pass for the MeshSDFLoss function.

        Args:
            ctx: Context object to save information for backward computation.
            verts (torch.Tensor): Tensor of shape [V, 3] containing the vertices.
            faces (torch.Tensor): Tensor of shape [F, 3] containing the faces.
            points (torch.Tensor): Tensor of shape [P, 3] containing the points.

        Returns:
            tuple: Loss, distances, and associations.
        """
        outputs = meshsdf_loss_cuda.forward(verts, faces, points)
        loss, dist, assoc, baryc = outputs
        variables = [verts, faces, points, assoc, baryc]
        ctx.save_for_backward(*variables)
        return loss, dist, assoc

    @staticmethod
    def backward(ctx, grad_loss, grad_dist=None, grad_assoc=None):
        """
        Backward pass for the MeshSDFLoss function.

        Args:
            ctx: Context object with saved variables.
            grad_loss (torch.Tensor): Gradient of the loss.
            grad_dist (torch.Tensor, optional): Gradient of the distances.
            grad_assoc (torch.Tensor, optional): Gradient of the associations.

        Returns:
            tuple: Gradients with respect to the inputs.
        """
        outputs = meshsdf_loss_cuda.backward(grad_loss, *ctx.saved_variables)
        d_verts = outputs[0]
        return d_verts, None, None


class MeshSDFLoss(torch.nn.Module):
    def __init__(self):
        """
        Initializes the MeshSDFLoss module.
        """
        super(MeshSDFLoss, self).__init__()

    def forward(self, verts, faces, points):
        """
        Forward pass for the MeshSDFLoss module.

        Args:
            verts (torch.Tensor): Tensor of shape [V, 3] containing the vertices.
            faces (torch.Tensor): Tensor of shape [F, 3] containing the faces.
            points (torch.Tensor): Tensor of shape [P, 3] containing the points.

        Returns:
            tuple: Loss, distances, and associations.
        """
        return MeshSDFLossFunction.apply(verts, faces, points)
