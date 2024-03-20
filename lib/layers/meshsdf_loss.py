import torch
import meshsdf_loss_cuda


class MeshSDFLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, verts, faces, points):
        outputs = meshsdf_loss_cuda.forward(verts, faces, points)
        loss = outputs[0]
        dist = outputs[1]
        assoc = outputs[2]
        baryc = outputs[3]
        variables = [verts, faces, points, assoc, baryc]
        ctx.save_for_backward(*variables)
        return loss, dist, assoc

    @staticmethod
    def backward(ctx, grad_loss, grad_dist, grad_assoc):
        outputs = meshsdf_loss_cuda.backward(grad_loss, *ctx.saved_variables)
        d_verts = outputs[0]
        return d_verts, None, None


class MeshSDFLoss(torch.nn.Module):
    def __init__(self):
        super(MeshSDFLoss, self).__init__()

    def forward(self, verts, faces, points):
        return MeshSDFLossFunction.apply(verts, faces, points)
