import numpy as np
import torch
from torch.nn import Module


class ObjectLayer(Module):
    def __init__(self, verts: np.ndarray, faces: np.ndarray, normals: np.ndarray):
        """
        Initializes the object layer.

        Args:
            verts (np.ndarray): A numpy array of shape [N, 3] containing the vertices.
            faces (np.ndarray): A numpy array of shape [N, 3] containing the faces.
            normals (np.ndarray): A numpy array of shape [N, 3] containing the normals.
        """
        super().__init__()
        self._num_verts = verts.shape[0]

        # Convert numpy arrays to torch tensors
        v = torch.from_numpy(verts.astype(np.float32).T)
        n = torch.from_numpy(normals.astype(np.float32).T)
        f = torch.from_numpy(faces.astype(np.int64).reshape((-1, 3)))

        # Register buffers for vertices, normals, and faces
        self.register_buffer("v", v)
        self.register_buffer("n", n)
        self.register_buffer("f", f)

    def forward(
        self, r: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward function.

        Args:
            r (torch.Tensor): A tensor of shape [B, 3] containing the rotation in axis-angle.
            t (torch.Tensor): A tensor of shape [B, 3] containing the translation.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                v: A tensor of shape [B, N, 3] containing the transformed vertices.
                n: A tensor of shape [B, N, 3] containing the transformed normals.
        """
        R = self.rv2dcm(r)
        v = torch.matmul(R, self.v).permute(0, 2, 1) + t.unsqueeze(1)
        n = torch.matmul(R, self.n).permute(0, 2, 1)
        return v, n

    def rv2dcm(self, rv: torch.Tensor) -> torch.Tensor:
        """
        Converts rotation vectors to direction cosine matrices.

        Args:
            rv (torch.Tensor): A tensor of shape [B, 3] containing the rotation vectors.

        Returns:
            torch.Tensor: A tensor of shape [B, 3, 3] containing the direction cosine matrices.
        """
        angle = torch.norm(rv + 1e-8, p=2, dim=1)
        axis = rv / angle.unsqueeze(1)
        s = torch.sin(angle).unsqueeze(1).unsqueeze(2)
        c = torch.cos(angle).unsqueeze(1).unsqueeze(2)
        I = torch.eye(3, device=rv.device).expand(rv.size(0), -1, -1)
        z = torch.zeros_like(angle)
        K = torch.stack(
            (
                torch.stack((z, -axis[:, 2], axis[:, 1]), dim=1),
                torch.stack((axis[:, 2], z, -axis[:, 0]), dim=1),
                torch.stack((-axis[:, 1], axis[:, 0], z), dim=1),
            ),
            dim=1,
        )
        dcm = I + s * K + (1 - c) * torch.bmm(K, K)
        return dcm

    @property
    def num_verts(self) -> int:
        """Return the number of vertices."""
        return self._num_verts
