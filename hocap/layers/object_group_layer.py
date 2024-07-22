import torch
from torch.nn import Module, ModuleList
from .object_layer import ObjectLayer
import numpy as np


class ObjectGroupLayer(Module):
    """Wrapper layer to hold a group of ObjectLayers."""

    def __init__(
        self,
        verts: list[np.ndarray],
        faces: list[np.ndarray],
        normals: list[np.ndarray],
    ):
        """
        Constructor.

        Args:
            verts (list[np.ndarray]): A list of numpy arrays of shape [N, 3] containing the vertices.
            faces (list[np.ndarray]): A list of numpy arrays of shape [N, 3] containing the faces.
            normals (list[np.ndarray]): A list of numpy arrays of shape [N, 3] containing the normals.
        """
        super(ObjectGroupLayer, self).__init__()

        self._layers = ModuleList(
            [ObjectLayer(v, f, n) for v, f, n in zip(verts, faces, normals)]
        )
        self._num_obj = len(verts)
        self._num_verts = [v.shape[0] for v in verts]

        # Initialize faces with offsets
        f = []
        offset = 0
        for i in range(self._num_obj):
            if i > 0:
                offset += self._layers[i - 1].v.size(1)
            f.append(self._layers[i].f + offset)
        f = torch.cat(f)
        self.register_buffer("f", f)

    @property
    def num_obj(self) -> int:
        """Return the number of objects."""
        return self._num_obj

    @property
    def num_verts(self) -> list[int]:
        """Return the number of vertices for each object."""
        return self._num_verts

    @property
    def count(self) -> list[int]:
        """Return the number of faces for each object."""
        return [l.f.numel() for l in self._layers]

    def forward(
        self, p: torch.Tensor, inds: list[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward function.

        Args:
            p (torch.Tensor): A tensor of shape [B, D] containing the pose vectors.
            inds (list[int], optional): A list of sub-layer indices. Default is None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                v: A tensor of shape [B, N, 3] containing the transformed vertices.
                n: A tensor of shape [B, N, 3] containing the transformed normals.
        """
        if inds is None:
            inds = range(self._num_obj)
        v = [torch.zeros((p.size(0), 0, 3), dtype=torch.float32, device=self.f.device)]
        n = [torch.zeros((p.size(0), 0, 3), dtype=torch.float32, device=self.f.device)]
        r, t = self.pose2rt(p)
        for i in inds:
            y = self._layers[i](r[:, i], t[:, i])
            v.append(y[0])
            n.append(y[1])
        v = torch.cat(v, dim=1)
        n = torch.cat(n, dim=1)
        return v, n

    def pose2rt(self, pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts rotations and translations from pose vectors.

        Args:
            pose (torch.Tensor): A tensor of shape [B, D] containing the pose vectors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                r: A tensor of shape [B, O, 3] containing the rotation vectors.
                t: A tensor of shape [B, O, 3] containing the translations.
        """
        r = torch.stack(
            [pose[:, 6 * i : 6 * i + 3] for i in range(self._num_obj)], dim=1
        )
        t = torch.stack(
            [pose[:, 6 * i + 3 : 6 * i + 6] for i in range(self._num_obj)], dim=1
        )
        return r, t

    def get_f_from_inds(self, inds: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gets faces from sub-layer indices.

        Args:
            inds (list[int]): A list of sub-layer indices.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                f: A tensor of shape [F, 3] containing the faces.
                m: A tensor of shape [F] containing the face to index mapping.
        """
        f = [torch.zeros((0, 3), dtype=self.f.dtype, device=self.f.device)]
        m = [torch.zeros((0,), dtype=torch.int64, device=self.f.device)]
        offset = 0
        for i, x in enumerate(inds):
            if i > 0:
                offset += self._layers[inds[i - 1]].v.size(1)
            f.append(self._layers[x].f + offset)
            m.append(
                x
                * torch.ones(
                    self._layers[x].f.size(0), dtype=torch.int64, device=self.f.device
                )
            )
        f = torch.cat(f)
        m = torch.cat(m)
        return f, m

    def get_num_verts_from_inds(self, inds: list[int]) -> int:
        """
        Gets number of vertices from sub-layer indices.

        Args:
            inds (list[int]): A non-empty list of sub-layer indices.

        Returns:
            int: The number of vertices.
        """
        return sum(self._layers[i].v.size(1) for i in inds)

    def get_vert_inds_from_inds(
        self, inds: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gets vertices from sub-layer indices.

        Args:
            inds (list[int]): A list of sub-layer indices.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                idx: A tensor of shape [N] containing the vertices.
                m: A tensor of shape [N] containing the vertex to index mapping.
        """
        idx = [torch.zeros((0,), dtype=torch.int64, device=self.f.device)]
        m = [torch.zeros((0,), dtype=torch.int64, device=self.f.device)]
        offset = 0
        for i in range(self._num_obj):
            if i > 0:
                offset += self._layers[i - 1].v.size(1)
            idx.append(
                torch.arange(self._layers[i].v.size(1), device=self.f.device) + offset
            )
            m.append(
                i
                * torch.ones(
                    self._layers[i].v.size(1), dtype=torch.int64, device=self.f.device
                )
            )
        idx = torch.cat(idx)
        m = torch.cat(m)
        return idx, m
