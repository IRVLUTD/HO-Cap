import numpy as np
import torch
from torch.nn import Module, ModuleList
from .mano_layer import MANOLayer


class MANOGroupLayer(Module):
    """Wrapper layer to hold a group of MANOLayers."""

    def __init__(self, sides: list[str], betas: list[np.ndarray]):
        """
        Constructor.

        Args:
            sides (list[str]): A list of MANO sides. 'right' or 'left'.
            betas (list[np.ndarray]): A list of numpy arrays of shape [10] containing the betas.
        """
        super(MANOGroupLayer, self).__init__()

        self._sides = sides
        self._betas = betas
        self._num_obj = len(self._sides)

        self._layers = ModuleList(
            [MANOLayer(s, b) for s, b in zip(self._sides, self._betas)]
        )

        # Register buffer for faces
        f = torch.cat([self._layers[i].f + 778 * i for i in range(self._num_obj)])
        self.register_buffer("f", f)

        # Register buffer for root translation
        r = torch.cat([l.root_trans for l in self._layers])
        self.register_buffer("root_trans", r)

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
                v: A tensor of shape [B, N, 3] containing the vertices.
                j: A tensor of shape [B, J, 3] containing the joints.
        """
        if inds is None:
            inds = range(self._num_obj)
        v = [torch.zeros((p.size(0), 0, 3), dtype=torch.float32, device=self.f.device)]
        j = [torch.zeros((p.size(0), 0, 3), dtype=torch.float32, device=self.f.device)]
        p, t = self.pose2pt(p)
        for i in inds:
            y = self._layers[i](p[:, i], t[:, i])
            v.append(y[0])
            j.append(y[1])
        v = torch.cat(v, dim=1)
        j = torch.cat(j, dim=1)
        return v, j

    def pose2pt(self, pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts pose and trans from pose vectors.

        Args:
            pose (torch.Tensor): A tensor of shape [B, D] containing the pose vectors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                p: A tensor of shape [B, O, 48] containing the pose.
                t: A tensor of shape [B, O, 3] containing the trans.
        """
        p = torch.stack(
            [pose[:, 51 * i : 51 * i + 48] for i in range(self._num_obj)], dim=1
        )
        t = torch.stack(
            [pose[:, 51 * i + 48 : 51 * i + 51] for i in range(self._num_obj)], dim=1
        )
        return p, t

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
        for i, x in enumerate(inds):
            f.append(self._layers[x].f + 778 * i)
            m.append(
                x
                * torch.ones(
                    self._layers[x].f.size(0), dtype=torch.int64, device=self.f.device
                )
            )
        f = torch.cat(f)
        m = torch.cat(m)
        return f, m

    @property
    def num_obj(self) -> int:
        """Return the number of objects."""
        return self._num_obj
