import numpy as np
import torch
from torch.nn import Module
from manopth.manolayer import ManoLayer


class MANOLayer(Module):
    """Wrapper layer for manopath ManoLayer."""

    def __init__(self, side, betas):
        """
        Constructor.
        Args:
            side: MANO hand type. 'right' or 'left'.
            betas: A numpy array of shape [10] containing the betas.
        """
        super(MANOLayer, self).__init__()

        self._side = side
        self._betas = betas

        self._mano_layer = ManoLayer(
            center_idx=0,
            flat_hand_mean=True,
            ncomps=45,
            side=side,
            use_pca=True,
            root_rot_mode="axisang",
            joint_rot_mode="axisang",
            robust_rot=True,
        )

        # register buffer for betas
        b = torch.from_numpy(betas).unsqueeze(0).float()
        self.register_buffer("b", b)

        # register buffer for faces
        f = self._mano_layer.th_faces
        self.register_buffer("f", f)

        # register buffer for root translation
        v = (
            torch.matmul(self._mano_layer.th_shapedirs, self.b.transpose(0, 1)).permute(
                2, 0, 1
            )
            + self._mano_layer.th_v_template
        )
        r = torch.matmul(self._mano_layer.th_J_regressor[0], v)
        self.register_buffer("root_trans", r)

    def forward(self, p, t):
        """
        Forward function.
        Args:
            p: A tensor of shape [B, 48] containing the pose.
            t: A tensor of shape [B, 3] containing the trans.
        Returns:
            v: A tensor of shape [B, 778, 3] containing the vertices.
            j: A tensor of shape [B, 21, 3] containing the joints.
        """
        v, j = self._mano_layer(p, self.b.expand(p.size(0), -1), t)

        # Convert to meters.
        v /= 1000.0
        j /= 1000.0
        return v, j

    @property
    def th_hands_mean(self):
        return self._mano_layer.th_hands_mean

    @property
    def th_selected_comps(self):
        return self._mano_layer.th_selected_comps

    @property
    def th_v_template(self):
        return self._mano_layer.th_v_template

    @property
    def side(self):
        return self._side

    @property
    def num_verts(self):
        return 778
