import numpy as np
import torch
import torchvision.transforms as transforms
import logging
from ..utils import *
from ..utils.common import init_logger

add_path(EXTERNAL_ROOT / "LoFTR")
from src.loftr import LoFTR, default_cfg

LOFTR_MODEL_PATH = PROJ_ROOT / "config/LoFTR/outdoor_ds.ckpt"


class LoftrWrapper:
    def __init__(self, batch_size=8, device="cuda", debug=False):
        self._batch_size = batch_size
        self._device = device
        self._debug = debug

        self._logger = init_logger(
            log_level="debug" if debug else "info", name="LoftrWrapper"
        )
        self._matcher = self._init_loftr_marcher()

        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
            ]
        )

    def _init_loftr_marcher(self):
        self._logger.info("Initializing LoFTR matcher...")
        default_cfg["match_coarse"]["thr"] = 0.2
        self._logger.debug(f"LoFTR config:\n{default_cfg}")
        matcher = LoFTR(config=default_cfg)
        matcher.load_state_dict(torch.load(LOFTR_MODEL_PATH)["state_dict"])
        matcher = matcher.eval().to(self._device)
        return matcher

    @torch.no_grad()
    def match(self, rgbAs, rgbBs, batch_size=16):
        """
        @rgbAs: (N,H,W,C)
        """

        if len(rgbAs) != len(rgbBs):
            raise ValueError("rgbAs and rgbBs must have the same length")

        last_data = {
            "mkpts0_f": [],
            "mkpts1_f": [],
            "mconf": [],
            "m_bids": [],
        }

        with torch.cuda.amp.autocast(enabled=True):
            i_b = 0
            for b in range(0, len(rgbAs), batch_size):
                image0 = torch.stack(
                    [self._transform(rgb) for rgb in rgbAs[b : b + batch_size]]
                ).to(self._device)
                image1 = torch.stack(
                    [self._transform(rgb) for rgb in rgbBs[b : b + batch_size]]
                ).to(self._device)
                tmp = {"image0": image0, "image1": image1}
                self._matcher(tmp)
                tmp["m_bids"] += i_b
                for k in last_data:
                    last_data[k].append(tmp[k])
                i_b += len(tmp["image0"])

        for k in last_data:
            last_data[k] = torch.cat(last_data[k], dim=0)

        mkpts0 = last_data["mkpts0_f"].cpu().numpy()
        mkpts1 = last_data["mkpts1_f"].cpu().numpy()
        mconf = last_data["mconf"].cpu().numpy()
        m_bids = last_data["m_bids"].cpu().numpy()

        corres = (
            np.concatenate(
                (mkpts0.reshape(-1, 2), mkpts1.reshape(-1, 2), mconf.reshape(-1, 1)),
                axis=-1,
            )
            .reshape(-1, 5)
            .astype(np.float32)
        )

        corres_tmp = []
        for i in range(len(rgbAs)):
            cur_corres = corres[m_bids == i]
            corres_tmp.append(cur_corres)
        corres = corres_tmp

        return corres
