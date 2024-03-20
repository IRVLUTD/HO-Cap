import numpy as np
import cv2
import shutil
from tqdm import tqdm
import logging

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from ..utils import *

add_path(EXTERNAL_ROOT / "XMem")

from model.network import XMem
from inference.inference_core import InferenceCore
from inference.data.mask_mapper import MaskMapper


class DataReader(Dataset):
    def __init__(self, color_files, first_mask_file):
        self.color_files = color_files
        self.mask_file = first_mask_file
        self.im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __getitem__(self, idx):
        data = {}
        info = {}

        # process color image
        img = cv2.imread(str(self.color_files[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.im_transform(img)
        data["rgb"] = img

        # process mask image
        if idx == 0:
            data["mask"] = cv2.imread(str(self.mask_file), cv2.IMREAD_GRAYSCALE)

        return data

    def __len__(self):
        return len(self.color_files)


class XMemWrapper:
    def __init__(
        self,
        single_object=False,
        enable_long_term=True,
        model_type="XMem-s012",
        device="cuda:0",
        debug=False,
    ) -> None:
        self._logger = self._init_logger(debug)
        self._config = {
            "key_dim": 64,
            "value_dim": 512,
            "hidden_dim": 64,
            "top_k": 30,
            "mem_every": 3,  # r in paper. Increase to improve running speed.
            "deep_update_every": -1,  # Leave -1 normally to synchronize with mem_every
            "single_object": single_object,
            "enable_long_term": enable_long_term,
            "enable_long_term_count_usage": enable_long_term,
            "max_mid_term_frames": 60,  # T_max in paper, decrease to save memory
            "min_mid_term_frames": 3,  # T_min in paper, increase to save memory
            "num_prototypes": 128,  # P in paper
            "max_long_term_elements": 10000,  # LT_max in paper, increase if objects disappear for a long time
        }
        self._device = device

        if model_type not in ["XMem", "XMem-s012", "XMem-no-sensory"]:
            self._logger.warning(
                f"Supported model types are 'XMem', 'XMem-s012' and 'XMem-no-sensory'. Got {model_type}. Will use XMem."
            )
            model_type = "XMem"
        model_path = PROJ_ROOT / f"config/XMem/{model_type}.pth"

        self._network = XMem(self._config, model_path=model_path).to(self._device)
        self._network.eval()

        self._mapper = MaskMapper()

        self._processor = InferenceCore(self._network, config=self._config)

        self._im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def reset(self):
        self._logger.debug("Resetting XMemWrapper...")
        # reset the processor
        self._processor.clear_memory()
        self._processor.all_labels = None
        # reset the mapper
        self._mapper.labels = []
        self._mapper.remappings = {}
        # initialize the first mask
        self._first_mask_loaded = False

    @torch.no_grad()
    def get_mask(self, rgb, mask=None):
        """Returns a mask for the given frame.

        Args:
            rgb (np.ndarray): RGB image of the frame.
            mask (np.ndarray, optional): Mask of the frame. Defaults to None.
        """
        with torch.cuda.amp.autocast(enabled=True):
            rgb = self._im_transform(rgb).to(self._device)

            if not self._first_mask_loaded:
                if mask is not None:
                    self._first_mask_loaded = True
                else:
                    return np.zeros(rgb.shape[1:3], dtype=np.uint8)

            if mask is not None:
                mask = mask.astype(np.uint8)
                mask, labels = self._mapper.convert_mask(mask)
                mask = torch.Tensor(mask).to(self._device)
                self._processor.set_all_labels(list(self._mapper.remappings.values()))
            else:
                labels = None

            prob = self._processor.step(rgb, mask, labels)
            torch.cuda.synchronize(device=self._device)

            # probability mask -> index mask
            out_mask = torch.max(prob, dim=0).indices
            out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)
            out_mask = self._mapper.remap_index_mask(out_mask)
            return out_mask

    @torch.no_grad()
    def process(self, color_files, first_mask_file, out_folder):
        """Processes a video and saves the masks.

        Args:
            color_files (list): List of paths to the color images.
            first_mask_file (str): Path to the first mask image.
            out_folder (str): Path to the folder where the masks will be saved.
        """
        if not Path(first_mask_file).exists():
            self._logger.error(f"Mask file does not exist! Exiting...")
            return

        self._save_folder = Path(out_folder).resolve()
        self._save_folder.mkdir(parents=True, exist_ok=True)

        # Start eval
        self.reset()
        vid_reader = DataReader(color_files, first_mask_file)
        loader = DataLoader(vid_reader, batch_size=1, shuffle=False, num_workers=2)
        vid_length = len(vid_reader)

        processor = InferenceCore(self._network, config=self._config)
        first_mask_loaded = False

        self._mapper.labels = []
        self._mapper.remappings = {}
        for ti, data in tqdm(enumerate(loader), total=vid_length, ncols=60):
            with torch.cuda.amp.autocast(enabled=True):
                rgb = data["rgb"].to(self._device)[0]
                msk = data.get("mask")
                if not self._first_mask_loaded:
                    if msk is not None:
                        self._first_mask_loaded = True
                    else:
                        # no point to do anything without a mask
                        continue

                # map possibly non-continuous labels to continuous ones
                if msk is not None:
                    msk, labels = self._mapper.convert_mask(msk[0].numpy())
                    msk = torch.Tensor(msk).to(self._device)
                    self._processor.set_all_labels(
                        list(self._mapper.remappings.values())
                    )
                else:
                    labels = None

                # run the model on this frame
                prob = self._processor.step(
                    rgb, msk, labels, end=(ti == vid_length - 1)
                )
                torch.cuda.synchronize()

                # probability mask -> index mask
                out_mask = torch.max(prob, dim=0).indices
                out_mask = out_mask.detach().cpu().numpy().astype(np.uint8)
                out_mask = self._mapper.remap_index_mask(out_mask)

                # save the mask
                cv2.imwrite(str(self._save_folder / f"mask_{ti:06d}.png"), out_mask)

    def _init_logger(self, debug):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if debug else logging.INFO)

        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger
