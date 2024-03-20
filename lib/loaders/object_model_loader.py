import numpy as np
import pandas as pd
import h5py
from ..utils import *

MODELS_ROOT_DIR = "/metadisk/HO101/object_collection"


class ObjectModelLoader:
    def __init__(self) -> None:
        self._data_folder = Path(MODELS_ROOT_DIR).resolve()
        self._metadata = self._read_metadata()
        self._models_info = self._retrive_model_folders_info()

    def _read_metadata(self):
        meta_file = self._data_folder / "Object_Models_Info_v1.xlsx"
        meta_df = pd.read_excel(meta_file)
        return meta_df

    def _retrive_model_folders_info(self):
        all_sequences = {}
        for folder in self._data_folder.iterdir():
            all_sequences[folder.name] = folder

        object_id_dict = self._metadata.set_index("Group_ID")["Sequence_Name"].to_dict()
        models_info = {}
        for object_id in object_id_dict:
            sequence_name = object_id_dict[object_id]
            if sequence_name in all_sequences:
                sequence_folder = all_sequences[sequence_name]
                models_info[object_id] = sequence_folder
        return models_info

    def get_model_info(self, object_id):
        sequence_folder = self._models_info[object_id]
        model_info = {}
        model_info["object_id"] = object_id
        model_info["sequence_name"] = sequence_folder.name
        model_info["cam_K_file"] = str(sequence_folder / "cam_K.txt")
        model_info["data_file"] = str(sequence_folder / "data.h5")
        model_info["models_folder"] = str(sequence_folder / "models")
        return model_info

    def get_num_frames(self, object_id):
        num_frames = -1
        with h5py.File(self._models_info[object_id] / "data.h5", "r") as hf:
            num_frames = hf["color_images"].shape[0]
        return num_frames

    def get_rgb_image(self, object_id, frame_id=None):
        img = None
        with h5py.File(self._models_info[object_id] / "data.h5", "r") as hf:
            if frame_id is None:
                img = hf["color_images"][()]
            else:
                img = hf["color_images"][frame_id]
        return img

    def get_depth_image(self, object_id, frame_id=None):
        img = None
        with h5py.File(self._models_info[object_id] / "data.h5", "r") as hf:
            if frame_id is None:
                img = hf["depth_images"][()]
            else:
                img = hf["depth_images"][frame_id]
        return img

    def get_mask_image(self, object_id, frame_id=None):
        img = None
        with h5py.File(self._models_info[object_id] / "data.h5", "r") as hf:
            if frame_id is None:
                img = hf["mask_images"][()]
            else:
                img = hf["mask_images"][frame_id]
        return img

    def get_K_matrix(self, object_id):
        K_file = self._models_info[object_id] / "cam_K.txt"
        K = np.loadtxt(str(K_file), dtype=np.float32).reshape(3, 3)
        return K

    def get_obj_in_cam(self, object_id, frame_id=None):
        pose = None
        with h5py.File(self._models_info[object_id] / "data.h5", "r") as hf:
            if frame_id is None:
                pose = hf["obj_in_cam"][()]
            else:
                pose = hf["obj_in_cam"][frame_id]
        return pose

    def get_point_cloud(self, object_id, frame_id):
        color_image = self.get_rgb_image(object_id, frame_id)
        depth_image = self.get_depth_image(object_id, frame_id)
        K = self.get_K_matrix(object_id)
        points = self._deproject_depth(depth_image, K)
        colors = color_image.reshape(-1, 3)
        return points, colors

    def _deproject_depth(self, depth_image, K):
        h, w = depth_image.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
        points = np.stack([u, v, np.ones_like(u)], axis=-1)
        points = points.reshape(-1, 3)
        points = points * depth_image.reshape(-1, 1)
        points = points @ np.linalg.inv(K).T
        return points

    def get_models_list(self):
        return sorted(self._models_info.keys())

    def save_metadata_to_csv(self, csv_file=None):
        if csv_file is None:
            csv_file = self._data_folder / "meta.csv"
        self._metadata.to_csv(csv_file, index=False)

    def save_metadata_to_html(self, html_file=None):
        if html_file is None:
            html_file = self._data_folder / "meta.html"
        self._metadata.to_html(html_file, index=False)
