from hocap.utils import *
from hocap.loaders import SequenceLoader
from hocap.wrappers import XMemWrapper


class XmemSegmentation:
    def __init__(self, sequence_folder) -> None:

        self._logger = get_logger(log_name="XmemSegmentation")
        self._data_folder = Path(sequence_folder).resolve()
        self._xmem_seg_folder = self._data_folder / "processed/segmentation/xmem"
        self._init_seg_folder = self._data_folder / "processed/segmentation/init"

        self._loader = SequenceLoader(sequence_folder)
        self._hl_serials = self._loader.holo_serials
        self._rs_serials = self._loader.rs_serials
        self._num_frames = self._loader.num_frames

        self._config = read_data_from_json(PROJ_ROOT / "config/xmem_config.json")

    def run_xmem_segmentation(self):
        self._xmem = XMemWrapper(self._config)

        for serial in self._rs_serials + self._hl_serials:
            self._logger.info(f"Running XMem segmentation for {serial}")
            mask_files = sorted((self._init_seg_folder / serial).glob("mask_*.png"))

            if len(mask_files) == 0:
                self._logger.warning(f"    ** No init masks found, skipping...")
                continue

            mask_inds = [int(mask_file.stem.split("_")[-1]) for mask_file in mask_files]

            save_folder = self._xmem_seg_folder / serial
            make_clean_folder(save_folder)

            tqdm.write(f"  - Reading RGB images...")
            rgb_images = [None] * self._num_frames
            tqbar = tqdm(total=self._num_frames, ncols=80)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        self._loader.get_rgb_image, frame_id, serial
                    ): frame_id
                    for frame_id in range(self._num_frames)
                }
                for future in concurrent.futures.as_completed(futures):
                    rgb_images[futures[future]] = future.result()
                    tqbar.update(1)
                    tqbar.refresh()
            tqbar.close()

            tqdm.write(f"  - Predicting XMem masks...")
            # self._xmem.reset()
            mask_images = [None] * self._num_frames
            tqbar = tqdm(total=self._num_frames, ncols=80)
            for i in range(len(mask_inds)):
                start = mask_inds[i]
                end = mask_inds[i + 1] if i + 1 < len(mask_inds) else self._num_frames
                for frame_id in range(start, end, 1):
                    mask_file = (
                        self._init_seg_folder / serial / f"mask_{frame_id:06d}.png"
                    )
                    mask_images[frame_id] = self._xmem.get_mask(
                        rgb=rgb_images[frame_id],
                        mask=(
                            read_mask_image(mask_file) if mask_file.exists() else None
                        ),
                        exhaustive=True,
                    )
                    tqbar.update(1)
                    tqbar.refresh()
            if mask_inds[0] > 0:
                # self._xmem.reset()
                for frame_id in range(mask_inds[0], -1, -1):
                    mask_file = (
                        self._init_seg_folder / serial / f"mask_{frame_id:06d}.png"
                    )
                    mask_images[frame_id] = self._xmem.get_mask(
                        rgb=rgb_images[frame_id],
                        mask=(
                            read_mask_image(mask_file) if mask_file.exists() else None
                        ),
                        exhaustive=True,
                    )
                    if frame_id != mask_inds[0]:
                        tqbar.update(1)
                        tqbar.refresh()
            tqbar.close()

            tqdm.write(f"  - Saving XMem masks...")
            tqbar = tqdm(total=self._num_frames, ncols=80)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        write_mask_image,
                        save_folder / f"mask_{frame_id:06d}.png",
                        mask_images[frame_id],
                    ): frame_id
                    for frame_id in range(self._num_frames)
                }
                for future in concurrent.futures.as_completed(futures):
                    future.result()
                    tqbar.update(1)
                    tqbar.refresh()
            tqbar.close()

            tqdm.write(f"  - Generating vis images...")
            vis_images = [None] * self._num_frames
            tqbar = tqdm(total=self._num_frames, ncols=80)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        draw_object_masks_overlay,
                        rgb_images[frame_id],
                        mask_images[frame_id],
                        alpha=0.65,
                        reduce_background=True,
                    ): frame_id
                    for frame_id in range(self._num_frames)
                }
                for future in concurrent.futures.as_completed(futures):
                    vis_images[futures[future]] = future.result()
                    tqbar.update(1)
                    tqbar.refresh()
            tqbar.close()

            del rgb_images, mask_images

            tqdm.write(f"  - Saving vis images...")
            tqbar = tqdm(total=self._num_frames, ncols=80)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        write_rgb_image,
                        image_path=save_folder / f"vis_{frame_id:06d}.png",
                        image=vis_images[frame_id],
                    ): frame_id
                    for frame_id in range(self._num_frames)
                }
                for future in concurrent.futures.as_completed(futures):
                    future.result()
                    tqbar.update(1)
                    tqbar.refresh()
            tqbar.close()

            tqdm.write(f"  - Saving vis video...")
            create_video_from_rgb_images(
                self._xmem_seg_folder / f"vis_{serial}.mp4", vis_images, fps=30
            )

            del vis_images

        self._logger.info("  * Done!!!")


if __name__ == "__main__":
    sequence_folder = PROJ_ROOT / "data/subject_1/20231025_165502"

    xmem_seg = XmemSegmentation(sequence_folder)
    xmem_seg.run_xmem_segmentation()
