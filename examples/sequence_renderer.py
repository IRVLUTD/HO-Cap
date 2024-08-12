"""Example of rendering a sequence."""

from hocap.utils import *
from hocap.renderers import SequenceRenderer


if __name__ == "__main__":
    sequence_folder = PROJ_ROOT / "data/subject_1/20231025_165502"
    render_folder = sequence_folder / "renders"

    renderer = SequenceRenderer(sequence_folder, device="cuda")

    for frame_id in tqdm(range(renderer.num_frames), desc="Rendering", ncols=80):
        # Render the scene and get the rendered images
        renderer.create_scene(frame_id)
        render_colors = renderer.get_render_colors()
        render_masks = renderer.get_render_masks()
        overlays = {
            serial: cv2.addWeighted(
                renderer.get_rgb_image(frame_id, serial), 0.4, render_color, 0.6, 0
            )
            for serial, render_color in render_colors.items()
        }

        # Save the rendered images
        for serial in render_colors:
            save_folder = render_folder / serial
            save_folder.mkdir(parents=True, exist_ok=True)
            write_rgb_image(save_folder / f"vis_{frame_id:06d}.jpg", overlays[serial])
            write_rgb_image(
                save_folder / f"color_{frame_id:06d}.jpg", render_colors[serial]
            )
            write_mask_image(
                save_folder / f"seg_{frame_id:06d}.png", render_masks[serial]
            )
