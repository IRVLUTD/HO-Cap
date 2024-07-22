"""Example of rendering a sequence."""

from hocap.utils import *
from hocap.renderers import SequenceRenderer


if __name__ == "__main__":
    sequence_folder = PROJ_ROOT / "data/subject_1/20231025_165502"
    renderer = SequenceRenderer(sequence_folder, device="cuda")

    frame_id = 150

    # Render the scene and get the rendered images
    renderer.create_scene(frame_id)
    render_colors = renderer.get_render_colors()
    render_masks = renderer.get_render_masks()

    # Display the rendered images
    pose_overlays = [
        cv2.addWeighted(
            renderer.get_rgb_image(frame_id, serial), 0.4, render_color, 0.6, 0
        )
        for serial, render_color in render_colors.items()
    ]

    display_all_camera_images(pose_overlays, list(render_colors.keys()))

    mask_overlays = [
        cv2.addWeighted(
            renderer.get_rgb_image(frame_id, serial), 0.4, render_mask, 0.6, 0
        )
        for serial, render_mask in render_masks.items()
    ]

    display_all_camera_images(mask_overlays, list(render_masks.keys()))
