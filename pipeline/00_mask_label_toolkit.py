import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from mobile_sam import sam_model_registry, SamPredictor
from hocap.utils import *


class ImageLabelToolkit:
    def __init__(
        self, model_type: str = "vit_t", device: str = "cpu", debug: bool = False
    ) -> None:
        """
        Initializes the ImageLabelToolkit.

        Args:
            model_type (str): The model type to use for SAM. Default is 'vit_l'.
            device (str): The device to run the model on ('cpu' or 'cuda').
            debug (bool): If True, enables debug mode with verbose logging.
        """
        self._logger = self._init_logger(debug)
        self._device = device
        self._predictor = self._init_sam_predictor(model_type)

        self._points = []
        self._undo_stack = []
        self._curr_mask = None
        self._curr_label = 0
        self._masks = []
        self._labels = []
        self._raw_image = None
        self._img_width = 640
        self._img_height = 480
        self._gui_image = o3d.geometry.Image(
            np.zeros((self._img_height, self._img_width, 3), dtype=np.uint8)
        )
        self._text = ""
        self._is_done = False

    def _init_logger(self, debug: bool):
        """Initializes the logger."""
        logger = logging.getLogger("ImageLabelTool")
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if debug else logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def _init_sam_predictor(self, model_type: str):
        """Initializes the SAM predictor."""
        chkpt = PROJ_ROOT / f"config/SAM/sam_{model_type}.pth"
        sam = sam_model_registry[model_type](checkpoint=chkpt)
        sam = sam.to(self._device)
        sam.eval()
        predictor = SamPredictor(sam)
        return predictor

    def run(self):
        """Runs the GUI application."""
        self._app = gui.Application.instance
        self._app.initialize()

        # Create window
        self._window = self._create_window()

        # Add callbacks
        self._window.set_on_layout(self._on_layout)
        self._window.set_on_close(self._on_close)
        self._window.set_on_key(self._on_key)
        self._widget3d.set_on_mouse(self._on_mouse_widget3d)

        self._app.run()

    def _create_window(
        self, title: str = "Image Label Tool", width: int = 800, height: int = 600
    ):
        """Creates the main GUI window."""
        window = gui.Application.instance.create_window(
            title=title, width=width, height=height
        )

        em = window.theme.font_size
        self._panel_width = 20 * em
        margin = 0.25 * em

        self._widget3d = gui.SceneWidget()
        self._widget3d.enable_scene_caching(True)
        self._widget3d.scene = rendering.Open3DScene(window.renderer)
        self._widget3d.scene.set_background(
            [1, 1, 1, 1], o3d.geometry.Image(self._gui_image)
        )
        window.add_child(self._widget3d)

        self._info = gui.Label("")
        self._info.visible = False
        window.add_child(self._info)

        self._panel = gui.Vert(margin, gui.Margins(margin, margin, margin, margin))
        self._add_file_chooser_to_panel()
        self._add_buttons_to_panel()
        self._add_mask_label_to_panel()
        window.add_child(self._panel)

        # Widget Proxy
        self._proxy = gui.WidgetProxy()
        self._proxy.set_widget(None)
        self._panel.add_child(self._proxy)

        return window

    def _add_file_chooser_to_panel(self):
        """Adds file chooser to the panel."""
        self._fileedit = gui.TextEdit()
        filedlgbutton = gui.Button("...")
        filedlgbutton.horizontal_padding_em = 0.5
        filedlgbutton.vertical_padding_em = 0
        filedlgbutton.set_on_clicked(self._on_filedlg_button)
        fileedit_layout = gui.Horiz()
        fileedit_layout.add_child(gui.Label("Image file"))
        fileedit_layout.add_child(self._fileedit)
        fileedit_layout.add_fixed(0.25)
        fileedit_layout.add_child(filedlgbutton)
        self._panel.add_child(fileedit_layout)

    def _add_buttons_to_panel(self):
        """Adds buttons to the panel."""
        button_layout = gui.Horiz(0, gui.Margins(0.25, 0.25, 0.25, 0.25))
        addButton = gui.Button("Add Mask")
        removeButton = gui.Button("Remove Mask")
        saveButton = gui.Button("Save Mask")
        addButton.set_on_clicked(self._on_add_mask)
        removeButton.set_on_clicked(self._on_remove_mask)
        saveButton.set_on_clicked(self._on_save_mask)
        button_layout.add_stretch()
        button_layout.add_child(addButton)
        button_layout.add_stretch()
        button_layout.add_child(removeButton)
        button_layout.add_stretch()
        button_layout.add_child(saveButton)
        button_layout.add_stretch()
        self._panel.add_child(button_layout)

    def _add_mask_label_to_panel(self):
        """Adds mask label to the panel."""
        blk = gui.Horiz(0, gui.Margins(0, 0, 0, 0))
        blk.add_stretch()
        blk.add_child(gui.Label(f"---Current Mask---"))
        blk.add_stretch()
        blk.add_child(gui.Label(f"Label:"))
        self._intedit = gui.NumberEdit(gui.NumberEdit.INT)
        self._intedit.int_value = 0
        self._intedit.set_on_value_changed(self._on_intedit_changed)
        blk.add_child(self._intedit)
        blk.add_stretch()
        self._panel.add_child(blk)

    def _on_intedit_changed(self, value: int):
        """Handles changes in the mask label."""
        self._curr_label = value

    def _mask_block(self):
        """Creates a block of mask labels."""
        if not self._labels:
            return None
        layout = gui.Vert(0, gui.Margins(0, 0, 0, 0))
        for idx, label in enumerate(self._labels):
            blk = gui.Horiz(0, gui.Margins(0, 0, 0, 0))
            blk.add_stretch()
            blk.add_child(gui.Label(f"Mask {idx}:"))
            blk.add_stretch()
            blk.add_child(gui.Label(f"Label: {label}"))
            blk.add_stretch()
            layout.add_child(blk)
        return layout

    def _on_layout(self, ctx):
        """Handles layout changes."""
        pref = self._info.calc_preferred_size(ctx, gui.Widget.Constraints())

        height = self._img_height

        self._widget3d.frame = gui.Rect(0, 0, self._img_width, height)
        self._panel.frame = gui.Rect(
            self._widget3d.frame.get_right(), 0, self._panel_width, height
        )
        self._info.frame = gui.Rect(
            self._widget3d.frame.get_left(),
            self._widget3d.frame.get_bottom() - pref.height,
            pref.width,
            pref.height,
        )
        self._window.size = gui.Size(self._img_width + self._panel_width, height)

    def _on_close(self):
        """Handles window close event."""
        self._is_done = True
        time.sleep(0.10)
        return True

    def _on_key(self, event):
        """Handles key events."""
        if event.key == gui.KeyName.Q:  # Quit
            if event.type == gui.KeyEvent.DOWN:
                self._window.close()
                return True

        if event.key == gui.KeyName.R:  # Reset points
            if event.type == gui.KeyEvent.DOWN:
                self._reset()
                return True

        return False

    def _on_add_mask(self):
        """Adds the current mask to the mask list."""
        self._masks.append(self._curr_mask)
        self._labels.append(self._intedit.int_value)
        self._curr_mask = None
        self._curr_label = 0
        self._reset()
        self._proxy.set_widget(self._mask_block())

    def _on_remove_mask(self):
        """Removes the last mask from the mask list."""
        if self._masks:
            self._masks.pop()
            self._labels.pop()
            self._proxy.set_widget(self._mask_block())

    def _on_save_mask(self):
        """Saves the current mask and its overlay."""
        self._save_folder.mkdir(parents=True, exist_ok=True)
        # Save mask
        mask = self.get_mask()
        write_mask_image(
            self._save_folder / f"{self._image_name.replace('color', 'mask')}.png", mask
        )
        # Save mask overlay
        vis_image = draw_object_masks_overlay(
            self._raw_image, mask, alpha=0.65, reduce_background=True
        )
        write_rgb_image(
            self._save_folder / f"{self._image_name.replace('color', 'vis')}.jpg",
            vis_image,
        )
        self._logger.info(f"Mask saved to {self._save_folder}")

    def _on_filedlg_button(self):
        """Handles file dialog button click."""
        filedlg = gui.FileDialog(gui.FileDialog.OPEN, "Select file", self._window.theme)
        filedlg.add_filter(".png .jpg .jpeg", "Image files (*.png;*.jpg;*.jpeg)")
        filedlg.add_filter("", "All files")
        filedlg.set_on_cancel(self._on_filedlg_cancel)
        filedlg.set_on_done(self._on_filedlg_done)
        self._window.show_dialog(filedlg)

    def _on_filedlg_cancel(self):
        """Handles file dialog cancel."""
        self._window.close_dialog()

    def _update_image(self):
        """Updates the displayed image."""

        def update_image():
            self._widget3d.scene.set_background(
                [1, 1, 1, 1], o3d.geometry.Image(self._gui_image)
            )
            self._widget3d.force_redraw()

        self._app.post_to_main_thread(self._window, update_image)

    def _on_filedlg_done(self, path: str):
        """Handles file dialog done."""
        self._fileedit.text_value = path
        path = Path(path).resolve()
        if not path.exists():
            return
        self._serial = path.parent.name
        self._image_name = path.stem
        self._save_folder = (
            path.parent.parent / "processed/segmentation/init_seg" / self._serial
        )
        img = cv2.imread(str(path))
        self._raw_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._img_height, self._img_width, _ = self._raw_image.shape
        self._gui_image = o3d.geometry.Image(self._raw_image)
        self._predictor.set_image(self._raw_image)
        self._masks = []
        self._labels = []
        self._curr_label = 0
        self._intedit.int_value = 0
        self._curr_mask = None
        self._reset()
        self._proxy.set_widget(self._mask_block())
        self._window.set_needs_layout()
        self._window.close_dialog()

    def _on_mouse_widget3d(self, event):
        """Handles mouse events in the 3D widget."""
        if (
            event.type == gui.MouseEvent.Type.BUTTON_DOWN
            and event.is_modifier_down(gui.KeyModifier.CTRL)
            and event.buttons == gui.MouseButton.LEFT.value
        ):
            x = int(event.x - self._widget3d.frame.x)
            y = int(event.y - self._widget3d.frame.y)

            self._points.append((x, y, True))
            self._undo_stack.append(("add", (x, y, True)))
            self._update_sam_mask()
            # current_image = self._overlay_mask(self._raw_image, self._curr_mask)
            current_image = draw_object_masks_overlay(self._raw_image, self._curr_mask)
            current_image = self._draw_points(current_image, self._points)
            self._gui_image = o3d.geometry.Image(current_image)
            self._update_image()

            return gui.Widget.EventCallbackResult.HANDLED

        if (
            event.type == gui.MouseEvent.Type.BUTTON_DOWN
            and event.is_modifier_down(gui.KeyModifier.CTRL)
            and event.buttons == gui.MouseButton.RIGHT.value
        ):
            x = int(event.x - self._widget3d.frame.x)
            y = int(event.y - self._widget3d.frame.y)

            self._points.append((x, y, False))
            self._undo_stack.append(("add", (x, y, False)))
            self._update_sam_mask()
            # current_image = self._overlay_mask(self._raw_image, self._curr_mask)
            current_image = draw_object_masks_overlay(self._raw_image, self._curr_mask)
            current_image = self._draw_points(current_image, self._points)
            self._gui_image = o3d.geometry.Image(current_image)
            self._update_image()

            return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED

    def _reset(self):
        """Resets the current points and mask."""
        self._points = []
        self._undo_stack = []
        self._gui_image = o3d.geometry.Image(self._raw_image)
        self._update_image()

    def _undo_last_step(self):
        """Undoes the last action."""
        if self._undo_stack:
            action, data = self._undo_stack.pop()
            if action == "add":
                if self._points and self._points[-1] == data:
                    self._points.pop()
            self._update_sam_mask()
            # current_image = self._overlay_mask(self._raw_image, self._curr_mask)
            current_image = draw_mask_overlay(self._raw_image, self._curr_mask)
            current_image = self._draw_points(current_image, self._points)
            self._gui_image = o3d.geometry.Image(current_image)
            self._update_image()

    def _update_sam_mask(self):
        """Updates the SAM mask."""
        if self._points:
            # Get mask from SAM
            input_points = np.array(self._points)[:, :2]
            input_labels = np.array(self._points)[:, 2]
            masks, scores, _ = self._predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )
            mask = masks[np.argmax(scores)]
            self._curr_mask = mask.astype(np.uint8) * 255
        else:
            self._curr_mask = np.zeros(self._raw_image.shape[:2], dtype=np.uint8)

    # def _overlay_mask(
    #     self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.4
    # ) -> np.ndarray:
    #     """Overlays the mask on the image."""
    #     unique_labels = np.unique(mask)
    #     if len(unique_labels) == 1:
    #         return image
    #     unique_labels = unique_labels[unique_labels != 0]  # Removing background
    #     # Map each object label in the mask to a color
    #     overlay = np.zeros_like(image)
    #     for label in unique_labels:
    #         mask_color = (
    #             (255, 255, 255)
    #             if label == 255
    #             else OBJ_CLASS_COLORS[label % len(OBJ_CLASS_COLORS)].rgb
    #         )
    #         overlay[mask == label] = mask_color
    #     # Blend the color image and the mask
    #     blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    #     return blended

    def _overlay_mask(
        self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.65
    ) -> np.ndarray:
        """Overlays the mask on the image."""
        unique_labels = np.unique(mask)

        if len(unique_labels) == 1:
            return image

        unique_labels = unique_labels[unique_labels != 0]  # Removing background
        # Map each object label in the mask to a color
        overlay = np.zeros_like(image)
        for label in unique_labels:
            mask_color = (
                (255, 255, 255)
                if label == 255
                else OBJ_CLASS_COLORS[label % len(OBJ_CLASS_COLORS)].rgb
            )
            overlay[mask == label] = mask_color
        # Blend the color image and the mask
        overlay = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        return overlay

    def _draw_points(
        self, img: np.ndarray, points: list, is_rgb: bool = True
    ) -> np.ndarray:
        """Draws points on the image."""
        img_copy = img.copy()
        for x, y, label in points:
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            if is_rgb:
                color = color[::-1]
            cv2.circle(img_copy, (x, y), 3, color, -1)
        return img_copy

    def get_mask(self) -> np.ndarray:
        """Gets the current mask."""
        mask = np.zeros(self._raw_image.shape[:2], dtype=np.uint8)
        if self._masks:
            for idx, m in enumerate(self._masks):
                mask[m == 255] = self._labels[idx]
        else:
            mask[self._curr_mask == 255] = self._curr_label
        return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Label Toolkit")
    parser.add_argument("--model_type", type=str, default="vit_l")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    toolkit = ImageLabelToolkit(args.model_type, args.device, args.debug)
    toolkit.run()
