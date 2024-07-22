from hocap.utils import *
from hocap.loaders import SequenceLoader
from hocap.wrappers import MPHandDetector

# Set the start method to 'spawn'
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

MP_CONFIG = read_data_from_json(PROJ_ROOT / "config/mp_config.json")


def runner_mp_hand_detector(rgb_images, mp_config=MP_CONFIG) -> np.ndarray:
    detector = MPHandDetector(mp_config)

    results = []

    for image in rgb_images:
        result = np.full((2, 21, 2), -1, dtype=np.int64)

        hand_marks, hand_sides, hand_scores = detector.detect_one(image)

        if not hand_sides:
            results.append(result)
            continue

        # update hand sides if there are two same hand sides
        if len(hand_sides) == 2 and hand_sides[0] == hand_sides[1]:
            if hand_scores[0] >= hand_scores[1]:
                hand_sides[1] = "right" if hand_sides[0] == "left" else "left"
            else:
                hand_sides[0] = "right" if hand_sides[1] == "left" else "left"

        # update hand marks result
        for i, hand_side in enumerate(hand_sides):
            side_index = 0 if hand_side == "right" else 1
            result[side_index] = hand_marks[i]

        results.append(result)

    results = np.stack(results).astype(np.int64)

    return results


def runner_hand_3d_joints_ransac_torch(
    handmarks, Ms, threshold=10.0, iterations=100, lr=1e-3, device="cuda"
):
    def triangulate_handmarks_batch(p1, p2, M1, M2):
        """
        Triangulate handmarks from two camera views using PyTorch for batch processing.

        Parameters:
        - p1: Batch of points from the first camera, torch tensor of shape (B, 21, 2).
        - p2: Batch of points from the second camera, torch tensor of shape (B, 21, 2).
        - M1, M2: Corresponding camera projection matrices, torch tensors of shape (B, 3, 4).

        Returns:
        - A torch tensor of shape (B, 21, 3) containing the triangulated 3D handmarks.
        """
        B, N, _ = p1.shape  # B: Batch size, N: Number of handmarks (21)
        X = torch.zeros((B, N, 3), dtype=torch.float32, device=device)

        for i in range(N):  # Process each handmark
            A = torch.zeros((B, 4, 4), dtype=torch.float32, device=device)
            A[:, 0, :] = p1[:, i, 0].unsqueeze(1) * M1[:, 2, :] - M1[:, 0, :]
            A[:, 1, :] = p1[:, i, 1].unsqueeze(1) * M1[:, 2, :] - M1[:, 1, :]
            A[:, 2, :] = p2[:, i, 0].unsqueeze(1) * M2[:, 2, :] - M2[:, 0, :]
            A[:, 3, :] = p2[:, i, 1].unsqueeze(1) * M2[:, 2, :] - M2[:, 1, :]

            # Perform SVD on A
            U, S, Vh = torch.linalg.svd(A, full_matrices=False)
            # Extract the solution from Vh, normalize by the last element
            X[:, i, :] = Vh[:, -1, :3] / Vh[:, -1, 3].unsqueeze(1)

        return X

    def project_3d_to_2d_parallel(p_3d, Ms):
        """
        Project a batch of 3D points to 2D across multiple camera views in parallel.

        Parameters:
        - p_3d: A torch tensor of 3D points of shape (N, 3).
        - Ms: Camera projection matrices, torch tensor of shape (C, 3, 4).

        Returns:
        - projected_2d: Projected 2D points, torch tensor of shape (C, N, 2).
        """
        C, N = Ms.shape[0], p_3d.shape[0]
        ones = torch.ones((N, 1), dtype=p_3d.dtype, device=device)
        p_3d_hom = (
            torch.cat((p_3d, ones), dim=1).unsqueeze(0).repeat(C, 1, 1)
        )  # Shape: (C, N, 4)
        p_2d_hom = torch.einsum("cij,ckj->cki", Ms, p_3d_hom)  # Shape: (C, N, 3)
        p_2d = p_2d_hom[:, :, :2] / p_2d_hom[:, :, 2:3]  # Normalize
        return p_2d

    def parallel_reprojection_loss(p_3d, uv_coords, Ms):
        """
        Compute the reprojection loss for all 21 joints across all camera views in parallel.

        Parameters:
        - p_3d: A torch tensor of 3D points of shape (N, 3).
        - uv_coords: Observed 2D points for all cameras, torch tensor of shape (C, N, 2).
        - Ms: Camera projection matrices, torch tensor of shape (C, 3, 4).

        Returns:
        - Total reprojection loss, a single scalar value.
        """
        p_2d = project_3d_to_2d_parallel(p_3d, Ms)
        loss = torch.norm(p_2d - uv_coords, dim=-1)  # Compute norm for each projection
        loss = loss.sum()
        loss /= uv_coords.size(0) * p_3d.size(0)  # Normalize by number of cameras
        return loss

    def optimize_all_joints(pts_3d, uv_coords, Ms):
        """
        Optimize the 3D coordinates of all 21 MANO joints in parallel.

        Parameters:
        - pts_3d: Batch of initial 3D coordinates for all joints, torch tensor of shape (B, N, 3).
        - uv_coords: Observed 2D points for all cameras, torch tensor of shape (C, N, 2).
        - Ms: Camera projection matrices, torch tensor of shape (C, 3, 4).
        - lr: Learning rate for the optimizer.
        - steps: Number of optimization steps.

        Returns:
        - Optimized 3D coordinates for all 21 joints, torch tensor of shape (N, 3).
        """
        B, N, _ = pts_3d.shape
        X = torch.zeros(
            (B, 3),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([X], lr=lr)
        opt_pts = []

        for i in range(N):
            X.data = pts_3d[:, i].clone()
            _uv_coords = uv_coords[:, i].unsqueeze(1).repeat(1, B, 1)
            for _ in range(iterations):
                optimizer.zero_grad()
                loss = parallel_reprojection_loss(X, _uv_coords, Ms)
                loss.backward()
                optimizer.step()
            opt_pts.append(X.detach().clone())

        opt_pts = torch.stack(opt_pts)
        return opt_pts

    handmarks_tensor = torch.from_numpy(handmarks).to(
        dtype=torch.float32, device=device
    )
    Ms_ts = torch.from_numpy(Ms).to(dtype=torch.float32, device=device)
    num_hands, num_cams, num_joints, _ = handmarks_tensor.shape

    hand_joints_3d = torch.full((num_hands, num_joints, 3), -1, dtype=torch.float32)

    for hand_idx in range(num_hands):
        marks = handmarks_tensor[hand_idx]  # shape (num_cameras, num_joints, 2)
        if torch.all(marks == -1):  # no hand detected
            continue
        valid_views = torch.where(torch.all(marks[..., 0] != -1, dim=1))
        num_valid_views = len(valid_views[0])
        if num_valid_views < 4:  # less than 4 views detected
            continue

        combinations = torch.tensor(
            list(itertools.combinations(valid_views[0].cpu().numpy(), 2)),
            device=device,
        )

        b_pts_3d = triangulate_handmarks_batch(
            marks[combinations[:, 0]],
            marks[combinations[:, 1]],
            Ms_ts[combinations[:, 0]],
            Ms_ts[combinations[:, 1]],
        )

        pts_3d_optim = optimize_all_joints(
            b_pts_3d, marks[valid_views], Ms_ts[valid_views]
        )

        best_points = torch.full((num_joints, 3), -1, dtype=torch.float32)

        for joint_idx in range(num_joints):
            best_loss = np.inf
            best_point = None
            best_inliers = 0
            for i in range(len(pts_3d_optim[joint_idx])):
                projected_2d = project_3d_to_2d_parallel(
                    pts_3d_optim[joint_idx, i].unsqueeze(0),
                    Ms_ts[valid_views],
                )
                loss = torch.norm(
                    projected_2d - marks[valid_views][:, joint_idx].unsqueeze(1), dim=-1
                )
                valid_mask = torch.where(loss < threshold)[0]
                inliers = len(valid_mask)

                if inliers > best_inliers:
                    best_inliers = inliers
                    best_point = pts_3d_optim[joint_idx, i]
                    best_loss = loss[valid_mask].mean().item()
                if (
                    best_inliers > 0
                    and inliers == best_inliers
                    and loss[valid_mask].mean().item() < best_loss
                ):
                    best_inliers = inliers
                    best_point = pts_3d_optim[joint_idx, i]
                    best_loss = loss[valid_mask].mean().item()

            if best_inliers > 1:
                best_points[joint_idx] = best_point.cpu()
        if torch.all(best_points != -1):
            hand_joints_3d[hand_idx] = best_points

    hand_joints_3d = hand_joints_3d.numpy().astype(np.float32)

    return hand_joints_3d


def runner_get_mp_vis_image(rgb_images, handmarks, serials):
    if not (len(rgb_images) == len(handmarks) == len(serials)):
        raise ValueError(
            "The length of rgb_images, handmarks, and serials must be the same."
        )

    vis_images = [
        draw_debug_image(
            rgb_images[i],
            hand_marks=handmarks[i],
            draw_boxes=True,
            draw_hand_sides=True,
        )
        for i in range(len(rgb_images))
    ]

    return display_images(vis_images, serials, return_array=True)


class HandDetector:
    def __init__(self, sequence_folder, device="cuda"):
        self._logger = get_logger("HandDetector")

        self._device = device
        self._data_folder = Path(sequence_folder).resolve()
        self._hand_det_folder = self._data_folder / "processed" / "hand_detection"
        self._hand_det_folder.mkdir(parents=True, exist_ok=True)

        # load variables from sequence loader
        self._loader = SequenceLoader(sequence_folder)
        self._rs_serials = self._loader.rs_serials
        self._rs_master = self._loader.rs_master
        self._num_frames = self._loader.num_frames
        self._rs_width = self._loader.rs_width
        self._rs_height = self._loader.rs_height
        self._mano_sides = self._loader.mano_sides
        self._num_cameras = self._loader.num_cameras
        self._M = self._loader.M2world.cpu().numpy()

    def run_mp_handmarks_detection(self):
        self._logger.info(">>>>>>>>>> Running MediaPipe Hand Detection <<<<<<<<<<")

        start_time = time.time()

        # check if hand detection results already exist
        mp_handmarks_file = self._hand_det_folder / "mp_handmarks.npz"
        if mp_handmarks_file.exists():
            self._logger.info("  ** Hand Detection Results already exist. Loading...")
            mp_handmarks = np.load(self._hand_det_folder / "mp_handmarks.npz")
        else:
            self._logger.info("Start running Hand Detection...")
            mp_handmarks = {serial: None for serial in self._rs_serials}
            tqbar = tqdm(total=len(self._rs_serials), ncols=80)
            with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
                futures = {
                    executor.submit(
                        runner_mp_hand_detector,
                        [
                            self._loader.get_rgb_image(frame_id, serial)
                            for frame_id in range(self._num_frames)
                        ],
                    ): serial
                    for serial in self._rs_serials
                }
                for future in concurrent.futures.as_completed(futures):
                    mp_handmarks[futures[future]] = future.result()
                    tqbar.update(1)
                    tqbar.refresh()
            tqbar.close()

            self._logger.info("Updating Hand Detection Results with 'mano_sides'...")
            if len(self._mano_sides) == 1:
                for serial in self._rs_serials:
                    for frame_id in range(self._num_frames):
                        if "right" in self._mano_sides:
                            if np.any(
                                mp_handmarks[serial][frame_id][0] == -1
                            ) and np.all(mp_handmarks[serial][frame_id][1] != -1):
                                mp_handmarks[serial][frame_id][0] = mp_handmarks[
                                    serial
                                ][frame_id][1]
                            mp_handmarks[serial][frame_id][1] = -1
                        if "left" in self._mano_sides:
                            if np.any(
                                mp_handmarks[serial][frame_id][1] == -1
                            ) and np.all(mp_handmarks[serial][frame_id][0] != -1):
                                mp_handmarks[serial][frame_id][1] = mp_handmarks[
                                    serial
                                ][frame_id][0]
                            mp_handmarks[serial][frame_id][0] = -1

            self._logger.info("Saving Hand Detection Results...")
            # Swap the axes to (2, num_frames, 21, 2)
            mp_handmarks = {
                serial: np.swapaxes(mp_handmarks[serial], 0, 1)
                for serial in self._rs_serials
            }
            np.savez_compressed(
                self._hand_det_folder / "mp_handmarks.npz", **mp_handmarks
            )

        self._logger.info("Rendering Hand Detection results...")

        tqdm.write("  - Generatiing vis images...")
        vis_images = [None] * self._num_frames
        tqbar = tqdm(total=self._num_frames, ncols=80)
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(
                    runner_get_mp_vis_image,
                    self._loader.get_rgb_image(frame_id),
                    [mp_handmarks[serial][:, frame_id] for serial in self._rs_serials],
                    self._rs_serials,
                ): frame_id
                for frame_id in range(self._num_frames)
            }
            for future in concurrent.futures.as_completed(futures):
                vis_images[futures[future]] = future.result()
                tqbar.update(1)
                tqbar.refresh()
        tqbar.close()

        del mp_handmarks

        tqdm.write("  - Saving vis images...")
        save_folder = self._hand_det_folder / "vis" / "mp_handmarks"
        make_clean_folder(save_folder)
        tqbar = tqdm(total=self._num_frames, ncols=80)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    write_rgb_image,
                    save_folder / f"vis_{frame_id:04d}.png",
                    vis_images[frame_id],
                ): frame_id
                for frame_id in range(self._num_frames)
            }
            for future in concurrent.futures.as_completed(futures):
                future.result()
                tqbar.update(1)
                tqbar.refresh()
        tqbar.close()

        tqdm.write("  - Saving vis video...")
        create_video_from_rgb_images(
            self._hand_det_folder / "vis" / "vis_mp_handmarks.mp4", vis_images, fps=30
        )

        del vis_images

        end_time = time.time()
        elapsed_time = end_time - start_time
        self._logger.info(f">>>>>>>>>> Done!!! ({elapsed_time:.2f}s) <<<<<<<<<<")

    def run_joints_3d_estimation(self):
        self._logger.info(">>>>>>>>>> Running Hand 3D Joints Estimation <<<<<<<<<<")

        start_time = time.time()

        # Load handmarks
        handmarks_file = self._hand_det_folder / "mp_handmarks.npz"
        if not handmarks_file.exists():
            self._logger.error("  ** Handmarks file does not exist!!!")
            return
        handmarks = np.load(handmarks_file)
        handmarks = np.stack([handmarks[s] for s in self._rs_serials], axis=2)
        self._logger.info(
            f"  ** Handmarks loaded: {handmarks.shape}, {handmarks.dtype}"
        )

        joints_3d_file = self._hand_det_folder / "mp_joints_3d_raw.npy"
        if joints_3d_file.exists():
            self._logger.info("Hand 3D joints results already exist. Loading...")
            hand_joints_3d = np.load(joints_3d_file)
        else:
            self._logger.info("Estimating hand 3D joints...")
            hand_joints_3d = [None] * self._num_frames
            tqbar = tqdm(total=self._num_frames, ncols=80)
            with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
                futures = {
                    executor.submit(
                        runner_hand_3d_joints_ransac_torch,
                        handmarks[:, frame_id],
                        self._M,
                        threshold=10.0,
                        iterations=100,
                        lr=0.001,
                        device=self._device,
                    ): frame_id
                    for frame_id in range(self._num_frames)
                }
                for future in concurrent.futures.as_completed(futures):
                    hand_joints_3d[futures[future]] = future.result()
                    tqbar.update(1)
                    tqbar.refresh()
            tqbar.close()
            hand_joints_3d = np.stack(hand_joints_3d, axis=1)
            self._logger.info(
                f"  ** Hand 3D joints: {hand_joints_3d.shape}, {hand_joints_3d.dtype}"
            )

            self._logger.info("Saving hand 3D joints results...")
            np.save(joints_3d_file, hand_joints_3d)

        self._logger.info("Interpoliting missing hand 3D joints...")
        hand_joints_3d = self._complete_joints_3d_by_cubic(hand_joints_3d, ratio=0.8)
        np.save(self._hand_det_folder / "mp_joints_3d_iterpolated.npy", hand_joints_3d)

        self._logger.info(f"Generating hand 3D joints projection results...")
        hand_joints_2d = np.stack(
            [
                [
                    self._points_3d_to_2d(joints_3d[frame_id], self._M)
                    for frame_id in range(self._num_frames)
                ]
                for joints_3d in hand_joints_3d
            ],
            axis=0,
            dtype=np.int64,
        )
        self._logger.info(f"  ** Hand 3D joints projection: {hand_joints_2d.shape}")
        hand_joints_2d = {
            serial: hand_joints_2d[:, :, cam_idx]
            for cam_idx, serial in enumerate(self._rs_serials)
        }
        np.savez_compressed(
            self._hand_det_folder / "mp_joints_3d_projection.npz", **hand_joints_2d
        )

        self._logger.info("Generating vis images...")
        vis_images = [None] * self._num_frames
        tqbar = tqdm(total=self._num_frames, ncols=80)
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(
                    runner_get_mp_vis_image,
                    self._loader.get_rgb_image(frame_id),
                    [hand_joints_2d[s][:, frame_id] for s in self._rs_serials],
                    self._rs_serials,
                ): frame_id
                for frame_id in range(self._num_frames)
            }
            for future in concurrent.futures.as_completed(futures):
                vis_images[futures[future]] = future.result()
                tqbar.update(1)
                tqbar.refresh()
        tqbar.close()

        del hand_joints_2d, hand_joints_3d

        self._logger.info("Saving vis images...")
        save_folder = self._hand_det_folder / "vis" / "mp_joints_3d"
        make_clean_folder(save_folder)
        tqbar = tqdm(total=self._num_frames, ncols=80)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    write_rgb_image,
                    save_folder / f"vis_{frame_id:04d}.png",
                    vis_images[frame_id],
                ): frame_id
                for frame_id in range(self._num_frames)
            }
            for future in concurrent.futures.as_completed(futures):
                future.result()
                tqbar.update(1)
                tqbar.refresh()
        tqbar.close()

        self._logger.info("Saving vis video...")
        create_video_from_rgb_images(
            self._hand_det_folder / "vis" / "vis_mp_joints_3d.mp4", vis_images, fps=30
        )

        del vis_images

        end_time = time.time()
        elapsed_time = end_time - start_time
        self._logger.info(f">>>>>>>>>> Done!!! ({elapsed_time:.2f}s) <<<<<<<<<<")

    def _complete_joints_3d_v1(self, joints_3d, ratio=0.5):
        N, hands, joints, coords = joints_3d.shape
        # Loop over each hand and each joint
        complete_joints = joints_3d.copy()
        self._mp_config["complete_frames"] = {
            "method": "v1_linear",
            "ratio": ratio,
        }
        for hand in range(hands):
            valid_frames = np.where(
                np.all(complete_joints[:, hand] != -1, axis=(1, 2))
            )[0]
            self._mp_config["complete_frames"][
                f"hand_{hand}"
            ] = f"{len(valid_frames)}/{N}"
            if len(valid_frames) < int(N * ratio):
                self._logger.warning(
                    f"    ** Not enough valid frames for interpolation for hand-{hand}. (#frames: {len(valid_frames)}/{N})"
                )
                continue
            for joint in range(joints):
                for coord in range(coords):
                    # Extract the current sequence for the joint's coordinate
                    valid_coords = complete_joints[:, hand, joint, coord][valid_frames]
                    # Create a spline interpolation function
                    interp_func = interp1d(
                        valid_frames,
                        valid_coords,
                        kind="linear",
                        bounds_error=False,
                        fill_value=(valid_coords[0], valid_coords[-1]),
                    )
                    # Interpolate missing points
                    interpolated_coords = interp_func(np.arange(N))
                    # Update the original array with interpolated values
                    complete_joints[:, hand, joint, coord] = interpolated_coords

        return complete_joints

    def _complete_joints_3d_by_cubic(self, joints_3d, ratio=0.5):
        complete_joints_3d = joints_3d.copy()
        num_hands, num_frames, num_joints, num_coords = complete_joints_3d.shape

        for hand_idx in range(num_hands):
            valid_frames = np.where(
                np.all(complete_joints_3d[hand_idx] != -1, axis=(1, 2))
            )[0]

            if len(valid_frames) < int(num_frames * ratio):
                self._logger.warning(
                    f"  ** Not enough valid frames for interpolation for hand-{hand_idx}. "
                    f"(#frames: {len(valid_frames)}/{num_frames})"
                )
                continue

            for joint_idx in range(num_joints):
                for coord_idx in range(num_coords):
                    # Extract the current sequence for the joint's coordinate
                    valid_coords = complete_joints_3d[
                        hand_idx, valid_frames, joint_idx, coord_idx
                    ]
                    # Create a cubic spline interpolation function
                    cs = CubicSpline(valid_frames, valid_coords, bc_type="clamped")
                    # Interpolate missing points
                    interpolated_coords = cs(np.arange(num_frames))
                    # Update the original array with interpolated values
                    complete_joints_3d[hand_idx, :, joint_idx, coord_idx] = (
                        interpolated_coords
                    )

        return complete_joints_3d

    def _complete_joints_3d_v3(self, joints_3d, ratio=0.5):
        def calculate_bone_lengths(joints_3d):
            """
            Calculate the bone lengths from the parent-child joint relationships.

            Parameters:
            - joints_3d: Observed 3D joints, shape (N, 21, 3).

            Returns:
            - Bone lengths for each joint, shape (21,).
            """
            bone_lengths = np.zeros(21)
            for i in range(1, 21):  # Skip the root joint, which has no parent
                parent_idx = HAND_JOINT_PARENTS[i]
                if parent_idx >= 0:  # Valid parent index
                    bone_lengths[i] = np.linalg.norm(
                        joints_3d[:, parent_idx] - joints_3d[:, i], axis=1
                    ).mean()
            return bone_lengths

        def adjust_joints_to_bone_lengths(joints_3d, bone_lengths):
            """
            Adjust the positions of the joints to respect the given bone lengths.

            Parameters:
            - joints_3d: Interpolated 3D joints, shape (N, 21, 3).
            - bone_lengths: Bone lengths to enforce, shape (21,).
            """
            for i in range(1, 21):  # Skip the root joint
                parent_idx = HAND_JOINT_PARENTS[i]
                if parent_idx >= 0:  # Valid parent index
                    direction = joints_3d[:, i] - joints_3d[:, parent_idx]
                    # Normalize the direction vector
                    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
                    joints_3d[:, i] = (
                        joints_3d[:, parent_idx] + direction * bone_lengths[i]
                    )

        N, hands, joints, coords = joints_3d.shape
        complete_joints = joints_3d.copy()
        self._mp_config["complete_frames"] = {"method": "v3_cubic_bone_length"}

        for hand in range(hands):
            valid_frames = np.where(
                np.all(complete_joints[:, hand] != -1, axis=(1, 2))
            )[0]

            self._mp_config["complete_frames"][
                f"hand_{hand}"
            ] = f"{len(valid_frames)}/{N}"

            if len(valid_frames) < int(N * ratio):
                print(
                    f"** Not enough valid frames for interpolation for hand-{hand}. (#frames: {len(valid_frames)}/{N})"
                )
                continue

            # Calculate average bone lengths from the first frame (assuming it's fully observed)
            bone_lengths = calculate_bone_lengths(joints_3d[valid_frames, hand])

            for joint in range(joints):
                for coord in range(coords):
                    valid_coords = complete_joints[valid_frames, hand, joint, coord]
                    cs = CubicSpline(valid_frames, valid_coords, bc_type="clamped")
                    interpolated_coords = cs(np.arange(N))
                    complete_joints[:, hand, joint, coord] = interpolated_coords
            # Adjust the completed joints to respect bone lengths
            adjust_joints_to_bone_lengths(complete_joints[:, hand], bone_lengths)

        return complete_joints

    def _smooth_joints_3d(self, joints_3d, alpha=0.5, iterations=10):
        """
        Smooths the 3D joints of the hand over a sequence.

        Parameters:
        - joints_3d: numpy array of shape (N, 2, 21, 3), the 3D hand joint positions
        - alpha: float, the smoothing factor, determines the weight of the adjacent frames in smoothing
        - iterations: int, the number of smoothing iterations to perform

        Returns:
        - Smoothed 3D hand joint positions array of the same shape as input.
        """

        def calculate_bone_lengths(joints_3d):
            """
            Calculate the bone lengths from the parent-child joint relationships.

            Parameters:
            - joints_3d: Observed 3D joints, shape (N, 21, 3).

            Returns:
            - Bone lengths for each joint, shape (21,).
            """
            bone_lengths = np.zeros(21)
            for i in range(1, 21):  # Skip the root joint, which has no parent
                parent_idx = HAND_JOINT_PARENTS[i]
                if parent_idx >= 0:  # Valid parent index
                    bone_lengths[i] = np.linalg.norm(
                        joints_3d[:, parent_idx] - joints_3d[:, i], axis=1
                    ).mean()
            return bone_lengths

        def adjust_joints_to_bone_lengths(joints_3d, bone_lengths):
            """
            Adjust the positions of the joints to respect the given bone lengths.

            Parameters:
            - joints_3d: Interpolated 3D joints, shape (N, 21, 3).
            - bone_lengths: Bone lengths to enforce, shape (21,).
            """
            for i in range(1, 21):  # Skip the root joint
                parent_idx = HAND_JOINT_PARENTS[i]
                if parent_idx >= 0:  # Valid parent index
                    direction = joints_3d[:, i] - joints_3d[:, parent_idx]
                    # Normalize the direction vector
                    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
                    joints_3d[:, i] = (
                        joints_3d[:, parent_idx] + direction * bone_lengths[i]
                    )

        self._mp_config["smoothing_frames"] = {"alpha": alpha, "iterations": iterations}

        N, hands, joints, coords = joints_3d.shape
        # Create a copy of the array to avoid modifying the original data
        smoothed_joints = joints_3d.copy()
        for hand in range(hands):
            if np.any(joints_3d[:, hand] == -1):
                self._logger.warning(
                    f"    ** Hand-{hand} not detected in some frames, skipping smoothing..."
                )
                continue
            bone_lengths = calculate_bone_lengths(joints_3d[:, hand])
            self._mp_config["smoothing_frames"][f"hand_{hand}"] = True
            for _ in range(iterations):
                for joint in range(joints):
                    for frame in range(
                        1, N - 1
                    ):  # Skip the first and last frame to avoid boundary issues
                        # Simple averaging of the current frame with its immediate neighbors
                        smoothed_joints[frame, hand, joint] = alpha * smoothed_joints[
                            frame, hand, joint
                        ] + (1 - alpha) * 0.5 * (
                            smoothed_joints[frame - 1, hand, joint]
                            + smoothed_joints[frame + 1, hand, joint]
                        )
                # Adjust the smoothed joints to respect bone lengths
                adjust_joints_to_bone_lengths(smoothed_joints[:, hand], bone_lengths)

        return smoothed_joints

    def _get_mano_sides(self, mp_handmarks):
        """Get mano sides from the first frame of each sub device"""
        mano_sides = []
        count_left = 0
        count_right = 0
        count_detected = 0
        for serial in self._rs_serials:
            for frame_id in range(self._num_frames):
                hand_marks = mp_handmarks[serial][frame_id]
                if not np.any(hand_marks[0] == -1):
                    count_right += 1
                if not np.any(hand_marks[1] == -1):
                    count_left += 1
                if not np.any(hand_marks[0] == -1) or not np.any(hand_marks[1] == -1):
                    count_detected += 1
        if count_right > count_detected // 2:
            mano_sides.append("right")
        if count_left > count_detected // 2:
            mano_sides.append("left")
        return mano_sides

    def _point_3d_to_2d(self, point_3d, M):
        point_2d = M @ np.append(point_3d, 1)
        point_2d /= point_2d[2]
        point_2d = point_2d[:2].astype(np.int64)
        # check if the point is within the image
        if (
            point_2d[0] < 0
            or point_2d[0] >= self._rs_width
            or point_2d[1] < 0
            or point_2d[1] >= self._rs_height
        ):
            return None
        return point_2d

    def _points_3d_to_2d(self, points_3d, Ms):
        points_2d = np.full((len(Ms), len(points_3d), 2), -1)
        for i, p_3d in enumerate(points_3d):
            if np.any(p_3d == -1):
                continue
            tmp_pts_2d = [self._point_3d_to_2d(p_3d, M) for M in Ms]
            # replace invalid points with -1
            points_2d[:, i] = [
                p_2d if p_2d is not None else np.array([-1, -1]) for p_2d in tmp_pts_2d
            ]
        return points_2d.astype(np.int64)

    def _on_finish(self):
        write_data_to_json(
            self._hand_det_folder / "detection_info.json", self._mp_config
        )


if __name__ == "__main__":
    sequence_folder = PROJ_ROOT / "data/subject_1/20231025_165502"

    detector = HandDetector(sequence_folder, device="cuda")
    detector.run_mp_handmarks_detection()
    detector.run_joints_3d_estimation()
