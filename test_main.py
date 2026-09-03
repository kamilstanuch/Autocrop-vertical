import unittest

import main


class PanMathTests(unittest.TestCase):
    def test_smoothstep_is_monotonic_and_clamped(self):
        values = [main.smoothstep(i / 20) for i in range(21)]
        self.assertEqual(main.smoothstep(-1), 0)
        self.assertEqual(main.smoothstep(2), 1)
        self.assertEqual(values[0], 0)
        self.assertEqual(values[-1], 1)
        self.assertTrue(all(a <= b for a, b in zip(values, values[1:])))

    def test_pan_starts_and_ends_at_requested_positions(self):
        positions = [
            main.interpolate_pan_x(10, 110, frame, 5)
            for frame in range(5)
        ]
        self.assertEqual(positions[0], 10)
        self.assertEqual(positions[-1], 110)
        self.assertTrue(all(a < b for a, b in zip(positions, positions[1:])))
        self.assertLess(positions[1] - positions[0], positions[2] - positions[1])
        self.assertGreater(positions[3] - positions[2], positions[4] - positions[3])

    def test_interpolate_pan_x_clamps_to_source_bounds(self):
        self.assertEqual(
            main.interpolate_pan_x(-20, 200, 0, 5, min_x=0, max_x=110),
            0,
        )
        self.assertEqual(
            main.interpolate_pan_x(-20, 200, 4, 5, min_x=0, max_x=110),
            110,
        )

    def test_crop_center_is_clamped_to_source_bounds(self):
        self.assertEqual(
            main.calculate_crop_box_for_center(0, 160, 90, 50),
            (0, 0, 50, 90),
        )
        self.assertEqual(
            main.calculate_crop_box_for_center(160, 160, 90, 50),
            (110, 0, 160, 90),
        )

    def test_track_to_track_crop_jump_pans_without_reading_pixels(self):
        scenes = [
            {
                "start_frame": 0,
                "end_frame": 10,
                "strategy": "TRACK",
                "target_box": [10, 0, 40, 90],
            },
            {
                "start_frame": 10,
                "end_frame": 20,
                "strategy": "TRACK",
                "target_box": [120, 0, 150, 90],
            },
        ]
        # video_path is unused for pan eligibility; production H.264 must
        # still pan when adjacent frames would fail a pixel hard-cut check.
        main.plan_pan_transitions(
            "/nonexistent.mp4", scenes, 160, 90, 10, pan_duration=0.4)

        self.assertEqual(scenes[1]["boundary_kind"], "pan")
        pan = scenes[1]["pan"]
        self.assertIsNotNone(pan)
        positions = [
            main.interpolate_pan_x(
                pan["from_x"], pan["to_x"], frame, pan["duration_frames"])
            for frame in range(pan["duration_frames"])
        ]
        self.assertEqual(len(positions), 4)
        self.assertEqual(positions[0], pan["from_x"])
        self.assertEqual(positions[-1], pan["to_x"])
        self.assertTrue(all(a < b for a, b in zip(positions, positions[1:])))

    def test_speaker_switch_and_layout_and_jitter(self):
        scenes = [
            {
                "start_frame": 0,
                "end_frame": 10,
                "strategy": "TRACK",
                "target_box": [10, 0, 40, 90],
            },
            {
                "start_frame": 10,
                "end_frame": 20,
                "strategy": "TRACK",
                "target_box": [120, 0, 150, 90],
            },
            {
                "start_frame": 20,
                "end_frame": 30,
                "strategy": "LETTERBOX",
                "target_box": None,
            },
        ]
        main.plan_pan_transitions(
            None, scenes, 160, 90, 10, pan_duration=0.4)

        self.assertEqual(scenes[1]["boundary_kind"], "pan")
        self.assertIsNotNone(scenes[1]["pan"])
        self.assertEqual(scenes[2]["boundary_kind"], "layout-switch")
        self.assertIsNone(scenes[2]["pan"])

        jitter_scenes = [
            {
                "start_frame": 0,
                "end_frame": 10,
                "strategy": "TRACK",
                "target_box": [70, 0, 90, 90],
            },
            {
                "start_frame": 10,
                "end_frame": 20,
                "strategy": "TRACK",
                "target_box": [72, 0, 92, 90],
            },
        ]
        main.plan_pan_transitions(
            None,
            jitter_scenes,
            160,
            90,
            10,
            pan_duration=0.4,
        )
        self.assertEqual(jitter_scenes[1]["boundary_kind"], "hold")
        self.assertIsNone(jitter_scenes[1]["pan"])

    def test_zero_pan_duration_disables_transitions(self):
        scenes = [
            {
                "start_frame": 0,
                "end_frame": 10,
                "strategy": "TRACK",
                "target_box": [10, 0, 40, 90],
            },
            {
                "start_frame": 10,
                "end_frame": 20,
                "strategy": "TRACK",
                "target_box": [120, 0, 150, 90],
            },
        ]
        main.plan_pan_transitions(
            None, scenes, 160, 90, 10, pan_duration=0)
        self.assertIsNone(scenes[1]["pan"])

    def test_summary_counts_pans_against_track_boundaries(self):
        scenes = [
            {"start_frame": 0, "end_frame": 10, "strategy": "TRACK",
             "target_box": [10, 0, 40, 90]},
            {"start_frame": 10, "end_frame": 20, "strategy": "TRACK",
             "target_box": [120, 0, 150, 90]},
            {"start_frame": 20, "end_frame": 30, "strategy": "TRACK",
             "target_box": [121, 0, 151, 90]},
            {"start_frame": 30, "end_frame": 40, "strategy": "LETTERBOX",
             "target_box": None},
        ]
        main.plan_pan_transitions(None, scenes, 160, 90, 10, pan_duration=0.4)
        summary = main.summarize_pan_plan(scenes)
        self.assertEqual(summary["track_to_track"], 2)
        self.assertEqual(summary["pan"], 1)
        self.assertEqual(summary["hold"], 1)
        self.assertEqual(summary["layout_switch"], 1)


class ProductionRenderPathTests(unittest.TestCase):
    """Drive the exact per-frame functions the encode loop uses."""

    WIDTH, HEIGHT, FPS = 160, 90, 10

    def two_scene_plan(self, pan_duration=0.4):
        scenes = [
            {"start_frame": 0, "end_frame": 10, "strategy": "TRACK",
             "target_box": [10, 0, 40, 90]},
            {"start_frame": 10, "end_frame": 30, "strategy": "TRACK",
             "target_box": [120, 0, 150, 90]},
        ]
        main.plan_pan_transitions(
            None, scenes, self.WIDTH, self.HEIGHT, self.FPS,
            pan_duration=pan_duration)
        return scenes

    def test_frame_crops_move_gradually_through_a_pan(self):
        scenes = self.two_scene_plan()
        positions = main.plan_frame_crops(scenes, 30, self.WIDTH, self.HEIGHT)

        before = positions[:10]
        pan_frames = scenes[1]["pan"]["duration_frames"]
        during = positions[10:10 + pan_frames]
        after = positions[10 + pan_frames:]

        self.assertEqual(set(before), {0})
        self.assertEqual(set(after), {110})
        self.assertEqual(during[0], 0)
        self.assertEqual(during[-1], 110)
        # A real pan has several distinct intermediate positions, not a snap.
        self.assertGreaterEqual(len(set(during)), 4)
        self.assertTrue(all(a < b for a, b in zip(during, during[1:])))

    def test_zero_pan_duration_snaps_in_one_frame(self):
        scenes = self.two_scene_plan(pan_duration=0)
        positions = main.plan_frame_crops(scenes, 30, self.WIDTH, self.HEIGHT)
        self.assertEqual(positions[9], 0)
        self.assertEqual(positions[10], 110)

    def test_scene_cursor_advances_and_never_rewinds(self):
        scenes = self.two_scene_plan()
        self.assertEqual(main.scene_index_for_frame(scenes, 0, 0), 0)
        self.assertEqual(main.scene_index_for_frame(scenes, 9, 0), 0)
        self.assertEqual(main.scene_index_for_frame(scenes, 10, 0), 1)
        self.assertEqual(main.scene_index_for_frame(scenes, 5, 1), 1)

    def test_rendered_pixels_shift_gradually_across_boundary(self):
        try:
            import cv2  # noqa: F401
            import numpy as np
        except ImportError as exc:
            raise unittest.SkipTest(f"OpenCV unavailable: {exc}")

        scenes = self.two_scene_plan()
        out_w, out_h = main.compute_output_size(self.HEIGHT)
        # Encode each source column's x coordinate as its pixel intensity so
        # the leftmost output pixel reports exactly where the crop landed.
        frame = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        frame[:, :, 0] = np.arange(self.WIDTH, dtype=np.uint8)[None, :]

        rendered_crop_x = []
        index = 0
        for frame_number in range(30):
            index = main.scene_index_for_frame(scenes, frame_number, index)
            out = main.render_output_frame(
                frame, scenes[index], frame_number,
                self.WIDTH, self.HEIGHT, out_w, out_h)
            self.assertEqual(out.shape, (out_h, out_w, 3))
            rendered_crop_x.append(int(out[0, 0, 0]))

        planned = main.plan_frame_crops(scenes, 30, self.WIDTH, self.HEIGHT)
        self.assertEqual(rendered_crop_x, planned)
        pan_frames = scenes[1]["pan"]["duration_frames"]
        during = rendered_crop_x[10:10 + pan_frames]
        self.assertGreaterEqual(len(set(during)), 4)
        self.assertTrue(all(a < b for a, b in zip(during, during[1:])))

    def test_serialized_plan_round_trips_through_json(self):
        import json
        scenes = self.two_scene_plan()
        for scene in scenes:
            scene["analysis"] = [{"person_box": [1, 2, 3, 4]}]
        payload = json.loads(json.dumps(main.serialize_plan(
            scenes, self.WIDTH, self.HEIGHT, self.FPS, "9:16")))
        self.assertEqual(payload["summary"]["pan"], 1)
        self.assertEqual(payload["scenes"][1]["boundary_kind"], "pan")
        self.assertEqual(payload["scenes"][1]["people"], 1)
        replayed = main.plan_frame_crops(
            payload["scenes"], 30, payload["width"], payload["height"])
        self.assertEqual(
            replayed, main.plan_frame_crops(scenes, 30, self.WIDTH, self.HEIGHT))


class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest(f"OpenCV test dependencies unavailable: {exc}")

    def test_frame_difference_separates_motion_from_hard_cut(self):
        import numpy as np

        before = np.full((90, 160, 3), 80, dtype=np.uint8)
        soft = before.copy()
        soft[30:50, 40:60] = 100
        hard = np.full((90, 160, 3), 240, dtype=np.uint8)

        self.assertLess(main.frame_difference_score(before, soft), 0.18)
        self.assertGreater(main.frame_difference_score(before, hard), 0.18)


if __name__ == "__main__":
    unittest.main()
