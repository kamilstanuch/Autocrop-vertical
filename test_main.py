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
