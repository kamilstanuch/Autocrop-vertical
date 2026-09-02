import os
import tempfile
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

    def test_crop_center_is_clamped_to_source_bounds(self):
        self.assertEqual(
            main.calculate_crop_box_for_center(0, 160, 90, 50),
            (0, 0, 50, 90),
        )
        self.assertEqual(
            main.calculate_crop_box_for_center(160, 160, 90, 50),
            (110, 0, 160, 90),
        )


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

    def test_synthetic_video_builds_gradual_track_to_track_pan(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, "continuous.avi")
            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"MJPG"),
                10,
                (160, 90),
            )
            self.assertTrue(writer.isOpened())
            for frame_number in range(20):
                frame = np.full((90, 160, 3), 80, dtype=np.uint8)
                x = 30 + frame_number
                frame[35:45, x:x + 10] = 100
                writer.write(frame)
            writer.release()

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
                video_path, scenes, 160, 90, 10, pan_duration=0.4)

            self.assertEqual(scenes[1]["boundary_kind"], "soft")
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

    def test_hard_cuts_layout_switches_and_jitter_do_not_pan(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, "cut.avi")
            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"MJPG"),
                10,
                (160, 90),
            )
            self.assertTrue(writer.isOpened())
            for frame_number in range(30):
                value = 20 if frame_number < 10 else 230
                writer.write(np.full((90, 160, 3), value, dtype=np.uint8))
            writer.release()

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
                video_path, scenes, 160, 90, 10, pan_duration=0.4)

            self.assertEqual(scenes[1]["boundary_kind"], "hard-cut")
            self.assertIsNone(scenes[1]["pan"])
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
            # Force this fixture's boundary to be treated as continuous so
            # the assertion isolates the minimum-distance jitter guard.
            main.plan_pan_transitions(
                video_path,
                jitter_scenes,
                160,
                90,
                10,
                pan_duration=0.4,
                hard_cut_threshold=1.1,
            )
            self.assertIsNone(jitter_scenes[1]["pan"])


if __name__ == "__main__":
    unittest.main()
