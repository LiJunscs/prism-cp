import json
import tempfile
import unittest
from pathlib import Path

from analyze_profile import summarize_profile_dir


class AnalyzeProfileTests(unittest.TestCase):
    def test_aggregates_leaf_components_and_step_times(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {
                "rank": 0,
                "stage": 0,
                "samples": [
                    {"name": "optimizer_step", "wall_ms": 10.0, "cuda_ms": 9.0},
                    {
                        "name": "layer_backward",
                        "layer_type": "gdn",
                        "cp_size": 2,
                        "wall_ms": 7.0,
                        "cuda_ms": 6.5,
                    },
                    {
                        "name": "layer_backward",
                        "layer_type": "gdn",
                        "cp_size": 2,
                        "wall_ms": 5.0,
                        "cuda_ms": 4.5,
                    },
                ],
            }
            (root / "profile_rank_0000.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            report = summarize_profile_dir(root)
            self.assertEqual(report["step_times"][0]["wall_mean_ms"], 10.0)
            component = report["components"][0]
            self.assertEqual(component["name"], "layer_backward")
            self.assertEqual(component["count"], 2)
            self.assertEqual(component["wall_total_ms"], 12.0)
            self.assertEqual(
                component["timing_semantics"], "inclusive_autograd_hook"
            )
            self.assertTrue(
                any("must not be added" in item for item in report["interpretation"])
            )


if __name__ == "__main__":
    unittest.main()
