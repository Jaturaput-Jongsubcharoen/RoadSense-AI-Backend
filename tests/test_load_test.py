import unittest

from evaluation.load_test import _percentile_95, run_load_test


class LoadTestTests(unittest.TestCase):
    def test_percentile_uses_nearest_rank(self):
        self.assertEqual(_percentile_95([10, 20, 30, 40]), 40)
        self.assertIsNone(_percentile_95([]))

    def test_mocked_load_test_reports_local_api_measurements(self):
        report = run_load_test(
            mode="mocked",
            request_count=4,
            concurrency=2,
            duration_seconds=10,
        )

        self.assertEqual(report["total_requests"], 4)
        self.assertEqual(report["successful_requests"], 4)
        self.assertEqual(report["failed_requests"], 0)
        self.assertEqual(report["error_rate"], 0)
        self.assertEqual(report["mode"], "mocked")
        self.assertEqual(report["model"], None)
        self.assertEqual(report["endpoint"], "POST /api/chat (loopback only)")

    def test_real_ollama_concurrency_is_limited(self):
        with self.assertRaisesRegex(ValueError, "concurrency must be"):
            run_load_test(
                mode="ollama",
                request_count=1,
                concurrency=4,
                duration_seconds=10,
            )


if __name__ == "__main__":
    unittest.main()
