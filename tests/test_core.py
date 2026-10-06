"""rollup.core 的验收测试。

只断言期望的下采样结果与不变量：桶的半开区间归属、乱序点在桶里的分布、
区间汇总与缺失桶补齐策略、聚合函数族的取值、按时插值、空内核与非法输入。
"""

import unittest

from rollup import Downsampler, median_of


def feed(kernel, points):
    """按顺序把 (时间戳, 数值) 写进降采样器。"""
    for timestamp, value in points:
        kernel.add(timestamp, value)


class BucketingTests(unittest.TestCase):

    def test_points_land_in_unique_half_open_buckets(self):
        kernel = Downsampler(interval=10, origin=0, aggregator="sum")
        self.assertTrue(kernel.is_empty())
        self.assertEqual(kernel.point_count(), 0)
        self.assertEqual(len(kernel), 0)
        self.assertEqual(kernel.indices(), [])
        self.assertEqual(kernel.buckets(), [])
        self.assertEqual(kernel.summary(), [])
        self.assertEqual(kernel.bucket_index(0), 0)
        self.assertEqual(kernel.bucket_index(9), 0)
        self.assertEqual(kernel.bucket_index(10), 1)
        self.assertEqual(kernel.bucket_start(2), 20)
        self.assertEqual(kernel.bucket_end(2), 30)
        self.assertEqual(kernel.bucket_bounds(2), (20, 30))

        self.assertEqual(kernel.add(3, 1.0), 0)
        self.assertEqual(kernel.add(9, 2.0), 0)
        self.assertEqual(kernel.add_many([(10, 3.0), (19, 4.0), (20, 5.0)]), 3)
        self.assertFalse(kernel.is_empty())
        self.assertEqual(kernel.point_count(), 5)
        self.assertEqual(len(kernel), 5)
        self.assertEqual(kernel.indices(), [0, 1, 2])
        self.assertEqual([bucket.index for bucket in kernel.buckets()],
                         [0, 1, 2])
        self.assertEqual([(bucket.start, bucket.end)
                          for bucket in kernel.buckets()],
                         [(0, 10), (10, 20), (20, 30)])
        self.assertEqual([len(bucket) for bucket in kernel.buckets()], [2, 2, 1])
        self.assertEqual(kernel.value(0), 3.0)
        self.assertEqual(kernel.value(1), 7.0)
        self.assertEqual(kernel.value(2), 5.0)

        seen = []
        for bucket in kernel.buckets():
            for stamp in bucket.timestamps():
                self.assertGreaterEqual(stamp, bucket.start)
                self.assertLess(stamp, bucket.end)
                seen.append(stamp)
        self.assertEqual(sorted(seen), [3, 9, 10, 19, 20])
        self.assertEqual(len(seen), len(set(seen)))

    def test_buckets_before_the_origin_get_their_own_index(self):
        kernel = Downsampler(interval=10, origin=0, aggregator="sum")
        feed(kernel, [(-25, 1.0), (-20, 2.0), (-11, 3.0), (-1, 4.0), (0, 5.0)])
        self.assertEqual([kernel.bucket_index(stamp)
                          for stamp in (-25, -20, -11, -10, -1, 0, 9)],
                         [-3, -2, -2, -1, -1, 0, 0])
        self.assertEqual(kernel.indices(), [-3, -2, -1, 0])
        self.assertEqual(kernel.value(-3), 1.0)
        self.assertEqual(kernel.value(-2), 5.0)
        self.assertEqual(kernel.value(-1), 4.0)
        self.assertEqual(kernel.value(0), 5.0)
        self.assertEqual(kernel.point_count(), 5)
        self.assertEqual(kernel.total(), 15.0)
        for bucket in kernel.buckets():
            self.assertEqual(bucket.start, kernel.bucket_start(bucket.index))
            self.assertEqual(bucket.end, kernel.bucket_end(bucket.index))
            for stamp in bucket.timestamps():
                self.assertGreaterEqual(stamp, bucket.start)
                self.assertLess(stamp, bucket.end)

    def test_late_points_keep_the_bucket_they_belong_to(self):
        kernel = Downsampler(interval=10, origin=0, aggregator="sum")
        kernel.add(5, 1.0)
        kernel.add(25, 3.0)
        kernel.add(15, 2.0)
        kernel.add(13, 4.0)
        kernel.add(4, 8.0)
        self.assertEqual(kernel.indices(), [0, 1, 2])
        self.assertEqual([len(bucket) for bucket in kernel.buckets()], [2, 2, 1])
        self.assertEqual(kernel.value(0), 9.0)
        self.assertEqual(kernel.value(1), 6.0)
        self.assertEqual(kernel.value(2), 3.0)
        self.assertEqual(kernel.point_count(), 5)
        self.assertEqual(kernel.total(), 18.0)
        for bucket in kernel.buckets():
            for stamp in bucket.timestamps():
                self.assertGreaterEqual(stamp, bucket.start)
                self.assertLess(stamp, bucket.end)

    def test_late_points_before_the_origin_need_the_whole_boundary(self):
        kernel = Downsampler(interval=10, origin=0, aggregator="sum")
        kernel.add(-1, 4.0)
        kernel.add(-15, 6.0)
        self.assertEqual(kernel.bucket_index(-15), -2)
        self.assertEqual(kernel.indices(), [-2, -1])
        self.assertEqual([bucket.index for bucket in kernel.buckets()], [-2, -1])
        self.assertEqual(kernel.value(-2), 6.0)
        self.assertEqual(kernel.value(-1), 4.0)
        self.assertEqual(kernel.point_count(), 2)
        self.assertEqual(kernel.total(), 10.0)
        for bucket in kernel.buckets():
            for stamp in bucket.timestamps():
                self.assertGreaterEqual(stamp, bucket.start)
                self.assertLess(stamp, bucket.end)


class SummaryTests(unittest.TestCase):

    def test_summary_covers_every_bucket_and_fills_gaps(self):
        kernel = Downsampler(interval=10, origin=0, aggregator="sum",
                             fill="previous")
        feed(kernel, [(1, 10.0), (11, 20.0), (31, 40.0)])
        entries = kernel.summary(0, 35)
        self.assertEqual([stamp for stamp, _ in entries], [0, 10, 20, 30])
        self.assertEqual([value for _, value in entries],
                         [10.0, 20.0, 20.0, 40.0])
        self.assertEqual(kernel.summary(11, 19), [(10, 20.0)])

        leading = Downsampler(interval=10, origin=0, aggregator="sum",
                              fill="previous")
        feed(leading, [(15, 3.0), (35, 7.0)])
        head = leading.summary(0, 35)
        self.assertEqual([value for _, value in head], [None, 3.0, 3.0, 7.0])
        plain = leading.summary(0, 35, fill="none")
        self.assertEqual([stamp for stamp, _ in plain], [0, 10, 20, 30])
        self.assertEqual([value for _, value in plain], [None, 3.0, None, 7.0])

    def test_zero_and_linear_fills_are_deterministic(self):
        kernel = Downsampler(interval=10, origin=0, aggregator="sum")
        feed(kernel, [(1, 0.0), (21, 30.0)])
        self.assertEqual(kernel.indices(), [0, 2])
        zeroed = kernel.summary(0, 35, fill="zero")
        self.assertEqual(zeroed[:3], [(0, 0.0), (10, 0.0), (20, 30.0)])
        linear = kernel.summary(0, 35, fill="linear")
        self.assertEqual(linear[:3], [(0, 0.0), (10, 15.0), (20, 30.0)])

        edge = Downsampler(interval=10, origin=0, aggregator="sum")
        feed(edge, [(15, 4.0), (25, 6.0)])
        head = edge.summary(0, 35, fill="linear")
        self.assertIsNone(head[0][1])
        self.assertEqual(head[1], (10, 4.0))
        self.assertEqual(head[2], (20, 6.0))
        self.assertEqual(edge.summary(0, 35, fill="zero")[0], (0, 0.0))
        self.assertEqual(edge.summary(0, 35, fill="none")[0], (0, None))


class AggregateTests(unittest.TestCase):

    def test_bucket_aggregates_match_independent_recomputation(self):
        kernel = Downsampler(interval=10, origin=0, aggregator="median")
        feed(kernel, [(1, 10.0), (2, 20.0), (3, 30.0), (4, 40.0)])
        self.assertEqual(kernel.value(0), 25.0)
        self.assertEqual(kernel.bucket(0).median(), 25.0)
        self.assertEqual(median_of([1.0, 2.0, 3.0, 4.0]), 2.5)
        self.assertEqual(median_of([7.0]), 7.0)
        self.assertEqual(median_of([3.0, 1.0, 2.0]), 2.0)
        self.assertIsNone(median_of([]))

        ordering = Downsampler(interval=10, origin=0, aggregator="last")
        ordering.add(18, 3.0)
        ordering.add(12, 7.0)
        ordering.add(15, 5.0)
        bucket = ordering.bucket(1)
        self.assertEqual(bucket.timestamps(), [18, 12, 15])
        self.assertEqual(bucket.first(), 7.0)
        self.assertEqual(bucket.last(), 3.0)
        self.assertEqual(bucket.minimum(), 3.0)
        self.assertEqual(bucket.maximum(), 7.0)
        self.assertEqual(ordering.value(1), 3.0)
        self.assertEqual(ordering.at(15), 5.0)

    def test_total_matches_the_sum_of_every_point(self):
        kernel = Downsampler(interval=10, origin=0, aggregator="mean")
        feed(kernel, [(1, 2.0), (2, 4.0), (11, 10.0), (12, 20.0), (13, 30.0)])
        self.assertEqual(kernel.point_count(), 5)
        self.assertEqual(kernel.value(0), 3.0)
        self.assertEqual(kernel.value(1), 20.0)
        self.assertEqual(kernel.total(), 66.0)
        self.assertEqual(kernel.total(), 2.0 + 4.0 + 10.0 + 20.0 + 30.0)

        counted = Downsampler(interval=5, origin=0, aggregator="count")
        feed(counted, [(1, 1.0), (6, 1.0), (7, 1.0)])
        self.assertEqual(counted.value(0), 1)
        self.assertEqual(counted.value(1), 2)
        self.assertEqual(counted.total(), 3.0)


class InterpolationTests(unittest.TestCase):

    def test_interpolation_uses_the_bracketing_points(self):
        kernel = Downsampler(interval=10, origin=0, aggregator="mean")
        feed(kernel, [(2, 10.0), (17, 40.0)])
        self.assertEqual(kernel.at(2), 10.0)
        self.assertEqual(kernel.at(17), 40.0)
        self.assertEqual(kernel.at(11), 28.0)
        self.assertIsNone(kernel.at(1))
        self.assertIsNone(kernel.at(18))

        close = Downsampler(interval=10, origin=0)
        feed(close, [(0, 0.0), (4, 8.0), (9, 18.0)])
        self.assertEqual(close.at(4), 8.0)
        self.assertEqual(close.at(6), 12.0)
        self.assertEqual(close.at(7), 14.0)
        self.assertIsNone(Downsampler(interval=10, origin=0).at(0))


class InputTests(unittest.TestCase):

    def test_empty_kernel_and_invalid_inputs_are_rejected(self):
        kernel = Downsampler(interval=10, origin=5)
        self.assertTrue(kernel.is_empty())
        self.assertEqual(kernel.point_count(), 0)
        self.assertEqual(len(kernel), 0)
        self.assertEqual(kernel.total(), 0)
        self.assertEqual(kernel.summary(), [])
        self.assertIsNone(kernel.at(7))
        for index in (-3, 0, 4):
            self.assertIsNone(kernel.value(index))
            self.assertIsNone(kernel.bucket(index))
        self.assertEqual(kernel.interval, 10)
        self.assertEqual(kernel.origin, 5)
        self.assertEqual(kernel.aggregator, "mean")
        self.assertEqual(kernel.fill, "none")

        self.assertRaises(ValueError, Downsampler, 0)
        self.assertRaises(ValueError, Downsampler, -10)
        self.assertRaises(TypeError, Downsampler, 10.0)
        self.assertRaises(TypeError, Downsampler, 10, True)
        self.assertRaises(ValueError, Downsampler, 10, 0, "average")
        self.assertRaises(ValueError, Downsampler, 10, 0, "mean", "forward")

        kernel.add(7, 1.0)
        self.assertFalse(kernel.is_empty())
        self.assertEqual(kernel.point_count(), 1)
        self.assertEqual(kernel.bucket_index(7), 0)
        self.assertEqual(kernel.bucket_bounds(0), (5, 15))

        self.assertRaises(TypeError, kernel.add, 1.5, 1.0)
        self.assertRaises(TypeError, kernel.add, True, 1.0)
        self.assertRaises(TypeError, kernel.add, 1, True)
        self.assertRaises(TypeError, kernel.add, 1, "1.0")
        self.assertRaises(TypeError, kernel.bucket_index, None)
        self.assertRaises(TypeError, kernel.bucket_start, 1.5)
        self.assertRaises(TypeError, kernel.at, 0.5)
        self.assertRaises(ValueError, kernel.summary, 0, 5, "nearest")
        self.assertRaises(TypeError, kernel.summary, 0.5, 5)

        kernel.clear()
        self.assertTrue(kernel.is_empty())
        self.assertEqual(kernel.point_count(), 0)
        self.assertEqual(kernel.total(), 0)
        self.assertEqual(kernel.summary(), [])


if __name__ == "__main__":
    unittest.main()
