"""时间序列降采样内核（纯标准库，行为完全确定）。

时间用整数刻度表示，桶的边界由调用方给出的间隔与起点决定：桶 i 覆盖半开区间
[start_i, end_i)，start_i = origin + i * interval。内核不使用真实时钟、线程、
网络与随机数，同一串写入永远得到同样的结果。

约定：每个采样点只属于一个桶，时间戳落在该桶的半开区间内；乱序到达的点仍归入
它自己的桶，桶列表按下标升序；桶的聚合值只用桶内的点算出；缺失桶按确定策略补齐
（none 留空、zero 补 0、previous 沿用前一个非空桶、linear 在左右最近的非空桶之间
按桶下标插值）；一个点都没写过时返回空结果。

对外接口：Downsampler、Bucket 与 median_of()。
"""

from bisect import bisect_left

__all__ = ["Bucket", "Downsampler", "median_of"]

AGGREGATORS = ("count", "sum", "mean", "min", "max", "first", "last", "median")
FILLS = ("none", "zero", "previous", "linear")


def _require_int(value, label):
    """确认参数是非布尔的整数。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("%s 必须是整数: %r" % (label, value))
    return value


def _require_number(value, label):
    """确认参数是非布尔的数值。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("%s 必须是数值: %r" % (label, value))
    return value


def _require_choice(value, label, allowed):
    """确认参数是允许的取值之一。"""
    if value not in allowed:
        raise ValueError("%s 必须是 %s 之一: %r"
                         % (label, "/".join(allowed), value))
    return value


def median_of(values):
    """数值序列的中位数；空序列返回 None，偶数个取中间两个数的平均。"""
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _apply_fill(entries, policy):
    """按填充策略把 entries 里缺失桶的值补上并返回。

    entries 是升序的 (桶下标, 聚合值) 列表，值为 None 表示那个桶没有点。"""
    if policy == "none":
        return list(entries)
    if policy == "zero":
        return [(index, 0.0 if value is None else value)
                for index, value in entries]
    if policy == "previous":
        filled = []
        carried = None
        for index, value in entries:
            if value is not None:
                carried = value
            filled.append((index, carried))
        return filled
    filled = list(entries)
    known = [spot for spot, (_, value) in enumerate(entries) if value is not None]
    for position, (index, _) in enumerate(entries):
        if filled[position][1] is not None:
            continue
        left = [spot for spot in known if spot < position]
        right = [spot for spot in known if spot > position]
        if not left or not right:
            continue
        left_index, left_value = entries[left[-1]]
        right_index, right_value = entries[right[0]]
        share = (index - left_index) / (right_index - left_index)
        filled[position] = (index, left_value + (right_value - left_value) * share)
    return filled


class Bucket:
    """一个桶：下标、半开区间 [start, end) 与桶内的采样点。"""

    __slots__ = ("_index", "_start", "_end", "_points")

    def __init__(self, index, start, end):
        self._index = index
        self._start = start
        self._end = end
        self._points = []

    @property
    def index(self):
        """桶下标。"""
        return self._index

    @property
    def start(self):
        """桶的起点时间（闭）。"""
        return self._start

    @property
    def end(self):
        """桶的终点时间（开）。"""
        return self._end

    def __len__(self):
        return len(self._points)

    def __repr__(self):
        return "Bucket(index=%d, points=%d)" % (self._index, len(self._points))

    def points(self):
        """桶内的点，按写入次序排列。"""
        return list(self._points)

    def timestamps(self):
        """桶内点的时间戳，按写入次序排列。"""
        return [timestamp for timestamp, _ in self._points]

    def values(self):
        """桶内点的数值，按写入次序排列。"""
        return [value for _, value in self._points]

    def add(self, timestamp, value):
        """把一个点放进桶里。"""
        self._points.append((timestamp, value))

    def first(self):
        """时间戳最早的那个点的数值；空桶返回 None。"""
        if not self._points:
            return None
        return min(self._points, key=lambda point: point[0])[1]

    def last(self):
        """时间戳最晚的那个点的数值；空桶返回 None。"""
        if not self._points:
            return None
        return max(self._points, key=lambda point: point[0])[1]

    def total(self):
        """桶内点的数值之和。"""
        return sum(self.values())

    def minimum(self):
        """桶内点的数值的最小值。"""
        return min(self.values())

    def maximum(self):
        """桶内点的数值的最大值。"""
        return max(self.values())

    def median(self):
        """桶内点的数值的中位数。"""
        return median_of(self.values())

    def aggregate(self, name):
        """按名字取聚合值；空桶返回 None。"""
        if not self._points:
            return None
        if name == "count":
            return len(self._points)
        if name == "sum":
            return self.total()
        if name == "mean":
            return self.total() / len(self._points)
        table = {"min": self.minimum, "max": self.maximum, "first": self.first,
                 "last": self.last, "median": self.median}
        if name not in table:
            raise ValueError("未知聚合名: %r" % (name,))
        return table[name]()


class Downsampler:
    """把时间序列按固定间隔分桶，并按聚合名给出每个桶的值。"""

    __slots__ = ("_interval", "_origin", "_aggregator", "_fill", "_buckets")

    def __init__(self, interval, origin=0, aggregator="mean", fill="none"):
        _require_int(interval, "间隔")
        _require_int(origin, "起点")
        if interval <= 0:
            raise ValueError("间隔必须为正: %r" % (interval,))
        _require_choice(aggregator, "聚合名", AGGREGATORS)
        _require_choice(fill, "填充策略", FILLS)
        self._interval = interval
        self._origin = origin
        self._aggregator = aggregator
        self._fill = fill
        self._buckets = {}

    @property
    def interval(self):
        """桶宽。"""
        return self._interval

    @property
    def origin(self):
        """桶边界的对齐起点。"""
        return self._origin

    @property
    def aggregator(self):
        return self._aggregator

    @property
    def fill(self):
        return self._fill

    def __len__(self):
        return self.point_count()

    def __repr__(self):
        return "Downsampler(interval=%d, buckets=%d)" % (self._interval,
                                                           len(self._buckets))

    def is_empty(self):
        """是否一个点都没写过。"""
        return not self._buckets

    def point_count(self):
        """写入的采样点总数。"""
        return sum(len(bucket) for bucket in self._buckets.values())

    def bucket_index(self, timestamp):
        """时间戳落在哪个桶里。"""
        _require_int(timestamp, "时间戳")
        return (timestamp - self._origin) // self._interval

    def bucket_start(self, index):
        """桶的起点时间（闭）。"""
        _require_int(index, "桶下标")
        return self._origin + index * self._interval

    def bucket_end(self, index):
        """桶的终点时间（开）。"""
        _require_int(index, "桶下标")
        return self._origin + (index + 1) * self._interval

    def bucket_bounds(self, index):
        """桶的半开区间 (起点, 终点)。"""
        return (self.bucket_start(index), self.bucket_end(index))

    def add(self, timestamp, value):
        """写入一个采样点，返回该点所属桶的下标。"""
        _require_int(timestamp, "时间戳")
        _require_number(value, "数值")
        index = self.bucket_index(timestamp)
        bucket = self._buckets.get(index)
        if bucket is None:
            bucket = Bucket(index, self.bucket_start(index),
                            self.bucket_end(index))
            self._buckets[index] = bucket
        bucket.add(timestamp, value)
        return bucket.index

    def add_many(self, points):
        """批量写入 (时间戳, 数值)，返回写入的点数。"""
        added = 0
        for timestamp, value in points:
            self.add(timestamp, value)
            added += 1
        return added

    def clear(self):
        """丢掉全部点。"""
        self._buckets = {}

    def indices(self):
        """所有非空桶的下标，按升序排列。"""
        return sorted(self._buckets)

    def buckets(self):
        """所有非空桶，按下标升序排列。"""
        return [self._buckets[index] for index in self.indices()]

    def bucket(self, index):
        """按下标取桶；没有点的桶返回 None。"""
        _require_int(index, "桶下标")
        return self._buckets.get(index)

    def value(self, index):
        """按下标取桶的聚合值；没有点的桶返回 None。"""
        bucket = self.bucket(index)
        if bucket is None:
            return None
        return bucket.aggregate(self._aggregator)

    def total(self):
        """全部采样点的数值之和。"""
        return sum(value for bucket in self._buckets.values()
                   for value in bucket.values())

    def summary(self, start=None, end=None, fill=None):
        """把 [start, end] 覆盖到的桶逐个列出来。

        元素是 (桶起点时间, 聚合值)；缺省覆盖第一个到最后一个非空桶；缺失桶
        按 fill 策略补齐；没有写过点时返回 []。"""
        policy = self._fill if fill is None else _require_choice(
            fill, "填充策略", FILLS)
        if not self._buckets:
            return []
        if start is None:
            first = self.indices()[0]
        else:
            first = self.bucket_index(start)
        if end is None:
            last = self.indices()[-1]
        else:
            last = self.bucket_index(end)
        if first > last:
            return []
        entries = [(index, self.value(index))
                   for index in range(first, last + 1)]
        filled = _apply_fill(entries, policy)
        return [(self.bucket_start(index), value) for index, value in filled]

    def at(self, timestamp):
        """在时间戳 timestamp 处取值。

        正好有采样点落在这个时间戳上时返回该点数值，否则用左右最近的两个采样点
        线性插值；时间戳落在采样范围之外或没有点时为 None。"""
        _require_int(timestamp, "时间戳")
        points = [point for bucket in self.buckets() for point in bucket.points()]
        if not points:
            return None
        points.sort(key=lambda point: point[0])
        stamps = [point[0] for point in points]
        position = bisect_left(stamps, timestamp)
        if position < len(points) and stamps[position] == timestamp:
            return points[position][1]
        if position == 0 or position == len(points):
            return None
        left_timestamp, left_value = points[position - 1]
        right_timestamp, right_value = points[position]
        if right_timestamp == left_timestamp:
            return right_value
        share = ((timestamp - left_timestamp)
                 / (right_timestamp - left_timestamp))
        return left_value + (right_value - left_value) * share
