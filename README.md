# pairwise-rollup

只依赖 Python 标准库的时间序列降采样内核：时间用整数刻度表示，桶的边界由调用方
给出的间隔与起点决定，没有任何真实时钟、线程、网络或随机数，同一串写入永远得到
同样的下采样结果。

- `rollup/core.py` — 内核：对齐分桶（半开区间）、聚合函数族（count/sum/mean/min/max/
  first/last/median）、缺失桶补齐策略（none/zero/previous/linear）、区间汇总与按时取值。
- `tests/test_core.py` — 验收用例。

## 运行测试

在项目根目录执行：

```
python3 -m unittest discover -s tests -v
```

Windows 上如果没有 `python3`，可用：

```
python -m unittest discover -s tests -v
```
