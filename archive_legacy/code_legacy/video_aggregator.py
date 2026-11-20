#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""顶层快捷入口 `code/video_aggregator.py` -> 转发至 `qingyu.video_aggregator` (严格匹配方案 7.2 节目录树)。"""
import os
import sys

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from qingyu.video_aggregator import (  # noqa: F401, E402
    TemporalHysteresisVideoAggregator,
    main,
    run_monte_carlo_video_benchmark,
)

if __name__ == "__main__":
    main()
