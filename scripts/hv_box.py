#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""归一化盒（normalization box）的口径冻结工具。

问题
----
HV 一律用「参考集归一化」重算：取**参与比较的所有臂、所有 run 的前沿并集**
作为归一化盒，`ref=(1.02, 1.02)`。这条带来的硬约束是
**盒随臂集变化 → 绝对 HV 不能跨批次直接比**。

同一个臂 `QPAS2_hv_wide` 在项目里就留下了三个数值（0.82014 / 0.82795 / 0.83590），
差异全部来自盒漂移——结果文件是增量长出来的，臂集变了盒就变。

用法
----
    from hv_box import make_box, write_box_sidecar, format_box

    box = make_box(fronts, arms=sorted(arms))
    print(format_box(box))
    write_box_sidecar(lab_json_path, box)

分析脚本必须把盒落盘（sidecar），引用绝对 HV 的地方同时引用盒的指纹。
只引用 ΔHV / p / wins 时不受盒影响，可安全跨批次比对。
"""
import hashlib
import io
import json
import os

import numpy as np

try:
    from rmoea_d.utils.metrics import estimate_hv_bounds
except ImportError:  # 允许在没有 sys.path 设置的场景下单独导入
    estimate_hv_bounds = None

DEFAULT_REF = (1.02, 1.02)


def _bounds(fronts):
    if estimate_hv_bounds is not None:
        return estimate_hv_bounds(fronts)
    arrs = [np.asarray(f, float) for f in fronts if len(f)]
    if not arrs:
        return np.zeros(2), np.ones(2)
    lo = np.min([a.min(axis=0) for a in arrs], axis=0)
    hi = np.max([a.max(axis=0) for a in arrs], axis=0)
    return lo, hi


def fronts_fingerprint(fronts):
    """对参与定盒的前沿集合取指纹：同一臂集 + 同一批数据 → 同一指纹。

    指纹变了就说明盒变了，此时**任何绝对 HV 的跨批比对都无效**。
    """
    h = hashlib.sha1()
    for f in fronts:
        a = np.asarray(f, float)
        if a.ndim != 2:
            continue
        h.update(np.ascontiguousarray(np.round(a, 6)).tobytes())
        h.update(b"|")
    return h.hexdigest()[:12]


def make_box(fronts, arms=None, ref=DEFAULT_REF, extra=None):
    """构造可落盘的盒描述。``arms`` 是参与定盒的臂名列表（务必传）。"""
    lo, hi = _bounds(fronts)
    box = {
        "lo": [float(x) for x in np.asarray(lo, float)],
        "hi": [float(x) for x in np.asarray(hi, float)],
        "ref": [float(x) for x in ref],
        "n_fronts": int(len(fronts)),
        "fronts_sha1": fronts_fingerprint(fronts),
        "arms": sorted(arms) if arms is not None else None,
    }
    if extra:
        box.update(extra)
    return box


def format_box(box):
    """单行摘要，供分析脚本打印。"""
    arms = box.get("arms")
    n_arm = len(arms) if arms else 0
    return ("NORM-BOX  lo=%s hi=%s ref=%s  fronts=%d arms=%d sha1=%s"
            % (np.round(box["lo"], 2).tolist(), np.round(box["hi"], 2).tolist(),
               box["ref"], box["n_fronts"], n_arm, box["fronts_sha1"]))


def write_box_sidecar(source_path, box, suffix=".box.json"):
    """把盒写到结果文件旁边，使"这份数字用的是哪个盒"可被事后复核。"""
    path = source_path + suffix
    text = json.dumps(box, ensure_ascii=False, indent=2, sort_keys=True)
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return path


def read_box_sidecar(source_path, suffix=".box.json"):
    path = source_path + suffix
    if not os.path.exists(path):
        return None
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def assert_same_box(box_a, box_b, label="box"):
    """跨批次比较绝对 HV 之前必须先过这一步。"""
    if box_a is None or box_b is None:
        raise ValueError("%s: 缺盒，不能比较绝对 HV" % label)
    if box_a["fronts_sha1"] != box_b["fronts_sha1"]:
        raise ValueError(
            "%s: 盒指纹不同（%s vs %s）——绝对 HV 不可比，只能比 ΔHV/p/wins"
            % (label, box_a["fronts_sha1"], box_b["fronts_sha1"]))
    return True
