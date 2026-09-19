#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Hurink 标准 FJSP 基准实例下载器（数据获取，不跑实验）。

为什么需要它
------------
本项目此前的全部结论建立在 Brandimarte Mk01–Mk10 上（**n=10**）。
凡是「在实例集上挑一个阈值 / 挑一个代理量」的结论，在 n=10 上都无法自我验证 ——
阈值是在同一批实例上事后挑的，等于样本内拟合。
本项目因此引入 Hurink, Jurisch & Thole (1994) 的 66 个标准柔性作业车间实例，
作为**独立留出集**：job/机器结构与 Brandimarte 无关，规模覆盖 6×6 到 30×10。

数据来源
--------
https://github.com/Lei-Kun/FJSP-benchmarks
（Kun Lei et al., ESWA 205:117796 (2022) 公开的基准镜像；
 目录 2a/2b/2c/2d 对应 Hurink 的 sdata / edata / rdata / vdata）
四个子集共享同一批 job/机器结构，**只改各工序的可用机器集合**（柔性递增）：
sdata < rdata < edata < vdata。

口径说明（重要）
----------------
这批文件是**确定时间**的标准 FJSP 实例（`.fjs`）；模糊加工时间仍由本项目
`rmoea_d.core.instance.parse_fjs` 用与 Mk01–Mk10 **完全相同**的规则现场生成
（TFN 以确定加工时间为最可能值，a∈[b/2,b]、c∈[b,3b/2]）。
因此引入它们**不改变问题的模糊化口径**，只是把实例集从 10 个扩到 76 个。

命名
----
落盘为 `data/hurink/<前缀><编号>.fjs`：edata→`Hed`、sdata→`Hsd`、
rdata→`Hrd`、vdata→`Hvd`（如 `Hed01`…`Hed66`）。
**编号是上游文件自己的序号，本项目不臆造文献实例名**（曾想按 la01…orb10 反推，
但实测 `HurinkEdata1` 是 6×6，而 la01 是 10×5 —— 顺序对不上，故只记录编号与规模，
真实对应关系以 `PROVENANCE.json` 里的 `shape` 字段为准）。

工程要点
--------
* **并发下载**：GitHub raw 在本地单连接约 13 s/文件，8 线程后降到 ~1.5 s/文件。
* **增量落盘**：每完成一个就重写一次 `PROVENANCE.json`，中断后可幂等续跑
  （已存在且 sha256 匹配的实例直接跳过，不再发请求）。
* **解析不过就不落盘**：由生产解析器 `parse_fjs` 校验，宁可少一个实例，
  也不要一个「下载成功但加载不了」的坏文件。

用法
----
    python scripts/fetch_hurink.py                          # 默认 edata 全部 66 个
    python scripts/fetch_hurink.py --subsets edata --limit 30
    python scripts/fetch_hurink.py --subsets sdata,rdata,vdata --force
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

BASE = "https://raw.githubusercontent.com/Lei-Kun/FJSP-benchmarks/main"
# 子集名 -> (上游目录, 上游文件名模板, 本项目前缀)
SUBSETS = {
    "edata": ("2b_Hurink_edata", "HurinkEdata{}.fjs", "Hed"),
    "sdata": ("2a_Hurink_sdata", "HurinkSdata{}.fjs", "Hsd"),
    "rdata": ("2c_Hurink_rdata", "HurinkRdata{}.fjs", "Hrd"),
    "vdata": ("2d_Hurink_vdata", "HurinkVdata{}.fjs", "Hvd"),
}
N_PER_SUBSET = 66
DEFAULT_OUT = os.path.join(ROOT, "data", "hurink")

META = {
    "source": "https://github.com/Lei-Kun/FJSP-benchmarks",
    "reference": "Kun Lei et al., Expert Systems with Applications 205 (2022) 117796",
    "upstream_instances": ("Hurink, Jurisch & Thole (1994), 66 instances "
                           "x 4 flexibility levels (sdata/rdata/edata/vdata)"),
    "note": ("确定时间的标准 FJSP 实例；模糊加工时间由 rmoea_d.core.instance.parse_fjs "
             "用与 Mk01-Mk10 完全相同的规则生成"),
}


def fetch(url, attempts=4, timeout=30):
    """取一个 URL 的文本，失败重试（指数退避）。"""
    last = None
    for k in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as fh:
                return fh.read().decode("utf-8", "replace")
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            last = exc
            if k < attempts - 1:
                time.sleep(1.5 * (k + 1))
    raise RuntimeError("下载失败 %s: %s" % (url, last))


def shape_of(text):
    """用项目自己的 parse_fjs 解析，返回规模摘要；解析失败返回 {"error": ...}。

    这里**必须复用生产解析器**，不能另写一份 —— 两份解析器会给出
    「下载成功但加载不了」这种最难查的失败（缺陷 19 同款）。
    """
    from rmoea_d.core.instance import parse_fjs

    try:
        inst = parse_fjs(text, seed=42)
    except Exception as exc:                       # noqa: BLE001 - 记录而非吞掉
        return {"error": "%s: %s" % (type(exc).__name__, exc)}
    if not inst["jobs"] or inst["total_ops"] <= 0:
        return {"error": "解析成功但结构为空"}
    alts = [len(op) for job in inst["jobs"] for op in job]
    if any(n <= 0 for n in alts):
        return {"error": "存在 0 个可选机器的工序"}
    if max(alts) > inst["n_machines"]:
        return {"error": "可选机器数 > 机器总数（%d > %d）"
                         % (max(alts), inst["n_machines"])}
    if any(alt[2] <= 0 for job in inst["jobs"] for op in job for alt in op):
        return {"error": "存在非正的加工时间"}
    return {
        "n_jobs": inst["n_jobs"],
        "n_machines": inst["n_machines"],
        "total_ops": inst["total_ops"],
        "ops_per_job_min": min(len(j) for j in inst["jobs"]),
        "ops_per_job_max": max(len(j) for j in inst["jobs"]),
        "flex_min": min(alts),
        "flex_max": max(alts),
        "flex_mean": round(sum(alts) / len(alts), 3),
    }


def save_prov(path, prov):
    """写 PROVENANCE.json（每次调用都带最新 _meta）。"""
    prov["_meta"] = dict(META,
                         fetched_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                         n_instances=sum(1 for k in prov if not k.startswith("_")))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(prov, fh, indent=1, ensure_ascii=False, sort_keys=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subsets", default="edata",
                    help="逗号分隔，可选 " + "/".join(SUBSETS) + "（默认 edata）")
    ap.add_argument("--limit", type=int, default=N_PER_SUBSET,
                    help="每个子集取前 N 个编号（默认 66）")
    ap.add_argument("--skip", type=int, default=0,
                    help="跳过前 N 个编号（默认 0）")
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=8,
                    help="并发下载线程数（默认 8）")
    ap.add_argument("--force", action="store_true",
                    help="即使已存在且校验通过也重新下载")
    args = ap.parse_args()

    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    bad = [s for s in subsets if s not in SUBSETS]
    if bad:
        print("[错误] 未知子集 %s，可选 %s" % (bad, list(SUBSETS)))
        return 2

    os.makedirs(args.out_dir, exist_ok=True)
    prov_path = os.path.join(args.out_dir, "PROVENANCE.json")
    prov = {}
    if os.path.exists(prov_path):
        with open(prov_path, encoding="utf-8") as fh:
            prov = json.load(fh)

    # ── 先筛出真正需要下载的（幂等续跑）──
    todo, n_skip = [], 0
    for sub in subsets:
        upstream_dir, tmpl, prefix = SUBSETS[sub]
        for i in range(args.skip + 1, args.skip + args.limit + 1):
            up_name = tmpl.format(i)
            key = "%s%02d" % (prefix, i)
            url = "%s/%s/%s" % (BASE, upstream_dir, up_name)
            path = os.path.join(args.out_dir, key + ".fjs")
            rec = prov.get(key, {})
            if os.path.exists(path) and not args.force and rec.get("sha256"):
                if hashlib.sha256(open(path, "rb").read()).hexdigest() == rec["sha256"]:
                    n_skip += 1
                    continue
            todo.append((key, sub, i, up_name, upstream_dir, url, path))

    print("待下载 %d 个，已存在且校验通过 %d 个（%d 线程）"
          % (len(todo), n_skip, args.workers))

    def work(task):
        key, sub, i, up_name, upstream_dir, url, path = task
        try:
            text = fetch(url)
        except RuntimeError as exc:
            return task, None, str(exc)
        return task, (text, shape_of(text)), None

    n_new, n_bad = 0, 0
    if todo:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(work, t) for t in todo]
            for fut in concurrent.futures.as_completed(futs):
                task, payload, err = fut.result()
                key, sub, i, up_name, upstream_dir, url, path = task
                if err:
                    print("  !! %-6s %s" % (key, err))
                    n_bad += 1
                    continue
                text, shp = payload
                if "error" in shp:
                    print("  !! %-6s 解析失败，已丢弃: %s" % (key, shp["error"]))
                    n_bad += 1
                    continue
                with open(path, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(text)
                prov[key] = {
                    "subset": sub,
                    "upstream_dir": upstream_dir,
                    "upstream_file": up_name,
                    "upstream_index": i,
                    "url": url,
                    "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
                    "chars": len(text),
                    "shape": shp,
                    "local": os.path.relpath(path, ROOT).replace(os.sep, "/"),
                }
                n_new += 1
                print("  ok %-6s %-22s %2dj x %2dm  ops=%-4d flex=[%d,%d]  mean=%.2f"
                      % (key, up_name, shp["n_jobs"], shp["n_machines"],
                         shp["total_ops"], shp["flex_min"], shp["flex_max"],
                         shp["flex_mean"]))
                save_prov(prov_path, prov)      # 增量落盘：中断也不丢进度

    save_prov(prov_path, prov)
    n_all = prov["_meta"]["n_instances"]
    print("\n新下载 %d 个，跳过 %d 个，丢弃 %d 个；共 %d 个实例在 %s"
          % (n_new, n_skip, n_bad, n_all, args.out_dir))
    print("provenance -> %s" % os.path.relpath(prov_path, ROOT))
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
