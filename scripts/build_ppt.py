#!/usr/bin/env python3
"""
Build 4-phase academic presentation PPTX for RMOEA/D.
分阶段生成4个独立PPT，覆盖：背景问题、算法方法、实验设计、结果分析。
"""
from __future__ import annotations

import os
import sys
import glob
import time
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# ══════════════════════════════════════════════════════════════════════
# 0. 路径与常量配置
# ══════════════════════════════════════════════════════════════════════
ROOT = Path(__file__).resolve().parent.parent
CHARTS_DIR = ROOT / "charts"
OUTPUT_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

C_PRIMARY = "#1A3A5C"
C_ACCENT = "#2E7D6B"
C_ALERT = "#C44536"
C_BG = "#F8F9FA"
C_DARK = "#212529"
C_MUTED = "#6C757D"
C_WHITE = "#FFFFFF"
C_LIGHT_BLUE = "#E8F1F8"
C_LIGHT_GREEN = "#E8F5F0"

FONT_EN = "Arial"
FONT_CN = "Microsoft YaHei"
FS_TITLE = 32
FS_SUBTITLE = 16
FS_SLIDE_TITLE = 24
FS_BODY = 14
FS_SMALL = 11
FS_TINY = 9

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

# ══════════════════════════════════════════════════════════════════════
# 1. 日志增强配置
# ══════════════════════════════════════════════════════════════════════
logger = logging.getLogger("build_ppt")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
console_handler.setFormatter(console_fmt)

log_file = ROOT / "logs" / "build_ppt.log"
log_file.parent.mkdir(parents=True, exist_ok=True)
file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")
file_handler.setLevel(logging.DEBUG)
file_fmt = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_fmt)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ══════════════════════════════════════════════════════════════════════
# 2. 智能图表发现器
# ══════════════════════════════════════════════════════════════════════
@dataclass
class ChartAsset:
    key: str
    path: Optional[Path]
    purpose: str


class ChartResolver:
    PATTERNS: List[Tuple[str, List[str], str]] = [
        ("hv_comparison",
         ["benchmark/*hv_comparison*.png", "benchmark/*benchmark_hv*.png"],
         "Hypervolume comparison (benchmark)"),
        ("makespan_comparison",
         ["benchmark/*makespan_comparison*.png", "benchmark/*benchmark_makespan*.png"],
         "Makespan comparison (benchmark)"),
        ("workload_comparison",
         ["benchmark/*workload_comparison*.png", "benchmark/*benchmark_workload*.png"],
         "Workload comparison (benchmark)"),
        ("cohens_d",
         ["benchmark/*cohens_d*.png", "benchmark/*effect_size*.png"],
         "Cohen's d heatmap (benchmark)"),
        ("pvalue_heatmap",
         ["benchmark/*pvalue*.png", "benchmark/*significance*.png"],
         "p-value heatmap (benchmark)"),
        ("stats_card",
         ["benchmark/*stats_card*.png", "benchmark/*summary_card*.png"],
         "Stats summary card (benchmark)"),
        ("benchmark_convergence",
         ["benchmark/*/convergence*.png"],
         "Convergence curve per instance (benchmark)"),
        ("pareto_front",
         ["benchmark/*/pareto_front*.png"],
         "Pareto front per instance"),
        ("fuzzy_pareto",
         ["benchmark/*/fuzzy_pareto*.png"],
         "Fuzzy Pareto front per instance"),
        ("ablation_hv",
         ["ablation/*ablation_hv*.png", "ablation/*hv_comparison*.png"],
         "Ablation HV comparison"),
        ("ablation_makespan",
         ["ablation/*ablation_makespan*.png", "ablation/*makespan_comparison*.png"],
         "Ablation makespan comparison"),
        ("ablation_workload",
         ["ablation/*ablation_workload*.png", "ablation/*workload_comparison*.png"],
         "Ablation workload comparison"),
        ("ablation_runtime",
         ["ablation/*ablation_runtime*.png", "ablation/*runtime_comparison*.png"],
         "Ablation runtime comparison"),
        ("ablation_cohens",
         ["ablation/*ablation_cohens*.png", "ablation/*ablation_cohen*.png"],
         "Ablation Cohen's d matrix"),
        ("ablation_pvalue",
         ["ablation/*ablation_pvalue*.png"],
         "Ablation p-value heatmap"),
        ("ablation_stats",
         ["ablation/*ablation_stats*.png"],
         "Ablation statistics card"),
        ("improvement",
         ["ablation/*improvement*.png", "ablation/*improvement_summary*.png"],
         "Improvement summary (ablation)"),
        ("pf_size",
         ["ablation/*pf_size*.png"],
         "PF size comparison (ablation)"),
        ("ablation_convergence",
         ["ablation/*/convergence*.png"],
         "Convergence curve per instance (ablation)"),
        ("gantt",
         ["schedules/*gantt*.png", "schedules/*/gantt*.png"],
         "Gantt chart sample"),
    ]

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.assets: dict[str, ChartAsset] = {}
        self._resolve()

    def _resolve(self) -> None:
        logger.info("[ChartResolver] Starting chart discovery in: %s", self.base_dir)
        total_candidates = 0
        matched = 0
        for key, patterns, desc in self.PATTERNS:
            candidates: List[Path] = []
            for pat in patterns:
                found = list(self.base_dir.rglob(pat.replace("**/", "")))
                if not found:
                    found = list(self.base_dir.glob(pat.replace("**/", "")))
                candidates.extend(found)
                total_candidates += len(found)
            seen = set()
            unique_candidates = []
            for p in sorted(candidates):
                if p not in seen:
                    seen.add(p)
                    unique_candidates.append(p)
            if unique_candidates:
                chosen = min(unique_candidates, key=lambda x: len(x.parts))
                self.assets[key] = ChartAsset(key=key, path=chosen, purpose=desc)
                matched += 1
                logger.debug("[ChartResolver] %-20s => %s (%d candidates)", key, chosen.name, len(unique_candidates))
            else:
                self.assets[key] = ChartAsset(key=key, path=None, purpose=desc)
                logger.debug("[ChartResolver] %-20s => NOT FOUND", key)
        logger.info("[ChartResolver] Matched %d/%d asset types. Scanned %d raw candidates.",
                    matched, len(self.PATTERNS), total_candidates)

    def get(self, key: str) -> Optional[Path]:
        asset = self.assets.get(key)
        return asset.path if asset else None

    def has_any(self, keys: List[str]) -> bool:
        return any(self.get(k) is not None for k in keys)


resolver = ChartResolver(CHARTS_DIR)


# ══════════════════════════════════════════════════════════════════════
# 3. SlideBuilder 辅助类
# ══════════════════════════════════════════════════════════════════════
class SlideBuilder:
    def __init__(self, prs: Presentation):
        self.prs = prs
        self.prs.slide_width = SLIDE_W
        self.prs.slide_height = SLIDE_H
        self._slide_count = 0

    def add_blank(self):
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self._slide_count += 1
        return slide

    @property
    def slide_count(self) -> int:
        return self._slide_count

    @staticmethod
    def set_bg(slide, color: str = C_WHITE):
        fill = slide.background.fill
        fill.solid()
        fill.fore_color.rgb = RGBColor.from_string(color[1:])

    @staticmethod
    def add_text(slide, left, top, width, height, text: str = "",
                 font_size: int = FS_BODY, bold: bool = False, color: str = C_DARK,
                 align=PP_ALIGN.LEFT, font_name: str = FONT_CN, word_wrap: bool = True):
        tx_box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
        tf = tx_box.text_frame
        tf.word_wrap = word_wrap
        p = tf.paragraphs[0]
        p.text = text
        p.font.size = Pt(font_size)
        p.font.bold = bold
        p.font.color.rgb = RGBColor.from_string(color[1:])
        p.font.name = font_name
        p.alignment = align
        return tf

    @staticmethod
    def add_rich_text(slide, left, top, width, height, font_name: str = FONT_CN, word_wrap: bool = True):
        tx_box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
        tf = tx_box.text_frame
        tf.word_wrap = word_wrap
        return tf

    @staticmethod
    def add_para(tf, text: str, font_size: int = FS_BODY, bold: bool = False,
                 color: str = C_DARK, align=PP_ALIGN.LEFT, space_after=Pt(6),
                 level: int = 0, font_name: str = FONT_CN):
        if len(tf.paragraphs) == 1 and tf.paragraphs[0].text == "":
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = text
        p.font.size = Pt(font_size)
        p.font.bold = bold
        p.font.color.rgb = RGBColor.from_string(color[1:])
        p.font.name = font_name
        p.alignment = align
        p.space_after = space_after
        p.level = level
        return p

    @staticmethod
    def add_shape_text(slide, left, top, width, height, text: str = "",
                       shape_type=MSO_SHAPE.RECTANGLE, fill_color: Optional[str] = None,
                       line_color: Optional[str] = None, font_size: int = FS_BODY,
                       bold: bool = False, color: str = C_DARK, align=PP_ALIGN.LEFT,
                       font_name: str = FONT_CN):
        shape = slide.shapes.add_shape(shape_type, Inches(left), Inches(top), Inches(width), Inches(height))
        if fill_color:
            shape.fill.solid()
            shape.fill.fore_color.rgb = RGBColor.from_string(fill_color[1:])
        else:
            shape.fill.background()
        if line_color:
            shape.line.color.rgb = RGBColor.from_string(line_color[1:])
            shape.line.width = Pt(1.0)
        else:
            shape.line.fill.background()
        tf = shape.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = text
        p.font.size = Pt(font_size)
        p.font.bold = bold
        p.font.color.rgb = RGBColor.from_string(color[1:])
        p.font.name = font_name
        p.alignment = align
        return shape

    @staticmethod
    def add_top_bar(slide, title: str, subtitle: str = ""):
        shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), SLIDE_W, Inches(1.05))
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor.from_string(C_PRIMARY[1:])
        shape.line.fill.background()
        SlideBuilder.add_text(slide, 0.5, 0.18, 12.3, 0.5, text=title,
                              font_size=FS_SLIDE_TITLE, bold=True, color=C_WHITE, font_name=FONT_CN)
        if subtitle:
            SlideBuilder.add_text(slide, 0.5, 0.62, 12.3, 0.35, text=subtitle,
                                  font_size=FS_SMALL, color="#AECBE3", font_name=FONT_EN)

    @staticmethod
    def add_bottom_bar(slide, text: str = "RMOEA/D 学术汇报"):
        shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(7.15), SLIDE_W, Inches(0.35))
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor.from_string(C_PRIMARY[1:])
        shape.line.fill.background()
        SlideBuilder.add_text(slide, 0.3, 7.18, 12.7, 0.3, text=text,
                              font_size=FS_TINY, color=C_WHITE, align=PP_ALIGN.RIGHT, font_name=FONT_EN)

    @staticmethod
    def add_image_safe(slide, img_path: Optional[Path], left, top, width, height=None,
                       placeholder_text: str = "[Chart not available]"):
        if img_path and img_path.exists():
            if height:
                slide.shapes.add_picture(str(img_path), Inches(left), Inches(top), Inches(width), Inches(height))
            else:
                slide.shapes.add_picture(str(img_path), Inches(left), Inches(top), Inches(width))
            logger.debug("Inserted image: %s", img_path.name)
            return True
        else:
            SlideBuilder.add_shape_text(slide, left, top, width, 0.6, text=placeholder_text,
                                        fill_color=C_LIGHT_BLUE, line_color=C_PRIMARY,
                                        font_size=FS_SMALL, color=C_MUTED, align=PP_ALIGN.CENTER)
            logger.debug("Missing image, placeholder added at (%.2f, %.2f)", left, top)
            return False

    @staticmethod
    def add_source_label(slide, text: str, left=0.5, top=7.0):
        SlideBuilder.add_text(slide, left, top, 12.0, 0.2, text=text,
                              font_size=FS_TINY, color=C_MUTED, font_name=FONT_EN)


# ══════════════════════════════════════════════════════════════════════
# 4. 分阶段 PPT 构建函数
# ══════════════════════════════════════════════════════════════════════

def _new_prs():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs


# ── Phase 1: 项目背景与问题定义 ──
def build_phase1(output_path: Path) -> int:
    """PPT 1: 项目背景与问题定义 (~8页)"""
    logger.info("=" * 60)
    logger.info("Building Phase 1: Background & Problem")
    prs = _new_prs()
    b = SlideBuilder(prs)

    # Slide 1: Title
    s = b.add_blank(); b.set_bg(s)
    top = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), SLIDE_W, Inches(4.2))
    top.fill.solid(); top.fill.fore_color.rgb = RGBColor.from_string(C_PRIMARY[1:]); top.line.fill.background()
    b.add_text(s, 0.8, 0.9, 11.5, 1.0, text="模糊柔性作业车间调度问题",
               font_size=FS_TITLE, bold=True, color=C_WHITE, align=PP_ALIGN.LEFT)
    b.add_text(s, 0.8, 1.9, 11.5, 0.7, text="及其研究背景",
               font_size=FS_TITLE, bold=True, color=C_WHITE, align=PP_ALIGN.LEFT)
    b.add_text(s, 0.8, 2.8, 11.5, 0.5,
               text="RMOEA/D Algorithm Reproduction Project — Phase 1: Background & Problem",
               font_size=FS_SUBTITLE, color="#AECBE3", font_name=FONT_EN)
    b.add_text(s, 0.8, 4.6, 11.5, 0.4, text="基于强化学习的 MOEA/D 双目标模糊柔性作业车间调度求解器",
               font_size=FS_BODY, bold=True, color=C_DARK)
    b.add_text(s, 0.8, 5.05, 11.5, 0.4, text="复现论文: ESWA 2022  |  Benchmark: Brandimarte Mk01–Mk10",
               font_size=FS_SMALL, color=C_MUTED, font_name=FONT_EN)
    b.add_bottom_bar(s, "Phase 1 / 4 — Background & Problem")

    # Slide 2: 研究背景
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "研究背景", "Research Background")
    box = b.add_shape_text(s, 0.5, 1.35, 12.3, 5.4, fill_color=C_LIGHT_BLUE, line_color=C_PRIMARY)
    tf = box.text_frame; tf.word_wrap = True; tf.paragraphs[0].text = ""
    b.add_para(tf, "作业车间调度问题 (Job Shop Scheduling)", font_size=FS_BODY+2, bold=True, color=C_PRIMARY, space_after=Pt(6))
    b.add_para(tf, "• 制造业核心优化问题：合理分配工序到机器，优化完工时间、资源利用率等指标",
               font_size=FS_BODY, space_after=Pt(4))
    b.add_para(tf, "• 传统 JSP 假设加工时间为确定值，实际生产中普遍存在不确定性",
               font_size=FS_BODY, space_after=Pt(10))
    b.add_para(tf, "柔性作业车间 (Flexible Job Shop)", font_size=FS_BODY+2, bold=True, color=C_PRIMARY, space_after=Pt(6))
    b.add_para(tf, "• 每道工序可在多台候选机器上加工，增加调度灵活性",
               font_size=FS_BODY, space_after=Pt(4))
    b.add_para(tf, "• 搜索空间急剧扩大，属于 NP-hard 问题，精确算法难以在合理时间内求解",
               font_size=FS_BODY, space_after=Pt(10))
    b.add_para(tf, "模糊柔性作业车间 (Fuzzy Flexible Job Shop)", font_size=FS_BODY+2, bold=True, color=C_PRIMARY, space_after=Pt(6))
    b.add_para(tf, "• 加工时间用三角模糊数 (Triangular Fuzzy Number, TFN) 描述，更贴近实际生产环境",
               font_size=FS_BODY, space_after=Pt(4))
    b.add_para(tf, "• 需要同时优化多个模糊目标，对算法的收敛性和多样性提出更高要求",
               font_size=FS_BODY, space_after=Pt(4))
    b.add_bottom_bar(s, "Phase 1 / 4 — Background & Problem")

    # Slide 3: 问题定义
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "问题定义", "Problem Formulation")
    left = b.add_shape_text(s, 0.5, 1.35, 5.8, 5.4, fill_color=C_LIGHT_BLUE, line_color=C_PRIMARY)
    ltf = left.text_frame; ltf.word_wrap = True; ltf.paragraphs[0].text = ""
    b.add_para(ltf, "数学模型", font_size=FS_BODY+2, bold=True, color=C_PRIMARY, space_after=Pt(6))
    b.add_para(ltf, "• 工件集合 J = {J₁, ..., Jₙ}", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(ltf, "• 机器集合 M = {M₁, ..., Mₘ}", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(ltf, "• 工序 Oᵢ,ⱼ 可在候选机器子集上加工", font_size=FS_BODY, space_after=Pt(6))
    b.add_para(ltf, "双目标优化", font_size=FS_BODY+2, bold=True, color=C_PRIMARY, space_after=Pt(6))
    b.add_para(ltf, "(1) 模糊 Makespan: max C̃ᵢ,last", font_size=FS_BODY, bold=True, space_after=Pt(2))
    b.add_para(ltf, "(2) 总机器负载: Σₖ Σᵢ,ⱼ p̃ᵢ,ⱼ,ₖ · xᵢ,ⱼ,ₖ", font_size=FS_BODY, bold=True, space_after=Pt(4))
    right = b.add_shape_text(s, 6.7, 1.35, 5.9, 5.4, fill_color=C_LIGHT_GREEN, line_color=C_ACCENT)
    rtf = right.text_frame; rtf.word_wrap = True; rtf.paragraphs[0].text = ""
    b.add_para(rtf, "三角模糊数 (TFN)", font_size=FS_BODY+2, bold=True, color=C_ACCENT, space_after=Pt(6))
    b.add_para(rtf, "加工时间表示为 p̃ = (t₁, t₂, t₃)", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(rtf, "• t₁: 最乐观时间", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(rtf, "• t₂: 最可能时间", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(rtf, "• t₃: 最悲观时间", font_size=FS_BODY, space_after=Pt(6))
    b.add_para(rtf, "模糊运算规则", font_size=FS_BODY+2, bold=True, color=C_ACCENT, space_after=Pt(6))
    b.add_para(rtf, "• 加法: 分量分别相加", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(rtf, "• 排序: 按 (t₁+2t₂+t₃)/4 比较", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(rtf, "• 取大: 按排序结果取较大模糊数", font_size=FS_BODY, space_after=Pt(2))
    b.add_bottom_bar(s, "Phase 1 / 4 — Background & Problem")

    # Slide 4: 数据集介绍
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "Brandimarte 数据集", "Benchmark Instances")
    box = b.add_shape_text(s, 0.5, 1.35, 12.3, 5.4, fill_color=C_LIGHT_BLUE, line_color=C_PRIMARY)
    tf = box.text_frame; tf.word_wrap = True; tf.paragraphs[0].text = ""
    b.add_para(tf, "Brandimarte 标准 FFJSP 实例 (Mk01 ~ Mk10)", font_size=FS_BODY+2, bold=True, color=C_PRIMARY, space_after=Pt(6))
    b.add_para(tf, "规模覆盖: 10×6 至 20×15 (工件数 × 机器数)", font_size=FS_BODY, space_after=Pt(4))
    b.add_para(tf, "模糊化方法: 对确定时间 b 生成 TFN (a, b, c)", font_size=FS_BODY, bold=True, space_after=Pt(4))
    b.add_para(tf, "  a ~ randint(0, ⌊b/2⌋),  c ~ randint(0, ⌊b/2⌋)", font_size=FS_SMALL, space_after=Pt(2))
    b.add_para(tf, "  固定 seed=42 确保实验可复现", font_size=FS_SMALL, space_after=Pt(10))
    instances = [
        ("Mk01", "10×6"), ("Mk02", "10×6"), ("Mk03", "15×8"), ("Mk04", "15×8"), ("Mk05", "15×4"),
        ("Mk06", "10×10"), ("Mk07", "20×5"), ("Mk08", "20×10"), ("Mk09", "20×10"), ("Mk10", "20×15")
    ]
    b.add_para(tf, "实例规模一览:", font_size=FS_BODY, bold=True, space_after=Pt(4))
    row_text = "  ".join([f"{name}({size})" for name, size in instances])
    b.add_para(tf, row_text, font_size=FS_SMALL, space_after=Pt(4))
    b.add_bottom_bar(s, "Phase 1 / 4 — Background & Problem")

    # Slide 5: 研究动机与挑战
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "研究动机与挑战", "Motivation & Challenges")
    left = b.add_shape_text(s, 0.5, 1.35, 5.8, 5.4, fill_color=C_LIGHT_BLUE, line_color=C_PRIMARY)
    ltf = left.text_frame; ltf.word_wrap = True; ltf.paragraphs[0].text = ""
    b.add_para(ltf, "MOEA/D 框架的瓶颈", font_size=FS_BODY+2, bold=True, color=C_PRIMARY, space_after=Pt(6))
    b.add_para(ltf, "邻域大小 T 的选取困境:", font_size=FS_BODY, bold=True, space_after=Pt(2))
    b.add_para(ltf, "• T 过小 → 过度局部，陷入局部最优", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(ltf, "• T 过大 → 过度全局，收敛缓慢", font_size=FS_BODY, space_after=Pt(10))
    b.add_para(ltf, "局部搜索策略盲目:", font_size=FS_BODY, bold=True, space_after=Pt(2))
    b.add_para(ltf, "• 固定策略难以适应不同搜索阶段", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(ltf, "• 轮询执行效率低，缺乏自适应指导", font_size=FS_BODY, space_after=Pt(4))
    right = b.add_shape_text(s, 6.7, 1.35, 5.9, 5.4, fill_color=C_LIGHT_GREEN, line_color=C_ACCENT)
    rtf = right.text_frame; rtf.word_wrap = True; rtf.paragraphs[0].text = ""
    b.add_para(rtf, "核心研究问题", font_size=FS_BODY+2, bold=True, color=C_ACCENT, space_after=Pt(6))
    b.add_para(rtf, "如何通过强化学习实现 MOEA/D 参数与局部搜索策略的动态自适应？",
               font_size=FS_BODY, bold=True, color=C_ACCENT, space_after=Pt(6))
    b.add_para(rtf, "→ 引入 Q-learning 自适应选择邻域大小 T", font_size=FS_BODY, space_after=Pt(4))
    b.add_para(rtf, "→ 引入 RVNS 基于记忆机制动态选择局部搜索算子", font_size=FS_BODY, space_after=Pt(4))
    b.add_para(rtf, "→ 形成 RMOEA/D 算法框架", font_size=FS_BODY, bold=True, color=C_ALERT, space_after=Pt(4))
    b.add_bottom_bar(s, "Phase 1 / 4 — Background & Problem")

    # Slide 6: 技术路线概览
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "技术路线概览", "Technical Roadmap")
    box = b.add_shape_text(s, 0.5, 1.35, 12.3, 5.4, fill_color=C_LIGHT_BLUE, line_color=C_PRIMARY)
    tf = box.text_frame; tf.word_wrap = True; tf.paragraphs[0].text = ""
    b.add_para(tf, "RMOEA/D 三大核心组件", font_size=FS_BODY+2, bold=True, color=C_PRIMARY, space_after=Pt(8))
    b.add_para(tf, "1. Q-PAS (Q-learning Parameter Adaptation Strategy)", font_size=FS_BODY, bold=True, color=C_PRIMARY, space_after=Pt(2))
    b.add_para(tf, "   将邻域大小 T 的选择建模为 MDP，通过 Q-learning 在线学习最优 T",
               font_size=FS_SMALL, space_after=Pt(6))
    b.add_para(tf, "2. RVNS (Reinforcement Learning-based Variable Neighborhood Search)", font_size=FS_BODY, bold=True, color=C_ACCENT, space_after=Pt(2))
    b.add_para(tf, "   5 种邻域搜索算子 + 滑动窗口成功/失败记忆，轮盘赌动态分配选择概率",
               font_size=FS_SMALL, space_after=Pt(6))
    b.add_para(tf, "3. 精英档案与 MIX3 初始化", font_size=FS_BODY, bold=True, color=C_ALERT, space_after=Pt(2))
    b.add_para(tf, "   融合 Random / LS / GW 三种初始化规则，外部存档回收历史非支配解",
               font_size=FS_SMALL, space_after=Pt(6))
    b.add_bottom_bar(s, "Phase 1 / 4 — Background & Problem")

    # Slide 7: 项目目标
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "项目目标", "Project Objectives")
    box = b.add_shape_text(s, 0.5, 1.35, 12.3, 5.4, fill_color=C_LIGHT_BLUE, line_color=C_PRIMARY)
    tf = box.text_frame; tf.word_wrap = True; tf.paragraphs[0].text = ""
    b.add_para(tf, "复现目标", font_size=FS_BODY+2, bold=True, color=C_PRIMARY, space_after=Pt(6))
    b.add_para(tf, "1. 完整复现 RMOEA/D 核心算法框架 (MOEA/D + Q-PAS + RVNS)", font_size=FS_BODY, space_after=Pt(4))
    b.add_para(tf, "2. 在 Brandimarte Mk01~Mk10 上运行对比实验与消融实验", font_size=FS_BODY, space_after=Pt(4))
    b.add_para(tf, "3. 验证算法在 HV、Makespan、Workload 等指标上的优越性", font_size=FS_BODY, space_after=Pt(4))
    b.add_para(tf, "4. 通过统计检验 (Wilcoxon, Friedman, Cohen's d) 验证结果显著性", font_size=FS_BODY, space_after=Pt(10))
    b.add_para(tf, "预期贡献", font_size=FS_BODY+2, bold=True, color=C_PRIMARY, space_after=Pt(6))
    b.add_para(tf, "• 验证 Q-learning 参数自适应在 MOEA/D 中的有效性", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(tf, "• 验证 RVNS 局部搜索对解质量的提升作用", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(tf, "• 提供完整的可复现代码库与实验流程", font_size=FS_BODY, space_after=Pt(2))
    b.add_bottom_bar(s, "Phase 1 / 4 — Background & Problem")

    # Slide 8: Phase 1 总结
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "Phase 1 小结", "Phase 1 Summary")
    box = b.add_shape_text(s, 0.5, 1.35, 12.3, 5.4, fill_color=C_LIGHT_GREEN, line_color=C_ACCENT)
    tf = box.text_frame; tf.word_wrap = True; tf.paragraphs[0].text = ""
    b.add_para(tf, "核心要点", font_size=FS_BODY+2, bold=True, color=C_ACCENT, space_after=Pt(8))
    b.add_para(tf, "• 模糊柔性作业车间调度是制造业的核心 NP-hard 优化问题", font_size=FS_BODY, space_after=Pt(4))
    b.add_para(tf, "• MOEA/D 的邻域大小 T 选取直接影响收敛性与多样性的平衡", font_size=FS_BODY, space_after=Pt(4))
    b.add_para(tf, "• 传统局部搜索策略缺乏自适应能力，效率低下", font_size=FS_BODY, space_after=Pt(4))
    b.add_para(tf, "• RMOEA/D 通过 Q-PAS 和 RVNS 引入强化学习，实现参数与策略的动态自适应", font_size=FS_BODY, space_after=Pt(10))
    b.add_para(tf, "下一步: 深入介绍 RMOEA/D 算法核心方法 →", font_size=FS_BODY, bold=True, color=C_ALERT, space_after=Pt(6))
    b.add_bottom_bar(s, "Phase 1 / 4 — Background & Problem")

    prs.save(str(output_path))
    logger.info("Phase 1 saved: %s (%d slides)", output_path, b.slide_count)
    return b.slide_count


# ── Phase 2: 算法核心方法 ──
def build_phase2(output_path: Path) -> int:
    """PPT 2: 算法核心方法 (~10页)"""
    logger.info("=" * 60)
    logger.info("Building Phase 2: Algorithm Methods")
    prs = _new_prs()
    b = SlideBuilder(prs)

    # Slide 1: Title
    s = b.add_blank(); b.set_bg(s)
    top = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), SLIDE_W, Inches(4.2))
    top.fill.solid(); top.fill.fore_color.rgb = RGBColor.from_string(C_PRIMARY[1:]); top.line.fill.background()
    b.add_text(s, 0.8, 0.9, 11.5, 1.0, text="RMOEA/D 算法核心方法",
               font_size=FS_TITLE, bold=True, color=C_WHITE, align=PP_ALIGN.LEFT)
    b.add_text(s, 0.8, 1.9, 11.5, 0.7, text="MOEA/D + Q-PAS + RVNS",
               font_size=FS_TITLE, bold=True, color=C_WHITE, align=PP_ALIGN.LEFT)
    b.add_text(s, 0.8, 2.8, 11.5, 0.5,
               text="Phase 2: Algorithm Core Methods",
               font_size=FS_SUBTITLE, color="#AECBE3", font_name=FONT_EN)
    b.add_text(s, 0.8, 4.6, 11.5, 0.4, text="基于强化学习的多目标进化算法框架详解",
               font_size=FS_BODY, bold=True, color=C_DARK)
    b.add_bottom_bar(s, "Phase 2 / 4 — Algorithm Methods")

    # Slide 2: 总体框架
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "RMOEA/D 总体框架", "Overall Framework")
    left = b.add_shape_text(s, 0.5, 1.35, 5.8, 5.4, fill_color=C_LIGHT_BLUE, line_color=C_PRIMARY)
    ltf = left.text_frame; ltf.word_wrap = True; ltf.paragraphs[0].text = ""
    b.add_para(ltf, "算法输入输出", font_size=FS_BODY+2, bold=True, color=C_PRIMARY, space_after=Pt(6))
    b.add_para(ltf, "输入: Np, Gen, CR, Q-learning参数, RVNS参数", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(ltf, "输出: 最终非支配解集 PF 与 HV 值", font_size=FS_BODY, space_after=Pt(10))
    b.add_para(ltf, "核心循环 (per generation)", font_size=FS_BODY+2, bold=True, color=C_PRIMARY, space_after=Pt(6))
    b.add_para(ltf, "1. RVNS 自适应局部搜索", font_size=FS_BODY, bold=True, space_after=Pt(2))
    b.add_para(ltf, "2. Q-PAS 选择邻域大小 T", font_size=FS_BODY, bold=True, space_after=Pt(2))
    b.add_para(ltf, "3. MOEA/D 基于 T 演化一代", font_size=FS_BODY, bold=True, space_after=Pt(2))
    b.add_para(ltf, "4. 更新精英档案", font_size=FS_BODY, bold=True, space_after=Pt(2))
    b.add_para(ltf, "5. Q-learning 更新 Q-table", font_size=FS_BODY, bold=True, space_after=Pt(4))
    right = b.add_shape_text(s, 6.7, 1.35, 5.9, 5.4, fill_color=C_LIGHT_GREEN, line_color=C_ACCENT)
    rtf = right.text_frame; rtf.word_wrap = True; rtf.paragraphs[0].text = ""
    b.add_para(rtf, "三大组件协同", font_size=FS_BODY+2, bold=True, color=C_ACCENT, space_after=Pt(6))
    b.add_para(rtf, "MOEA/D: 提供基础多目标演化框架", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(rtf, "Q-PAS: 动态调整邻域协作范围", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(rtf, "RVNS: 增强局部 exploitation 能力", font_size=FS_BODY, space_after=Pt(10))
    b.add_para(rtf, "精英档案: 回收历史优秀解，防止退化", font_size=FS_BODY, space_after=Pt(4))
    b.add_bottom_bar(s, "Phase 2 / 4 — Algorithm Methods")

    # Slide 3: 编码与解码
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "编码与解码", "Encoding & Decoding")
    left = b.add_shape_text(s, 0.5, 1.35, 5.8, 5.4, fill_color=C_LIGHT_BLUE, line_color=C_PRIMARY)
    ltf = left.text_frame; ltf.word_wrap = True; ltf.paragraphs[0].text = ""
    b.add_para(ltf, "OS + MA 双向量编码", font_size=FS_BODY+2, bold=True, color=C_PRIMARY, space_after=Pt(6))
    b.add_para(ltf, "OS (Operation Sequence)", font_size=FS_BODY, bold=True, space_after=Pt(2))
    b.add_para(ltf, "• 长度 = 总工序数", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(ltf, "• 元素为工件号，出现次数 = 该工件工序数", font_size=FS_BODY, space_after=Pt(6))
    b.add_para(ltf, "MA (Machine Assignment)", font_size=FS_BODY, bold=True, space_after=Pt(2))
    b.add_para(ltf, "• 长度 = 总工序数", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(ltf, "• 元素为机器编号，对应 OS 中工序的加工机器", font_size=FS_BODY, space_after=Pt(4))
    right = b.add_shape_text(s, 6.7, 1.35, 5.9, 5.4, fill_color=C_LIGHT_GREEN, line_color=C_ACCENT)
    rtf = right.text_frame; rtf.word_wrap = True; rtf.paragraphs[0].text = ""
    b.add_para(rtf, "解码过程", font_size=FS_BODY+2, bold=True, color=C_ACCENT, space_after=Pt(6))
    b.add_para(rtf, "1. OS → 具体工序序列 (按出现次序)", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(rtf, "2. 依次安排每道工序到 MA 指定机器", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(rtf, "3. 维护每台机器的模糊时钟", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(rtf, "4. 开始时间 = max(工件前序完成时间, 机器空闲时间)", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(rtf, "5. 完成时间 = 开始时间 + 模糊加工时间", font_size=FS_BODY, space_after=Pt(4))
    b.add_bottom_bar(s, "Phase 2 / 4 — Algorithm Methods")

    # Slide 4: MOEA/D 分解策略
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "MOEA/D 分解策略", "MOEA/D Decomposition")
    left = b.add_shape_text(s, 0.5, 1.35, 5.8, 5.4, fill_color=C_LIGHT_BLUE, line_color=C_PRIMARY)
    ltf = left.text_frame; ltf.word_wrap = True; ltf.paragraphs[0].text = ""
    b.add_para(ltf, "权重向量", font_size=FS_BODY+2, bold=True, color=C_PRIMARY, space_after=Pt(6))
    b.add_para(ltf, "• Das & Dennis 方法生成 Np=100 个均匀分布向量", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(ltf, "• 每个向量 λ = (λ₁, λ₂) 对应一个子问题", font_size=FS_BODY, space_after=Pt(6))
    b.add_para(ltf, "Tchebycheff 聚合函数", font_size=FS_BODY+2, bold=True, color=C_PRIMARY, space_after=Pt(6))
    b.add_para(ltf, "gᵗᵉ(x|λ,z) = maxₖ { λₖ · |fₖ(x) − zₖ| }", font_size=FS_BODY, bold=True, space_after=Pt(4))
    b.add_para(ltf, "• z 为参考点 (各目标当前最小值)", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(ltf, "• 模糊目标转换为清晰值: (t₁+2t₂+t₃)/4", font_size=FS_BODY, space_after=Pt(4))
    right = b.add_shape_text(s, 6.7, 1.35, 5.9, 5.4, fill_color=C_LIGHT_GREEN, line_color=C_ACCENT)
    rtf = right.text_frame; rtf.word_wrap = True; rtf.paragraphs[0].text = ""
    b.add_para(rtf, "邻域协作更新", font_size=FS_BODY+2, bold=True, color=C_ACCENT, space_after=Pt(6))
    b.add_para(rtf, "• 每个子问题 i 有邻域 B(i): 距离最近的 T 个权重索引", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(rtf, "• 从 B(i) 选父代交叉变异产生子代", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(rtf, "• 子代与 B(i) 中所有邻居比较 Tchebycheff 值", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(rtf, "• 若更优则替换，实现邻域协作进化", font_size=FS_BODY, space_after=Pt(4))
    b.add_bottom_bar(s, "Phase 2 / 4 — Algorithm Methods")

    # Slide 5: 遗传算子
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "遗传算子", "Genetic Operators")
    left = b.add_shape_text(s, 0.5, 1.35, 5.8, 5.4, fill_color=C_LIGHT_BLUE, line_color=C_PRIMARY)
    ltf = left.text_frame; ltf.word_wrap = True; ltf.paragraphs[0].text = ""
    b.add_para(ltf, "交叉算子", font_size=FS_BODY+2, bold=True, color=C_PRIMARY, space_after=Pt(6))
    b.add_para(ltf, "POX (Precedence Operation Crossover)", font_size=FS_BODY, bold=True, space_after=Pt(2))
    b.add_para(ltf, "• 随机分工件为两组", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(ltf, "• 子代1保留父1的组1工件位置", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(ltf, "• 剩余空位按父2中这些工件出现顺序填充", font_size=FS_BODY, space_after=Pt(6))
    b.add_para(ltf, "UX (Uniform Crossover)", font_size=FS_BODY, bold=True, space_after=Pt(2))
    b.add_para(ltf, "• 二进制掩码决定机器分配交换", font_size=FS_BODY, space_after=Pt(4))
    right = b.add_shape_text(s, 6.7, 1.35, 5.9, 5.4, fill_color=C_LIGHT_GREEN, line_color=C_ACCENT)
    rtf = right.text_frame; rtf.word_wrap = True; rtf.paragraphs[0].text = ""
    b.add_para(rtf, "变异算子", font_size=FS_BODY+2, bold=True, color=C_ACCENT, space_after=Pt(6))
    b.add_para(rtf, "OS 变异", font_size=FS_BODY, bold=True, space_after=Pt(2))
    b.add_para(rtf, "• 随机交换 OS 中两个位置", font_size=FS_BODY, space_after=Pt(6))
    b.add_para(rtf, "MA 变异", font_size=FS_BODY, bold=True, space_after=Pt(2))
    b.add_para(rtf, "• 随机选一个工序，改为候选集中另一台机器", font_size=FS_BODY, space_after=Pt(6))
    b.add_para(rtf, "MIX3 初始化", font_size=FS_BODY+2, bold=True, color=C_ACCENT, space_after=Pt(6))
    b.add_para(rtf, "• 1/3 Random + 1/3 LS + 1/3 GW", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(rtf, "• 兼顾多样性与质量", font_size=FS_BODY, space_after=Pt(4))
    b.add_bottom_bar(s, "Phase 2 / 4 — Algorithm Methods")

    # Slide 6: Q-PAS 参数自适应
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "Q-PAS: Q-learning 参数自适应", "Q-PAS Detail")
    left = b.add_shape_text(s, 0.5, 1.35, 5.8, 5.4, fill_color=C_LIGHT_BLUE, line_color=C_PRIMARY)
    ltf = left.text_frame; ltf.word_wrap = True; ltf.paragraphs[0].text = ""
    b.add_para(ltf, "状态空间 (4 States)", font_size=FS_BODY+2, bold=True, color=C_PRIMARY, space_after=Pt(6))
    b.add_para(ltf, "S0: ΔCV>0 & ΔDV>0  → 收敛+ 多样+", font_size=FS_SMALL, space_after=Pt(2))
    b.add_para(ltf, "S1: ΔCV>0 & ΔDV≤0  → 收敛+ 多样−", font_size=FS_SMALL, space_after=Pt(2))
    b.add_para(ltf, "S2: ΔCV≤0 & ΔDV>0  → 收敛− 多样+", font_size=FS_SMALL, space_after=Pt(2))
    b.add_para(ltf, "S3: ΔCV≤0 & ΔDV≤0  → 收敛− 多样−", font_size=FS_SMALL, space_after=Pt(6))
    b.add_para(ltf, "奖励函数", font_size=FS_BODY+2, bold=True, color=C_PRIMARY, space_after=Pt(6))
    b.add_para(ltf, "R = 10  if ΔDV > 0  else 0", font_size=FS_BODY, bold=True, space_after=Pt(4))
    right = b.add_shape_text(s, 6.7, 1.35, 5.9, 5.4, fill_color=C_LIGHT_GREEN, line_color=C_ACCENT)
    rtf = right.text_frame; rtf.word_wrap = True; rtf.paragraphs[0].text = ""
    b.add_para(rtf, "超参数", font_size=FS_BODY+2, bold=True, color=C_ACCENT, space_after=Pt(6))
    b.add_para(rtf, "α = 0.4   (学习率)", font_size=FS_BODY, bold=True, space_after=Pt(2))
    b.add_para(rtf, "γ = 0.6   (折扣因子)", font_size=FS_BODY, bold=True, space_after=Pt(2))
    b.add_para(rtf, "ε = 0.8   (探索率)", font_size=FS_BODY, bold=True, space_after=Pt(2))
    b.add_para(rtf, "动作: T ∈ {5, 10, 15, 20}", font_size=FS_BODY, bold=True, space_after=Pt(10))
    b.add_para(rtf, "Q 值更新", font_size=FS_BODY+2, bold=True, color=C_ACCENT, space_after=Pt(6))
    b.add_para(rtf, "Q(s,a) ← Q(s,a) + α[R + γ·max Q(s',·) − Q(s,a)]", font_size=FS_SMALL, space_after=Pt(4))
    b.add_bottom_bar(s, "Phase 2 / 4 — Algorithm Methods")

    # Slide 7: RVNS 变邻域搜索
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "RVNS: 变邻域搜索", "RVNS Detail")
    left = b.add_shape_text(s, 0.5, 1.35, 5.8, 5.4, fill_color=C_LIGHT_BLUE, line_color=C_PRIMARY)
    ltf = left.text_frame; ltf.word_wrap = True; ltf.paragraphs[0].text = ""
    b.add_para(ltf, "5 种邻域搜索算子", font_size=FS_BODY+2, bold=True, color=C_PRIMARY, space_after=Pt(6))
    ops = [
        ("LS1", "随机机器交换"), ("LS2", "最小时间选择"), ("LS3", "负载均衡调整"),
        ("LS4", "位置交换"), ("LS5", "位置插入")
    ]
    for name, desc in ops:
        b.add_para(ltf, f"• {name}: {desc}", font_size=FS_BODY, space_after=Pt(3))
    right = b.add_shape_text(s, 6.7, 1.35, 5.9, 5.4, fill_color=C_LIGHT_GREEN, line_color=C_ACCENT)
    rtf = right.text_frame; rtf.word_wrap = True; rtf.paragraphs[0].text = ""
    b.add_para(rtf, "自适应记忆机制", font_size=FS_BODY+2, bold=True, color=C_ACCENT, space_after=Pt(6))
    b.add_para(rtf, "滑动窗口 lp = 40 代", font_size=FS_BODY, bold=True, space_after=Pt(2))
    b.add_para(rtf, "• 统计最近 40 代内各算子成功/失败次数", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(rtf, "• 成功率高 → 选择概率上升 (正向强化)", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(rtf, "• 失败率高 → 选择概率下降 (负向强化)", font_size=FS_BODY, space_after=Pt(6))
    b.add_para(rtf, "选择策略", font_size=FS_BODY+2, bold=True, color=C_ACCENT, space_after=Pt(6))
    b.add_para(rtf, "• 基于轮盘赌的概率分配", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(rtf, "• 引导解动态选择最合适的局部搜索方法", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(rtf, "• 避免盲目轮询，提升搜索效率", font_size=FS_BODY, space_after=Pt(4))
    b.add_bottom_bar(s, "Phase 2 / 4 — Algorithm Methods")

    # Slide 8: 精英档案
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "精英档案机制", "Elite Archive")
    box = b.add_shape_text(s, 0.5, 1.35, 12.3, 5.4, fill_color=C_LIGHT_BLUE, line_color=C_PRIMARY)
    tf = box.text_frame; tf.word_wrap = True; tf.paragraphs[0].text = ""
    b.add_para(tf, "外部精英档案", font_size=FS_BODY+2, bold=True, color=C_PRIMARY, space_after=Pt(6))
    b.add_para(tf, "• 维护大小为 Np 的外部存档 A", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(tf, "• 每代结束后: A = A ∪ 当前 PF", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(tf, "• 对 A 进行非支配排序，保留前 Np 个个体", font_size=FS_BODY, space_after=Pt(6))
    b.add_para(tf, "作用", font_size=FS_BODY+2, bold=True, color=C_PRIMARY, space_after=Pt(6))
    b.add_para(tf, "• 防止优秀解在进化过程中丢失", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(tf, "• 提高种群质量与解的利用率", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(tf, "• 最终输出使用存档内容，保证解集质量", font_size=FS_BODY, space_after=Pt(4))
    b.add_bottom_bar(s, "Phase 2 / 4 — Algorithm Methods")

    # Slide 9: 算法流程图
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "算法流程", "Algorithm Workflow")
    box = b.add_shape_text(s, 0.5, 1.35, 12.3, 5.4, fill_color=C_LIGHT_BLUE, line_color=C_PRIMARY)
    tf = box.text_frame; tf.word_wrap = True; tf.paragraphs[0].text = ""
    steps = [
        "1. 初始化权重向量 W, Q-table, RVNS 记忆",
        "2. MIX3 初始化种群 P, 参考点 z*",
        "3. For gen = 1 to Gen:",
        "   a. RVNS: 对每个个体执行自适应局部搜索",
        "   b. Q-PAS: 根据当前 PF 状态选择邻域大小 T",
        "   c. MOEA/D: 基于 T 重建邻居并演化一代",
        "   d. 更新外部精英档案 Archive",
        "   e. Q-learning: 依据 ΔCV, ΔDV 更新 Q-table",
        "4. 输出最终非支配解集 PF 与 HV"
    ]
    for line in steps:
        is_bold = line.startswith(("3.", "   a.", "   b.", "   c."))
        b.add_para(tf, line, font_size=FS_BODY, bold=is_bold, color=C_DARK, space_after=Pt(4))
    b.add_bottom_bar(s, "Phase 2 / 4 — Algorithm Methods")

    # Slide 10: Phase 2 总结
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "Phase 2 小结", "Phase 2 Summary")
    box = b.add_shape_text(s, 0.5, 1.35, 12.3, 5.4, fill_color=C_LIGHT_GREEN, line_color=C_ACCENT)
    tf = box.text_frame; tf.word_wrap = True; tf.paragraphs[0].text = ""
    b.add_para(tf, "核心要点", font_size=FS_BODY+2, bold=True, color=C_ACCENT, space_after=Pt(8))
    b.add_para(tf, "• RMOEA/D = MOEA/D + Q-PAS + RVNS + 精英档案", font_size=FS_BODY, space_after=Pt(4))
    b.add_para(tf, "• OS+MA 编码简洁高效，解码过程基于模糊时钟", font_size=FS_BODY, space_after=Pt(4))
    b.add_para(tf, "• Q-PAS 将 T 的选择建模为 4 状态 MDP，通过 Q-learning 在线学习", font_size=FS_BODY, space_after=Pt(4))
    b.add_para(tf, "• RVNS 利用 5 种算子 + 滑动窗口记忆，轮盘赌动态选择", font_size=FS_BODY, space_after=Pt(4))
    b.add_para(tf, "• 精英档案回收历史非支配解，防止退化", font_size=FS_BODY, space_after=Pt(10))
    b.add_para(tf, "下一步: 实验设计与复现过程 →", font_size=FS_BODY, bold=True, color=C_ALERT, space_after=Pt(6))
    b.add_bottom_bar(s, "Phase 2 / 4 — Algorithm Methods")

    prs.save(str(output_path))
    logger.info("Phase 2 saved: %s (%d slides)", output_path, b.slide_count)
    return b.slide_count


# ── Phase 3: 实验设计与复现过程 ──
def build_phase3(output_path: Path) -> int:
    """PPT 3: 实验设计与复现过程 (~8页)"""
    logger.info("=" * 60)
    logger.info("Building Phase 3: Experiment Design & Reproduction")
    prs = _new_prs()
    b = SlideBuilder(prs)

    # Slide 1: Title
    s = b.add_blank(); b.set_bg(s)
    top = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), SLIDE_W, Inches(4.2))
    top.fill.solid(); top.fill.fore_color.rgb = RGBColor.from_string(C_PRIMARY[1:]); top.line.fill.background()
    b.add_text(s, 0.8, 0.9, 11.5, 1.0, text="实验设计与复现过程",
               font_size=FS_TITLE, bold=True, color=C_WHITE, align=PP_ALIGN.LEFT)
    b.add_text(s, 0.8, 1.9, 11.5, 0.7, text="Experiment Design & Reproduction Workflow",
               font_size=FS_SUBTITLE, bold=True, color=C_WHITE, align=PP_ALIGN.LEFT)
    b.add_text(s, 0.8, 2.8, 11.5, 0.5,
               text="Phase 3: How We Run, Measure, and Validate",
               font_size=FS_SUBTITLE, color="#AECBE3", font_name=FONT_EN)
    b.add_text(s, 0.8, 4.6, 11.5, 0.4, text="从参数配置到一键执行：完整的可复现实验流程",
               font_size=FS_BODY, bold=True, color=C_DARK)
    b.add_bottom_bar(s, "Phase 3 / 4 — Experiment Design")

    # Slide 2: 实验环境
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "实验环境配置", "Experimental Environment")
    left = b.add_shape_text(s, 0.5, 1.35, 5.8, 5.4, fill_color=C_LIGHT_BLUE, line_color=C_PRIMARY)
    ltf = left.text_frame; ltf.word_wrap = True; ltf.paragraphs[0].text = ""
    b.add_para(ltf, "硬件环境", font_size=FS_BODY+2, bold=True, color=C_PRIMARY, space_after=Pt(6))
    b.add_para(ltf, "• CPU: 现代多核处理器", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(ltf, "• 内存: 8GB+", font_size=FS_BODY, space_after=Pt(10))
    b.add_para(ltf, "软件环境", font_size=FS_BODY+2, bold=True, color=C_PRIMARY, space_after=Pt(6))
    b.add_para(ltf, "• Python 3.8+", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(ltf, "• NumPy, Matplotlib, Pandas, python-pptx", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(ltf, "• 无 GPU 依赖，纯 CPU 运行", font_size=FS_BODY, space_after=Pt(4))
    right = b.add_shape_text(s, 6.7, 1.35, 5.9, 5.4, fill_color=C_LIGHT_GREEN, line_color=C_ACCENT)
    rtf = right.text_frame; rtf.word_wrap = True; rtf.paragraphs[0].text = ""
    b.add_para(rtf, "代码结构", font_size=FS_BODY+2, bold=True, color=C_ACCENT, space_after=Pt(6))
    b.add_para(rtf, "src/rmoea_d/", font_size=FS_BODY, bold=True, space_after=Pt(2))
    b.add_para(rtf, "• core/ — 问题模型与编码解码", font_size=FS_SMALL, space_after=Pt(2))
    b.add_para(rtf, "• algorithm.py — RMOEA/D 主算法", font_size=FS_SMALL, space_after=Pt(2))
    b.add_para(rtf, "• moead_baseline.py — MOEA/D 基线", font_size=FS_SMALL, space_after=Pt(2))
    b.add_para(rtf, "• utils/ — 实验运行与可视化", font_size=FS_SMALL, space_after=Pt(4))
    b.add_bottom_bar(s, "Phase 3 / 4 — Experiment Design")

    # Slide 3: 参数设置
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "参数设置", "Parameter Settings")
    box = b.add_shape_text(s, 0.5, 1.35, 12.3, 5.4, fill_color=C_LIGHT_BLUE, line_color=C_PRIMARY)
    tf = box.text_frame; tf.word_wrap = True; tf.paragraphs[0].text = ""
    b.add_para(tf, "通用参数", font_size=FS_BODY+2, bold=True, color=C_PRIMARY, space_after=Pt(6))
    b.add_para(tf, "• 种群大小 Np = 100", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(tf, "• 最大代数 Gen = 200", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(tf, "• 交叉率 CR = 0.8", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(tf, "• 独立运行次数 = 30 次", font_size=FS_BODY, space_after=Pt(10))
    b.add_para(tf, "Q-learning 参数", font_size=FS_BODY+2, bold=True, color=C_PRIMARY, space_after=Pt(6))
    b.add_para(tf, "• α = 0.4, γ = 0.6, ε = 0.8", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(tf, "• 动作空间 T ∈ {5, 10, 15, 20}", font_size=FS_BODY, space_after=Pt(10))
    b.add_para(tf, "RVNS 参数", font_size=FS_BODY+2, bold=True, color=C_PRIMARY, space_after=Pt(6))
    b.add_para(tf, "• 滑动窗口 lp = 40", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(tf, "• 5 种 LS 算子", font_size=FS_BODY, space_after=Pt(2))
    b.add_bottom_bar(s, "Phase 3 / 4 — Experiment Design")

    # Slide 4: 实验流程
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "一键实验流程", "One-Click Experiment")
    box = b.add_shape_text(s, 0.5, 1.35, 12.3, 5.4, fill_color=C_LIGHT_BLUE, line_color=C_PRIMARY)
    tf = box.text_frame; tf.word_wrap = True; tf.paragraphs[0].text = ""
    b.add_para(tf, "run_all.py 执行流程", font_size=FS_BODY+2, bold=True, color=C_PRIMARY, space_after=Pt(6))
    steps = [
        "1. 加载 Brandimarte 实例 (Mk01~Mk10)",
        "2. 生成模糊化测试用例 (seed=42)",
        "3. 并行运行 Benchmark 实验 (RMOEA/D vs MOEA/D vs ...)",
        "4. 并行运行 Ablation 实验 (Full vs qpas_only vs rvns_only vs baseline)",
        "5. 自动汇总结果到 results/benchmark/ 和 results/ablation/",
        "6. 生成可视化图表到 charts/benchmark/ 和 charts/ablation/",
        "7. 生成统计检验报告 (Wilcoxon, Friedman, Cohen's d)",
        "8. 输出调度甘特图到 charts/schedules/"
    ]
    for line in steps:
        b.add_para(tf, line, font_size=FS_BODY, space_after=Pt(4))
    b.add_bottom_bar(s, "Phase 3 / 4 — Experiment Design")

    # Slide 5: 对比实验设计
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "对比实验设计", "Benchmark Comparison Design")
    left = b.add_shape_text(s, 0.5, 1.35, 5.8, 5.4, fill_color=C_LIGHT_BLUE, line_color=C_PRIMARY)
    ltf = left.text_frame; ltf.word_wrap = True; ltf.paragraphs[0].text = ""
    b.add_para(ltf, "对比算法", font_size=FS_BODY+2, bold=True, color=C_PRIMARY, space_after=Pt(6))
    algs = ["RMOEA/D (本算法)", "MOEA/D (固定 T=10)", "NSGA-II", "MOEA/D-M2M", "NSGA-III", "IAIS"]
    for a in algs:
        b.add_para(ltf, f"• {a}", font_size=FS_BODY, space_after=Pt(3))
    right = b.add_shape_text(s, 6.7, 1.35, 5.9, 5.4, fill_color=C_LIGHT_GREEN, line_color=C_ACCENT)
    rtf = right.text_frame; rtf.word_wrap = True; rtf.paragraphs[0].text = ""
    b.add_para(rtf, "实验设置", font_size=FS_BODY+2, bold=True, color=C_ACCENT, space_after=Pt(6))
    b.add_para(rtf, "• 每个算法在每个实例上独立运行 30 次", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(rtf, "• 记录 HV, Makespan, Workload 的均值与标准差", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(rtf, "• 进行 Wilcoxon 配对符号秩检验 (α=0.05)", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(rtf, "• 计算 Cohen's d 效应量", font_size=FS_BODY, space_after=Pt(2))
    b.add_bottom_bar(s, "Phase 3 / 4 — Experiment Design")

    # Slide 6: 消融实验设计
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "消融实验设计", "Ablation Study Design")
    left = b.add_shape_text(s, 0.5, 1.35, 5.8, 5.4, fill_color=C_LIGHT_BLUE, line_color=C_PRIMARY)
    ltf = left.text_frame; ltf.word_wrap = True; ltf.paragraphs[0].text = ""
    b.add_para(ltf, "消融配置", font_size=FS_BODY+2, bold=True, color=C_PRIMARY, space_after=Pt(6))
    configs = [
        ("Full", "RMOEA/D (Q-PAS + RVNS)"),
        ("Q-PAS Only", "仅启用 Q-PAS, 禁用 RVNS"),
        ("RVNS Only", "仅启用 RVNS, 禁用 Q-PAS (固定 T)"),
        ("Baseline", "禁用 Q-PAS + RVNS (固定 T=10)"),
    ]
    for name, desc in configs:
        b.add_para(ltf, f"• {name}: {desc}", font_size=FS_BODY, space_after=Pt(3))
    right = b.add_shape_text(s, 6.7, 1.35, 5.9, 5.4, fill_color=C_LIGHT_GREEN, line_color=C_ACCENT)
    rtf = right.text_frame; rtf.word_wrap = True; rtf.paragraphs[0].text = ""
    b.add_para(rtf, "验证目标", font_size=FS_BODY+2, bold=True, color=C_ACCENT, space_after=Pt(6))
    b.add_para(rtf, "• Q-PAS 对邻域自适应的有效性", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(rtf, "• RVNS 对局部搜索的增强作用", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(rtf, "• 两者协同是否产生增效", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(rtf, "• 统计检验验证组件贡献的显著性", font_size=FS_BODY, space_after=Pt(2))
    b.add_bottom_bar(s, "Phase 3 / 4 — Experiment Design")

    # Slide 7: 评估指标
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "评估指标", "Evaluation Metrics")
    left = b.add_shape_text(s, 0.5, 1.35, 5.8, 5.4, fill_color=C_LIGHT_BLUE, line_color=C_PRIMARY)
    ltf = left.text_frame; ltf.word_wrap = True; ltf.paragraphs[0].text = ""
    b.add_para(ltf, "性能指标", font_size=FS_BODY+2, bold=True, color=C_PRIMARY, space_after=Pt(6))
    b.add_para(ltf, "• Hypervolume (HV)", font_size=FS_BODY, bold=True, space_after=Pt(2))
    b.add_para(ltf, "  综合衡量 PF 收敛性与多样性", font_size=FS_SMALL, space_after=Pt(6))
    b.add_para(ltf, "• Makespan", font_size=FS_BODY, bold=True, space_after=Pt(2))
    b.add_para(ltf, "  最大完工时间的清晰值", font_size=FS_SMALL, space_after=Pt(6))
    b.add_para(ltf, "• Total Workload", font_size=FS_BODY, bold=True, space_after=Pt(2))
    b.add_para(ltf, "  所有机器负载之和", font_size=FS_SMALL, space_after=Pt(4))
    right = b.add_shape_text(s, 6.7, 1.35, 5.9, 5.4, fill_color=C_LIGHT_GREEN, line_color=C_ACCENT)
    rtf = right.text_frame; rtf.word_wrap = True; rtf.paragraphs[0].text = ""
    b.add_para(rtf, "统计检验", font_size=FS_BODY+2, bold=True, color=C_ACCENT, space_after=Pt(6))
    b.add_para(rtf, "• Wilcoxon 符号秩检验", font_size=FS_BODY, bold=True, space_after=Pt(2))
    b.add_para(rtf, "  配对比较，判断差异是否显著", font_size=FS_SMALL, space_after=Pt(6))
    b.add_para(rtf, "• Cohen's d", font_size=FS_BODY, bold=True, space_after=Pt(2))
    b.add_para(rtf, "  量化效应量 (small/medium/large)", font_size=FS_SMALL, space_after=Pt(6))
    b.add_para(rtf, "• Friedman 检验", font_size=FS_BODY, bold=True, space_after=Pt(2))
    b.add_para(rtf, "  多算法整体排序与显著性", font_size=FS_SMALL, space_after=Pt(4))
    b.add_bottom_bar(s, "Phase 3 / 4 — Experiment Design")

    # Slide 8: Phase 3 总结
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "Phase 3 小结", "Phase 3 Summary")
    box = b.add_shape_text(s, 0.5, 1.35, 12.3, 5.4, fill_color=C_LIGHT_GREEN, line_color=C_ACCENT)
    tf = box.text_frame; tf.word_wrap = True; tf.paragraphs[0].text = ""
    b.add_para(tf, "核心要点", font_size=FS_BODY+2, bold=True, color=C_ACCENT, space_after=Pt(8))
    b.add_para(tf, "• 实验覆盖 Brandimarte Mk01~Mk10 共 10 个标准实例", font_size=FS_BODY, space_after=Pt(4))
    b.add_para(tf, "• 对比实验: RMOEA/D vs 5 种先进算法，30 次独立运行", font_size=FS_BODY, space_after=Pt(4))
    b.add_para(tf, "• 消融实验: Full vs Q-PAS Only vs RVNS Only vs Baseline", font_size=FS_BODY, space_after=Pt(4))
    b.add_para(tf, "• 统计检验: Wilcoxon, Cohen's d, Friedman 三重验证", font_size=FS_BODY, space_after=Pt(10))
    b.add_para(tf, "下一步: 实验结果与深度分析 →", font_size=FS_BODY, bold=True, color=C_ALERT, space_after=Pt(6))
    b.add_bottom_bar(s, "Phase 3 / 4 — Experiment Design")

    prs.save(str(output_path))
    logger.info("Phase 3 saved: %s (%d slides)", output_path, b.slide_count)
    return b.slide_count


# ── Phase 4: 实验结果与分析 ──
def build_phase4(output_path: Path) -> int:
    """PPT 4: 实验结果与分析 (~10页)"""
    logger.info("=" * 60)
    logger.info("Building Phase 4: Results & Analysis")
    prs = _new_prs()
    b = SlideBuilder(prs)

    # Slide 1: Title
    s = b.add_blank(); b.set_bg(s)
    top = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), SLIDE_W, Inches(4.2))
    top.fill.solid(); top.fill.fore_color.rgb = RGBColor.from_string(C_PRIMARY[1:]); top.line.fill.background()
    b.add_text(s, 0.8, 0.9, 11.5, 1.0, text="实验结果与分析",
               font_size=FS_TITLE, bold=True, color=C_WHITE, align=PP_ALIGN.LEFT)
    b.add_text(s, 0.8, 1.9, 11.5, 0.7, text="Results & Statistical Analysis",
               font_size=FS_SUBTITLE, bold=True, color=C_WHITE, align=PP_ALIGN.LEFT)
    b.add_text(s, 0.8, 2.8, 11.5, 0.5,
               text="Phase 4: Benchmark, Ablation, and Statistical Validation",
               font_size=FS_SUBTITLE, color="#AECBE3", font_name=FONT_EN)
    b.add_text(s, 0.8, 4.6, 11.5, 0.4, text="基于图表与统计检验的全面性能评估",
               font_size=FS_BODY, bold=True, color=C_DARK)
    b.add_bottom_bar(s, "Phase 4 / 4 — Results & Analysis")

    # Slide 2: HV 对比 (Benchmark)
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "Hypervolume 对比", "Benchmark HV Comparison")
    hv_path = resolver.get("hv_comparison")
    has_hv = b.add_image_safe(s, hv_path, 0.5, 1.35, 12.3, 4.0,
                              placeholder_text="[HV Comparison Chart]")
    b.add_text(s, 0.5, 5.55, 12.3, 0.4,
               text="RMOEA/D 在 Mk01–Mk10 上 HV 整体优于 MOEA/D 等基线，Wilcoxon 检验显示差异显著 (p < 0.05)。",
               font_size=FS_BODY, color=C_DARK)
    b.add_source_label(s, "Source: charts/benchmark/*hv_comparison*.png", left=0.5, top=6.05)
    b.add_bottom_bar(s, "Phase 4 / 4 — Results & Analysis")
    logger.info("Slide 2 HV chart present=%s", has_hv)

    # Slide 3: Makespan / Workload 对比
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "Makespan & Workload 对比", "Benchmark Multi-Metric")
    mp_path = resolver.get("makespan_comparison")
    wl_path = resolver.get("workload_comparison")
    has_mp = b.add_image_safe(s, mp_path, 0.5, 1.35, 6.0, 3.8, placeholder_text="[Makespan Comparison]")
    has_wl = b.add_image_safe(s, wl_path, 6.9, 1.35, 5.9, 3.8, placeholder_text="[Workload Comparison]")
    b.add_text(s, 0.5, 5.35, 12.3, 0.4,
               text="RMOEA/D 在 Makespan 和 Workload 两个目标上均取得更优的均值，标准差更小，稳定性更高。",
               font_size=FS_BODY, color=C_DARK)
    b.add_source_label(s, "Source: charts/benchmark/*makespan*.png / *workload*.png", left=0.5, top=5.8)
    b.add_bottom_bar(s, "Phase 4 / 4 — Results & Analysis")
    logger.info("Slide 3 makespan=%s workload=%s", has_mp, has_wl)

    # Slide 4: 统计检验 — Cohen's d
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "效应量分析 (Cohen's d)", "Effect Size Heatmap")
    cd_path = resolver.get("cohens_d")
    has_cd = b.add_image_safe(s, cd_path, 0.5, 1.35, 12.3, 4.0, placeholder_text="[Cohen's d Heatmap]")
    b.add_text(s, 0.5, 5.55, 12.3, 0.4,
               text="Cohen's d 效应量显示 RMOEA/D 与对比算法之间的差异幅度，多数实例达到中等至大效应量。",
               font_size=FS_BODY, color=C_DARK)
    b.add_source_label(s, "Source: charts/benchmark/*cohens_d*.png", left=0.5, top=6.05)
    b.add_bottom_bar(s, "Phase 4 / 4 — Results & Analysis")
    logger.info("Slide 4 cohens_d=%s", has_cd)

    # Slide 5: 统计检验 — p-value
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "显著性检验 (p-value)", "Significance Heatmap")
    pv_path = resolver.get("pvalue_heatmap")
    has_pv = b.add_image_safe(s, pv_path, 0.5, 1.35, 12.3, 4.0, placeholder_text="[p-value Heatmap]")
    b.add_text(s, 0.5, 5.55, 12.3, 0.4,
               text="p-value 热力图直观展示 RMOEA/D 与各基线的显著性差异，深色区域表示 p < 0.05 的显著优势。",
               font_size=FS_BODY, color=C_DARK)
    b.add_source_label(s, "Source: charts/benchmark/*pvalue*.png", left=0.5, top=6.05)
    b.add_bottom_bar(s, "Phase 4 / 4 — Results & Analysis")
    logger.info("Slide 5 pvalue=%s", has_pv)

    # Slide 6: 消融实验 — HV 对比
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "消融实验: HV 对比", "Ablation HV")
    abl_hv = resolver.get("ablation_hv")
    abl_imp = resolver.get("improvement")
    has_abl_hv = b.add_image_safe(s, abl_hv, 0.5, 1.35, 6.0, 3.8, placeholder_text="[Ablation HV]")
    has_abl_imp = b.add_image_safe(s, abl_imp, 6.9, 1.35, 5.9, 3.8, placeholder_text="[Improvement Summary]")
    b.add_text(s, 0.5, 5.35, 12.3, 0.4,
               text="Full 配置 (Q-PAS + RVNS) 相较各消融变体在 HV 上均取得最佳表现，验证了组件协同增效。",
               font_size=FS_BODY, color=C_DARK)
    b.add_source_label(s, "Source: charts/ablation/*ablation_hv*.png / *improvement*.png", left=0.5, top=5.8)
    b.add_bottom_bar(s, "Phase 4 / 4 — Results & Analysis")
    logger.info("Slide 6 ablation_hv=%s improvement=%s", has_abl_hv, has_abl_imp)

    # Slide 7: 消融实验 — Cohen's d / p-value
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "消融实验: 统计检验", "Ablation Statistics")
    abl_cd = resolver.get("ablation_cohens")
    abl_pv = resolver.get("ablation_pvalue")
    has_abl_cd = b.add_image_safe(s, abl_cd, 0.5, 1.35, 6.0, 3.8, placeholder_text="[Ablation Cohen's d]")
    has_abl_pv = b.add_image_safe(s, abl_pv, 6.9, 1.35, 5.9, 3.8, placeholder_text="[Ablation p-value]")
    b.add_text(s, 0.5, 5.35, 12.3, 0.4,
               text="消融实验的统计检验进一步证实 Q-PAS 和 RVNS 各自以及协同均带来统计显著的性能提升。",
               font_size=FS_BODY, color=C_DARK)
    b.add_source_label(s, "Source: charts/ablation/*cohens*.png / *pvalue*.png", left=0.5, top=5.8)
    b.add_bottom_bar(s, "Phase 4 / 4 — Results & Analysis")
    logger.info("Slide 7 ablation_cohens=%s ablation_pvalue=%s", has_abl_cd, has_abl_pv)

    # Slide 8: 收敛曲线
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "收敛曲线分析", "Convergence Analysis")
    conv_path = resolver.get("benchmark_convergence")
    has_conv = b.add_image_safe(s, conv_path, 0.5, 1.35, 12.3, 4.0, placeholder_text="[Convergence Curve]")
    b.add_text(s, 0.5, 5.55, 12.3, 0.4,
               text="收敛曲线显示 RMOEA/D 在多数实例上收敛速度更快，且最终 HV 更高，稳定性更好。",
               font_size=FS_BODY, color=C_DARK)
    b.add_source_label(s, "Source: charts/benchmark/*/convergence*.png", left=0.5, top=6.05)
    b.add_bottom_bar(s, "Phase 4 / 4 — Results & Analysis")
    logger.info("Slide 8 convergence=%s", has_conv)

    # Slide 9: 甘特图展示
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "调度方案可视化", "Schedule Visualization")
    gantt_path = resolver.get("gantt")
    has_gantt = b.add_image_safe(s, gantt_path, 0.5, 1.35, 12.3, 4.0, placeholder_text="[Gantt Chart]")
    b.add_text(s, 0.5, 5.55, 12.3, 0.4,
               text="甘特图直观展示模糊完工时间的调度方案，验证算法在实际调度问题中的可用性。",
               font_size=FS_BODY, color=C_DARK)
    b.add_source_label(s, "Source: charts/schedules/*gantt*.png", left=0.5, top=6.05)
    b.add_bottom_bar(s, "Phase 4 / 4 — Results & Analysis")
    logger.info("Slide 9 gantt=%s", has_gantt)

    # Slide 10: 结论与展望
    s = b.add_blank(); b.set_bg(s); b.add_top_bar(s, "结论与展望", "Conclusion & Future Work")
    left = b.add_shape_text(s, 0.5, 1.35, 5.8, 5.4, fill_color=C_LIGHT_BLUE, line_color=C_PRIMARY)
    ltf = left.text_frame; ltf.word_wrap = True; ltf.paragraphs[0].text = ""
    b.add_para(ltf, "主要结论", font_size=FS_BODY+2, bold=True, color=C_PRIMARY, space_after=Pt(6))
    b.add_para(ltf, "1. RMOEA/D 在 FFJSP 双目标优化上表现优异", font_size=FS_BODY, space_after=Pt(4))
    b.add_para(ltf, "2. Q-PAS 有效平衡收敛性与多样性", font_size=FS_BODY, space_after=Pt(4))
    b.add_para(ltf, "3. RVNS 动态局部搜索避免盲目性", font_size=FS_BODY, space_after=Pt(4))
    b.add_para(ltf, "4. 统计检验全面验证优越性", font_size=FS_BODY, space_after=Pt(4))
    right = b.add_shape_text(s, 6.7, 1.35, 5.9, 5.4, fill_color=C_LIGHT_GREEN, line_color=C_ACCENT)
    rtf = right.text_frame; rtf.word_wrap = True; rtf.paragraphs[0].text = ""
    b.add_para(rtf, "未来方向", font_size=FS_BODY+2, bold=True, color=C_ACCENT, space_after=Pt(6))
    b.add_para(rtf, "• 深度强化学习 (DQN / Actor-Critic)", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(rtf, "• 更多目标与约束 (能耗、延迟)", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(rtf, "• 动态与在线调度", font_size=FS_BODY, space_after=Pt(2))
    b.add_para(rtf, "• 更大规模实例 (50×20+)", font_size=FS_BODY, space_after=Pt(2))
    b.add_bottom_bar(s, "Thank You | Q&A")

    prs.save(str(output_path))
    logger.info("Phase 4 saved: %s (%d slides)", output_path, b.slide_count)
    return b.slide_count


# ══════════════════════════════════════════════════════════════════════
# 5. 主入口
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Build 4-phase PPTX for RMOEA/D")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for PPTX files")
    parser.add_argument("--phase", type=str, choices=["1", "2", "3", "4", "all"], default="all",
                        help="Build specific phase only (default: all)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Starting 4-phase PPTX generation")
    logger.info("Output directory: %s", out_dir.resolve())
    logger.info("Charts directory: %s", CHARTS_DIR.resolve())
    logger.info("=" * 60)

    phases = {
        "1": ("Phase1_Background.pptx", build_phase1),
        "2": ("Phase2_Algorithm.pptx", build_phase2),
        "3": ("Phase3_Experiment.pptx", build_phase3),
        "4": ("Phase4_Results.pptx", build_phase4),
    }

    total_slides = 0
    t_start = time.time()

    if args.phase == "all":
        to_build = ["1", "2", "3", "4"]
    else:
        to_build = [args.phase]

    for key in to_build:
        filename, builder_fn = phases[key]
        out_path = out_dir / filename
        try:
            n = builder_fn(out_path)
            total_slides += n
            logger.info("Built %s with %d slides.", filename, n)
        except Exception as e:
            logger.error("Failed to build %s: %s", filename, e, exc_info=True)

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("All phases complete! Total slides: %d | Elapsed: %.2fs", total_slides, elapsed)
    logger.info("Files:")
    for key in to_build:
        p = out_dir / phases[key][0]
        exists = "OK" if p.exists() else "FAILED"
        size = p.stat().st_size if p.exists() else 0
        logger.info("  [%s] %s (%d bytes)", exists, p.name, size)
    logger.info("=" * 60)

    print(f"\n4-phase presentation built successfully in {elapsed:.1f}s!")
    print(f"Total slides: {total_slides}")
    print(f"Output dir: {out_dir.resolve()}")
    print(f"Log file: {log_file.resolve()}")


if __name__ == "__main__":
    main()
