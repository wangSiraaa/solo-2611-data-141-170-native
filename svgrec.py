#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""svgrec.py — 从导出的 SVG 折线图恢复可核查数值（Python 3.11 标准库）。

设计要点
--------
* 统一坐标：所有曲线顶点、刻度锚点、裁剪边界都变换到"共享绘图区坐标"
  （manifest 的 plot_frame 元素局部坐标；plot_frame 为空时取全部刻度元素的
  最深公共祖先的局部坐标）。轴标定、裁剪、数值恢复都在该空间完成，
  不使用屏幕横纵坐标做两个独立比例（图内存在旋转与非等比缩放）。
* 刻度数值只读 text/tspan 文本；text 的 (x,y) 属性是刻度锚点，变换到绘图区
  坐标后，垂直于本轴的标签偏移不参与该轴拟合。
* 每轴用全部刻度联合最小二乘拟合（log10 轴在对数值上拟合），残差超容差
  整图拒绝——不任选两个刻度掩盖矛盾。
* 裁剪按其声明空间（userSpaceOnUse：引用元素的用户空间）解释，线段与矩形
  求交；交点只表示图形截断位置，不冒充采样观测；隐藏顶点保留并标注。
* 白名单特性：单一 SVG 命名空间；translate/scale/rotate；text/tspan；
  polyline；仅含绝对 M/L 的 path；userSpaceOnUse 单矩形 clipPath；嵌套 g。
  其余元素/属性/命令一律明确拒绝，不建设通用渲染器。

用法
----
    python3 svgrec.py extract --manifest fixtures/manifest.json --out out
    python3 svgrec.py verify  --manifest fixtures/manifest.json --out out
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------

class RecovError(Exception):
    """恢复过程基础异常。"""

class UnsupportedFeature(RecovError):
    """遇到白名单之外的 SVG 特性。"""

class DataError(RecovError):
    """输入数据缺失或无法解析。"""

class AxisInconsistent(RecovError):
    """刻度标注相互不一致（含 log 轴非正刻度），阻断可信数值输出。"""


# ---------------------------------------------------------------------------
# 常量与白名单
# ---------------------------------------------------------------------------

ALLOWED_ELEMENTS = {
    "svg", "g", "defs", "clipPath", "rect",
    "text", "tspan", "polyline", "path", "title", "desc",
}

NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
NUM_RE = re.compile(NUM)
NUM_FULL = re.compile(NUM + r"$")
TRANSFORM_RE = re.compile(
    r"\s*(translate|scale|rotate|matrix|skewX|skewY)\s*\(([^)]*)\)")

# 刻度拟合容差（声明值）：绘图区单位，取轴刻度跨度的 1e-6（下限 1e-9）。
FIT_TOL_REL = 1e-6
FIT_TOL_ABS = 1e-9

PRECISION_NOTES = {
    "svg_numbers": (
        "SVG 数字按 IEEE-754 双精度解析；源文件坐标若保留 k 位小数，其量化误差 "
        "≤ 0.5×10^-k（元素局部单位），经变换链缩放后进入绘图区坐标。"),
    "tick_fit": (
        "轴标定使用全部刻度联合最小二乘拟合；残差（绘图区单位）反映刻度标注与"
        "轴几何的一致程度，超过声明容差即整图拒绝，不任选两个刻度掩盖矛盾。"),
    "value_error": (
        "恢复值误差 ≈ |拟合斜率| × 绘图区坐标误差（线性轴）；log10 轴的相对误差 "
        "≈ ln(10) × |斜率| × 坐标误差。刻度拟合残差以同样方式传播到恢复值。"),
    "no_inference": (
        "不从文件推断原始数据未知的绘图/采样精度；恢复值仅描述图形的绘制几何。"
        "裁剪交点仅为图形截断位置，不是额外采样观测。"),
}


# ---------------------------------------------------------------------------
# 2D 仿射矩阵 (a,b,c,d,e,f): x' = a x + c y + e ; y' = b x + d y + f
# ---------------------------------------------------------------------------

IDENT = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

def mmul(m1, m2):
    """返回 m1 ∘ m2（先应用 m2）。"""
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (a1 * a2 + c1 * b2, b1 * a2 + d1 * b2,
            a1 * c2 + c1 * d2, b1 * c2 + d1 * d2,
            a1 * e2 + c1 * f2 + e1, b1 * e2 + d1 * f2 + f1)

def minv(m):
    a, b, c, d, e, f = m
    det = a * d - b * c
    if det == 0 or not math.isfinite(det):
        raise DataError("变换矩阵不可逆（退化的 scale?）")
    return (d / det, -b / det, -c / det, a / det,
            (c * f - d * e) / det, (b * e - a * f) / det)

def mapply(m, p):
    a, b, c, d, e, f = m
    x, y = p
    return (a * x + c * y + e, b * x + d * y + f)

def m_translate(tx, ty=0.0):
    return (1.0, 0.0, 0.0, 1.0, tx, ty)

def m_scale(sx, sy=None):
    return (sx, 0.0, 0.0, sx if sy is None else sy, 0.0, 0.0)

def m_rotate(deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return (c, s, -s, c, 0.0, 0.0)

def m_max_scale(m):
    """矩阵线性部分的最大奇异值（局部单位 → 目标单位的放大上界）。"""
    a, b, c, d, _, _ = m
    s1 = a * a + b * b + c * c + d * d
    s2 = math.hypot(a * a + b * b - c * c - d * d, 2 * (a * c + b * d))
    return math.sqrt(max(0.0, (s1 + s2) / 2.0))


# ---------------------------------------------------------------------------
# 解析辅助
# ---------------------------------------------------------------------------

def fnum(text, who):
    try:
        return float(text)
    except (TypeError, ValueError):
        raise DataError(f"{who}: 无法解析数值 {text!r}") from None

def split_tag(tag):
    if tag.startswith("{"):
        ns, _, local = tag[1:].partition("}")
        if ns != SVG_NS:
            raise UnsupportedFeature(f"非 SVG 命名空间元素: {tag}")
        return local
    raise UnsupportedFeature(f"无命名空间元素（要求单一 SVG 命名空间）: {tag}")

def parse_transform(s, who):
    """解析 transform 属性；只允许 translate/scale/rotate，保持声明顺序。"""
    if s is None:
        return IDENT
    mats = []
    pos = 0
    for m in TRANSFORM_RE.finditer(s):
        if s[pos:m.start()].strip(" \t\r\n,"):
            raise UnsupportedFeature(
                f"{who}: transform 含无法解析的片段 {s[pos:m.start()]!r}")
        pos = m.end()
        fn, argstr = m.group(1), m.group(2)
        if not re.fullmatch(rf"\s*{NUM}(?:[\s,]+{NUM})*\s*", argstr):
            raise DataError(f"{who}: transform {fn} 参数无法解析: {argstr!r}")
        args = [float(a) for a in NUM_RE.findall(argstr)]
        if fn == "translate":
            if len(args) not in (1, 2):
                raise DataError(f"{who}: translate 参数个数错误: {args}")
            mats.append(m_translate(*args))
        elif fn == "scale":
            if len(args) not in (1, 2):
                raise DataError(f"{who}: scale 参数个数错误: {args}")
            mats.append(m_scale(*args))
        elif fn == "rotate":
            if len(args) == 1:
                mats.append(m_rotate(args[0]))
            elif len(args) == 3:
                mats.append(mmul(m_translate(args[1], args[2]),
                                 mmul(m_rotate(args[0]),
                                      m_translate(-args[1], -args[2]))))
            else:
                raise DataError(f"{who}: rotate 参数个数错误: {args}")
        else:
            raise UnsupportedFeature(
                f"{who}: 不支持的变换 {fn}（仅支持 translate/scale/rotate）")
    if s[pos:].strip(" \t\r\n,"):
        raise UnsupportedFeature(
            f"{who}: transform 含无法解析的片段 {s[pos:]!r}")
    m = IDENT
    for t in mats:  # SVG 变换列表：先声明的为外层
        m = mmul(m, t)
    return m

def parse_points(s, who):
    if re.sub(NUM, "", s).strip(" \t\r\n,"):
        raise DataError(f"{who}: polyline points 含无法解析的内容")
    vals = [float(t) for t in NUM_RE.findall(s)]
    if len(vals) < 4 or len(vals) % 2:
        raise DataError(f"{who}: polyline points 坐标不成对或点数不足")
    return list(zip(vals[0::2], vals[1::2]))

def parse_path(d, who):
    """仅接受绝对 M/L（含隐式后续 lineto）。返回子路径列表。"""
    if re.sub(r"[A-Za-z]|" + NUM, "", d).strip(" \t\r\n,"):
        raise DataError(f"{who}: path d 含无法解析的内容")
    tokens = re.findall(r"[A-Za-z]|" + NUM, d)

    def num_at(i):
        if i >= len(tokens) or tokens[i].isalpha():
            raise DataError(f"{who}: path 坐标缺失或不成对")
        return float(tokens[i])

    subpaths, cur, cmd, i = [], None, None, 0
    while i < len(tokens):
        t = tokens[i]
        if t.isalpha():
            if t in ("m", "l"):
                raise UnsupportedFeature(
                    f"{who}: 相对 path 命令 {t!r} 不支持（仅绝对 M/L）")
            if t not in ("M", "L"):
                raise UnsupportedFeature(
                    f"{who}: path 命令 {t!r} 不支持（仅绝对 M/L，无曲线）")
            cmd = t
            i += 1
            x, y = num_at(i), num_at(i + 1)
            i += 2
            if cmd == "M":
                cur = [(x, y)]
                subpaths.append(cur)
                cmd = "L"  # M 后续坐标对为隐式 lineto
            else:
                if cur is None:
                    raise DataError(f"{who}: L 之前缺少 M")
                cur.append((x, y))
        else:
            if cmd != "L" or cur is None:
                raise DataError(f"{who}: 坐标出现在有效命令之前")
            x, y = num_at(i), num_at(i + 1)
            i += 2
            cur.append((x, y))
    if not subpaths:
        raise DataError(f"{who}: path 为空")
    return subpaths

def max_decimals(s):
    k = 0
    for m in re.finditer(r"\.(\d+)", s):
        k = max(k, len(m.group(1)))
    return k


# ---------------------------------------------------------------------------
# 文档遍历：建立元素表、CTM、裁剪链
# ---------------------------------------------------------------------------

class El:
    __slots__ = ("tag", "attrs", "ctm", "parent", "node", "tchain", "clip", "label")

    def __init__(self, tag, attrs, ctm, parent, node, tchain, clip, label):
        self.tag, self.attrs, self.ctm = tag, attrs, ctm
        self.parent, self.node, self.tchain = parent, node, tchain
        self.clip, self.label = clip, label

class Clip:
    __slots__ = ("rect", "space", "ref", "who")

    def __init__(self, rect, space, ref, who):
        self.rect, self.space, self.ref, self.who = rect, space, ref, who

class Doc:
    def __init__(self):
        self.by_id = {}
        self.root = None


def describe(node, tag):
    ident = node.get("id")
    return f"<{tag} id={ident!r}>" if ident else f"<{tag}>"


def parse_clip_path(node, who):
    units = node.get("clipPathUnits", "userSpaceOnUse")
    if units != "userSpaceOnUse":
        raise UnsupportedFeature(
            f"{who}: 仅支持 clipPathUnits=userSpaceOnUse，得到 {units!r}")
    kids = list(node)
    if len(kids) != 1 or split_tag(kids[0].tag) != "rect":
        raise UnsupportedFeature(f"{who}: clipPath 仅支持单个子元素 rect")
    r = kids[0]
    if list(r):
        raise UnsupportedFeature(f"{who}: clip rect 不应有子元素")
    if r.get("transform"):
        raise UnsupportedFeature(f"{who}: clip rect 不支持 transform")
    if fnum(r.get("rx", "0"), who) != 0 or fnum(r.get("ry", "0"), who) != 0:
        raise UnsupportedFeature(f"{who}: 不支持圆角 clip rect")
    x = fnum(r.get("x", "0"), who)
    y = fnum(r.get("y", "0"), who)
    if r.get("width") is None or r.get("height") is None:
        raise DataError(f"{who}: clip rect 缺少 width/height")
    w, h = fnum(r.get("width"), who), fnum(r.get("height"), who)
    if w < 0 or h < 0:
        raise DataError(f"{who}: clip rect 尺寸为负")
    return (x, y, w, h)


def walk_document(root):
    if split_tag(root.tag) != "svg":
        raise UnsupportedFeature("根元素不是 <svg>")
    doc = Doc()
    doc.root = root

    # 预扫描 clipPath 定义（可在文档任意位置）
    clipdefs = {}
    for node in root.iter():
        if split_tag(node.tag) == "clipPath":
            cid = node.get("id")
            if not cid:
                raise DataError("clipPath 缺少 id")
            if cid in clipdefs:
                raise DataError(f"clipPath id 重复: {cid}")
            clipdefs[cid] = parse_clip_path(node, f"clipPath {cid!r}")

    def rec(node, ctm, clip, parent_el, tchain):
        tag = split_tag(node.tag)
        if tag not in ALLOWED_ELEMENTS:
            raise UnsupportedFeature(f"不支持的元素 <{tag}>")
        who = describe(node, tag)
        for k in node.attrib:
            if "}" in k:
                raise UnsupportedFeature(f"{who}: 外命名空间属性 {k}")
            if k in ("display", "visibility"):
                raise UnsupportedFeature(
                    f"{who}: 不支持属性 {k}（可见性控制超出范围）")
        if tag == "rect" and (parent_el is None or parent_el.tag != "clipPath"):
            raise UnsupportedFeature(f"{who}: rect 仅允许出现在 clipPath 内")
        m = parse_transform(node.get("transform"), who)
        ctm2 = mmul(ctm, m)
        clip2 = clip
        cp = node.get("clip-path")
        if cp is not None:
            if cp.strip() == "none":
                clip2 = None
            else:
                mm = re.fullmatch(r"url\(#([^)]+)\)", cp.strip())
                if not mm:
                    raise UnsupportedFeature(f"{who}: 不支持的 clip-path 值 {cp!r}")
                if clip is not None:
                    raise UnsupportedFeature(
                        f"{who}: 嵌套 clip-path 求交不支持（已有 {clip.ref!r}）")
                cid = mm.group(1)
                if cid not in clipdefs:
                    raise DataError(f"{who}: 引用的 clipPath 不存在: {cid!r}")
                # userSpaceOnUse：矩形声明于引用元素的用户空间（含其自身 transform）
                clip2 = Clip(clipdefs[cid], ctm2, cid, who)
        tchain2 = tchain + (((who, node.get("transform")),)
                            if node.get("transform") else ())
        el = El(tag, dict(node.attrib), ctm2, parent_el, node, tchain2,
                clip2, who)
        if tag == "clipPath":
            return  # 子元素已在 parse_clip_path 校验
        ident = node.get("id")
        if ident:
            if ident in doc.by_id:
                raise DataError(f"元素 id 重复: {ident!r}")
            doc.by_id[ident] = el
        for ch in node:
            rec(ch, ctm2, clip2, el, tchain2)

    rec(root, IDENT, None, None, ())
    return doc


def ancestor_chain(el):
    chain = []
    while el is not None:
        chain.append(el)
        el = el.parent
    return chain[::-1]


# ---------------------------------------------------------------------------
# 轴标定
# ---------------------------------------------------------------------------

class Axis:
    def __init__(self, name, scale, coord, a, b, span, tol, ticks,
                 max_resid_plot, max_resid_value):
        self.name, self.scale, self.coord = name, scale, coord
        self.a, self.b, self.span, self.tol = a, b, span, tol
        self.ticks = ticks
        self.max_resid_plot = max_resid_plot
        self.max_resid_value = max_resid_value

    def to_value(self, p):
        if self.scale == "linear":
            return self.a * p + self.b
        return 10.0 ** (self.a * p + self.b)

    def to_plot(self, v):
        y = v
        if self.scale == "log10":
            if v <= 0:
                raise DataError(f"轴 {self.name}: log10 轴无法映射非正值 {v}")
            y = math.log10(v)
        return (y - self.b) / self.a


def fit_axis(name, spec, tick_entries):
    """tick_entries: [(tick_id, axis_pos, value)]；全部刻度联合最小二乘。"""
    scale, coord = spec["scale"], spec["coordinate"]
    if scale not in ("linear", "log10"):
        raise UnsupportedFeature(f"轴 {name}: 不支持的刻度类型 {scale!r}")
    n = len(tick_entries)
    if n < 2:
        raise DataError(f"轴 {name}: 至少需要两个刻度，得到 {n}")
    pairs = []
    for tid, pos, val in tick_entries:
        if scale == "log10" and val <= 0:
            raise AxisInconsistent(
                f"轴 {name}: log10 轴刻度 {tid} 标注非正值 {val}，无法取对数")
        pairs.append((pos, val if scale == "linear" else math.log10(val),
                      tid, val))
    ps = [p for p, _, _, _ in pairs]
    ys = [y for _, y, _, _ in pairs]
    sp, sy = sum(ps), sum(ys)
    sxx = sum(p * p for p in ps)
    sxy = sum(p * y for p, y in zip(ps, ys))
    den = n * sxx - sp * sp
    if abs(den) <= 1e-300:
        raise DataError(f"轴 {name}: 刻度位置重合，无法拟合")
    a = (n * sxy - sp * sy) / den
    b = (sy - a * sp) / n
    if not math.isfinite(a) or a == 0:
        raise DataError(f"轴 {name}: 拟合斜率退化")
    span = max(ps) - min(ps)
    tol = max(FIT_TOL_ABS, FIT_TOL_REL * max(1.0, span))
    ticks, max_rp, max_rv = [], 0.0, 0.0
    bad = []
    for p, y, tid, val in pairs:
        r_val_space = (a * p + b) - y          # 线性：值单位；log：对数单位
        r_plot = r_val_space / a               # 折算绘图区单位
        if scale == "log10":
            r_value = 10.0 ** (a * p + b) - val
        else:
            r_value = r_val_space
        max_rp = max(max_rp, abs(r_plot))
        max_rv = max(max_rv, abs(r_value))
        ticks.append({"id": tid, "value": val, "axis_pos": p,
                      "residual_plot": r_plot, "residual_value": r_value})
        if abs(r_plot) > tol:
            bad.append((tid, val, p, r_plot))
    if bad:
        detail = "; ".join(
            f"{tid}(值={val}, 位置={p:.6g}, 残差={r:.6g} 绘图单位)"
            for tid, val, p, r in bad)
        raise AxisInconsistent(
            f"轴 {name}: {len(bad)}/{n} 个刻度与联合拟合不一致"
            f"（容差 {tol:.3g} 绘图单位）: {detail}。"
            f"已阻断可信数值输出，不任选两个刻度掩盖矛盾。")
    return Axis(name, scale, coord, a, b, span, tol, ticks, max_rp, max_rv)


# ---------------------------------------------------------------------------
# 裁剪与片段
# ---------------------------------------------------------------------------

def inside(p, rect, eps):
    x, y = p
    x0, y0, w, h = rect
    return (x0 - eps <= x <= x0 + w + eps) and (y0 - eps <= y <= y0 + h + eps)


def clip_segment(p0, p1, rect):
    """Liang–Barsky；返回 (t0, t1) 或 None。"""
    x0, y0 = p0
    dx, dy = p1[0] - x0, p1[1] - y0
    rx, ry, rw, rh = rect
    t0, t1 = 0.0, 1.0
    for p_, q_ in ((-dx, x0 - rx), (dx, rx + rw - x0),
                   (-dy, y0 - ry), (dy, ry + rh - y0)):
        if p_ == 0:
            if q_ < 0:
                return None
        else:
            t = q_ / p_
            if p_ < 0:
                if t > t1:
                    return None
                if t > t0:
                    t0 = t
            else:
                if t < t0:
                    return None
                if t < t1:
                    t1 = t
    return t0, t1


def boundary_of(p, rect):
    x, y = p
    x0, y0, w, h = rect
    cands = (("xmin", abs(x - x0)), ("xmax", abs(x - (x0 + w))),
             ("ymin", abs(y - y0)), ("ymax", abs(y - (y0 + h))))
    return min(cands, key=lambda t: t[1])[0]


def lerp(p0, p1, t):
    return (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)


# ---------------------------------------------------------------------------
# 单图恢复
# ---------------------------------------------------------------------------

def extract_chart(source_text, spec, source_name, want_overlay=False):
    """返回结果字典；失败抛 RecovError 子类（调用方负责阻断输出）。"""
    try:
        root = ET.fromstring(source_text)
    except ET.ParseError as e:
        raise DataError(f"{source_name}: XML 解析失败: {e}") from None
    doc = walk_document(root)

    axes_spec = spec["axes"]
    all_tick_ids = [t for ax in axes_spec.values() for t in ax["ticks"]]

    # --- 统一坐标：绘图区空间 ---
    frame_id = spec.get("plot_frame")
    if frame_id:
        frame_el = doc.by_id.get(frame_id)
        if frame_el is None:
            raise DataError(f"{source_name}: plot_frame 元素 {frame_id!r} 不存在")
        space_desc = {"kind": "plot_frame", "element": frame_id}
    else:
        chains = []
        for tid in all_tick_ids:
            el = doc.by_id.get(tid)
            if el is None:
                raise DataError(f"{source_name}: 刻度元素 {tid!r} 不存在")
            chains.append(ancestor_chain(el))
        common = None
        for level in zip(*chains):
            if all(e is level[0] for e in level):
                common = level[0]
            else:
                break
        if common is None:
            raise DataError(f"{source_name}: 刻度元素无公共祖先")
        frame_el = common
        space_desc = {"kind": "shared_tick_ancestor",
                      "element": common.label}
    inv_plot = minv(frame_el.ctm)

    # --- 刻度：读文本、变换锚点、联合拟合 ---
    axes = {}
    axes_evidence = {}
    for ax_name, ax_spec in axes_spec.items():
        coord = ax_spec["coordinate"]
        if coord not in ("x", "y"):
            raise DataError(f"轴 {ax_name}: coordinate 必须为 'x'/'y'")
        entries, tick_ev = [], []
        for tid in ax_spec["ticks"]:
            el = doc.by_id.get(tid)
            if el is None:
                raise DataError(f"轴 {ax_name}: 刻度元素 {tid!r} 不存在")
            if el.tag != "text":
                raise DataError(f"轴 {ax_name}: 刻度 {tid!r} 不是 <text>")
            raw = "".join(el.node.itertext()).strip()
            if not NUM_FULL.fullmatch(raw):
                raise DataError(
                    f"轴 {ax_name}: 刻度 {tid} 文本非纯数值: {raw!r}")
            val = float(raw)
            ax = fnum(el.attrs.get("x", "0"), f"刻度 {tid}")
            ay = fnum(el.attrs.get("y", "0"), f"刻度 {tid}")
            plot_pt = mapply(mmul(inv_plot, el.ctm), (ax, ay))
            axis_pos = plot_pt[0] if coord == "x" else plot_pt[1]
            entries.append((tid, axis_pos, val))
            tick_ev.append({
                "id": tid, "raw_text": raw, "value": val,
                "attr_xy": [el.attrs.get("x", "0"), el.attrs.get("y", "0")],
                "transform_chain": [
                    {"element": w, "transform": t} for w, t in el.tchain],
                "plot_xy": list(plot_pt),
                "axis_coord_used": coord,
                "note": "垂直于本轴的标签偏移不参与拟合",
            })
        axis = fit_axis(ax_name, ax_spec, entries)
        axes[ax_name] = axis
        for ev, tk in zip(tick_ev, axis.ticks):
            ev["residual_plot"] = tk["residual_plot"]
            ev["residual_value"] = tk["residual_value"]
        axes_evidence[ax_name] = {
            "scale": axis.scale, "coordinate": coord,
            "fit": {
                "form": ("value = a*plot_%s + b" % coord) if axis.scale == "linear"
                        else ("log10(value) = a*plot_%s + b" % coord),
                "a": axis.a, "b": axis.b, "n_ticks": len(entries),
                "plot_span": axis.span, "tolerance_plot": axis.tol,
                "max_abs_residual_plot": axis.max_resid_plot,
                "max_abs_residual_value": axis.max_resid_value,
                "warning": ("仅两个刻度，拟合必然精确，无交叉校验能力"
                            if len(entries) == 2 else None),
                "status": "ok",
            },
            "ticks": tick_ev,
        }
    xax = axes["x"] if "x" in axes else axes[[k for k in axes if axes[k].coord == "x"][0]]
    yax = axes["y"] if "y" in axes else axes[[k for k in axes if axes[k].coord == "y"][0]]

    # --- 曲线：顶点、可见性、片段 ---
    series_out = {}
    for sid in spec["series"]:
        el = doc.by_id.get(sid)
        if el is None:
            raise DataError(f"{source_name}: 曲线元素 {sid!r} 不存在")
        who = el.label
        if el.tag == "polyline":
            raw_attr = el.attrs.get("points", "")
            subpaths = [parse_points(raw_attr, who)]
        elif el.tag == "path":
            raw_attr = el.attrs.get("d", "")
            subpaths = parse_path(raw_attr, who)
        else:
            raise UnsupportedFeature(
                f"{who}: 曲线元素仅支持 polyline/path，得到 <{el.tag}>")

        m_to_plot = mmul(inv_plot, el.ctm)
        clip = el.clip
        clip_corners_plot = None
        if clip is not None:
            m_to_clip = mmul(minv(clip.space), el.ctm)
            m_clip_to_plot = mmul(inv_plot, clip.space)
            rect = clip.rect
            eps = 1e-9 * max(1.0, rect[2], rect[3])
            rx, ry, rw, rh = rect
            clip_corners_plot = [list(mapply(m_clip_to_plot, p)) for p in
                                 ((rx, ry), (rx + rw, ry),
                                  (rx + rw, ry + rh), (rx, ry + rh))]
        diagnostics = []
        source_points = []
        fragments = []
        gidx = 0
        for sidx, sub in enumerate(subpaths):
            if len(sub) < 2 and clip is None:
                raise DataError(f"{who}: 子路径点数不足")
            idxs = list(range(gidx, gidx + len(sub)))
            gidx += len(sub)
            pts_plot = [mapply(m_to_plot, p) for p in sub]
            if clip is not None:
                pts_clip = [mapply(m_to_clip, p) for p in sub]
                vis = [inside(p, rect, eps) for p in pts_clip]
            else:
                pts_clip = [None] * len(sub)
                vis = [True] * len(sub)
            for j, p in enumerate(sub):
                source_points.append({
                    "index": idxs[j], "subpath": sidx, "local": list(p),
                    "plot": list(pts_plot[j]), "visible": vis[j],
                })
            hidden = [idxs[j] for j in range(len(sub)) if not vis[j]]
            if hidden:
                diagnostics.append(
                    f"顶点 {hidden} 在裁剪区外（保留于 source_points，"
                    f"用于解释片段端点）")

            # 片段构建（在裁剪声明空间内求交）
            cur_frag = None

            def emit_node(kind, pt_clip, j=None, seg=None):
                if kind == "vertex":
                    plot_pt = pts_plot[j]
                    node = {"kind": "vertex", "point_index": idxs[j]}
                else:
                    plot_pt = mapply(m_clip_to_plot, pt_clip)
                    node = {"kind": "intersection", "segment_index": idxs[seg],
                            "boundary": boundary_of(pt_clip, rect),
                            "note": "图形截断位置，非采样观测"}
                node["plot"] = list(plot_pt)
                return node

            if clip is None:
                frag = {"id": f"{sid}#frag{len(fragments)}", "subpath": sidx,
                        "points": [emit_node("vertex", None, j=j)
                                   for j in range(len(sub))]}
                fragments.append(frag)
            else:
                for j in range(len(sub) - 1):
                    r = clip_segment(pts_clip[j], pts_clip[j + 1], rect)
                    if r is None or r[1] - r[0] <= 1e-12:
                        if r is not None:
                            diagnostics.append(
                                f"线段 {idxs[j]}→{idxs[j+1]} 与裁剪边界相切，忽略")
                        cur_frag = None
                        continue
                    t0, t1 = r
                    start = (emit_node("vertex", None, j=j) if t0 == 0.0
                             else emit_node("intersection", lerp(pts_clip[j], pts_clip[j + 1], t0), seg=j))
                    end = (emit_node("vertex", None, j=j + 1) if t1 == 1.0
                           else emit_node("intersection", lerp(pts_clip[j], pts_clip[j + 1], t1), seg=j))
                    if (cur_frag is not None and start["kind"] == "vertex"
                            and cur_frag["points"]
                            and cur_frag["points"][-1]["kind"] == "vertex"
                            and cur_frag["points"][-1]["point_index"]
                            == start["point_index"]):
                        cur_frag["points"].append(end)
                    else:
                        cur_frag = {"id": f"{sid}#frag{len(fragments)}",
                                    "subpath": sidx, "points": [start, end]}
                        fragments.append(cur_frag)
                if len(sub) == 1 and vis[0]:
                    fragments.append({
                        "id": f"{sid}#frag{len(fragments)}", "subpath": sidx,
                        "points": [emit_node("vertex", None, j=0)],
                        "note": "孤立可见顶点"})

        # 数值恢复（绘图区坐标 → 轴值）
        def recover(plot_pt):
            return [xax.to_value(plot_pt[0]), yax.to_value(plot_pt[1])]

        for sp in source_points:
            sp["value"] = recover(sp["plot"])
        for frag in fragments:
            for node in frag["points"]:
                node["value"] = recover(node["plot"])
            first, last = frag["points"][0], frag["points"][-1]
            frag["from"] = (f"顶点 {first['point_index']}" if first["kind"] == "vertex"
                            else f"线段 {first['segment_index']} 与 {first['boundary']} 的交点")
            frag["to"] = (f"顶点 {last['point_index']}" if last["kind"] == "vertex"
                          else f"线段 {last['segment_index']} 与 {last['boundary']} 的交点")

        frag_pts = [nd for fr in fragments for nd in fr["points"]]
        xs = [nd["value"][0] for nd in frag_pts]
        ys = [nd["value"][1] for nd in frag_pts]
        pxs = [nd["plot"][0] for nd in frag_pts]
        # 坐标量化上界（按源文件实际打印小数位，事实性说明，非精度推断）
        k = max_decimals(raw_attr)
        d_local = 0.5 * 10.0 ** (-k) if k else 0.0
        d_plot = d_local * m_max_scale(m_to_plot)
        series_out[sid] = {
            "element": el.tag,
            "transform_chain": [
                {"element": w, "transform": t} for w, t in el.tchain],
            "clip": ({"id": clip.ref, "rect_plot_space_note":
                      "矩形声明于引用元素用户空间，已按该空间解释",
                      "rect": list(clip.rect),
                      "corners_plot": clip_corners_plot} if clip else None),
            "source_points": source_points,
            "fragments": fragments,
            "visible_interval": {
                "x_value": [min(xs), max(xs)] if xs else None,
                "y_value": [min(ys), max(ys)] if ys else None,
                "plot_x": [min(pxs), max(pxs)] if pxs else None,
            },
            "n_vertices": len(source_points),
            "n_visible_vertices": sum(1 for sp in source_points if sp["visible"]),
            "coordinate_quantization": {
                "printed_decimals": k, "delta_local": d_local,
                "delta_plot_bound": d_plot,
                "note": "源文件打印位数的事实性量化上界，非原始采样精度推断",
            },
            "diagnostics": diagnostics,
        }

    result = {
        "file": source_name,
        "plot_space": space_desc,
        "axes": axes_evidence,
        "series": series_out,
        "precision": PRECISION_NOTES,
    }
    if want_overlay:
        result["_overlay_svg"] = build_overlay(
            root, frame_el, axes, axes_evidence, series_out)
    # 供 verify 使用的内部对象
    result["_axes"] = axes
    result["_inv_plot"] = inv_plot
    return result


# ---------------------------------------------------------------------------
# 证据叠加 SVG
# ---------------------------------------------------------------------------

def build_overlay(root, frame_el, axes, axes_evidence, series_out):
    xax, yax = axes["x"], axes["y"]
    span = max(xax.span, yax.span, 1.0)
    r = 0.012 * span
    g = ET.SubElement(frame_el.node, f"{{{SVG_NS}}}g",
                      {"id": "recovery-overlay", "fill": "none"})

    def fmt(v):
        return f"{v:.6g}"

    def sub(tag, attrs):
        return ET.SubElement(g, f"{{{SVG_NS}}}{tag}",
                             {k: fmt(v) if isinstance(v, float) else str(v)
                              for k, v in attrs.items()})

    # 裁剪边界（声明空间四角 → 绘图区空间，旋转时为多边形）
    drawn_clip = set()
    for sid, s in series_out.items():
        if not s["clip"] or s["clip"]["id"] in drawn_clip:
            continue
        drawn_clip.add(s["clip"]["id"])
        pts = " ".join(f"{fmt(x)},{fmt(y)}" for x, y in
                       s["clip"]["corners_plot"])
        sub("polygon", {"points": pts, "stroke": "#f80",
                        "stroke-width": r * 0.4,
                        "stroke-dasharray": f"{fmt(r)},{fmt(r)}"})

    # 刻度证据：黑圈=变换后实际锚点；蓝十字=拟合预测位置
    for ax_name, ax in (("x", xax), ("y", yax)):
        for ev in axes_evidence[ax_name]["ticks"]:
            px, py = ev["plot_xy"]
            sub("circle", {"cx": px, "cy": py, "r": r * 0.8,
                           "stroke": "#000", "stroke-width": r * 0.25})
            pred = ax.to_plot(ev["value"])
            cx, cy = (pred, py) if ax.coord == "x" else (px, pred)
            sub("path", {"d": f"M {fmt(cx - r)} {fmt(cy)} L {fmt(cx + r)} {fmt(cy)} "
                              f"M {fmt(cx)} {fmt(cy - r)} L {fmt(cx)} {fmt(cy + r)}",
                         "stroke": "#06c", "stroke-width": r * 0.3})
            sub("text", {"x": cx + r, "y": cy - r, "font-size": 3 * r,
                         "fill": "#06c", "stroke": "none"}).text = ev["raw_text"]

    # 曲线证据
    for sid, s in series_out.items():
        for frag in s["fragments"]:
            pts = " ".join(f"{fmt(nd['plot'][0])},{fmt(nd['plot'][1])}"
                           for nd in frag["points"])
            sub("polyline", {"points": pts, "stroke": "#d62728",
                             "stroke-width": r * 0.6, "opacity": 0.65})
            for nd in frag["points"]:
                x, y = nd["plot"]
                if nd["kind"] == "vertex":
                    sub("circle", {"cx": x, "cy": y, "r": r * 0.7,
                                   "fill": "#2ca02c", "stroke": "none"})
                else:
                    sub("path", {"d": f"M {fmt(x - r)} {fmt(y)} L {fmt(x)} {fmt(y - r)} "
                                      f"L {fmt(x + r)} {fmt(y)} L {fmt(x)} {fmt(y + r)} Z",
                                 "fill": "#ff7f0e", "stroke": "none"})
        for sp in s["source_points"]:
            if not sp["visible"]:
                x, y = sp["plot"]
                sub("circle", {"cx": x, "cy": y, "r": r * 0.7,
                               "stroke": "#888", "stroke-width": r * 0.25,
                               "stroke-dasharray": f"{fmt(r*0.5)},{fmt(r*0.5)}"})
    # 图例：放在绘图区内容（裁剪框与刻度锚点）下方，避免遮挡证据
    leg_y = 4 * r
    for s in series_out.values():
        if s["clip"]:
            leg_y = max(leg_y, max(y for _, y in s["clip"]["corners_plot"]) + 6 * r)
    for ax_name in ("x", "y"):
        for ev in axes_evidence[ax_name]["ticks"]:
            leg_y = max(leg_y, ev["plot_xy"][1] + 6 * r)
    legend = sub("text", {"x": r, "y": leg_y, "font-size": 3 * r,
                          "fill": "#333", "stroke": "none"})
    legend.text = ("绿点=原始顶点 橙菱形=截断交点(非观测) 灰虚线圈=裁剪区外顶点 "
                   "黑圈=刻度锚点 蓝十字=刻度拟合位置")
    return ET.tostring(root, encoding="unicode")


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def public_result(result):
    """去掉内部键，得到可序列化结果。"""
    return {k: v for k, v in result.items() if not k.startswith("_")}


def write_outputs(result, out_dir):
    base = Path(result["file"]).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    pub = public_result(result)

    axes_json = {"chart": result["file"], "plot_space": pub["plot_space"],
                 "axes": pub["axes"], "precision": pub["precision"]}
    (out_dir / f"{base}.axes.json").write_text(
        json.dumps(axes_json, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(out_dir / f"{base}.points.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["chart", "series", "fragment", "order", "kind",
                    "point_index", "segment_index", "boundary",
                    "plot_x", "plot_y", "x_value", "y_value",
                    "visible", "note"])
        for sid, s in pub["series"].items():
            in_frag = set()
            for frag in s["fragments"]:
                for order, nd in enumerate(frag["points"]):
                    if nd["kind"] == "vertex":
                        in_frag.add(nd["point_index"])
                        w.writerow([result["file"], sid, frag["id"], order,
                                    "vertex", nd["point_index"], "", "",
                                    repr(nd["plot"][0]), repr(nd["plot"][1]),
                                    repr(nd["value"][0]), repr(nd["value"][1]),
                                    1, "原始路径顶点"])
                    else:
                        w.writerow([result["file"], sid, frag["id"], order,
                                    "intersection", "", nd["segment_index"],
                                    nd["boundary"],
                                    repr(nd["plot"][0]), repr(nd["plot"][1]),
                                    repr(nd["value"][0]), repr(nd["value"][1]),
                                    1, "图形截断位置，非采样观测"])
            for sp in s["source_points"]:
                if not sp["visible"]:
                    w.writerow([result["file"], sid, "", "", "vertex",
                                sp["index"], "", "",
                                repr(sp["plot"][0]), repr(sp["plot"][1]),
                                repr(sp["value"][0]), repr(sp["value"][1]),
                                0, "裁剪区外原始顶点（保留以解释片段端点）"])

    segments_json = {"chart": result["file"],
                     "caveat": "intersection 点仅为图形截断位置，不是额外采样观测",
                     "series": {sid: {
                         "element": s["element"],
                         "transform_chain": s["transform_chain"],
                         "clip": s["clip"],
                         "fragments": s["fragments"],
                         "hidden_vertices": [sp for sp in s["source_points"]
                                             if not sp["visible"]],
                         "diagnostics": s["diagnostics"],
                     } for sid, s in pub["series"].items()}}
    (out_dir / f"{base}.segments.json").write_text(
        json.dumps(segments_json, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {"chart": result["file"], "status": "ok",
              "visible_interval": {sid: s["visible_interval"]
                                   for sid, s in pub["series"].items()},
              "series_summary": {sid: {
                  "n_vertices": s["n_vertices"],
                  "n_visible_vertices": s["n_visible_vertices"],
                  "n_fragments": len(s["fragments"]),
                  "coordinate_quantization": s["coordinate_quantization"],
              } for sid, s in pub["series"].items()},
              "precision": pub["precision"]}
    (out_dir / f"{base}.report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if "_overlay_svg" in result and result["_overlay_svg"]:
        (out_dir / f"{base}.overlay.svg").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n' + result["_overlay_svg"],
            encoding="utf-8")
    return base


def write_rejected(source_name, err, out_dir):
    base = Path(source_name).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{base}.rejected.json").write_text(json.dumps({
        "chart": source_name, "status": "rejected",
        "error_type": type(err).__name__, "error": str(err),
        "note": "刻度/输入不满足一致性要求，已阻断可信数值输出",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return base


# ---------------------------------------------------------------------------
# extract 命令
# ---------------------------------------------------------------------------

def run_extract(manifest_path, out_dir, want_overlay=True):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    mdir = Path(manifest_path).parent
    results, failures = {}, []
    for spec in manifest["charts"]:
        src = (mdir / spec["file"]).read_text(encoding="utf-8")
        try:
            res = extract_chart(src, spec, spec["file"], want_overlay=want_overlay)
        except RecovError as e:
            write_rejected(spec["file"], e, out_dir)
            failures.append((spec["file"], e))
            print(f"[extract] {spec['file']}: 拒绝 — {e}")
            continue
        base = write_outputs(res, out_dir)
        results[spec["file"]] = res
        for sid, s in res["series"].items():
            vi = s["visible_interval"]
            print(f"[extract] {spec['file']}:{sid} 片段="
                  f"{len(s['fragments'])} 顶点={s['n_vertices']}"
                  f"(可见 {s['n_visible_vertices']}) "
                  f"可见x区间={vi['x_value']}")
        for axn, ax in res["axes"].items():
            print(f"[extract] {spec['file']} 轴{axn}: "
                  f"最大残差={ax['fit']['max_abs_residual_plot']:.3g} 绘图单位 "
                  f"(容差 {ax['fit']['tolerance_plot']:.3g})")
        print(f"[extract] {spec['file']}: 输出 {base}.*")
    return results, failures


# ---------------------------------------------------------------------------
# verify 命令
# ---------------------------------------------------------------------------

def rigid_wrap(source_text):
    """给整图套一个刚性变换（平移+旋转），用于不变性检验。"""
    root = ET.fromstring(source_text)
    g = ET.Element(f"{{{SVG_NS}}}g",
                   {"transform": "translate(37.5 -12.25) rotate(23)"})
    for child in list(root):
        root.remove(child)
        g.append(child)
    root.append(g)
    return ET.tostring(root, encoding="unicode")


def close(a, b, tol):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def verify_chart_values(res_a, res_b, tol, what):
    """比较两次恢复的数值结果（用于刚性不变性）。"""
    errs = []
    for sid, sa in res_a["series"].items():
        sb = res_b["series"].get(sid)
        if sb is None:
            errs.append(f"{what}: {sid} 缺失")
            continue
        if [sp["visible"] for sp in sa["source_points"]] != \
           [sp["visible"] for sp in sb["source_points"]]:
            errs.append(f"{what}: {sid} 顶点可见性不一致")
        for spa, spb in zip(sa["source_points"], sb["source_points"]):
            for k in (0, 1):
                if not close(spa["value"][k], spb["value"][k], tol):
                    errs.append(f"{what}: {sid} 顶点 {spa['index']} 值 "
                                f"{spa['value']} != {spb['value']}")
        fa = [[(nd["kind"], nd.get("point_index"), nd.get("segment_index"))
               for nd in fr["points"]] for fr in sa["fragments"]]
        fb = [[(nd["kind"], nd.get("point_index"), nd.get("segment_index"))
               for nd in fr["points"]] for fr in sb["fragments"]]
        if fa != fb:
            errs.append(f"{what}: {sid} 片段结构不一致")
    return errs


def cmd_verify(manifest_path, out_dir):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mdir = manifest_path.parent
    checks = []

    def check(name, ok, detail=""):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(f"[verify] {'PASS' if ok else 'FAIL'} {name}"
              + (f" — {detail}" if detail else ""))

    results, failures = run_extract(manifest_path, out_dir, want_overlay=True)
    check("extract:all_charts", not failures,
          "; ".join(f"{f}: {e}" for f, e in failures))

    for spec in manifest["charts"]:
        fname = spec["file"]
        if fname not in results:
            continue
        res = results[fname]
        axes = res["_axes"]
        xax, yax = axes["x"], axes["y"]
        base = Path(fname).stem

        # 1) 恢复数值映回绘图区，核对几何位置（读回写出的 CSV/axes.json）
        axj = json.loads((out_dir / f"{base}.axes.json").read_text("utf-8"))
        fits = {k: v["fit"] for k, v in axj["axes"].items()}
        errs = []
        with open(out_dir / f"{base}.points.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                px, py = float(row["plot_x"]), float(row["plot_y"])
                xv, yv = float(row["x_value"]), float(row["y_value"])
                fx, fy = fits["x"], fits["y"]
                if fx["form"].startswith("log10"):
                    pbx = (math.log10(xv) - fx["b"]) / fx["a"]
                else:
                    pbx = (xv - fx["b"]) / fx["a"]
                if fy["form"].startswith("log10"):
                    pby = (math.log10(yv) - fy["b"]) / fy["a"]
                else:
                    pby = (yv - fy["b"]) / fy["a"]
                tol = 1e-9 * max(1.0, xax.span, yax.span)
                if abs(pbx - px) > tol or abs(pby - py) > tol:
                    errs.append(f"{row['series']} {row['kind']} "
                                f"{row['point_index'] or row['segment_index']}: "
                                f"映回误差 ({abs(pbx-px):.3g},{abs(pby-py):.3g})")
        check(f"roundtrip:{fname}", not errs, "; ".join(errs[:3]))

        # 2) 手算复核：用首末两个刻度做两点式独立重算每个恢复值
        errs = []
        for ax in (xax, yax):
            t0, t1 = ax.ticks[0], ax.ticks[-1]
            p0, v0 = t0["axis_pos"], t0["value"]
            p1, v1 = t1["axis_pos"], t1["value"]

            def manual(p, ax=ax, p0=p0, v0=v0, p1=p1, v1=v1):
                if ax.scale == "linear":
                    return v0 + (v1 - v0) * (p - p0) / (p1 - p0)
                l0, l1 = math.log10(v0), math.log10(v1)
                return 10.0 ** (l0 + (l1 - l0) * (p - p0) / (p1 - p0))

            for sid, s in res["series"].items():
                for nd in (nd for fr in s["fragments"] for nd in fr["points"]):
                    pv = nd["plot"][0] if ax.coord == "x" else nd["plot"][1]
                    got = nd["value"][0] if ax.coord == "x" else nd["value"][1]
                    want = manual(pv)
                    if not close(got, want, 1e-6):
                        errs.append(f"{sid} 轴{ax.name}: {got} != 手算 {want}")
        check(f"hand_two_point:{fname}", not errs, "; ".join(errs[:3]))

        # 3) log 轴十倍关系：相邻刻度值比为 10 时，位置间距必须相等
        for ax in (xax, yax):
            if ax.scale != "log10":
                continue
            errs = []
            tks = ax.ticks
            for u, v in zip(tks, tks[1:]):
                ratio = v["value"] / u["value"]
                if ratio <= 0:
                    errs.append("非正刻度值")
                    continue
                # 拟合推出的位置间距应与对数差一致
                dp_fit = math.log10(ratio) / ax.a
                dp_act = v["axis_pos"] - u["axis_pos"]
                if not close(dp_fit, dp_act, 1e-9):
                    errs.append(f"{u['id']}→{v['id']}: 间距 {dp_act} "
                                f"与十倍关系 {dp_fit} 不符")
            # 恢复值的十倍检查：沿拟合斜率方向相隔一个 decade 宽度的两点，
            # 值比恰为 10
            decade = 1.0 / ax.a  # 带符号：log10(value) 恰好增加 1
            pmid = sum(t["axis_pos"] for t in tks) / len(tks)
            r10 = ax.to_value(pmid + decade) / ax.to_value(pmid)
            if not close(r10, 10.0, 1e-9):
                errs.append(f"一个 decade 宽度对应值比 {r10} != 10")
            check(f"log_decade:{fname}:{ax.name}", not errs, "; ".join(errs))

        # 4) 几何一致性：交点带合法边界标记；隐藏顶点不出现在任何片段中
        errs = []
        for sid, s in res["series"].items():
            if not s["clip"]:
                continue
            for fr in s["fragments"]:
                for nd in fr["points"]:
                    if nd["kind"] == "intersection" and nd["boundary"] not in (
                            "xmin", "xmax", "ymin", "ymax"):
                        errs.append(f"{sid}: 非法边界 {nd['boundary']}")
            for sp in s["source_points"]:
                if not sp["visible"] and any(
                        sp["index"] == nd.get("point_index")
                        for fr in s["fragments"] for nd in fr["points"]):
                    errs.append(f"{sid}: 隐藏顶点 {sp['index']} 出现在片段中")
        check(f"geometry:{fname}", not errs, "; ".join(errs[:3]))

        # 5) 刚性变换不变性
        src = (mdir / fname).read_text(encoding="utf-8")
        try:
            res2 = extract_chart(rigid_wrap(src), spec, fname + " (rigid)")
            errs = verify_chart_values(res, res2, 1e-9, "rigid")
        except RecovError as e:
            errs = [f"刚性变换后提取失败: {e}"]
        check(f"rigid_invariance:{fname}", not errs, "; ".join(errs[:3]))

        # 6) 叠加证据文件存在且可解析
        ov = out_dir / f"{base}.overlay.svg"
        ok = ov.exists()
        if ok:
            try:
                ET.parse(ov)
            except ET.ParseError:
                ok = False
        check(f"overlay:{fname}", ok)

    # 7) 负例：错误刻度必须阻断可信数值输出
    case = mdir.parent / "cases" / "inconsistent_ticks.svg"
    if not case.exists():
        case = mdir / "cases" / "inconsistent_ticks.svg"
    if case.exists():
        text = case.read_text(encoding="utf-8")
        ids = set(re.findall(r'id="([^"]+)"', text))
        cloned = None
        for spec in manifest["charts"]:
            need = set(spec["series"]) | {
                t for ax in spec["axes"].values() for t in ax["ticks"]}
            if spec.get("plot_frame"):
                need.add(spec["plot_frame"])
            if need <= ids:
                cloned = copy.deepcopy(spec)
                break
        if cloned is None:
            check("negative:inconsistent_ticks", False, "无匹配的 manifest 规格")
        else:
            try:
                extract_chart(text, cloned, case.name)
                check("negative:inconsistent_ticks", False,
                      "错误刻度未被拒绝，产生了可信数值")
            except AxisInconsistent as e:
                check("negative:inconsistent_ticks", True, str(e)[:120])
            except RecovError as e:
                check("negative:inconsistent_ticks", False,
                      f"拒绝原因类型不符: {type(e).__name__}: {e}")
    else:
        check("negative:inconsistent_ticks", False, f"找不到 {case}")

    all_ok = all(c["ok"] for c in checks)
    (out_dir / "verify.json").write_text(json.dumps(
        {"ok": all_ok, "checks": checks}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"[verify] {'全部通过' if all_ok else '存在失败'} "
          f"({sum(c['ok'] for c in checks)}/{len(checks)})")
    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="SVG 折线图数值恢复工具")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", required=True)
        p.add_argument("--out", default="out")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    if args.cmd == "extract":
        _, failures = run_extract(Path(args.manifest), out_dir)
        return 1 if failures else 0
    return cmd_verify(Path(args.manifest), out_dir)


if __name__ == "__main__":
    sys.exit(main())
