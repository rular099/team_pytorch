#!/usr/bin/env python3
"""Generate a PPT-friendly SVG architecture diagram for train_light.py/eval_checkpoint.py."""
from html import escape

W, H = 1920, 1080
FONT = "Noto Sans CJK SC, Microsoft YaHei, SimHei, Arial, sans-serif"

boxes = []
arrows = []


def box(id_, x, y, w, h, lines, fill, stroke, fs=25, title=False):
    boxes.append(dict(id=id_, x=x, y=y, w=w, h=h, lines=lines, fill=fill, stroke=stroke, fs=fs, title=title))


def arrow(src, dst, dashed=False, label=None, bend=None):
    arrows.append(dict(src=src, dst=dst, dashed=dashed, label=label, bend=bend))


def center(b):
    return b["x"] + b["w"] / 2, b["y"] + b["h"] / 2


def right_mid(b):
    return b["x"] + b["w"], b["y"] + b["h"] / 2


def left_mid(b):
    return b["x"], b["y"] + b["h"] / 2


def bottom_mid(b):
    return b["x"] + b["w"] / 2, b["y"] + b["h"]


def top_mid(b):
    return b["x"] + b["w"] / 2, b["y"]


# Main boxes: PPT compressed version
box("A", 60, 420, 250, 130, ["日本强震", "多台站数据", "HDF5"], "#E8F3FF", "#2F80ED")
box("B", 365, 400, 300, 170, ["样本构造", "PreloadedEventGenerator", "25 输入台站", "15 PGA 目标台站"], "#E8F3FF", "#2F80ED", fs=23)
box("C", 370, 165, 290, 145, ["单台站预训练", "Mag / 距震中距离 / PGA", "Huber Loss"], "#FFF4E6", "#F2994A", fs=23)
box("D", 735, 390, 300, 190, ["DiTing 波形编码器", "backbone_attn_pool", "冻结预训练 backbone", "训练 attention pool adapter"], "#EAF8EF", "#27AE60", fs=23)
box("E", 1100, 395, 280, 180, ["台站表征融合", "波形 embedding", "+ 幅值特征", "+ 坐标位置编码"], "#EAF8EF", "#27AE60", fs=23)
box("F", 1440, 370, 315, 230, ["TEAM Transformer", "2 layers / 10 heads", "Event Token", "Station Tokens", "PGA Query Tokens"], "#F2EAFE", "#9B51E0", fs=23)

box("G1", 1460, 690, 255, 105, ["震级预测", "Magnitude", "PointOutput"], "#FFEFF3", "#EB5757", fs=23)
box("G2", 1130, 690, 255, 105, ["震源位置预测", "Lat / Lon / Depth", "PointOutput"], "#FFEFF3", "#EB5757", fs=23)
box("G3", 800, 690, 255, 105, ["PGA 预测", "15 目标台站", "PointOutput"], "#FFEFF3", "#EB5757", fs=23)
box("H", 1010, 875, 300, 120, ["三任务联合训练", "Huber Loss", "权重 0.2 / 0.2 / 1.0"], "#F7F7F7", "#333333", fs=23)
box("I", 1390, 875, 230, 120, ["Checkpoint", "full_model_*.pth"], "#F7F7F7", "#333333", fs=24)
box("J", 1680, 875, 200, 120, ["eval_checkpoint.py", "Train / Val 推理", "eval_results.npz"], "#F7F7F7", "#333333", fs=21)

# Flow arrows
arrow("A", "B")
arrow("B", "D")
arrow("B", "C", dashed=True)
arrow("C", "D", dashed=True, label="初始化 adapter / scale / layernorm")
arrow("D", "E")
arrow("E", "F")
arrow("F", "G1")
arrow("F", "G2")
arrow("F", "G3")
arrow("G1", "H")
arrow("G2", "H")
arrow("G3", "H")
arrow("H", "I")
arrow("I", "J")

box_by_id = {b["id"]: b for b in boxes}


def arrow_points(src, dst):
    s = box_by_id[src]
    d = box_by_id[dst]
    sx, sy = center(s)
    dx, dy = center(d)
    # Decide side based on relative displacement
    if abs(dx - sx) >= abs(dy - sy):
        if dx >= sx:
            return (*right_mid(s), *left_mid(d))
        return (*left_mid(s), *right_mid(d))
    else:
        if dy >= sy:
            return (*bottom_mid(s), *top_mid(d))
        return (*top_mid(s), *bottom_mid(d))


def draw_arrow(a):
    x1, y1, x2, y2 = arrow_points(a["src"], a["dst"])
    dashed = ' stroke-dasharray="10 8"' if a["dashed"] else ""
    # Orthogonal-ish paths for vertical-ish links, straight for horizontal.
    if abs(x2 - x1) > 80 and abs(y2 - y1) > 40:
        mx = (x1 + x2) / 2
        path = f"M{x1:.1f},{y1:.1f} C{mx:.1f},{y1:.1f} {mx:.1f},{y2:.1f} {x2:.1f},{y2:.1f}"
    else:
        path = f"M{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f}"
    label_svg = ""
    if a.get("label"):
        lx, ly = (x1 + x2) / 2, (y1 + y2) / 2 - 12
        label_svg = (
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="20" fill="#8A5A00">{escape(a["label"])}</text>'
        )
    return f'<path d="{path}" fill="none" stroke="#555" stroke-width="3" marker-end="url(#arrow)"{dashed}/>{label_svg}'


def draw_box(b):
    x, y, w, h = b["x"], b["y"], b["w"], b["h"]
    rect = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" ry="18" fill="{b["fill"]}" stroke="{b["stroke"]}" stroke-width="3"/>'
    lines = b["lines"]
    fs = b["fs"]
    total = len(lines)
    line_h = fs * 1.28
    start_y = y + h / 2 - (total - 1) * line_h / 2 + fs * 0.35
    tspans = []
    for i, line in enumerate(lines):
        weight = "700" if i == 0 else "500"
        tspans.append(
            f'<tspan x="{x + w/2:.1f}" y="{start_y + i*line_h:.1f}" font-weight="{weight}">{escape(line)}</tspan>'
        )
    text = f'<text text-anchor="middle" font-family="{FONT}" font-size="{fs}" fill="#111">' + "".join(tspans) + '</text>'
    return rect + text


svg = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    '<defs>',
    '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">',
    '<path d="M 0 0 L 10 5 L 0 10 z" fill="#555"/>',
    '</marker>',
    '<filter id="shadow" x="-10%" y="-10%" width="120%" height="120%"><feDropShadow dx="0" dy="3" stdDeviation="3" flood-opacity="0.18"/></filter>',
    '</defs>',
    '<rect width="100%" height="100%" fill="white"/>',
    f'<text x="{W/2}" y="70" text-anchor="middle" font-family="{FONT}" font-size="42" font-weight="800" fill="#111">TEAM-PyTorch 训练与推理模型架构</text>',
    f'<text x="{W/2}" y="112" text-anchor="middle" font-family="{FONT}" font-size="24" fill="#555">transformer_japan_overfit.json + diting_1200m_backbone_attnpool.yml · PointOutput · 三任务联合预测</text>',
]

# Draw arrows under boxes
for a in arrows:
    svg.append(draw_arrow(a))

# Draw boxes with a group shadow effect
svg.append('<g filter="url(#shadow)">')
for b in boxes:
    svg.append(draw_box(b))
svg.append('</g>')

# Bottom note
svg.append(
    f'<text x="60" y="1035" font-family="{FONT}" font-size="21" fill="#555">'
    '注：PGA query token 使用目标台站坐标编码；训练时 pga_target_valid mask 只计算有效目标台站损失；eval_checkpoint.py 加载 checkpoint 后输出 train/val 预测结果。'
    '</text>'
)
svg.append('</svg>')

open('team_pytorch/docs/figures/team_pytorch_model_architecture.svg', 'w', encoding='utf-8').write('\n'.join(svg))
print('Wrote team_pytorch/docs/figures/team_pytorch_model_architecture.svg')
