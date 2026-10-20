"""Generate the Fingerspell overview figure (light + dark animated SVG):
camera -> palm detector (NPU) -> 21 keypoints (NPU) -> letter MLP (M55) -> LCD, all on the Alif E8.

The hand photo (hand.png, a test fixture) and the board render (alif_e8_board_top.png) are embedded as base64,
which GitHub blocks inside an SVG <img>. The README therefore uses overview.png / overview-dark.png: animated PNGs
(40 frames, 4 s loop, transparent) made from these SVGs:
  python tools/render_frames.py overview.svg <dir>/light 40    # frames on black and white, paused at t = 0 .. 3.9 s
  python tools/apng_delta.py <dir>/light overview.png 40       # alpha from the black/white pair; frames store only changes
Run: python documentation/media/make_overview_svg.py documentation/media
Every animation duration divides LOOP, so the 40-frame loop has no jump.
"""
import base64
import io
import sys
from pathlib import Path

from PIL import Image

W, H = 1500, 560
HERE = Path(__file__).parent
LOOP = 4.0                      # one letter of "HELLO" every 0.8 s
WORD = "HELLO"
# top-3 MLP guesses per letter (illustrative)
GUESSES = [("H", .94, "G", .03, "U", .02), ("E", .91, "S", .05, "O", .02), ("L", .96, "D", .02, "G", .01),
           ("L", .95, "D", .03, "X", .01), ("O", .89, "C", .06, "E", .03)]
HUES = {"cam": "6,182,212", "det": "124,92,255", "lm": "34,197,94", "mlp": "245,158,11", "lcd": "59,130,246",
        "board": "236,72,153"}
THEMES = {
    "light": dict(title="#2b2b2b", tools="#3a3a3a", role="#6b6b6b", stroke="#4b4b4b", arrow="#9a9a9a",
                  label="#6b6b6b", fill_a=0.13, edge_a=0.75, chip_a=0.28, sheen_a=0.65),
    "dark": dict(title="#e6edf3", tools="#c9d1d9", role="#8b949e", stroke="#c9d1d9", arrow="#6e7681",
                 label="#8b949e", fill_a=0.18, edge_a=0.85, chip_a=0.40, sheen_a=0.10),
}
# finger colours: thumb, index, middle, ring, pinky (palm lines use the theme stroke)
FINGER = ["244,63,94", "249,115,22", "234,179,8", "20,184,166", "139,92,246"]
# MediaPipe hand topology, 21 points of an open right hand in a 200 x 200 box
PTS = [(100, 186), (72, 170), (54, 146), (42, 124), (34, 104), (80, 110), (74, 78), (70, 56), (67, 34),
       (100, 104), (100, 70), (100, 46), (100, 22), (120, 108), (124, 77), (127, 56), (130, 37),
       (138, 120), (148, 98), (155, 82), (161, 64)]
PALM = [(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]

# 24x24 line icons
ICONS = {
    "camera": '<path d="M3 8h4l2-3h6l2 3h4v11H3z"/><circle cx="12" cy="13" r="3.5"/>',
    "npu": '<rect x="6" y="6" width="12" height="12" rx="1.5"/>'
           '<path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>'
           '<rect x="9.5" y="9.5" width="5" height="5" rx="1" fill="{stroke}" stroke="none">'
           '<animate attributeName="opacity" values="1;0.25;1" dur="0.8s" repeatCount="indefinite"/></rect>',
    "hand": '<path d="M8 13V5.5a1.5 1.5 0 0 1 3 0V11M11 10V4a1.5 1.5 0 0 1 3 0v7M14 10.5V5.5a1.5 1.5 0 0 1 3 0V13'
            'M17 12v-2.5a1.5 1.5 0 0 1 3 0V14a7 7 0 0 1-7 7h-1a7 7 0 0 1-5.6-2.8L3.3 14a1.5 1.5 0 0 1 2.4-1.8L8 15"/>',
    "chip": '<rect x="6" y="6" width="12" height="12" rx="1.5"/>'
            '<path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/><path d="M9 13l1.5-3 2 4 1.5-2"/>',
    "lcd": '<rect x="2" y="4" width="20" height="14" rx="1.5"/><path d="M6 9h6M6 13h9"/><path d="M8 21h8"/>',
    "imu": '<circle cx="12" cy="13" r="2"/><path d="M12 11V3M10 5l2-2 2 2M13.7 14l6.3 4M17.5 18.6l2.5-.6-.6-2.4'
           'M10.3 14L4 18M6.5 18.6L4 18l.6-2.4"/>',
}


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def b64(path, width):
    img = Image.open(HERE / path)
    img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode(), img.height


def image(x, y, w, h, data, extra=""):
    return (f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="data:image/png;base64,{data}" '
            f'xlink:href="data:image/png;base64,{data}"{extra}/>')


def glass(t, hue, x, y, w, h, rx=16, dash=False):
    c = HUES[hue]
    d = ' stroke-dasharray="7 6"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="rgba({c},{t["fill_a"]})" '
            f'stroke="rgba({c},{t["edge_a"]})" stroke-width="1.6"{d}/>'
            f'<rect x="{x + 1.5}" y="{y + 1.5}" width="{w - 3}" height="{min(h * 0.4, 110)}" rx="{rx - 1.5}" fill="url(#sheen)"/>')


def head(t, hue, x, y, w, icon, title, tag):
    c = HUES[hue]
    tw = 12 + 8.4 * len(tag)
    return (f'<g transform="translate({x + 14},{y + 14}) scale(1.1)" fill="none" stroke="{t["stroke"]}" stroke-width="1.8" '
            f'stroke-linecap="round" stroke-linejoin="round">{ICONS[icon].replace("{stroke}", t["stroke"])}</g>'
            f'<text x="{x + 50}" y="{y + 34}" font-size="15.5" font-weight="700" letter-spacing="0.3" fill="{t["title"]}">{esc(title)}</text>'
            f'<rect x="{x + w - tw - 12}" y="{y + 16}" width="{tw}" height="24" rx="12" fill="rgba({c},{t["chip_a"]})" '
            f'stroke="rgba({c},{t["edge_a"]})" stroke-width="1"/>'
            f'<text x="{x + w - 12 - tw / 2}" y="{y + 32.5}" font-size="12" font-weight="700" text-anchor="middle" '
            f'fill="{t["title"]}">{esc(tag)}</text>')


def lines(t, x, y, rows, size=14, gap=20):
    return "".join(f'<text x="{x}" y="{y + gap * i}" font-size="{size}" fill="{t["tools"] if i == 0 else t["role"]}">{esc(r)}</text>'
                   for i, r in enumerate(rows))


def arrow(t, d, label=None, lx=0, ly=0):
    s = (f'<path d="{d}" fill="none" stroke="{t["arrow"]}" stroke-width="2.4" stroke-dasharray="7 5" '
         f'marker-end="url(#ah)"><animate attributeName="stroke-dashoffset" from="24" to="0" dur="0.8s" '
         f'repeatCount="indefinite"/></path>')
    if label:
        s += f'<text x="{lx}" y="{ly}" font-size="12.5" text-anchor="middle" fill="{t["label"]}">{esc(label)}</text>'
    return s


def phase(k, n=len(WORD)):
    """Discrete opacity animation: visible only during letter k of the loop."""
    vals = ";".join("1" if i == k else "0" for i in range(n)) + ";" + ("1" if k == 0 else "0")
    kts = ";".join(f"{i / n:.3f}" for i in range(n)) + ";1"
    return (f'<animate attributeName="opacity" values="{vals}" keyTimes="{kts}" calcMode="discrete" '
            f'dur="{LOOP}s" repeatCount="indefinite"/>')


def upto(k, n=len(WORD)):
    """Discrete opacity: visible from letter k to the end of the loop."""
    vals = ";".join("1" if i >= k else "0" for i in range(n)) + ";" + ("1" if k == 0 else "0")
    kts = ";".join(f"{i / n:.3f}" for i in range(n)) + ";1"
    return (f'<animate attributeName="opacity" values="{vals}" keyTimes="{kts}" calcMode="discrete" '
            f'dur="{LOOP}s" repeatCount="indefinite"/>')


def skeleton(t, ox, oy, s=1.0, joints=True):
    """21-point hand skeleton; joints light up wrist to fingertip in a 2 s wave."""
    P = [(ox + x * s, oy + y * s) for x, y in PTS]
    out = []
    for a, b in PALM:
        out.append(f'<line x1="{P[a][0]:.1f}" y1="{P[a][1]:.1f}" x2="{P[b][0]:.1f}" y2="{P[b][1]:.1f}" '
                   f'stroke="{t["stroke"]}" stroke-width="{2.2 * s:.1f}" stroke-linecap="round" opacity="0.7"/>')
    for f in range(5):
        idx = [0] + list(range(1 + 4 * f, 5 + 4 * f)) if f == 0 else list(range(1 + 4 * f, 5 + 4 * f))
        d = "M" + " L".join(f"{P[i][0]:.1f},{P[i][1]:.1f}" for i in idx)
        out.append(f'<path d="{d}" fill="none" stroke="rgb({FINGER[f]})" stroke-width="{3 * s:.1f}" '
                   f'stroke-linecap="round" stroke-linejoin="round"/>')
    if joints:
        for i, (x, y) in enumerate(P):
            f = 0 if i == 0 else (i - 1) // 4
            depth = 0 if i == 0 else (i - 1) % 4 + 1
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{4.2 * s:.1f}" fill="rgb({FINGER[f]})" stroke="#ffffff" '
                       f'stroke-width="{1.4 * s:.1f}"><animate attributeName="r" values="{4.2 * s:.1f};{7 * s:.1f};{4.2 * s:.1f};{4.2 * s:.1f}" '
                       f'keyTimes="0;0.12;0.25;1" dur="2s" begin="{0.2 * depth - 2:.1f}s" repeatCount="indefinite"/></circle>')
    return "".join(out)


def build(theme):
    t = THEMES[theme]
    hand, hand_h = b64("hand.png", 200)
    board, board_h = b64("alif_e8_board_top.png", 128)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {W} {H}" '
         f'width="{W}" height="{H}" font-family="Inter, -apple-system, \'Segoe UI\', Helvetica, Arial, sans-serif">',
         '<defs><linearGradient id="sheen" x1="0" y1="0" x2="0" y2="1">'
         f'<stop offset="0" stop-color="#ffffff" stop-opacity="{t["sheen_a"]}"/>'
         '<stop offset="1" stop-color="#ffffff" stop-opacity="0"/></linearGradient>'
         '<linearGradient id="scan" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#22d3ee" stop-opacity="0"/>'
         '<stop offset="1" stop-color="#22d3ee" stop-opacity="0.55"/></linearGradient>'
         '<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
         f'<path d="M0,0 L10,5 L0,10 z" fill="{t["arrow"]}"/></marker>'
         '<clipPath id="photo1"><rect width="200" height="200" rx="10"/></clipPath>'
         '<clipPath id="photo2"><rect width="200" height="200" rx="10"/></clipPath></defs>']

    cy, ch, cw, gap = 20, 380, 248, 55
    X = [20 + i * (cw + gap) for i in range(5)]

    # 1. Camera
    x = X[0]
    p.append(glass(t, "cam", x, cy, cw, ch))
    p.append(head(t, "cam", x, cy, cw, "camera", "CAMERA", "INPUT"))
    px, py = x + 24, cy + 62
    p.append(f'<g transform="translate({px},{py})"><g clip-path="url(#photo1)">{image(0, 0, 200, hand_h, hand)}'
             f'<rect x="0" y="-60" width="200" height="60" fill="url(#scan)">'
             f'<animateTransform attributeName="transform" type="translate" values="0 0;0 260" dur="2s" repeatCount="indefinite"/></rect>'
             '</g>')
    for cx_, cy_, dx, dy in ((8, 8, 1, 1), (192, 8, -1, 1), (8, 192, 1, -1), (192, 192, -1, -1)):
        p.append(f'<path d="M{cx_},{cy_ + 18 * dy} V{cy_} H{cx_ + 18 * dx}" fill="none" stroke="#ffffff" stroke-width="3" stroke-linecap="round"/>')
    p.append('<circle cx="182" cy="24" r="6" fill="#ef4444"><animate attributeName="opacity" values="1;0.15;1" dur="1s" '
             'repeatCount="indefinite"/></circle></g>')
    p.append(lines(t, x + 24, cy + 300, ["MT9M114 on the DevKit", "320×320 RGB565 frames", "about 15 fps"]))

    # 2. Palm detector
    x = X[1]
    p.append(glass(t, "det", x, cy, cw, ch))
    p.append(head(t, "det", x, cy, cw, "npu", "PALM DETECTOR", "U85"))
    px, py = x + 24, cy + 62
    p.append(f'<g transform="translate({px},{py})"><g clip-path="url(#photo2)">{image(0, 0, 200, hand_h, hand)}'
             '<rect width="200" height="200" fill="#000" opacity="0.25"/></g>')
    # hand box snaps on, holds, then re-detects
    p.append('<rect x="14" y="18" width="150" height="172" rx="6" fill="rgba(167,139,250,.18)" stroke="#a78bfa" '
             'stroke-width="3" stroke-dasharray="10 6"><animate attributeName="stroke-dashoffset" from="32" to="0" '
             'dur="0.8s" repeatCount="indefinite"/></rect>'
             '<rect x="14" y="-4" width="86" height="20" rx="4" fill="#a78bfa"/>'
             '<text x="57" y="10.5" font-size="11.5" font-weight="700" text-anchor="middle" fill="#1e1036">hand 0.98</text></g>')
    p.append(lines(t, x + 24, cy + 300, ["BlazePalm, 192×192 int8", "2,016 anchors, best box", "≈ 23 ms on the NPU"]))

    # 3. Keypoints
    x = X[2]
    p.append(glass(t, "lm", x, cy, cw, ch))
    p.append(head(t, "lm", x, cy, cw, "hand", "21 KEYPOINTS", "U85"))
    p.append(f'<rect x="{x + 24}" y="{cy + 62}" width="200" height="200" rx="10" fill="rgba(34,197,94,.08)" '
             f'stroke="rgba(34,197,94,.45)" stroke-width="1" stroke-dasharray="3 4"/>')
    p.append(skeleton(t, x + 24, cy + 62 - 4))
    p.append(lines(t, x + 24, cy + 300, ["hand_landmark_lite, 224×224", "21 × (x, y, z), presence", "≈ 6 ms on the NPU"]))

    # 4. Letter MLP
    x = X[3]
    p.append(glass(t, "mlp", x, cy, cw, ch))
    p.append(head(t, "mlp", x, cy, cw, "chip", "LETTER MLP", "M55"))
    bx, by = x + 24, cy + 62
    p.append(f'<rect x="{bx}" y="{by}" width="200" height="200" rx="10" fill="rgba(245,158,11,.08)" '
             f'stroke="rgba(245,158,11,.45)" stroke-width="1" stroke-dasharray="3 4"/>')
    for k, (a, pa, b, pb, c, pc) in enumerate(GUESSES):
        g = [f'<g opacity="{1 if k == 0 else 0}">{phase(k)}',
             f'<text x="{bx + 100}" y="{by + 92}" font-size="76" font-weight="800" text-anchor="middle" fill="#f59e0b">{a}</text>']
        for r, (L, pr) in enumerate(((a, pa), (b, pb), (c, pc))):
            yy = by + 122 + r * 24
            col = "#22c55e" if r == 0 else t["arrow"]
            g.append(f'<text x="{bx + 18}" y="{yy + 13}" font-size="13" font-weight="700" fill="{t["tools"]}" '
                     f'font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">{L}</text>'
                     f'<rect x="{bx + 36}" y="{yy + 2}" width="112" height="13" rx="6.5" fill="{t["arrow"]}" opacity="0.18"/>'
                     f'<rect x="{bx + 36}" y="{yy + 2}" width="{max(4, 112 * pr):.0f}" height="13" rx="6.5" fill="{col}"/>'
                     f'<text x="{bx + 182}" y="{yy + 13}" font-size="12" text-anchor="end" fill="{t["role"]}" '
                     f'font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">{pr:.2f}</text>')
        g.append('</g>')
        p.append("".join(g))
    p.append(lines(t, x + 24, cy + 300, ["63 features, 63-128-24 MLP", "24 still letters; Z and J", "from the fingertip's motion"]))

    # 5. LCD
    x = X[4]
    p.append(glass(t, "lcd", x, cy, cw, ch))
    p.append(head(t, "lcd", x, cy, cw, "lcd", "LCD", "OUTPUT"))
    sx, sy = x + 20, cy + 62
    p.append(f'<rect x="{sx}" y="{sy}" width="208" height="200" rx="12" fill="#2a2f3a"/>'
             f'<rect x="{sx + 8}" y="{sy + 8}" width="192" height="184" rx="6" fill="#0d1220"/>')
    # mini camera view with the skeleton, as the panel shows it
    p.append(f'<rect x="{sx + 18}" y="{sy + 18}" width="80" height="80" rx="4" fill="#161d30" stroke="#2c3550"/>'
             f'{skeleton(t | {"stroke": "#9aa4b8"}, sx + 22, sy + 18, 0.38, joints=False)}')
    p.append(f'<text x="{sx + 110}" y="{sy + 36}" font-size="11" fill="#9aa4b8">letter</text>')
    for k, (a, pa, *_rest) in enumerate(GUESSES):
        p.append(f'<g opacity="{1 if k == 0 else 0}">{phase(k)}'
                 f'<text x="{sx + 110}" y="{sy + 80}" font-size="40" font-weight="800" fill="#5eead4">{a}</text>'
                 f'<text x="{sx + 146}" y="{sy + 80}" font-size="12" fill="#5eead4" '
                 f'font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">{pa:.2f}</text></g>')
    p.append(f'<path d="M{sx + 18},{sy + 112} H{sx + 190}" stroke="#2c3550"/>'
             f'<text x="{sx + 18}" y="{sy + 132}" font-size="11" fill="#9aa4b8">text</text>')
    for k in range(len(WORD)):
        p.append(f'<text x="{sx + 18 + k * 30}" y="{sy + 170}" font-size="34" font-weight="800" fill="#ffffff" '
                 f'opacity="{1 if k == 0 else 0}">{WORD[k]}{upto(k)}</text>')
    caret = ";".join(str(sx + 18 + (k + 1) * 30 + 2) for k in range(len(WORD))) + f";{sx + 50}"
    p.append(f'<rect x="{sx + 50}" y="{sy + 142}" width="3" height="32" fill="#5eead4">'
             f'<animate attributeName="x" values="{caret}" keyTimes="{";".join(f"{i / 5:.3f}" for i in range(5))};1" '
             f'calcMode="discrete" dur="{LOOP}s" repeatCount="indefinite"/>'
             '<animate attributeName="opacity" values="1;0;1" keyTimes="0;0.5;1" calcMode="discrete" dur="0.8s" '
             'repeatCount="indefinite"/></rect>')
    p.append(lines(t, x + 24, cy + 300, ["4.3\" panel: live view,", "letter, confidence, text", "joystick: space, delete"]))

    # arrows between the cards
    labels = ["frame", "hand box", "keypoints", "letter"]
    for i in range(4):
        x0, x1 = X[i] + cw + 4, X[i + 1] - 6
        p.append(arrow(t, f"M{x0},{cy + 162} H{x1}", labels[i], (x0 + x1) / 2, cy + 150))

    # board band: everything above runs on the E8
    yb, hb = 430, 112
    p.append(glass(t, "board", 20, yb, W - 40, hb))
    p.append(image(36, yb + (hb - board_h) / 2, 128, board_h, board))
    p.append(f'<text x="184" y="{yb + 40}" font-size="16" font-weight="700" letter-spacing="0.4" fill="{t["title"]}">'
             'ALIF ENSEMBLE E8 DEVKIT</text>')
    p.append(lines(t, 184, yb + 66, ["One ExecuTorch program, two int8 methods, every operator on the Ethos-U85;",
                                     "Cortex-M55 runs the MLP, decoder and LCD. No phone, no cloud, no network."], 14, 22))
    # IMU chip feeding the MLP
    ix, iw = X[3] - 2, cw + 4
    p.append(f'<rect x="{ix}" y="{yb + 22}" width="{iw}" height="68" rx="12" fill="rgba({HUES["mlp"]},{t["chip_a"]})" '
             f'stroke="rgba({HUES["mlp"]},{t["edge_a"]})" stroke-width="1.2"/>'
             f'<g transform="translate({ix + 12},{yb + 34}) scale(0.95)" fill="none" stroke="{t["stroke"]}" stroke-width="1.8" '
             f'stroke-linecap="round" stroke-linejoin="round">{ICONS["imu"]}</g>'
             f'<text x="{ix + 46}" y="{yb + 49}" font-size="14" font-weight="700" fill="{t["title"]}">IMU (ICM-42670-P)</text>'
             f'<text x="{ix + 46}" y="{yb + 72}" font-size="13" fill="{t["role"]}">levels the hand by gravity</text>')
    p.append(f'<path d="M{X[3] + cw / 2},{yb + 20} V{cy + ch + 8}" fill="none" stroke="{t["arrow"]}" stroke-width="2.2" '
             'stroke-dasharray="4 6" marker-end="url(#ah)"><animate attributeName="stroke-dashoffset" from="20" to="0" '
             'dur="0.8s" repeatCount="indefinite"/></path>')

    p.append("</svg>")
    return "\n".join(p) + "\n"


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "overview.svg").write_text(build("light"))
    (out / "overview-dark.svg").write_text(build("dark"))
    print("wrote", out / "overview.svg", out / "overview-dark.svg")
