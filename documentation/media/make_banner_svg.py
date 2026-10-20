"""Generate the README banner (2048 x 768): a neon hand skeleton whose fingertips stream into the letters of HELLO.
Same palette and glow as the CorteX ARM banner. Render to PNG with headless Chrome:
  python documentation/media/make_banner_svg.py documentation/media
  chrome --headless=new --screenshot=banner.png --window-size=2048,768 banner.svg   (see render_banner below)
"""
import math
import random
import subprocess
import sys
from pathlib import Path

W, H = 2048, 768
# 21 MediaPipe points of an open right hand, 200 x 200 box (same shape as the overview figure)
PTS = [(100, 186), (72, 170), (54, 146), (42, 124), (34, 104), (80, 110), (74, 78), (70, 56), (67, 34),
       (100, 104), (100, 70), (100, 46), (100, 22), (120, 108), (124, 77), (127, 56), (130, 37),
       (138, 120), (148, 98), (155, 82), (161, 64)]
PALM = [(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]
FINGER = ["#ff7a18", "#ff3d8b", "#c13cff", "#6b5bff", "#22d3ee"]   # thumb .. pinky, orange to cyan
LETTERS = "HELLO"
TILE = ["#ff7a18", "#ff3d8b", "#c13cff", "#6b5bff", "#22d3ee"]


def build():
    rnd = random.Random(7)
    s, ox, oy = 3.3, 90, 50           # hand scale and offset
    P = [(ox + x * s, oy + y * s) for x, y in PTS]
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         "font-family=\"Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif\">",
         '<defs>'
         '<radialGradient id="gL" cx="0.12" cy="0.45" r="0.55"><stop offset="0" stop-color="#ff6a00" stop-opacity="0.85"/>'
         '<stop offset="0.5" stop-color="#b8370a" stop-opacity="0.35"/><stop offset="1" stop-color="#000" stop-opacity="0"/></radialGradient>'
         '<radialGradient id="gM" cx="0.45" cy="0.15" r="0.45"><stop offset="0" stop-color="#c13cff" stop-opacity="0.55"/>'
         '<stop offset="1" stop-color="#000" stop-opacity="0"/></radialGradient>'
         '<radialGradient id="gT" cx="0.42" cy="0.95" r="0.45"><stop offset="0" stop-color="#14b8a6" stop-opacity="0.45"/>'
         '<stop offset="1" stop-color="#000" stop-opacity="0"/></radialGradient>'
         '<radialGradient id="gR" cx="0.88" cy="0.5" r="0.5"><stop offset="0" stop-color="#1f6feb" stop-opacity="0.85"/>'
         '<stop offset="0.6" stop-color="#0b2a6b" stop-opacity="0.35"/><stop offset="1" stop-color="#000" stop-opacity="0"/></radialGradient>'
         '<filter id="glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="9" result="b"/>'
         '<feMerge><feMergeNode in="b"/><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
         '<filter id="soft" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="3.5" result="b"/>'
         '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
         '<filter id="haze" x="-30%" y="-60%" width="160%" height="220%"><feGaussianBlur stdDeviation="28"/></filter>'
         '<linearGradient id="flow" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#ff7a18"/>'
         '<stop offset="0.5" stop-color="#c13cff"/><stop offset="1" stop-color="#22d3ee"/></linearGradient>'
         '</defs>',
         f'<rect width="{W}" height="{H}" fill="#04040a"/>',
         f'<rect width="{W}" height="{H}" fill="url(#gL)"/><rect width="{W}" height="{H}" fill="url(#gM)"/>'
         f'<rect width="{W}" height="{H}" fill="url(#gT)"/><rect width="{W}" height="{H}" fill="url(#gR)"/>']

    # sparkles
    for _ in range(260):
        x, y = rnd.uniform(0, W), rnd.uniform(0, H)
        c = rnd.choice(["#ffb36b", "#ff7ac0", "#d38bff", "#8fb4ff", "#7ff0ff", "#ffffff"])
        p.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{rnd.uniform(0.6, 2.2):.1f}" fill="{c}" opacity="{rnd.uniform(0.2, 0.8):.2f}"/>')

    # streams from the five fingertips to the letter tiles
    tips = [P[4], P[8], P[12], P[16], P[20]]
    tx0, ty, tw, tg = 1090, 240, 140, 20
    centers = [(tx0 + i * (tw + tg) + tw / 2, ty + tw / 2) for i in range(5)]
    for f, ((x0, y0), (x1, y1)) in enumerate(zip(tips, centers)):
        x1, y1 = tx0 - 14, ty + tw / 2 + (f - 2) * 6   # all five streams meet at the first tile
        for k, (wd, op) in enumerate(((7, 0.18), (2.6, 0.95))):
            d = []
            n = 140
            for i in range(n + 1):
                u = i / n
                bx = x0 + (x1 - x0) * u
                by = y0 + (y1 - y0) * (3 * u * u - 2 * u * u * u)
                # an oscillating "signal" that grows mid-way, like the EEG traces of the sibling banner
                amp = 30 * math.sin(math.pi * u) ** 2
                by += amp * math.sin(2 * math.pi * (7 + f) * u + f) * (0.6 + 0.4 * math.sin(19 * u + f))
                d.append(f"{bx:.1f},{by:.1f}")
            flt = ' filter="url(#soft)"' if k else ""
            p.append(f'<path d="M{" L".join(d)}" fill="none" stroke="{FINGER[f]}" stroke-width="{wd}" '
                     f'stroke-linejoin="round" opacity="{op}"{flt}/>')

    # hand skeleton, glowing
    g = ['<g filter="url(#glow)" stroke-linecap="round" stroke-linejoin="round">']
    for a, b in PALM:
        g.append(f'<line x1="{P[a][0]:.0f}" y1="{P[a][1]:.0f}" x2="{P[b][0]:.0f}" y2="{P[b][1]:.0f}" stroke="#ffd2a8" stroke-width="5" opacity="0.8"/>')
    for f in range(5):
        idx = ([0] if f == 0 else []) + list(range(1 + 4 * f, 5 + 4 * f))
        g.append(f'<path d="M{" L".join(f"{P[i][0]:.0f},{P[i][1]:.0f}" for i in idx)}" fill="none" stroke="{FINGER[f]}" stroke-width="9"/>')
    for i, (x, y) in enumerate(P):
        f = 0 if i == 0 else (i - 1) // 4
        r = 15 if i in (4, 8, 12, 16, 20) else 11
        g.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r}" fill="{FINGER[f]}"/><circle cx="{x:.0f}" cy="{y:.0f}" r="{r * 0.45:.1f}" fill="#fff"/>')
    g.append('</g>')
    p += g

    # letter tiles
    for i, ((cx, cy), L) in enumerate(zip(centers, LETTERS)):
        x, y = cx - tw / 2, ty
        p.append(f'<g filter="url(#glow)"><rect x="{x}" y="{y}" width="{tw}" height="{tw}" rx="26" fill="{TILE[i]}" fill-opacity="0.16" '
                 f'stroke="{TILE[i]}" stroke-width="5"/>'
                 f'<text x="{cx}" y="{cy + 36}" font-size="104" font-weight="800" text-anchor="middle" fill="#ffffff" '
                 f'stroke="{TILE[i]}" stroke-width="2">{L}</text></g>')
    # caret after the word
    cx = centers[-1][0] + tw / 2 + 26
    p.append(f'<rect x="{cx}" y="{ty + 18}" width="10" height="{tw - 36}" rx="3" fill="#7ff0ff" filter="url(#glow)"/>')

    # title with a dark haze behind it, as on the sibling banner
    p.append(f'<ellipse cx="1180" cy="612" rx="470" ry="120" fill="#000" opacity="0.85" filter="url(#haze)"/>')
    p.append('<text x="1180" y="640" font-size="132" font-weight="700" text-anchor="middle" fill="#ffffff" '
             'letter-spacing="1">Fingerspell</text>')
    p.append('<text x="1180" y="712" font-size="46" font-weight="500" text-anchor="middle" fill="#f2f2f2" '
             'letter-spacing="4">Hands into words</text>')
    p.append("</svg>")
    return "\n".join(p) + "\n"


def render_banner(svg, png):
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    html = svg.with_suffix(".html")
    html.write_text('<html><body style="margin:0;background:#000">' + svg.read_text() + "</body></html>")
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--screenshot={png}",
                    f"--window-size={W},{H}", html.resolve().as_uri()], capture_output=True, check=True)
    html.unlink()


if __name__ == "__main__":
    out = Path(sys.argv[1])
    svg = out / "banner.svg"
    svg.write_text(build())
    render_banner(svg, out / "banner.png")
    print("wrote", svg, out / "banner.png")
