"""
Generate the Deep-Space Photonics Thermal Advisor hero banner.

Produces a rich infographic (docs/thermals.svg + docs/thermals.png) with real,
legible labels — material properties, space-environment conditions, physics
thermal-drift surfaces, the integrated prediction engine, agentic tool-use
analytics, predicted drift charts, and mitigation strategies.

Run:  python docs/make_banner.py

Author: A Taylor
"""

import math
from pathlib import Path

W, H = 1240, 690

# ---- thermal colour ramp (cold blue -> cyan -> amber -> hot red) ----
_RAMP = [
    (0.00, (46, 92, 196)),
    (0.35, (38, 178, 196)),
    (0.62, (240, 198, 66)),
    (1.00, (222, 64, 52)),
]


def thermal_color(t):
    """Map t in [0, 1] to a hex colour along the thermal ramp."""
    t = max(0.0, min(1.0, t))
    for (t0, c0), (t1, c1) in zip(_RAMP, _RAMP[1:]):
        if t <= t1:
            f = 0 if t1 == t0 else (t - t0) / (t1 - t0)
            r = round(c0[0] + (c1[0] - c0[0]) * f)
            g = round(c0[1] + (c1[1] - c0[1]) * f)
            b = round(c0[2] + (c1[2] - c0[2]) * f)
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#de4034"


def surface(cx, cy, n, cell, hz, fn, width=1.4, opacity=0.95):
    """Build an isometric wireframe surface coloured by height.

    Args:
        cx, cy: screen anchor (top-centre-ish) of the surface.
        n: grid resolution (n x n cells).
        cell: horizontal cell size in px.
        hz: vertical exaggeration in px.
        fn: height function fn(i, j, n) -> float.
        width: stroke width.
        opacity: stroke opacity.

    Returns:
        SVG markup string of coloured line segments (painter-ordered).
    """
    hs = {(i, j): fn(i, j, n) for i in range(n + 1) for j in range(n + 1)}
    hmin, hmax = min(hs.values()), max(hs.values())
    span = (hmax - hmin) or 1.0

    def pt(i, j):
        h = hs[(i, j)]
        sx = cx + (i - j) * cell
        sy = cy + (i + j) * (cell * 0.5) - (h - hmin) * hz
        return sx, sy

    segs = []
    for i in range(n + 1):
        for j in range(n + 1):
            for di, dj in ((1, 0), (0, 1)):
                ni, nj = i + di, j + dj
                if ni > n or nj > n:
                    continue
                x1, y1 = pt(i, j)
                x2, y2 = pt(ni, nj)
                t = ((hs[(i, j)] + hs[(ni, nj)]) / 2 - hmin) / span
                depth = i + j  # painter order: far (small) first
                segs.append((depth, x1, y1, x2, y2, thermal_color(t)))

    segs.sort(key=lambda s: s[0])
    out = [f'<g stroke-width="{width}" stroke-linecap="round" opacity="{opacity}">']
    for _, x1, y1, x2, y2, col in segs:
        out.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}"/>'
        )
    out.append("</g>")
    return "\n".join(out)


def polyline(points, color, width=2.2, dash=None, opacity=1.0):
    """Return an SVG polyline for a list of (x, y) points."""
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline points="{pts}" fill="none" stroke="{color}" '
        f'stroke-width="{width}"{d} opacity="{opacity}" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
    )


def panel(x, y, w, h, title):
    """Return SVG for a titled panel frame."""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" '
        f'fill="url(#panel)" stroke="#2f4675" stroke-width="1.5"/>'
        f'<rect x="{x}" y="{y}" width="{w}" height="30" rx="12" fill="#21345f"/>'
        f'<rect x="{x}" y="{y+16}" width="{w}" height="14" fill="#21345f"/>'
        f'<text x="{x+w/2}" y="{y+20}" text-anchor="middle" font-size="14" '
        f'font-weight="700" letter-spacing="0.6" fill="#7fe3d6">{title}</text>'
    )


def txt(x, y, s, size=13, fill="#dce6f5", anchor="start", weight="400", mono=False, ls=0):
    """Return an SVG text element."""
    fam = ' font-family="\'DejaVu Sans Mono\', monospace"' if mono else ""
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}" letter-spacing="{ls}"{fam}>{s}</text>'
    )


def build():
    s = []
    s.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f"font-family=\"'DejaVu Sans','Segoe UI',Arial,sans-serif\">"
    )

    # ---- defs ----
    s.append("""<defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#070b1c"/><stop offset="1" stop-color="#101c3a"/>
    </linearGradient>
    <linearGradient id="panel" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#16264a"/><stop offset="1" stop-color="#101b38"/>
    </linearGradient>
    <radialGradient id="sun" cx="0.5" cy="0.5" r="0.5">
      <stop offset="0" stop-color="#fff3c4"/><stop offset="0.5" stop-color="#ffb347"/>
      <stop offset="1" stop-color="#b85c00" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="jupiter" cx="0.4" cy="0.4" r="0.7">
      <stop offset="0" stop-color="#e8cba0"/><stop offset="1" stop-color="#9c6b3f"/>
    </radialGradient>
    <radialGradient id="moon" cx="0.4" cy="0.4" r="0.7">
      <stop offset="0" stop-color="#cfd8e6"/><stop offset="1" stop-color="#6b7689"/>
    </radialGradient>
    <linearGradient id="chip" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#2a3f6e"/><stop offset="1" stop-color="#16233f"/>
    </linearGradient>
    <linearGradient id="accent" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#4fd1c5"/><stop offset="1" stop-color="#63b3ed"/>
    </linearGradient>
    <marker id="arrow" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#4fd1c5"/>
    </marker>
    </defs>""")

    # ---- background + stars + faint grid ----
    s.append(f'<rect width="{W}" height="{H}" fill="url(#bg)"/>')
    stars = [(90, 60, 1.3), (250, 40, 1), (430, 70, 1.4), (610, 36, 1),
             (790, 64, 1.2), (980, 44, 1), (1150, 60, 1.3), (60, 150, 1),
             (1190, 130, 1.1), (520, 150, 0.9), (700, 120, 1), (340, 130, 0.9)]
    s.append('<g fill="#ffffff" opacity="0.5">')
    for x, y, r in stars:
        s.append(f'<circle cx="{x}" cy="{y}" r="{r}"/>')
    s.append("</g>")

    # ---- title ----
    s.append(txt(W / 2, 40, "DEEP-SPACE PHOTONICS THERMAL ADVISOR", size=27,
                 fill="#ffffff", anchor="middle", weight="700", ls=1.2))
    s.append(txt(W / 2, 64, "Tool-Using Agent  +  Physics Simulator  for Thermal Mitigation in Photonic Instruments",
                 size=13, fill="#8fa6c8", anchor="middle"))

    # ============================================================ TOP-LEFT
    px, py, pw, ph = 20, 80, 590, 150
    s.append(panel(px, py, pw, ph, "DETAILED MATERIAL PROPERTIES"))
    # micro-structure swatches
    for k, col in enumerate(["#7d8aa0", "#9aa7bd", "#5c6b86"]):
        sx = px + 16 + k * 52
        s.append(f'<rect x="{sx}" y="{py+44}" width="44" height="44" rx="4" fill="{col}"/>')
        s.append(f'<rect x="{sx}" y="{py+44}" width="44" height="44" rx="4" fill="none" stroke="#0d1730" stroke-width="3"/>')
    s.append(txt(px + 16, py + 104, "Chip micro-structure", size=11, fill="#8fa6c8"))
    s.append(txt(px + 16, py + 122, "(layered photonic stack)", size=11, fill="#8fa6c8"))
    # materials table
    tx = px + 210
    cols = [tx, tx + 150, tx + 250]
    s.append(txt(cols[0], py + 50, "Material", size=12, fill="#7fe3d6", weight="700"))
    s.append(txt(cols[1], py + 50, "dn/dT (K⁻¹)", size=12, fill="#7fe3d6", weight="700"))
    s.append(txt(cols[2], py + 50, "α / CTE (K⁻¹)", size=12, fill="#7fe3d6", weight="700"))
    rows = [
        ("Silicon", "1.86×10⁻⁴", "2.6×10⁻⁶"),
        ("Silicon Nitride", "2.45×10⁻⁵", "8.0×10⁻⁷"),
        ("Polymer", "1.1×10⁻⁴", "2.2×10⁻⁶"),
        ("Indium Phosphide", "3.4×10⁻⁴", "4.6×10⁻⁶"),
    ]
    for r, (m, dn, al) in enumerate(rows):
        ry = py + 72 + r * 18
        s.append(txt(cols[0], ry, m, size=12, fill="#dce6f5"))
        s.append(txt(cols[1], ry, dn, size=12, fill="#cdd9ec"))
        s.append(txt(cols[2], ry, al, size=12, fill="#cdd9ec"))

    # ============================================================ TOP-RIGHT
    px, py, pw, ph = 630, 80, 590, 150
    s.append(panel(px, py, pw, ph, "COMPLEX SPACE ENVIRONMENT CONDITIONS"))
    # celestial scene
    cxn, cyn = px + 95, py + 92
    s.append(f'<circle cx="{px+40}" cy="{py+58}" r="34" fill="url(#sun)"/>')
    s.append(f'<ellipse cx="{cxn}" cy="{cyn}" rx="78" ry="30" fill="none" stroke="#3a4f7a" stroke-width="1.2"/>')
    s.append(f'<ellipse cx="{cxn}" cy="{cyn}" rx="50" ry="19" fill="none" stroke="#3a4f7a" stroke-width="1.2"/>')
    s.append(f'<circle cx="{cxn+78}" cy="{cyn}" r="14" fill="url(#jupiter)"/>')
    s.append(f'<circle cx="{cxn-50}" cy="{cyn}" r="7" fill="url(#moon)"/>')
    s.append(txt(px + 16, py + 130, "Solar flux · orbital thermal cycling · radiative load",
                 size=11, fill="#8fa6c8"))
    # environment table
    tx = px + 300
    s.append(txt(tx, py + 50, "Environment", size=12, fill="#7fe3d6", weight="700"))
    s.append(txt(tx + 165, py + 50, "ΔT (K)", size=12, fill="#7fe3d6", weight="700"))
    s.append(txt(tx + 225, py + 50, "Severity", size=12, fill="#7fe3d6", weight="700"))
    envs = [
        ("Near-Earth Deep Space", "120", "Moderate", "#7fe3d6"),
        ("Mars Transit", "150", "Moderate", "#7fe3d6"),
        ("Jovian System", "180", "High", "#f0c642"),
        ("Outer Solar System", "240", "Critical", "#de6b52"),
    ]
    for r, (e, dt, sev, col) in enumerate(envs):
        ry = py + 72 + r * 18
        s.append(txt(tx, ry, e, size=12, fill="#dce6f5"))
        s.append(txt(tx + 165, ry, dt, size=12, fill="#cdd9ec"))
        s.append(txt(tx + 225, ry, sev, size=12, fill=col, weight="600"))

    # ============================================================ MIDDLE LEFT (sim surface)
    px, py = 20, 250
    s.append(txt(px + 170, py + 4, "HIGH-FIDELITY PHYSICS-BASED", size=12, fill="#7fe3d6",
                 weight="700", anchor="middle"))
    s.append(txt(px + 170, py + 20, "THERMAL SIMULATION", size=12, fill="#7fe3d6",
                 weight="700", anchor="middle"))
    s.append(surface(px + 150, py + 70, 12, 13,
                     hz=78, fn=lambda i, j, n: math.exp(-(((i-n/2)/(n/2))**2 + ((j-n/2)/(n/2))**2) * 2.1)))
    s.append(txt(px + 170, py + 196, "Δn = dn/dT × ΔT   ·   ε = α × ΔT",
                 size=13, fill="#dce6f5", anchor="middle"))

    # ============================================================ MIDDLE CENTER (engine)
    cx0, cy0 = 430, 250
    cw = 380
    # 3D prediction surface above the chip
    s.append(surface(cx0 + cw / 2, cy0 + 70, 12, 12,
                     hz=60, fn=lambda i, j, n: 0.5 + 0.5 * math.sin(i * 0.85) * math.cos(j * 0.8),
                     opacity=0.9))
    # chip body
    chx, chy, chw, chph = cx0 + 60, cy0 + 96, cw - 120, 96
    for px_ in range(int(chx) + 14, int(chx + chw) - 10, 18):  # pins top/bottom
        s.append(f'<rect x="{px_}" y="{chy-7}" width="8" height="7" fill="#3c5891"/>')
        s.append(f'<rect x="{px_}" y="{chy+chph}" width="8" height="7" fill="#3c5891"/>')
    s.append(f'<rect x="{chx}" y="{chy}" width="{chw}" height="{chph}" rx="8" fill="url(#chip)" stroke="#4f6fa8" stroke-width="1.5"/>')
    s.append(txt(cx0 + cw / 2, chy + 34, "INTEGRATED", size=15, fill="#ffffff", anchor="middle", weight="700", ls=1))
    s.append(txt(cx0 + cw / 2, chy + 56, "PREDICTION ENGINE", size=15, fill="#ffffff", anchor="middle", weight="700", ls=1))
    s.append(txt(cx0 + cw / 2, chy + 78, "agent reason → act tool-use loop", size=11, fill="#9fd8ce", anchor="middle"))

    # ============================================================ MIDDLE RIGHT (analytics)
    px, py, pw, ph = 860, 244, 360, 196
    s.append(panel(px, py, pw, ph, "AGENTIC TOOL-USE ANALYTICS"))
    code = [
        ("agent.run(scenario)", "#9fd8ce"),
        ("→ simulate_thermal_drift()", "#cdd9ec"),
        ("→ classify_strategy()  [XGBoost]", "#cdd9ec"),
        ("→ search_thermal_knowledge()", "#cdd9ec"),
        ("backend: Bedrock | Local GGUF", "#8fa6c8"),
        ("✔ grounded recommendation", "#7fe3d6"),
    ]
    for r, (line, col) in enumerate(code):
        s.append(txt(px + 16, py + 56 + r * 22, line, size=12.5, fill=col, mono=True))

    # ---- connector arrows ----
    s.append('<g stroke="#4fd1c5" stroke-width="2" fill="none" opacity="0.85">')
    s.append(f'<line x1="315" y1="230" x2="430" y2="300" marker-end="url(#arrow)"/>')      # mat -> engine
    s.append(f'<line x1="925" y1="230" x2="810" y2="300" marker-end="url(#arrow)"/>')      # env -> engine
    s.append(f'<line x1="980" y1="240" x2="980" y2="200" opacity="0"/>')
    s.append(f'<line x1="560" y1="460" x2="430" y2="500" marker-end="url(#arrow)"/>')      # engine -> drift
    s.append(f'<line x1="700" y1="460" x2="840" y2="500" marker-end="url(#arrow)"/>')      # engine -> mitigation
    s.append("</g>")

    # ============================================================ BOTTOM LEFT (drift charts)
    px, py, pw, ph = 20, 470, 590, 198
    s.append(panel(px, py, pw, ph, "PREDICTED THERMAL DRIFT PATTERNS"))
    # chart 1: spectral drift vs time (damped oscillation)
    ax, ay, aw, ahh = px + 50, py + 48, 230, 110
    s.append(f'<line x1="{ax}" y1="{ay}" x2="{ax}" y2="{ay+ahh}" stroke="#3a4f7a" stroke-width="1.2"/>')
    s.append(f'<line x1="{ax}" y1="{ay+ahh}" x2="{ax+aw}" y2="{ay+ahh}" stroke="#3a4f7a" stroke-width="1.2"/>')
    pts1 = [(ax + k / 60 * aw, ay + ahh / 2 - math.sin(k / 60 * 6.5) * math.exp(-k / 90) * ahh * 0.42)
            for k in range(61)]
    s.append(polyline(pts1, "#4fd1c5"))
    s.append(txt(ax, py + 40, "Spectral drift Δλ  vs  time", size=11, fill="#8fa6c8"))
    s.append(txt(ax - 6, ay + ahh + 16, "t", size=11, fill="#8fa6c8"))
    # chart 2: temperature ramp + cycling
    bx = px + 330
    s.append(f'<line x1="{bx}" y1="{ay}" x2="{bx}" y2="{ay+ahh}" stroke="#3a4f7a" stroke-width="1.2"/>')
    s.append(f'<line x1="{bx}" y1="{ay+ahh}" x2="{bx+aw}" y2="{ay+ahh}" stroke="#3a4f7a" stroke-width="1.2"/>')
    pts2 = [(bx + k / 60 * aw, ay + ahh - (1 - math.exp(-k / 28)) * ahh * 0.7 - math.sin(k / 6) * 6)
            for k in range(61)]
    s.append(polyline(pts2, "#f0a043"))
    s.append(txt(bx, py + 40, "Temperature ΔT  vs  time", size=11, fill="#8fa6c8"))
    s.append(txt(bx - 6, ay + ahh + 16, "t", size=11, fill="#8fa6c8"))

    # ============================================================ BOTTOM RIGHT (mitigation)
    px, py, pw, ph = 630, 470, 590, 198
    s.append(panel(px, py, pw, ph, "MITIGATION STRATEGIES"))
    s.append(txt(px + 16, py + 52, "✔ Active Thermal Control", size=13, fill="#7fe3d6", weight="700"))
    for r, line in enumerate([
        "Multi-Layer Insulation (MLI) geometry",
        "Active TCS feedback loops",
        "Thermal coating selection and application",
    ]):
        s.append(txt(px + 30, py + 72 + r * 17, f"• {line}", size=11.5, fill="#cdd9ec"))
    s.append(txt(px + 16, py + 138, "✔ Advanced Material Selection", size=13, fill="#7fe3d6", weight="700"))
    for r, line in enumerate([
        "Carbon-fiber (CFRP) laminate lay-up",
        "Zerodur ULE glass-ceramic substrate",
    ]):
        s.append(txt(px + 30, py + 158 + r * 17, f"• {line}", size=11.5, fill="#cdd9ec"))
    # impact index bars
    bx0 = px + 380
    s.append(txt(bx0, py + 52, "Strategy Impact Index", size=12, fill="#7fe3d6", weight="700"))
    bars = [("Passive", 0.55, "#63b3ed"), ("Active", 0.78, "#4fd1c5"), ("Hybrid", 0.94, "#9ae6b4")]
    for r, (lab, val, col) in enumerate(bars):
        by = py + 74 + r * 30
        s.append(txt(bx0, by + 11, lab, size=12, fill="#dce6f5"))
        s.append(f'<rect x="{bx0+62}" y="{by}" width="120" height="14" rx="7" fill="#1c2c50"/>')
        s.append(f'<rect x="{bx0+62}" y="{by}" width="{120*val:.0f}" height="14" rx="7" fill="{col}"/>')

    s.append("</svg>")
    return "\n".join(s)


def main():
    out_dir = Path(__file__).resolve().parent
    svg_path = out_dir / "thermals.svg"
    png_path = out_dir / "thermals.png"
    svg = build()
    svg_path.write_text(svg, encoding="utf-8")
    print(f"wrote {svg_path}")
    try:
        import cairosvg

        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path),
                         output_width=2 * W, output_height=2 * H)
        print(f"wrote {png_path}")
    except ImportError:
        print("cairosvg not installed — SVG written; install cairosvg to render PNG")


if __name__ == "__main__":
    main()
