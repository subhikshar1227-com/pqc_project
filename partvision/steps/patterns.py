"""Hole patterns: circles of holes and straight rows, found among holes of the same shape and size.

Inputs come from two sources that may both be present:
  holes      - generic hole finder (ctx.results["holes"]["items"]): any part shape, all feature types
  bolt_circle - specialist bolt-circle finder (ctx.data["holes"] + ctx.data["off_circle_holes"]): round
                parts only, bolt holes and off-circle holes measured by the wall-fit method

Both sources produce round holes with x, y, d, and n fields; patterns analyses the merged list so that
bolt-circle holes and generic holes are considered together. The step runs when either source is available.

Settings in config analysis.patterns."""
import itertools
import math

import numpy as np

from ..config import C
from ..core import angle_deg, fit_circle
from ..registry import step

BLUE = (40, 170, 255)


def _collect_items(ctx):
    """Merge hole items from all available sources into one list for pattern analysis.

    bolt_circle holes use integer n (1, 2, …) and off-circle holes use string n ("A1", …).
    generic holes use string n ("H1", …).
    To avoid collisions the bolt-circle holes are re-labelled "B<n>" and off-circle holes "A<n>"
    here so pattern names are unambiguous.  The original labels are preserved in each item's
    "source_n" field for traceability.
    """
    items = []

    # Generic holes from holes.py (always present when holes ran)
    for h in ctx.results.get("holes", {}).get("items", []):
        items.append(h)

    # Bolt-circle holes from bolt_circle.py – only when that step ran and succeeded
    bc = ctx.results.get("bolt_circle", {})
    if bc.get("ok"):
        # Bolt-circle holes store their data in ctx.data, not ctx.results, because bolt_circle.py
        # predates the holes/patterns split.  Read from ctx.data directly.
        for h in ctx.data.get("holes", []):
            # Avoid double-counting: a round hole that holes.py already found at the same place.
            # Same-place test: centre within 0.5 x the smaller diameter of an existing item.
            already = any(
                math.hypot(h["x"] - q["x"], h["y"] - q["y"])
                <= 0.5 * min(h["d"], q["d"])
                for q in items
            )
            if already:
                continue
            # Normalise the schema to match holes.py items (add fields that patterns needs).
            item = dict(h)
            item.setdefault("shape", "round")
            item.setdefault("eq_d", h["d"])
            item.setdefault("length", h["d"])
            item.setdefault("width", h["d"])
            # Re-label to avoid clashing with H# labels from holes.py
            item["source_n"] = h["n"]
            item["n"] = f"B{h['n']}"
            items.append(item)

        for h in ctx.data.get("off_circle_holes", []):
            already = any(
                math.hypot(h["x"] - q["x"], h["y"] - q["y"])
                <= 0.5 * min(h["d"], q["d"])
                for q in items
            )
            if already:
                continue
            item = dict(h)
            item.setdefault("shape", "round")
            item.setdefault("eq_d", h["d"])
            item.setdefault("length", h["d"])
            item.setdefault("width", h["d"])
            item["source_n"] = h["n"]
            # off_circle holes already use "A<n>" labels – keep them
            items.append(item)

    return items


def _groups(items):
    """Holes of the same shape whose sizes agree within analysis.patterns.size_group_ratio."""
    k = C["analysis"]["patterns"]
    out = []
    for shape in sorted({h["shape"] for h in items}):
        hs = sorted([h for h in items if h["shape"] == shape], key=lambda h: h["d"])
        cur = []
        for h in hs:
            if cur and h["d"] > k["size_group_ratio"] * np.median([q["d"] for q in cur]):
                out.append(cur)
                cur = []
            cur.append(h)
        if cur:
            out.append(cur)
    return [g for g in out if len(g) >= k["min_holes"]]


def _circular(ctx, g):
    k = C["analysis"]["patterns"]
    pts = np.array([[h["x"], h["y"]] for h in g])
    cx, cy, R, std, _, _ = fit_circle(pts, iters=1)
    size = ctx.part["r"]
    if not (R < k["max_circle_radius_frac"] * size and std <= max(k["circle_fit_tol_px"], k["circle_fit_tol_frac"] * R)):
        return None
    angs = sorted((angle_deg(h["x"], h["y"], cx, cy), h["n"]) for h in g)
    pitches = [((angs[(i + 1) % len(angs)][0] - angs[i][0]) % 360) for i in range(len(angs))]
    if len(g) == 4 and max(pitches) - min(pitches) > k["even_pitch_tol_deg"]:
        return None  # any rectangle's corners are concyclic: only an even 4 counts as a circle
    if len(g) == 3:  # three points always fit a circle: accept only an even, centred triple
        ox, oy = ctx.origin[:2]
        if max(abs(p - 120) for p in pitches) > k["even_pitch_tol_deg"] \
                or math.hypot(cx - ox, cy - oy) > k["triple_centre_tol_frac"] * size:
            return None
    return dict(kind="circular", holes=[n for _, n in angs], count=len(g), cx=cx, cy=cy, d=2 * R, fit_std=std,
                pitches_deg=pitches, mean_pitch_deg=float(np.mean(pitches)),
                uniform=bool(max(pitches) - min(pitches) < k["uniform_pitch_deg"]),
                centre_offset_from_origin=float(math.hypot(cx - ctx.origin[0], cy - ctx.origin[1])))


def _rect4(ctx, g):
    """Four holes at the corners of a rectangle (sides and diagonals agree within tolerance)."""
    pts = np.array([[h["x"], h["y"]] for h in g])
    c = pts.mean(0)
    order = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    q = pts[order]
    s = [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]
    d1, d2 = float(np.linalg.norm(q[2] - q[0])), float(np.linalg.norm(q[3] - q[1]))
    k = C["analysis"]["patterns"]
    tol = max(k["rect_tol_px"], k["rect_tol_frac"] * max(s))
    if abs(d1 - d2) > tol or abs(s[0] - s[2]) > tol or abs(s[1] - s[3]) > tol:
        return None
    a, b = (s[0] + s[2]) / 2, (s[1] + s[3]) / 2
    e = q[1] - q[0] if a >= b else q[2] - q[1]
    fr = ctx.frame
    rel = (math.degrees(math.atan2(e[1], e[0])) - (fr["angle"] if fr else 0) + 90) % 180 - 90
    return dict(kind="rectangle", holes=[g[i]["n"] for i in order], count=4, long_side=max(a, b),
                short_side=min(a, b), diagonal_mismatch=abs(d1 - d2), direction_vs_long_axis_deg=rel,
                corners=q.tolist())


def _linear(ctx, g):
    """Greedy: repeatedly take the line through two holes with the most holes on it (>= 3)."""
    k = C["analysis"]["patterns"]
    left = list(g)
    tol = max(k["row_tol_px"], k["row_tol_diameters"] * np.median([h["d"] for h in g]))
    out = []
    while len(left) >= k["min_holes"]:
        best = None
        for a, b in itertools.combinations(left, 2):
            d = np.array([b["x"] - a["x"], b["y"] - a["y"]])
            n = np.linalg.norm(d)
            if n < 1e-6:
                continue
            d /= n
            on = [h for h in left if abs((h["x"] - a["x"]) * d[1] - (h["y"] - a["y"]) * d[0]) <= tol]
            if best is None or len(on) > len(best[0]):
                best = (on, d)
        if not best or len(best[0]) < k["min_holes"]:
            break
        on, d = best
        on.sort(key=lambda h: h["x"] * d[0] + h["y"] * d[1])
        steps = [math.hypot(on[i + 1]["x"] - on[i]["x"], on[i + 1]["y"] - on[i]["y"]) for i in range(len(on) - 1)]
        fr = ctx.frame
        rel = math.degrees(math.atan2(d[1], d[0])) - (fr["angle"] if fr else 0)
        rel = (rel + 90) % 180 - 90
        out.append(dict(kind="row", holes=[h["n"] for h in on], count=len(on), direction_vs_long_axis_deg=rel,
                        spacings_px=steps, mean_spacing_px=float(np.mean(steps)),
                        uniform=bool(max(steps) - min(steps) <= max(k["uniform_spacing_px"], k["uniform_spacing_frac"] * np.mean(steps))),
                        ends=[[on[0]["x"], on[0]["y"]], [on[-1]["x"], on[-1]["y"]]]))
        left = [h for h in left if h not in on]
    return out


def _has_holes(ctx):
    """patterns runs when at least one hole source produced results."""
    return bool(ctx.results.get("holes", {}).get("items")) \
        or bool(ctx.data.get("holes")) \
        or bool(ctx.data.get("off_circle_holes"))


@step("patterns", needs=("outline",), when=_has_holes)
def patterns(ctx):
    """Circular patterns first, then straight rows, within each same-shape/same-size group.

    Analyses holes from both the generic hole finder (holes.py) and the bolt-circle finder
    (bolt_circle.py) so that every detected hole contributes to pattern analysis regardless of
    which specialist step found it."""
    items = _collect_items(ctx)
    found = []
    for g in _groups(items):
        c = _circular(ctx, g) or (_rect4(ctx, g) if len(g) == 4 else None)
        if c:
            found.append(c)
        else:
            found += _linear(ctx, g)
    rows = []
    for i, p in enumerate(found, 1):
        p["name"] = {"circular": "PC", "rectangle": "RC", "row": "R"}[p["kind"]] + str(i)
        if p["kind"] == "rectangle":
            rows.append([p["name"], "rectangle", " ".join(p["holes"]), 4, "", "", "", f"{p['long_side']:.2f}",
                         f"{p['short_side']:.2f}", f"diagonals differ {p['diagonal_mismatch']:.2f} px", ""])
            ctx.draw("polyline", z=0, points=p["corners"], color=BLUE, width=1, closed=True)
            cx_, cy_ = np.mean(p["corners"], axis=0)
            ctx.draw("text", z=6, x=float(cx_), y=float(cy_), text=p["name"], color=BLUE, size=13)
            ctx.say(f"pattern {p['name']}: rectangle of 4 ({' '.join(p['holes'])}), {p['long_side']:.1f} x "
                    f"{p['short_side']:.1f} px centre-to-centre, {p['direction_vs_long_axis_deg']:.1f} deg to long axis")
        elif p["kind"] == "circular":
            rows.append([p["name"], "circular", " ".join(p["holes"]), p["count"], f"{p['d']:.2f}", "",
                         f"{p['mean_pitch_deg']:.2f}", "", "", f"{max(p['pitches_deg']) - min(p['pitches_deg']):.2f} deg",
                         "yes" if p["uniform"] else "no"])
            ctx.draw("circle", z=0, x=p["cx"], y=p["cy"], r=p["d"] / 2, color=BLUE, width=1)
            ctx.draw("text", z=6, x=p["cx"], y=p["cy"] - p["d"] / 2, text=p["name"], color=BLUE, size=13, dy=-12)
            ctx.say(f"pattern {p['name']}: circle of {p['count']} ({' '.join(p['holes'])}), d={p['d']:.1f} px, "
                    f"pitches {[round(x, 1) for x in p['pitches_deg']]}")
        else:
            rows.append([p["name"], "row", " ".join(p["holes"]), p["count"], "", f"{p['mean_spacing_px']:.2f}", "", "",
                         "", f"{max(p['spacings_px']) - min(p['spacings_px']):.2f} px", "yes" if p["uniform"] else "no"])
            ctx.draw("line", z=0, points=p["ends"], color=BLUE, width=1)
            (x0, y0), (x1, y1) = p["ends"]
            n_ = max(math.hypot(x1 - x0, y1 - y0), 1e-6)
            ext = 0.08 * ctx.part["r"]
            ctx.draw("text", z=6, x=x1 + (x1 - x0) / n_ * ext, y=y1 + (y1 - y0) / n_ * ext, text=p["name"],
                     color=BLUE, size=13)
            ctx.say(f"pattern {p['name']}: row of {p['count']} ({' '.join(p['holes'])}), spacing "
                    f"{[round(x, 1) for x in p['spacings_px']]} px, {p['direction_vs_long_axis_deg']:.1f} deg to long axis")
    ctx.data["patterns"] = found
    if rows:
        ctx.table("hole patterns (holes of the same shape and size)",
                  ["pattern", "kind", "holes", "count", "circle_diameter_px", "mean_spacing_px", "mean_pitch_deg",
                   "rect_long_side_px", "rect_short_side_px", "spread", "uniform"], rows,
                  length_cols=["circle_diameter_px", "mean_spacing_px", "rect_long_side_px", "rect_short_side_px"])
        ctx.add_legend("patterns", "Blue = hole patterns (PC# = circle of holes, RC# = rectangle, R# = row)", BLUE,
                       order=25)
        circ = [p for p in found if p["kind"] == "circular"]
        if circ:
            ctx.dims["bolt"] = max(circ, key=lambda p: p["count"])["d"]
    return dict(patterns=found)
