"""Holes, openings and other closed features of any shape inside a part of any shape.

Applies to every part. Candidates come from pluggable sources (config analysis.holes.sources, in order;
a later candidate at an already found feature is a duplicate):
  background - regions of background colour enclosed by the part: through-openings
  surface    - spots darker (or brighter, --hole-polarity) than the surface around them: holes, blind holes
  edges      - regions closed by a sharp edge all round (the same edge map that finds the outline): windows whose
               backdrop is in shadow, pins/bosses, pockets; anything a brightness rule cannot separate
Every candidate is then measured the same way: a round one by a wall/edge fit on rays from its centre (as the bore
and bolt holes of a round part), a rectangular one by a line fit on each of its four edges. Add a source by
decorating a function with @source("name") and listing it in the config.

On round parts where bolt_circle also ran, holes.py detects features that the bolt-circle specialist does not
cover: slots, openings, blind holes off the bolt circle, pins/bosses and other non-round features.  The two steps
complement each other; patterns.py merges both sets for pattern analysis.

Settings in config analysis.holes (sizes relative to the part radius), analysis.rect_fit."""
import math

import cv2
import numpy as np

from .. import segment as seg
from ..core import angle_deg, odd
from ..holefit import hole_wall_fit, rect_edge_fit
from ..config import C
from ..registry import step
from .outline import is_round

GREEN, YEL = (40, 220, 40), (255, 240, 40)

SOURCES = {}


def source(name):
    """Register fn(ctx, k) -> [(contour, label)] as a candidate source for the holes step."""
    def deco(fn):
        SOURCES[name] = fn
        return fn
    return deco


def _round_fit(ctx, x, y, rm):
    """Wall fit (level crossing between core and surround), else the strongest radial edge (a pin or boss has no
    dark core). Returns the fit and its method, or (None, None) when neither is a clean circle near rm."""
    kf = C["analysis"]["holes"]["round_fit"]
    # scatter is judged on the fitted radius: the candidate region may be only the core of the feature
    ok = lambda f: f and f["std"] <= max(kf["max_scatter_px"], kf["max_scatter_radii"] * max(rm, f["r"])) \
        and f["rays"] >= kf["min_ray_frac"] * f["of"] and abs(f["r"] - rm) < kf["max_radius_error"] * rm
    inside = ctx.disc_stat(x, y, 0, max(2, kf["core_radii"] * rm))
    around = ctx.disc_stat(x, y, kf["surface_ring_radii"][0] * rm, kf["surface_ring_radii"][1] * rm)
    pol = -1.0 if inside < around else 1.0
    lo, hi = kf["wall_search_radii"]
    f = hole_wall_fit(ctx, x, y, (inside + around) / 2, pol, lo * rm, hi * rm,
                      floor=max(kf["fit_floor_px"], kf["fit_floor_radii"] * rm))
    if ok(f):
        return f, "wall fit"
    if kf["edge_fallback"]:
        f = ctx.ray_edge_fit(x, y, lo * rm, hi * rm, n=C["analysis"]["hole_fit"]["rays"],
                             floor=max(kf["fit_floor_px"], kf["fit_floor_radii"] * rm))
        if ok(f):
            return f, "edge fit"
    return None, None


def _describe(ctx, c, source):
    """Shape descriptors of one feature, then a precise fit: round -> circle on rays, rectangular -> 4 edge lines."""
    k = C["analysis"]["holes"]
    area = cv2.contourArea(c)
    if area <= 0:
        return None
    M = cv2.moments(c)
    x, y = M["m10"] / M["m00"], M["m01"] / M["m00"]
    perim = cv2.arcLength(c, True)
    circ = 4 * math.pi * area / max(perim ** 2, 1)
    solidity = area / max(cv2.contourArea(cv2.convexHull(c)), 1)
    rect = cv2.minAreaRect(c)
    (rcx, rcy), (rw, rh), rang = rect
    L, Wd = max(rw, rh), min(rw, rh)
    aspect = L / max(Wd, 1e-6)
    fill = area / max(rw * rh, 1)
    approx = cv2.approxPolyDP(c, k["corner_approx_frac"] * perim, True)
    if circ >= k["round_min_circularity"] and aspect <= k["round_max_aspect"]:
        shape = "round"
    elif aspect >= k["slot_min_aspect"] and solidity >= k["slot_min_solidity"]:
        shape = "slot"
    elif len(approx) == 4 and fill >= k["rect_min_fill"]:
        shape = "rectangular"
    else:
        shape = "irregular"
    box = cv2.boxPoints(((rcx, rcy), (rw, rh), rang))
    e = box[1] - box[0] if np.linalg.norm(box[1] - box[0]) >= np.linalg.norm(box[2] - box[1]) else box[2] - box[1]
    h = dict(x=float(x), y=float(y), area=float(area), eq_d=float(2 * math.sqrt(area / math.pi)), length=float(L),
             width=float(Wd), aspect=float(aspect), circularity=float(circ), solidity=float(solidity),
             axis_deg=float(math.degrees(math.atan2(e[1], e[0])) % 180), shape=shape, source=source, fit=None)
    h["d"] = h["eq_d"]
    # a ragged but compact blob (a hole seen with its chamfer or a reflection) is tried as a circle too
    if shape == "round" or (shape == "irregular" and aspect <= k["round_fit"]["compact_max_aspect"]
                            and fill < k["rect_candidate"]["min_fill"]):
        f, how = _round_fit(ctx, x, y, h["eq_d"] / 2)
        if f:
            h.update(x=f["x"], y=f["y"], d=f["d"], shape="round",
                     fit=dict(method=how, std=f["std"], rays=f["rays"], of=f["of"]))
    elif (shape == "rectangular" or (shape == "irregular" and fill >= k["rect_candidate"]["min_fill"]
                                     and solidity >= k["rect_candidate"]["min_solidity"])) \
            and Wd >= k["rect_candidate"]["min_side_frac"] * ctx.part["r"]:
        f = rect_edge_fit(ctx, rect)
        if f and f["ok"]:
            h.update(x=f["x"], y=f["y"], length=f["length"], width=f["width"], axis_deg=f["axis_deg"],
                     aspect=f["length"] / max(f["width"], 1e-6), shape="rectangular",
                     d=float(2 * math.sqrt(f["length"] * f["width"] / math.pi)),
                     fit=dict(method="4-edge line fit", std=f["std"], rays=f["rays"], of=f["of"]))
            c = np.round(np.array(f["corners"])).astype(np.int32).reshape(-1, 1, 2)
    if h["fit"]:  # inside vs the material around it: tells a hole (darker) from a pin/boss or bright recess
        rm = max(h["width"], h["d"]) / 2 if h["shape"] != "round" else h["d"] / 2
        kf = k["round_fit"]
        inside = ctx.disc_stat(h["x"], h["y"], 0, max(2, kf["core_radii"] * rm))
        around = ctx.disc_stat(h["x"], h["y"], kf["surface_ring_radii"][0] * rm, kf["surface_ring_radii"][1] * rm)
        h["inside_vs_around"] = "darker" if inside < around else "brighter"
    return h, c


def _mask(ctx, h, c, ring_radii=0.0):
    """Pixels of a feature: its fitted circle (plus a counterbore/chamfer ring of `ring_radii` hole radii) or
    rectangle, else its region (for overlap tests)."""
    m = np.zeros((ctx.H, ctx.W), np.uint8)
    if h["fit"] and h["shape"] == "round":
        cv2.circle(m, (int(round(h["x"])), int(round(h["y"]))), max(1, int(round(h["d"] / 2 * (1 + ring_radii)))), 1, -1)
    else:
        cv2.drawContours(m, [c], -1, 1, -1)
    return m


def _resolve(ctx, cands, k):
    """One feature per place. Measured (fitted) features first, larger first; a candidate is dropped when it is
    concentric with an accepted one, or unmeasured and overlapping a measured one (a shadowed corner inside a
    window, a fleck beside a hole) or too small to matter; two measured features that only overlap are both kept
    (a hole in a pocket)."""
    ko = k["overlap"]
    order = sorted(range(len(cands)), key=lambda i: (cands[i][0]["fit"] is None, -cands[i][0]["area"]))
    kept, masks = [], []
    for i in order:
        h, c = cands[i]
        m = _mask(ctx, h, c, ko["measured_ring_radii"])
        a = max(int(_mask(ctx, h, c).sum()), 1)
        drop = h["fit"] is None and h["eq_d"] / 2 < ko["unmeasured_min_radius_frac"] * ctx.part["r"]
        for (q, _), qm in zip(kept, masks):
            if math.hypot(h["x"] - q["x"], h["y"] - q["y"]) <= k["same_hole_diameters"] * min(h["d"], q["d"]):
                drop = True  # the same feature seen by two sources
            elif h["fit"] is None and int((_mask(ctx, h, c) & qm).sum()) > ko["unmeasured_max_overlap"] * a:
                drop = True
            if drop:
                break
        if not drop:
            kept.append((h, c))
            masks.append(m)
    return kept


@source("background")
def _background(ctx, k):
    """Openings showing background: background-coloured regions enclosed by the part (from the part step)."""
    p = ctx.part
    bg = float(np.median(ctx.gray[p["mask_outer"] == 0]))
    surf = float(np.median(ctx.gray[p["mask_material"] > 0]))
    out, skipped = [], 0
    for c in p["inner"]:
        # A background-coloured region inside the outline is an opening only if it has a solid interior
        # that matches the background; thin shiny highlights also pass Otsu and are dropped here.
        x, y, w, h = cv2.boundingRect(c)
        m = np.zeros((h + 2, w + 2), np.uint8)
        cv2.drawContours(m, [c - [x - 1, y - 1]], -1, 1, -1)
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        rin = float(dt.max())
        roi = ctx.gray[y - 1:y + h + 1, x - 1:x + w + 1]
        core = roi[dt[:roi.shape[0], :roi.shape[1]] >= k["core_depth"] * max(rin, 1)]
        solidity = cv2.contourArea(c) / max(cv2.contourArea(cv2.convexHull(c)), 1)
        if rin < k["opening_min_inner_frac"] * p["r"] or not core.size \
                or abs(np.median(core) - bg) > k["opening_max_background_diff"] * abs(surf - bg) \
                or (rin < k["highlight_max_inner_frac"] * p["r"]
                    and solidity < k["highlight_max_solidity"]):  # thin and curved: a highlight, not a slot
            skipped += 1
            continue
        out.append((c, "opening (background visible)"))
    if skipped:
        ctx.note(f"{skipped} light region(s) inside the outline were not counted as openings (thin or not "
                 "background-coloured: highlights or reflections).")
    return out


@source("surface")
def _surface(ctx, k):
    """Holes darker (or brighter) than the surface around them, on the normalised image: local contrast against
    a wide median of the part's own material."""
    p = ctx.part
    sN, NR = ctx.sN, ctx.NORM_R
    mat = cv2.resize(p["mask_material"], (ctx.gN.shape[1], ctx.gN.shape[0]), interpolation=cv2.INTER_NEAREST)
    er = cv2.erode(mat, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(k["material_erode_frac"] * NR),) * 2))
    g = ctx.gN.astype(np.float32)
    # Background estimate from part material only: outside the part, substitute the local mean of nearby
    # material (normalised convolution), so corners and edges are not compared with the backdrop.
    kw = odd(k["background_window_frac"] * NR)
    m = (mat > 0).astype(np.float32)
    local = cv2.blur(g * m, (2 * kw, 2 * kw)) / np.maximum(cv2.blur(m, (2 * kw, 2 * kw)), 1e-3)
    filled = np.where(mat > 0, g, local).clip(0, 255).astype(np.uint8)
    bgm = cv2.medianBlur(filled, kw).astype(np.float32)
    pol_opt = ctx.opt_get("hole_polarity", "dark") or "dark"
    diff = (bgm - g) if pol_opt == "dark" else (g - bgm)
    vals = g[er > 0]
    out = []
    if not vals.size:
        return out
    pq = k["extreme_percentile"]  # the darkest (or brightest) level on the part: inside a hole
    spread = (np.percentile(vals, 50) - np.percentile(vals, pq)) if pol_opt == "dark" else \
        (np.percentile(vals, 100 - pq) - np.percentile(vals, 50))
    thr = max(float(C["image"]["min_part_contrast"]), k["threshold_spread"] * spread)
    black = np.percentile(vals, pq) if pol_opt == "dark" else np.percentile(vals, 100 - pq)
    cand = ((diff > thr) & (er > 0)).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand)
    amin, amax = math.pi * (k["min_radius_frac"] * NR) ** 2, k["max_area_frac"] * (er > 0).sum()
    label = "blind/surface hole (%s than surface)" % ("darker" if pol_opt == "dark" else "brighter")
    for i in range(1, n):
        if not amin <= st[i, 4] <= amax:
            continue
        x_, y_, w_, h_ = st[i, :4]
        comp = (lab[y_:y_ + h_, x_:x_ + w_] == i).astype(np.uint8)
        dc = diff[y_:y_ + h_, x_:x_ + w_][comp > 0]
        if dc.max() < max(k["min_peak_contrast"], k["min_peak_spread"] * spread) \
                or np.median(dc) < k["min_median_over_threshold"] * thr:
            continue  # not hole-dark across its area: a stain, streak or shadow
        # A real hole has a solid interior close to the darkest level on the part. Streaks, scratches and
        # shadow lines are thin: their inscribed radius is small and they have no dark core.
        dt = cv2.distanceTransform(np.pad(comp, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        rin = float(dt.max())
        if rin < k["min_inner_frac"] * NR:
            continue  # thinner than this fraction of the part radius across: a scratch or edge line
        core = g[y_:y_ + h_, x_:x_ + w_][dt >= k["core_depth"] * rin]
        sign = 1.0 if pol_opt == "dark" else -1.0
        if (np.median(core) - black) * sign > k["core_max_from_extreme"] * spread:
            continue  # interior not hole-dark (a stain or shadow); darker than the darkest surface is fine
        cc, _ = cv2.findContours(np.pad(comp, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cc, key=cv2.contourArea).astype(np.float32)
        c[:, 0, 0] += x_ - 1
        c[:, 0, 1] += y_ - 1
        c = (c / sN).astype(np.int32)
        if cv2.contourArea(c) and cv2.contourArea(c) / max(cv2.contourArea(cv2.convexHull(c)), 1) < k["min_solidity"]:
            continue  # ragged outline: rust or texture
        out.append((c, label))
    ctx.data["surface_hole_threshold"] = dict(threshold=thr, spread=spread, polarity=pol_opt)
    return out


@source("edges")
def _edges(ctx, k):
    """Regions inside the part closed all round by a sharp edge. Texture and scratches also close small regions,
    but those are ragged (low solidity) and their borders weak; a feature's border is a strong, clean edge."""
    ke = k["edges"]
    p = ctx.part
    E, noise = seg.sharp_edges(ctx.gray)
    g1 = seg.fine_gradient(ctx.gray)
    gap = max(1, int(round(ke["gap_frac"] * p["r"])))
    disk = seg._disk
    solid = p["mask_outer"]
    inner = cv2.erode(solid, disk(ke["outline_clearance_gaps"] * gap))
    free = ((cv2.dilate(E, disk(gap)) == 0) & (inner > 0)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    band = (solid > 0) & (cv2.erode(solid, disk(ke["outline_band_gaps"] * gap)) == 0)
    band = cv2.dilate(band.astype(np.uint8), disk(gap)) > 0
    out, rejected = [], 0
    for i in range(1, n):
        x_, y_, w_, h_ = st[i, :4]
        amin, amax = ke["min_area_frac"] * p["area"], ke["max_area_frac"] * p["area"]
        if (w_ + 2 * gap) * (h_ + 2 * gap) < amin or st[i, 4] > amax:
            continue  # even grown back by the gap it cannot reach the minimum
        pad = 2 * gap + 1
        xa, ya = max(x_ - pad, 0), max(y_ - pad, 0)
        xb, yb = min(x_ + w_ + pad, ctx.W), min(y_ + h_ + pad, ctx.H)
        m = cv2.dilate((lab[ya:yb, xa:xb] == i).astype(np.uint8), disk(gap))
        if (m.astype(bool) & band[ya:yb, xa:xb]).any():
            continue  # reaches the outline: a face of the part, not a feature inside it
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea) + np.array([xa, ya])
        area = cv2.contourArea(c)
        if area < amin:
            continue
        if area / max(cv2.contourArea(cv2.convexHull(c)), 1) < ke["min_solidity"]:
            rejected += 1
            continue
        bnd = (cv2.dilate(m, disk(gap)) > 0) & (cv2.erode(m, disk(gap)) == 0)
        if float(np.median(g1[ya:yb, xa:xb][bnd])) < ke["min_boundary_strength"] * noise:
            rejected += 1
            continue
        out.append((c.astype(np.int32), "closed sharp edge"))
    ctx.data["edge_regions"] = dict(gap_px=gap, kept=len(out), rejected_texture=rejected)
    return out


@step("holes", needs=("part", "outline"))
def holes(ctx):
    """Holes, openings, windows, pins: candidates from every configured source, each measured by a precise fit."""
    k = C["analysis"]["holes"]
    if is_round(ctx) and not ctx.results.get("bolt_circle", {}).get("ok"):
        ctx.warn("Round part without a bolt circle: also ran the generic hole finder.")
    cands, unfitted = [], 0
    for name in k["sources"]:
        if name not in SOURCES:
            raise SystemExit(f"unknown hole source '{name}'. Available: {', '.join(sorted(SOURCES))}")
        need_fit = isinstance(k.get(name), dict) and k[name].get("require_fit")
        for c, label in SOURCES[name](ctx, k):
            r = _describe(ctx, c, label)
            if r and need_fit and r[0]["fit"] is None:
                unfitted += 1  # this source's regions count only when they measure as a clean circle/rectangle
                continue
            if r:
                cands.append(r)
    found = _resolve(ctx, cands, k)
    if unfitted:
        ctx.data.setdefault("edge_regions", {})["not_fitted"] = unfitted

    # number in reading order along the part's own axes (u along the long side)
    med = np.median([h["eq_d"] for h, _ in found]) if found else 1
    for h, _ in found:
        h["u"], h["v"] = ctx.to_frame(h["x"], h["y"])
        if h["u"] is None:
            h["u"], h["v"] = h["x"], h["y"]
    found.sort(key=lambda hc: (round(hc[0]["v"] / max(med, 1)), hc[0]["u"]))
    ox, oy = ctx.origin[:2]
    items = []
    for i, (h, c) in enumerate(found, 1):
        h["n"] = f"H{i}"
        h["R"] = float(math.hypot(h["x"] - ox, h["y"] - oy))
        h["ang"] = angle_deg(h["x"], h["y"], ox, oy)
        h["contains_origin"] = cv2.pointPolygonTest(c, (float(ox), float(oy)), False) >= 0
        items.append(h)
        if h["shape"] == "round":
            ctx.draw("circle", z=4, x=h["x"], y=h["y"], r=h["d"] / 2 + 0.15 * h["d"], color=GREEN, width=2.5)
        else:
            ctx.draw("polyline", z=4, points=c[:, 0, :].tolist(), color=GREEN, width=2.5, closed=True)
        ctx.draw("text", z=6, x=h["x"], y=h["y"] - 0.5 * h["width"] - 0.03 * ctx.part["r"], text=h["n"], color=YEL,
                 size=13)
    ctx.data["holes_generic"] = items
    shapes = {}
    for h in items:
        shapes[h["shape"]] = shapes.get(h["shape"], 0) + 1
    rows = [[h["n"], h["shape"], h["source"], f'{h["x"]:.2f}', f'{h["y"]:.2f}', f'{h["u"]:.2f}', f'{h["v"]:.2f}',
             f'{h["d"]:.2f}', f'{h["length"]:.2f}', f'{h["width"]:.2f}', f'{h["axis_deg"]:.1f}',
             f'{h["circularity"]:.3f}', h["fit"]["method"] if h["fit"] else "area",
             h.get("inside_vs_around", "")] for h in items]
    ctx.table("holes and openings; u/v = position along the part's long/short axis from frame origin 0; "
              "diameter = fitted for round features, else area-equivalent; length/width fitted for rectangles",
              ["hole", "shape", "source", "x_px", "y_px", "u_px", "v_px", "diameter_px", "length_px", "width_px",
               "axis_deg", "circularity", "measured_by", "inside_vs_around"], rows,
              length_cols=["u_px", "v_px", "diameter_px", "length_px", "width_px"])
    round_d = [h["d"] for h in items if h["shape"] == "round"]
    if round_d:
        ctx.dims["hole"] = float(np.median(round_d))
    central = [h for h in items if h["contains_origin"] and h["shape"] == "round"]
    if central:
        ctx.dims["bore"] = central[0]["d"]
    ctx.add_legend("holes", f"Green + H# = hole/opening ({len(items)}: "
                   + ", ".join(f"{v} {k_}" for k_, v in shapes.items()) + "), numbered along the part's long axis",
                   GREEN, order=10)
    if any(h["shape"] == "irregular" for h in items):
        ctx.warn(f"{sum(h['shape'] == 'irregular' for h in items)} irregular opening(s): check they are real "
                 "features, not shadows or stains.")
    if not items:
        ctx.note("No holes or openings found.")
    ctx.say(f"holes/openings: {len(items)} ({shapes})")
    for h in items:
        ctx.say("%4s %-11s d=%7.2f L=%7.2f W=%7.2f at (%7.1f,%7.1f) u,v=(%7.1f,%7.1f) circ=%.2f %s" % (
            h["n"], h["shape"], h["d"], h["length"], h["width"], h["x"], h["y"], h["u"], h["v"], h["circularity"],
            f"[{h['fit']['method']}, {h.get('inside_vs_around', '')}]" if h["fit"] else ""))
    return dict(items=items)
