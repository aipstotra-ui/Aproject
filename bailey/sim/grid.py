"""Model II geometry: rasterise the Bailey Hall schematic into a graph of 0.5 m cells.

The same constants as Model I (bailey/geometry/schematic.py) define the regions:

  layer 0 (first floor)  orchestra | mezzanine | corridor | foyer | NW/NE vestibules | stage-escape passages
  layer 1 (balcony)      balcony seating ring
  stairs                 rectangular strips of cells, joined to the floors by portals

Cells in different regions are joined ONLY through apertures (doors, access stairs, portals). Each
aperture is k cells wide, k = round(effective width / cell), with the effective width from the SFPE
boundary layers. CA door capacity is then (cells x calibrated flow per cell), consistent with Model I's
capacity = Fsm x effective width. Every aperture carries the id of the matching Model I edge, so
obstructions are specified identically in both models.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from bailey.geometry import schematic as S
from bailey.params import effective_width, load_params

EXITS = ["OUT_FRONT", "OUT_NW", "OUT_NE", "OUT_P1", "OUT_P2", "OUT_P3", "OUT_P4"]
SEAT_REGIONS = {"orch", "mezz", "balc"}

# 8-neighbourhood (dx, dy)
NB = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def polar_theta(x, y):
    """Polar angle in the schematic convention: 0 = west, 90 = south, 180 = east (north side wraps)."""
    th = np.degrees(np.arctan2(-y, -x))
    return np.where(th < -90, th + 360, th)


@dataclass
class Aperture:
    id: str
    pairs: list[tuple[int, int]]          # undirected cell pairs that this opening adds
    center: tuple[int, float, float]      # (layer, x, y) for discovery / plotting
    cells: int                            # width in cells
    lengths: list[float]                  # step length of each pair [m]


@dataclass
class CellGrid:
    h: float
    x: np.ndarray
    y: np.ndarray
    layer: np.ndarray                     # 0 first floor, 1 balcony, 2 stair
    region: np.ndarray                    # region name per cell (object array)
    speed: np.ndarray                     # speed factor per cell (1 = free level walking)
    base_edges: np.ndarray                # (M, 2) undirected neighbour pairs inside regions
    base_len: np.ndarray                  # (M,) step length [m]
    apertures: dict[str, Aperture]
    exit_cells: dict[str, np.ndarray]
    zone_cells: dict[str, np.ndarray]
    plot_xy: np.ndarray = field(default=None)  # coordinates used for plotting (stairs drawn along their line)

    @property
    def n(self) -> int:
        return len(self.x)

    def edges(self, blocked: set[str] = frozenset()) -> tuple[np.ndarray, np.ndarray]:
        """All undirected walkable pairs (and their lengths), without the apertures in `blocked`."""
        ap = [a for a in self.apertures.values() if a.id not in blocked]
        extra = [p for a in ap for p in a.pairs]
        extra_len = [l for a in ap for l in a.lengths]
        return (np.vstack([self.base_edges, np.array(extra, dtype=int).reshape(-1, 2)]),
                np.concatenate([self.base_len, np.array(extra_len, dtype=float)]))

    def is_exit(self) -> np.ndarray:
        out = np.zeros(self.n, bool)
        for c in self.exit_cells.values():
            out[c] = True
        return out

    def exit_index(self) -> np.ndarray:
        idx = -np.ones(self.n, int)
        for i, e in enumerate(EXITS):
            idx[self.exit_cells[e]] = i
        return idx

    def nearest_cell(self, layer: int, x: float, y: float, regions: set[str] | None = None) -> int:
        m = self.layer == layer
        if regions:
            m &= np.isin(self.region, list(regions))
        idx = np.where(m)[0]
        return int(idx[np.argmin((self.x[idx] - x) ** 2 + (self.y[idx] - y) ** 2)])


# ---------------------------------------------------------------------------------------------- builder
def _region_floor(x, y):
    r = np.hypot(x, y)
    reg = np.full(x.shape, "", dtype=object)
    south = y <= 0
    reg[south & (r < S.R_BARRIER)] = "orch"
    reg[south & (r >= S.R_BARRIER) & (r < 19.5)] = "mezz"
    r_out = 19.5 + S.CORRIDOR_WIDTH
    reg[south & (r >= 19.5) & (r < r_out)] = "corr"
    reg[(np.abs(x) <= 9.0) & (y >= -30.0) & (y < 0) & (r >= r_out)] = "foyer"
    for side, s in (("W", -1), ("E", 1)):
        reg[(s * x >= 18.5) & (s * x <= 24.0) & (y > 0) & (y <= 6.0)] = f"vest_{side}"
        reg[(s * x > 11.0) & (s * x < 18.5) & (y > 0) & (y <= 2.0)] = f"pass_{side}"
        reg[(s * x > 24.0) & (s * x <= 24.5) & (y > 0) & (y <= 6.0)] = f"OUT_N{side}"
    reg[(np.abs(x) <= 9.5) & (y >= -30.5) & (y < -30.0)] = "OUT_FRONT"
    th = polar_theta(x, y)
    ring = south & (r >= r_out) & (r < r_out + 0.5) & (reg == "")
    for k, a in enumerate(S.PORTICO_DOOR_ANGLES, start=1):
        cx, cy = S.polar(r_out, a)
        reg[ring & (np.hypot(x - cx, y - cy) < 2.0)] = f"OUT_P{k}"
    return reg


def _region_balcony(x, y):
    r = np.hypot(x, y)
    th = polar_theta(x, y)
    reg = np.full(x.shape, "", dtype=object)
    reg[(r >= 14.5) & (r < 20.5) & (th >= -10) & (th <= 190)] = "balc"
    return reg


def _k(width_eff: float, h: float) -> int:
    return max(1, int(round(width_eff / h)))


def build_grid(params: dict | None = None) -> CellGrid:
    P = params or load_params()
    h = P["ca"]["cell"]
    ew = lambda kind, w, n=1: effective_width(P, kind, w, n)  # noqa: E731

    xs, ys, layers, regs = [], [], [], []
    index: dict[tuple[int, int, int], int] = {}  # (layer, i, j) -> cell id

    def add_layer(layer, x0, x1, y0, y1, region_fn):
        gx = np.arange(x0 + h / 2, x1, h)
        gy = np.arange(y0 + h / 2, y1, h)
        X, Y = np.meshgrid(gx, gy, indexing="ij")
        R = region_fn(X, Y)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if R[i, j]:
                    index[(layer, i, j)] = len(xs)
                    xs.append(X[i, j]); ys.append(Y[i, j]); layers.append(layer); regs.append(R[i, j])
        return X.shape

    add_layer(0, -25.0, 25.0, -31.0, 7.0, _region_floor)
    add_layer(1, -21.0, 21.0, -21.0, 4.0, _region_balcony)
    n_floor = len(xs)

    # in-region neighbour pairs; cross-region pairs are remembered as candidate aperture pairs
    base, base_len, cross = [], [], {}
    for (L, i, j), c in index.items():
        for di, dj in NB:
            o = index.get((L, i + di, j + dj))
            if o is None or o < c:
                continue
            if regs[c] == regs[o]:
                base.append((c, o)); base_len.append(h * math.hypot(di, dj))
            else:
                key = frozenset((regs[c], regs[o]))
                cross.setdefault(key, []).append((c, o, di == 0 or dj == 0))
    x = np.array(xs); y = np.array(ys)

    apertures: dict[str, Aperture] = {}

    def door(aid, ra, rb, cx, cy, width_eff, layer=0):
        """k-cell opening between regions ra and rb, centred near (cx, cy)."""
        k = _k(width_eff, h)
        cand = cross.get(frozenset((ra, rb)), [])
        if not cand:
            raise ValueError(f"{aid}: regions {ra}/{rb} do not touch")
        sideA = {c if regs[c] == ra else o for c, o, _ in cand}
        sideB = {o if regs[o] == rb else c for c, o, _ in cand}
        pick = lambda side: set(sorted(side, key=lambda c: (x[c] - cx) ** 2 + (y[c] - cy) ** 2)[:k])  # noqa: E731
        A, B = pick(sideA), pick(sideB)
        pairs = [(c, o) for c, o, _ in cand if (c in A and o in B) or (c in B and o in A)]
        lengths = [math.hypot(x[c] - x[o], y[c] - y[o]) for c, o in pairs]
        apertures[aid] = Aperture(aid, pairs, (layer, cx, cy), k, lengths)

    r_out = 19.5 + S.CORRIDOR_WIDTH
    for k_, th in enumerate(S.ACCESS_ANGLES, start=1):
        m = min(range(9), key=lambda i: abs(S.MEZZ_DOOR_ANGLES[i] - th)) + 1
        door(f"A{k_}->M{m}", "orch", "mezz", *S.polar(S.R_BARRIER, th), ew("aisle_stair", S.W_ACCESS_STAIR))
    for k_, th in enumerate(S.MEZZ_DOOR_ANGLES, start=1):
        door(f"M{k_}->C{th:03d}", "mezz", "corr", *S.polar(19.5, th), ew("door", S.W_MEZZ_DOOR))
    for th in S.FOYER_DOOR_ANGLES:
        door(f"C{th:03d}->FOYER", "corr", "foyer", *S.polar(r_out, th), ew("door", S.W_FOYER_INNER))
    for i in range(S.N_FOYER_OUTER):
        cx = (i - (S.N_FOYER_OUTER - 1) / 2) * 3.4
        door(f"FOYER->OUT_FRONT#{i}", "foyer", "OUT_FRONT", cx, -30.0, ew("door", S.W_FOYER_OUTER))
    for k_, th in enumerate(S.PORTICO_DOOR_ANGLES, start=1):
        door(f"C{th:03d}->OUT_P{k_}", "corr", f"OUT_P{k_}", *S.polar(r_out, th), ew("door", S.W_PORTICO_DOOR))
    for side, s, th in (("W", -1, 0), ("E", 1, 180)):
        door(f"C{th:03d}->V_{side}", "corr", f"vest_{side}", s * 20.7, 0.0, ew("door", S.W_CORRIDOR_END))
        door(f"V_{side}->OUT_N{side}", f"vest_{side}", f"OUT_N{side}", s * 24.0, 4.0, ew("door", S.W_VESTIBULE_EXIT))
        door(f"SE_{side}->V_{side}", f"pass_{side}", f"vest_{side}", s * 18.5, 1.0, ew("door", S.W_STAGE_ESCAPE))
        door(f"O->SE_{side}", "orch", f"pass_{side}", s * 12.0, 0.0, ew("aisle", S.W_AISLE))

    # ---------------- stairs: strips of cells + portals at both ends
    stair_specs = []
    for side, s in (("W", -1), ("E", 1)):
        th_top = {"W": 62, "E": 118}[side]
        stair_specs.append((f"BS_{side}->C{S.BALC_STAIR_LANDING[side]:03d}", th_top,
                            ("corr", S.polar(20.7, S.BALC_STAIR_LANDING[side])),
                            ew("stair_2lane", S.W_BALC_FOYER_STAIR)))
        stair_specs.append((f"BN_{side}->V_{side}", {"W": -8, "E": 188}[side],
                            (f"vest_{side}", (s * 21.0, 5.0)), ew("stair", S.W_BALC_STAGE_STAIR)))
    plot_xy = [(xx, yy) for xx, yy in zip(xs, ys)]
    for aid, th_top, (reg_bot, (bx, by)), weff in stair_specs:
        k = _k(weff, h)
        n_len = int(round(S.L_BALC_STAIR / h))
        tx, ty = S.polar(20.0, th_top)
        ids = np.zeros((n_len, k), int)
        for a in range(n_len):
            for b in range(k):
                ids[a, b] = len(xs)
                f = (a + 0.5) / n_len
                xs.append(np.nan); ys.append(np.nan); layers.append(2); regs.append(f"stair:{aid}")
                plot_xy.append((tx + f * (bx - tx) + (b - k / 2) * h * 0.6, ty + f * (by - ty)))
        for a in range(n_len):
            for b in range(k):
                for da, db in NB:
                    a2, b2 = a + da, b + db
                    if 0 <= a2 < n_len and 0 <= b2 < k and ids[a2, b2] > ids[a, b]:
                        base.append((ids[a, b], ids[a2, b2])); base_len.append(h * math.hypot(da, db))
        x = np.array(xs, dtype=float); y = np.array(ys, dtype=float)
        lay = np.array(layers)
        reg_arr = np.array(regs, dtype=object)

        def ends(layer, region, cx, cy):
            idx = np.where((lay == layer) & (reg_arr == region))[0]
            return idx[np.argsort((x[idx] - cx) ** 2 + (y[idx] - cy) ** 2)[:k]]

        top = ends(1, "balc", *S.polar(20.25, th_top))
        bot = ends(0, reg_bot, bx, by)
        # order end cells along their line so lanes pair up 1:1 (+ diagonals)
        top = top[np.lexsort((y[top], x[top]))]
        bot = bot[np.lexsort((y[bot], x[bot]))]
        pairs, lengths = [], []
        for b in range(k):
            for d in (-1, 0, 1):
                if 0 <= b + d < k:
                    pairs.append((int(top[b]), int(ids[0, b + d])))
                    pairs.append((int(ids[n_len - 1, b]), int(bot[b + d])))
                    lengths += [h * math.hypot(1, d)] * 2
        apertures[aid] = Aperture(aid, pairs, (1, tx, ty), k, lengths)

    x = np.array(xs, dtype=float); y = np.array(ys, dtype=float)
    lay = np.array(layers); reg_arr = np.array(regs, dtype=object)
    speed = np.ones(len(xs))
    speed[np.isin(reg_arr, list(SEAT_REGIONS))] = P["ca"]["seat_speed"]
    speed[lay == 2] = P["ca"]["stair_speed"]

    exit_cells = {e: np.where(reg_arr == e)[0] for e in EXITS}

    # ---------------- seat zones (same split as Model I)
    th = polar_theta(np.nan_to_num(x), np.nan_to_num(y))
    r = np.hypot(np.nan_to_num(x), np.nan_to_num(y))
    zone_cells: dict[str, np.ndarray] = {}
    orch = (reg_arr == "orch")
    zone_cells["O_W"] = np.where(orch & (th < 60))[0]
    centre = np.where(orch & (th >= 60) & (th <= 120))[0]
    centre = centre[np.argsort(r[centre])]          # split the centre block front/back by seat share
    f_front = S.ORCH_ZONES["O_CF"][2] / (S.ORCH_ZONES["O_CF"][2] + S.ORCH_ZONES["O_CB"][2])
    n_front = int(round(f_front * len(centre)))
    zone_cells["O_CF"], zone_cells["O_CB"] = centre[:n_front], centre[n_front:]
    zone_cells["O_E"] = np.where(orch & (th > 120))[0]
    mezz = (reg_arr == "mezz")
    for k_, a in enumerate(S.MEZZ_DOOR_ANGLES, start=1):
        zone_cells[f"Z_M{k_}"] = np.where(mezz & (np.abs(th - a) <= 10) & (r < 19.0))[0]
    balc = (reg_arr == "balc") & (r < 19.5)  # keep the rear cross-aisle free
    for zid, lo, hi in (("B_W", -11, 35), ("B_CW", 35, 90), ("B_CE", 90, 145), ("B_E", 145, 191)):
        zone_cells[zid] = np.where(balc & (th >= lo) & (th < hi))[0]

    return CellGrid(h=h, x=x, y=y, layer=lay, region=reg_arr, speed=speed,
                    base_edges=np.array(base, dtype=int), base_len=np.array(base_len), apertures=apertures, exit_cells=exit_cells,
                    zone_cells=zone_cells, plot_xy=np.array(plot_xy))
