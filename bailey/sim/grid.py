"""Model II, step 1: draw the building as a grid of 0.5 m x 0.5 m squares ("cells").

One person fits in one cell. The building is made of:
  * floor 0 (first floor): orchestra, mezzanine, corridor, foyer, two stair rooms (vestibules) and two
    short stage-escape passages
  * floor 1: the balcony
  * floor 2: the four balcony staircases, each drawn as a long strip of cells

Every cell gets a label ("region"), e.g. "orch", "corr", "foyer", or an exit name like "OUT_P2".
People may step to any of the 8 neighbouring cells, but only if the neighbour has the SAME label.
That is how walls work. The only way from one region to another is through a door ("aperture"):
a few chosen cell pairs that are allowed to connect anyway. A door is as many cells wide as
(usable door width / 0.5 m). Each door is named like the matching Model I edge (e.g. "C090->FOYER"),
so blocking a door means the same thing in both models.
"""
import math

import numpy as np

from bailey.geometry import schematic as S
from bailey.params import effective_width, load_params

EXITS = ["OUT_FRONT", "OUT_NW", "OUT_NE", "OUT_P1", "OUT_P2", "OUT_P3", "OUT_P4"]
SEAT_REGIONS = {"orch", "mezz", "balc"}

# the 8 neighbours of a cell, as (step in x, step in y)
NB = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def polar_theta(x, y):
    """Angle around the hall centre, in degrees: 0 = west, 90 = south (foyer side), 180 = east.
    Works on single numbers or numpy arrays."""
    angle = np.degrees(np.arctan2(-y, -x))
    return np.where(angle < -90, angle + 360, angle)


class Aperture:
    """A door (or stair end) between two regions."""

    def __init__(self, id, pairs, center, cells, lengths):
        self.id = id                # name, same as the Model I edge
        self.pairs = pairs          # list of (cell, cell) that this door connects
        self.center = center        # (floor, x, y) of the door, used to "see" a blocked door
        self.cells = cells          # door width in cells
        self.lengths = lengths      # step length of each pair in metres (0.5 straight, 0.71 diagonal)


class CellGrid:
    """The finished grid."""

    def __init__(self, h, x, y, layer, region, speed, base_edges, base_len, apertures, exit_cells,
                 zone_cells, plot_xy=None):
        self.h = h                      # cell size (0.5 m)
        self.x = x                      # x position of each cell (nan for stair cells)
        self.y = y                      # y position of each cell
        self.layer = layer              # 0 first floor, 1 balcony, 2 stairs
        self.region = region            # label of each cell
        self.speed = speed              # walking speed factor (1 = normal, 0.8 seats, 0.71 stairs)
        self.base_edges = base_edges    # neighbour pairs inside a region (no doors)
        self.base_len = base_len        # their step lengths
        self.apertures = apertures      # door name -> Aperture
        self.exit_cells = exit_cells    # exit name -> cells of that exit
        self.zone_cells = zone_cells    # seating zone name -> its cells
        self.plot_xy = plot_xy          # positions used for drawing (stairs drawn along their line)

    @property
    def n(self):
        """Number of cells."""
        return len(self.x)

    def edges(self, blocked=frozenset()):
        """All pairs of cells you can step between, and their step lengths.
        Doors whose name is in `blocked` are left out (you cannot walk through them)."""
        pairs = list(map(tuple, self.base_edges))
        lengths = list(self.base_len)
        for door in self.apertures.values():
            if door.id in blocked:
                continue
            pairs += door.pairs
            lengths += door.lengths
        return np.array(pairs, dtype=int).reshape(-1, 2), np.array(lengths, dtype=float)

    def is_exit(self):
        """True for every exit cell."""
        flags = np.zeros(self.n, bool)
        for cells in self.exit_cells.values():
            flags[cells] = True
        return flags

    def exit_index(self):
        """For every cell: the number of its exit in EXITS, or -1 if it is not an exit."""
        numbers = -np.ones(self.n, int)
        for number, exit_name in enumerate(EXITS):
            numbers[self.exit_cells[exit_name]] = number
        return numbers

    def nearest_cell(self, layer, x, y, regions=None):
        """The cell on floor `layer` closest to the point (x, y)."""
        best_cell, best_distance = None, None
        for cell in range(self.n):
            if self.layer[cell] != layer:
                continue
            if regions and self.region[cell] not in regions:
                continue
            distance = (self.x[cell] - x) ** 2 + (self.y[cell] - y) ** 2
            if best_distance is None or distance < best_distance:
                best_cell, best_distance = cell, distance
        return int(best_cell)


# ============================================================================= labelling the cells
def region_first_floor(x, y):
    """Which part of the first floor is the point (x, y) in? "" = wall / outside.
    Later rules overwrite earlier ones."""
    r = math.hypot(x, y)                      # distance from the hall centre
    corridor_outer = 19.5 + S.CORRIDOR_WIDTH
    region = ""
    if y <= 0:                                # the auditorium is the southern half-circle
        if r < S.R_BARRIER:
            region = "orch"
        elif r < 19.5:
            region = "mezz"
        elif r < corridor_outer:
            region = "corr"
    if abs(x) <= 9.0 and -30.0 <= y < 0 and r >= corridor_outer:
        region = "foyer"
    for side, sign in (("W", -1), ("E", 1)):
        if 18.5 <= sign * x <= 24.0 and 0 < y <= 6.0:
            region = "vest_" + side           # stair room next to the stage
        if 11.0 < sign * x < 18.5 and 0 < y <= 2.0:
            region = "pass_" + side           # stage-escape passage
        if 24.0 < sign * x <= 24.5 and 0 < y <= 6.0:
            region = "OUT_N" + side           # exterior door of the stair room
    if abs(x) <= 9.5 and -30.5 <= y < -30.0:
        region = "OUT_FRONT"                  # outside the five foyer doors
    # the four portico doors: a thin strip of exit cells just outside the corridor
    if y <= 0 and corridor_outer <= r < corridor_outer + 0.5 and region == "":
        for number, angle in enumerate(S.PORTICO_DOOR_ANGLES, start=1):
            door_x, door_y = S.polar(corridor_outer, angle)
            if math.hypot(x - door_x, y - door_y) < 2.0:
                region = "OUT_P" + str(number)
    return region


def region_balcony(x, y):
    """The balcony is a ring between 14.5 m and 20.5 m from the centre."""
    r = math.hypot(x, y)
    angle = float(polar_theta(x, y))
    if 14.5 <= r < 20.5 and -10 <= angle <= 190:
        return "balc"
    return ""


def cells_across(width, h):
    """How many cells wide a door of this (usable) width is: at least 1."""
    return max(1, int(round(width / h)))


# ============================================================================= building the grid
def build_grid(params=None):
    if params is None:
        params = load_params()
    h = params["ca"]["cell"]

    def usable_width(kind, width, n=1):
        return effective_width(params, kind, width, n)

    # lists with one entry per cell
    xs, ys, layers, regions = [], [], [], []
    cell_at = {}                              # (floor, column, row) -> cell number

    # ---- Step 1: lay a 0.5 m grid over each floor and keep the cells that are inside the building ----
    def add_floor(floor, x_min, x_max, y_min, y_max, region_of):
        columns = np.arange(x_min + h / 2, x_max, h)
        rows = np.arange(y_min + h / 2, y_max, h)
        for i in range(len(columns)):
            for j in range(len(rows)):
                region = region_of(columns[i], rows[j])
                if region:
                    cell_at[(floor, i, j)] = len(xs)
                    xs.append(columns[i])
                    ys.append(rows[j])
                    layers.append(floor)
                    regions.append(region)

    add_floor(0, -25.0, 25.0, -31.0, 7.0, region_first_floor)
    add_floor(1, -21.0, 21.0, -21.0, 4.0, region_balcony)

    # ---- Step 2: connect neighbouring cells ----
    # same region -> always connected; different regions -> remembered as a possible door spot
    pairs, pair_lengths = [], []
    door_candidates = {}                      # {region A, region B} -> list of (cell, cell, straight?)
    for (floor, i, j), cell in cell_at.items():
        for di, dj in NB:
            other = cell_at.get((floor, i + di, j + dj))
            if other is None or other < cell:   # each pair only once
                continue
            if regions[cell] == regions[other]:
                pairs.append((cell, other))
                pair_lengths.append(h * math.hypot(di, dj))
            else:
                key = frozenset((regions[cell], regions[other]))
                door_candidates.setdefault(key, []).append((cell, other, di == 0 or dj == 0))
    x = np.array(xs)
    y = np.array(ys)

    # ---- Step 3: doors ----
    apertures = {}

    def make_door(name, region_a, region_b, centre_x, centre_y, width, floor=0):
        """Open a door `width` metres wide between two regions, near the point (centre_x, centre_y).
        We take the k cells on each side that are closest to that point and connect them."""
        k = cells_across(width, h)
        candidates = door_candidates.get(frozenset((region_a, region_b)), [])
        if not candidates:
            raise ValueError(f"{name}: regions {region_a}/{region_b} do not touch")
        side_a = set()
        side_b = set()
        for c, o, _ in candidates:
            side_a.add(c if regions[c] == region_a else o)
            side_b.add(o if regions[o] == region_b else c)

        def closest_k(side):
            ordered = sorted(side, key=lambda cell: (x[cell] - centre_x) ** 2 + (y[cell] - centre_y) ** 2)
            return set(ordered[:k])

        chosen_a = closest_k(side_a)
        chosen_b = closest_k(side_b)
        door_pairs = []
        for c, o, _ in candidates:
            if (c in chosen_a and o in chosen_b) or (c in chosen_b and o in chosen_a):
                door_pairs.append((c, o))
        lengths = [math.hypot(x[c] - x[o], y[c] - y[o]) for c, o in door_pairs]
        apertures[name] = Aperture(name, door_pairs, (floor, centre_x, centre_y), k, lengths)

    corridor_outer = 19.5 + S.CORRIDOR_WIDTH

    # 6 access stairs from the orchestra up to the mezzanine
    for number, angle in enumerate(S.ACCESS_ANGLES, start=1):
        nearest_door = min(range(9), key=lambda i: abs(S.MEZZ_DOOR_ANGLES[i] - angle)) + 1
        make_door(f"A{number}->M{nearest_door}", "orch", "mezz", *S.polar(S.R_BARRIER, angle),
                  usable_width("aisle_stair", S.W_ACCESS_STAIR))
    # 9 mezzanine doors into the corridor
    for number, angle in enumerate(S.MEZZ_DOOR_ANGLES, start=1):
        make_door(f"M{number}->C{angle:03d}", "mezz", "corr", *S.polar(19.5, angle),
                  usable_width("door", S.W_MEZZ_DOOR))
    # 3 doors from the corridor into the foyer
    for angle in S.FOYER_DOOR_ANGLES:
        make_door(f"C{angle:03d}->FOYER", "corr", "foyer", *S.polar(corridor_outer, angle),
                  usable_width("door", S.W_FOYER_INNER))
    # 5 front doors out of the foyer
    for i in range(S.N_FOYER_OUTER):
        door_x = (i - (S.N_FOYER_OUTER - 1) / 2) * 3.4
        make_door(f"FOYER->OUT_FRONT#{i}", "foyer", "OUT_FRONT", door_x, -30.0,
                  usable_width("door", S.W_FOYER_OUTER))
    # 4 portico side doors
    for number, angle in enumerate(S.PORTICO_DOOR_ANGLES, start=1):
        make_door(f"C{angle:03d}->OUT_P{number}", "corr", f"OUT_P{number}", *S.polar(corridor_outer, angle),
                  usable_width("door", S.W_PORTICO_DOOR))
    # doors on the west and east sides near the stage
    for side, sign, angle in (("W", -1, 0), ("E", 1, 180)):
        make_door(f"C{angle:03d}->V_{side}", "corr", f"vest_{side}", sign * 20.7, 0.0,
                  usable_width("door", S.W_CORRIDOR_END))
        make_door(f"V_{side}->OUT_N{side}", f"vest_{side}", f"OUT_N{side}", sign * 24.0, 4.0,
                  usable_width("door", S.W_VESTIBULE_EXIT))
        make_door(f"SE_{side}->V_{side}", f"pass_{side}", f"vest_{side}", sign * 18.5, 1.0,
                  usable_width("door", S.W_STAGE_ESCAPE))
        make_door(f"O->SE_{side}", "orch", f"pass_{side}", sign * 12.0, 0.0,
                  usable_width("aisle", S.W_AISLE))

    # ---- Step 4: the four balcony staircases ----
    # (name, angle of the top door, (region at the bottom, point at the bottom), usable width)
    stairs = []
    for side, sign in (("W", -1), ("E", 1)):
        top_angle = 62 if side == "W" else 118
        landing = S.BALC_STAIR_LANDING[side]
        stairs.append((f"BS_{side}->C{landing:03d}", top_angle, ("corr", S.polar(20.7, landing)),
                       usable_width("stair_2lane", S.W_BALC_FOYER_STAIR)))
        stairs.append((f"BN_{side}->V_{side}", -8 if side == "W" else 188, (f"vest_{side}", (sign * 21.0, 5.0)),
                       usable_width("stair", S.W_BALC_STAGE_STAIR)))

    plot_xy = [(xx, yy) for xx, yy in zip(xs, ys)]
    for name, top_angle, (bottom_region, (bottom_x, bottom_y)), width in stairs:
        lanes = cells_across(width, h)
        length = int(round(S.L_BALC_STAIR / h))          # 14 m -> 28 cells
        top_x, top_y = S.polar(20.0, top_angle)

        # 4a. the strip of stair cells (they have no real x, y; for drawing we place them on a line)
        stair_cell = np.zeros((length, lanes), int)
        for a in range(length):
            for b in range(lanes):
                stair_cell[a, b] = len(xs)
                along = (a + 0.5) / length
                xs.append(np.nan)
                ys.append(np.nan)
                layers.append(2)
                regions.append("stair:" + name)
                plot_xy.append((top_x + along * (bottom_x - top_x) + (b - lanes / 2) * h * 0.6,
                                top_y + along * (bottom_y - top_y)))
        # 4b. connect neighbouring stair cells
        for a in range(length):
            for b in range(lanes):
                for da, db in NB:
                    a2, b2 = a + da, b + db
                    if 0 <= a2 < length and 0 <= b2 < lanes and stair_cell[a2, b2] > stair_cell[a, b]:
                        pairs.append((stair_cell[a, b], stair_cell[a2, b2]))
                        pair_lengths.append(h * math.hypot(da, db))

        # 4c. join the top of the stair to the balcony and the bottom to the floor below
        x = np.array(xs, dtype=float)
        y = np.array(ys, dtype=float)
        layer_array = np.array(layers)
        region_array = np.array(regions, dtype=object)

        def closest_cells(floor, region, point_x, point_y):
            candidates = np.where((layer_array == floor) & (region_array == region))[0]
            distances = (x[candidates] - point_x) ** 2 + (y[candidates] - point_y) ** 2
            return candidates[np.argsort(distances)[:lanes]]

        top_cells = closest_cells(1, "balc", *S.polar(20.25, top_angle))
        bottom_cells = closest_cells(0, bottom_region, bottom_x, bottom_y)
        # sort them along the stair's width so lane 1 meets lane 1, lane 2 meets lane 2, ...
        top_cells = top_cells[np.lexsort((y[top_cells], x[top_cells]))]
        bottom_cells = bottom_cells[np.lexsort((y[bottom_cells], x[bottom_cells]))]
        stair_pairs, stair_lengths = [], []
        for b in range(lanes):
            for shift in (-1, 0, 1):                  # straight on, or one lane to the side
                if 0 <= b + shift < lanes:
                    stair_pairs.append((int(top_cells[b]), int(stair_cell[0, b + shift])))
                    stair_pairs.append((int(stair_cell[length - 1, b]), int(bottom_cells[b + shift])))
                    stair_lengths += [h * math.hypot(1, shift)] * 2
        apertures[name] = Aperture(name, stair_pairs, (1, top_x, top_y), lanes, stair_lengths)

    # ---- Step 5: final arrays, walking speed per cell, exits ----
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    layer_array = np.array(layers)
    region_array = np.array(regions, dtype=object)

    speed = np.ones(len(xs))
    for cell in range(len(xs)):
        if region_array[cell] in SEAT_REGIONS:
            speed[cell] = params["ca"]["seat_speed"]      # squeezing past seats: 80 %
        elif layer_array[cell] == 2:
            speed[cell] = params["ca"]["stair_speed"]     # stairs: 71 %

    exit_cells = {}
    for exit_name in EXITS:
        exit_cells[exit_name] = np.where(region_array == exit_name)[0]

    # ---- Step 6: seating zones (the same split as Model I) ----
    angle = polar_theta(np.nan_to_num(x), np.nan_to_num(y))
    r = np.hypot(np.nan_to_num(x), np.nan_to_num(y))
    zone_cells = {}

    orchestra = region_array == "orch"
    zone_cells["O_W"] = np.where(orchestra & (angle < 60))[0]
    # centre block: the front part gets as many cells as its share of the seats
    centre = np.where(orchestra & (angle >= 60) & (angle <= 120))[0]
    centre = centre[np.argsort(r[centre])]                 # from front to back
    front_share = S.ORCH_ZONES["O_CF"][2] / (S.ORCH_ZONES["O_CF"][2] + S.ORCH_ZONES["O_CB"][2])
    front_count = int(round(front_share * len(centre)))
    zone_cells["O_CF"] = centre[:front_count]
    zone_cells["O_CB"] = centre[front_count:]
    zone_cells["O_E"] = np.where(orchestra & (angle > 120))[0]

    mezzanine = region_array == "mezz"
    for number, door_angle in enumerate(S.MEZZ_DOOR_ANGLES, start=1):
        zone_cells[f"Z_M{number}"] = np.where(mezzanine & (np.abs(angle - door_angle) <= 10) & (r < 19.0))[0]

    balcony_seats = (region_array == "balc") & (r < 19.5)  # the back row is kept free as an aisle
    for zone, low, high in (("B_W", -11, 35), ("B_CW", 35, 90), ("B_CE", 90, 145), ("B_E", 145, 191)):
        zone_cells[zone] = np.where(balcony_seats & (angle >= low) & (angle < high))[0]

    return CellGrid(h=h, x=x, y=y, layer=layer_array, region=region_array, speed=speed,
                    base_edges=np.array(pairs, dtype=int), base_len=np.array(pair_lengths),
                    apertures=apertures, exit_cells=exit_cells, zone_cells=zone_cells, plot_xy=np.array(plot_xy))
