"""Parametric schematic of Bailey Hall -> data/building.json.

This file is the single source of truth for the building abstraction shared by Model I
(network flow) and Model II (cellular automaton). Edit the constants below, then run

    python -m bailey.geometry.schematic

to regenerate data/building.json.

Coordinate system (metres): origin at the centre of the auditorium's semicircle on the stage line.
+y points north (towards the stage), -y south (towards the foyer / main entrance).
Polar angle theta (degrees) runs 0 = west, 90 = south, 180 = east:  x = -r cos(theta), y = -r sin(theta).
z = floor elevation (0 first floor / mezzanine, 5 balcony).

All dimensions are estimates scaled from the CMCM drawings (first-floor gross area 18,602 sq ft,
seating-plan dimension strings such as 14'-0" row lengths). Each estimate is tagged ASSUMPTION in
the "note" field so it can be listed in the paper's assumptions table and stress-tested.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "building.json"

# ---------------------------------------------------------------- radii (m)
R_BARRIER = 13.0      # orchestra / mezzanine barrier
R_MEZZ_ZONE = 16.5    # centroid radius of mezzanine seat sectors
R_MEZZ_AISLE = 19.0   # mezzanine rear cross-aisle (nodes M_k, just inside the 9 doors)
R_CORRIDOR = 20.7     # corridor centre line
CORRIDOR_WIDTH = 2.4  # ASSUMPTION: clear width of the semicircular corridor
R_BALC_ZONE = 17.0
R_BALC_DOOR = 20.0
Z_BALCONY = 5.0

# ---------------------------------------------------------------- widths (m, clear width per leaf/channel)
W_AISLE = 1.2            # ASSUMPTION: seating aisles (orchestra, mezzanine, balcony)
W_ACCESS_STAIR = 1.2     # ASSUMPTION: the six orchestra -> mezzanine access steps
W_MEZZ_DOOR = 1.6        # ASSUMPTION: the nine mezzanine exit doors (pair of leaves)
W_FOYER_INNER = 1.8      # ASSUMPTION: the three corridor -> foyer doors
W_FOYER_OUTER = 1.8      # ASSUMPTION: each of the five large foyer exterior doors
N_FOYER_OUTER = 5
W_PORTICO_DOOR = 1.6     # ASSUMPTION: the four side doors corridor -> portico
W_STAGE_ESCAPE = 1.6     # ASSUMPTION: stage-side escape doors from the auditorium
W_CORRIDOR_END = 1.6     # ASSUMPTION: corridor end doors into the NW/NE stair vestibules
W_VESTIBULE_EXIT = 1.8   # ASSUMPTION: NW/NE exterior doors
W_BALC_DOOR = 1.5        # ASSUMPTION: balcony doors 20001-20004
W_BALC_FOYER_STAIR = 2.0  # ASSUMPTION: each foyer-side balcony stair (two lanes, centre barrier)
W_BALC_STAGE_STAIR = 1.3  # ASSUMPTION: stage-side balcony stairs (emergency only)
L_BALC_STAIR = 14.0      # ASSUMPTION: along-the-slope length of two flights + landing (~5 m rise)

# ---------------------------------------------------------------- feature angles (deg)
MEZZ_DOOR_ANGLES = [10 + 20 * i for i in range(9)]       # 9 doors evenly around the semicircle
ACCESS_ANGLES = [12, 45, 78, 102, 135, 168]               # 6 orchestra -> mezzanine access points
FOYER_DOOR_ANGLES = [80, 90, 100]                         # 3 doors corridor -> foyer (south)
PORTICO_DOOR_ANGLES = [25, 50, 130, 155]                  # 4 rarely used side doors
BALC_STAIR_LANDING = {"W": 65, "E": 115}                  # ASSUMPTION: foyer balcony stairs land in corridor
CORRIDOR_ENDS = {"W": 0, "E": 180}

# seat counts (CMCM statement): orchestra 527, mezzanine 486, balcony 316 -> 1329
ORCH_ZONES = {  # id: (theta, r, seats)
    "O_W": (30, 8.0, 110),
    "O_CF": (90, 5.0, 150),
    "O_CB": (90, 10.5, 157),
    "O_E": (150, 8.0, 110),
}
MEZZ_SEATS = [54] * 9  # 486 split evenly over the 9 door sectors (ASSUMPTION)
BALC_ZONES = {
    "B_W": (10, 66),
    "B_CW": (60, 92),
    "B_CE": (120, 92),
    "B_E": (170, 66),
}

AISLE_DETOUR = 1.3  # walking distance / straight-line distance inside seating (ASSUMPTION)


def polar(r: float, theta_deg: float) -> tuple[float, float]:
    t = math.radians(theta_deg)
    return (-r * math.cos(t), -r * math.sin(t))


class Builder:
    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: list[dict] = []

    def node(self, nid: str, xy, kind: str, z: float = 0.0, area: float | None = None,
             seats: int = 0, label: str = "") -> str:
        assert nid not in self.nodes, nid
        self.nodes[nid] = {"id": nid, "x": round(xy[0], 3), "y": round(xy[1], 3), "z": z,
                           "kind": kind, "area": area, "seats": seats, "label": label}
        return nid

    def dist(self, a: str, b: str) -> float:
        na, nb = self.nodes[a], self.nodes[b]
        return math.dist((na["x"], na["y"], na["z"]), (nb["x"], nb["y"], nb["z"]))

    def edge(self, a: str, b: str, kind: str, width: float, *, n: int = 1, length: float | None = None,
             use: str = "normal", blockable: bool = False, group: str = "", note: str = "",
             both: bool = False, detour: float = 1.0) -> None:
        if length is None:
            length = max(self.dist(a, b) * detour, 1.0)
        for u, v in ((a, b), (b, a)) if both else ((a, b),):
            self.edges.append({"id": f"{u}->{v}", "from": u, "to": v, "kind": kind,
                               "width": width, "n": n, "length": round(length, 2), "use": use,
                               "blockable": blockable, "group": group, "note": note})


def build() -> dict:
    B = Builder()

    # ---------------- corridor ring: one node per feature angle
    feature_angles = sorted(set(MEZZ_DOOR_ANGLES + FOYER_DOOR_ANGLES + PORTICO_DOOR_ANGLES
                                + list(BALC_STAIR_LANDING.values()) + list(CORRIDOR_ENDS.values())))
    cnode = {}
    for i, th in enumerate(feature_angles):
        prev_th = feature_angles[i - 1] if i > 0 else th
        next_th = feature_angles[i + 1] if i + 1 < len(feature_angles) else th
        arc = R_CORRIDOR * math.radians((next_th - prev_th) / 2)
        cnode[th] = B.node(f"C{th:03d}", polar(R_CORRIDOR, th), "corridor",
                           area=round(CORRIDOR_WIDTH * max(arc, 1.0), 2), label=f"corridor {th} deg")
    for a, b in zip(feature_angles[:-1], feature_angles[1:]):
        B.edge(cnode[a], cnode[b], "corridor", CORRIDOR_WIDTH,
               length=R_CORRIDOR * math.radians(b - a), both=True, group="corridor",
               note="ASSUMPTION corridor width; counter-flow ignored")

    # ---------------- exits and rooms outside the auditorium
    B.node("FOYER", (0.0, -26.0), "room", area=144.0, label="foyer 10040 (ASSUMPTION 18 m x 8 m)")
    B.node("OUT_FRONT", (0.0, -31.0), "exit", label="five foyer doors -> monumental stair")
    for side, sx in (("W", -1), ("E", 1)):
        B.node(f"V_{side}", (sx * 21.0, 3.0), "room", area=25.0, label=f"N{side} stair vestibule")
        B.node(f"OUT_N{side}", (sx * 24.0, 5.0), "exit", label=f"N{side} exterior door")
        B.node(f"SE_{side}", (sx * 13.5, 1.0), "junction", area=6.0, label=f"stage escape {side}")

    for th in FOYER_DOOR_ANGLES:
        B.edge(cnode[th], "FOYER", "door", W_FOYER_INNER, length=2.0, blockable=True,
               group="foyer_inner", note="ASSUMPTION width")
    B.edge("FOYER", "OUT_FRONT", "door", W_FOYER_OUTER, n=N_FOYER_OUTER, length=4.0, blockable=True,
           group="foyer_outer", note="five leaves; blocking removes one")
    for i, th in enumerate(PORTICO_DOOR_ANGLES, start=1):
        B.node(f"OUT_P{i}", polar(R_CORRIDOR + 3.5, th), "exit", label=f"portico door {i}")
        B.edge(cnode[th], f"OUT_P{i}", "door", W_PORTICO_DOOR, length=2.0, use="emergency",
               blockable=True, group="portico", note="ASSUMPTION width; rarely used in normal operation")
    for side, th in CORRIDOR_ENDS.items():
        B.edge(cnode[th], f"V_{side}", "door", W_CORRIDOR_END, use="emergency", blockable=True,
               group="corridor_end", note="ASSUMPTION corridor connects to stage-side stair hall")
        B.edge(f"V_{side}", f"OUT_N{side}", "door", W_VESTIBULE_EXIT, length=3.0, use="emergency",
               blockable=True, group="vestibule_exit")
        B.edge(f"SE_{side}", f"V_{side}", "door", W_STAGE_ESCAPE, use="emergency", blockable=True,
               group="stage_escape", note="stage escape route (first-floor plan)")

    # ---------------- mezzanine: 9 sectors, 9 doors, rear cross aisle
    for k, th in enumerate(MEZZ_DOOR_ANGLES, start=1):
        B.node(f"M{k}", polar(R_MEZZ_AISLE, th), "junction", area=10.0, label=f"mezzanine aisle at door {k}")
        B.node(f"Z_M{k}", polar(R_MEZZ_ZONE, th), "zone", seats=MEZZ_SEATS[k - 1], label=f"mezzanine sector {k}")
        B.edge(f"Z_M{k}", f"M{k}", "aisle", W_AISLE, n=2, detour=AISLE_DETOUR, group="seating")
        B.edge(f"M{k}", cnode[th], "door", W_MEZZ_DOOR, length=1.5, blockable=True, group="mezz_door",
               note="ASSUMPTION width")
    for k in range(1, 9):
        B.edge(f"M{k}", f"M{k + 1}", "aisle", W_AISLE,
               length=R_MEZZ_AISLE * math.radians(20), both=True, group="mezz_cross_aisle")

    # ---------------- orchestra: zones -> 6 access stairs -> mezzanine aisle (and stage escapes)
    for j, th in enumerate(ACCESS_ANGLES, start=1):
        B.node(f"A{j}", polar(R_BARRIER, th), "junction", area=4.0, label=f"access point {j}")
        k = min(range(9), key=lambda i: abs(MEZZ_DOOR_ANGLES[i] - th)) + 1
        B.edge(f"A{j}", f"M{k}", "aisle_stair", W_ACCESS_STAIR, group="access",
               blockable=True, note="ASSUMPTION width; steps up through the barrier")
    for zid, (th, r, seats) in ORCH_ZONES.items():
        B.node(zid, polar(r, th), "zone", seats=seats, label=f"orchestra {zid}")
    orch_links = {
        "O_W": ["A1", "A2"], "O_CF": ["A2", "A3", "A4", "A5"],
        "O_CB": ["A3", "A4"], "O_E": ["A5", "A6"],
    }
    for zid, acc in orch_links.items():
        for a in acc:
            B.edge(zid, a, "aisle", W_AISLE, detour=AISLE_DETOUR, group="seating")
    for zid, side in (("O_W", "W"), ("O_CF", "W"), ("O_CF", "E"), ("O_E", "E")):
        B.edge(zid, f"SE_{side}", "aisle", W_AISLE, detour=AISLE_DETOUR, use="emergency", group="seating")

    # ---------------- balcony
    for zid, (th, seats) in BALC_ZONES.items():
        B.node(zid, polar(R_BALC_ZONE, th), "zone", z=Z_BALCONY, seats=seats, label=f"balcony {zid}")
    bdoors = {"BN_W": -8, "BS_W": 62, "BS_E": 118, "BN_E": 188}
    for bid, th in bdoors.items():
        B.node(bid, polar(R_BALC_DOOR, th), "junction", z=Z_BALCONY, area=8.0, label=f"balcony door {bid}")
    balc_links = [("B_W", "BN_W", 1, "emergency"), ("B_W", "BS_W", 1, "normal"),
                  ("B_CW", "BS_W", 2, "normal"), ("B_CW", "BS_E", 1, "normal"),
                  ("B_CE", "BS_E", 2, "normal"), ("B_CE", "BS_W", 1, "normal"),
                  ("B_E", "BN_E", 1, "emergency"), ("B_E", "BS_E", 1, "normal")]
    for z, d, n, use in balc_links:
        B.edge(z, d, "aisle", W_AISLE, n=n, detour=AISLE_DETOUR, use=use, group="seating")
    for side in ("W", "E"):
        B.edge(f"BS_{side}", cnode[BALC_STAIR_LANDING[side]], "stair_2lane", W_BALC_FOYER_STAIR,
               length=L_BALC_STAIR, blockable=True, group="balc_foyer_stair",
               note="ASSUMPTION lands in corridor next to the foyer doors")
        B.edge(f"BN_{side}", f"V_{side}", "stair", W_BALC_STAGE_STAIR, length=L_BALC_STAIR,
               use="emergency", blockable=True, group="balc_stage_stair")

    total = sum(n["seats"] for n in B.nodes.values())
    assert total == 1329, total
    return {
        "meta": {
            "description": "Schematic network of Bailey Hall (CMCM 2026). Generated by bailey/geometry/schematic.py",
            "units": "metres; edge width = clear width of ONE channel/leaf, n = number of parallel leaves",
            "seats_total": total,
        },
        "nodes": list(B.nodes.values()),
        "edges": B.edges,
        "exits": [nid for nid, n in B.nodes.items() if n["kind"] == "exit"],
    }


def main() -> None:
    data = build()
    OUT.write_text(json.dumps(data, indent=1))
    print(f"wrote {OUT}: {len(data['nodes'])} nodes, {len(data['edges'])} edges, "
          f"{data['meta']['seats_total']} seats, exits={data['exits']}")


if __name__ == "__main__":
    main()
