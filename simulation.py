"""
Bailey Hall evacuation - a simple grid simulation.

The hall floor is cut into squares of 0.5 m x 0.5 m (about the space one person needs).
Every 0.42 seconds each person may step to one of the 8 squares around them.
We repeat this until everybody is out.

Main rules (numbered like the code below):
  1. Draw the hall as a grid of squares: wall, floor, seat or exit.
  2. For every exit, count how many steps every square is away from it (a "distance map").
  3. Put the 1,329 people in their seats and give each one a type:
        60 % guided  - follow the ushers, who spread people over all exits
        25 % habit   - go back the way they came in (the front foyer exit)
        15 % nearest - go to the closest exit
  4. Each person picks an exit.
  5. Every time step, each person chooses a square at random, but some squares are much more
     likely than others (see choose_square). If two people choose the same square,
     one of them (picked at random) gets it and the others wait.
  6. Two stuck people who face each other (each wants the other's square) swap places half of the time.
  7. Bottlenecks use our measured data: each stair (orchestra -> mezzanine), mezzanine door and
     foyer door has a "ticket machine" that hands out tickets at the measured flow rate.
     You may only step into the doorway if a ticket is available.
  8. The final time is CALCULATED (calculated_escape_time): from the doorways each person used,
     we count the people per doorway, divide by the doorway's flow rate, and chain the doorways
     from the seats to the exits to find when the last person is completely out.
"""
import csv
import heapq
import math
import os
import random
from collections import deque

import numpy as np

# ---------------------------------------------------------------- settings you can change
CELL = 0.5                 # size of one square in metres
WALK_SPEED = 1.19          # normal walking speed in metres per second
TIME_STEP = CELL / WALK_SPEED   # time to walk one square: about 0.42 s

STAIR_WALK_TIME = 16.0     # extra seconds for balcony people to walk down their stairs
MAX_TIME = 600.0           # stop the simulation after this many seconds

# The situations we simulate: the two emergency cases of our data (case 3 and case 4).
# "reaction" = seconds before people start to move.
# guided = follow the ushers, habit = back to the front foyer, nearest = closest exit.
CASES = [
    # "case" = the label we show, "data_case" = which case of our data files it uses
    {"case": "3", "data_case": 3, "name": "Students, emergency",
     "reaction": 5.0, "guided": 0.0, "habit": 0.40, "nearest": 0.60},
    {"case": "4", "data_case": 4, "name": "Concert, emergency",
     "reaction": 10.0, "guided": 0.60, "habit": 0.25, "nearest": 0.15},
    # Crowd Managers: they halve the people who ignore the ushers (40 % -> 20 %),
    # keeping the same habit : nearest ratio (25 : 15). Not 100 %: some people never listen.
    {"case": "4 + CM", "data_case": 4, "name": "Concert, emergency, Crowd Managers",
     "reaction": 10.0, "guided": 0.80, "habit": 0.125, "nearest": 0.075},
]

# our measured bottleneck data (one file per place, 4 cases x 500 runs each)
DATA_FOLDER = "data"
DATA_FILES = {"stairs": "stairs.csv", "corridor": "corridor.csv", "foyer": "foyer_doors.csv"}
# Doorways we have no data for get: (measured flow per metre at the mezzanine doors) x (their width).
# Widths in metres (estimated from the CMCM drawings).
NO_DATA_WIDTHS = {
    "P1": 1.6, "P2": 1.6, "P3": 1.6, "P4": 1.6,          # portico side doors
    "NW": 1.5, "NE": 1.5,                                # side exits next to the stage
    "FRONT": 5 * 1.8,                                    # the 5 front doors of the foyer together
    "stage W": 1.5, "stage E": 1.5,                      # openings from the orchestra to the stage passages
    "balcony NW stair": 1.3, "balcony W foyer stair": 2.0,
    "balcony E foyer stair": 2.0, "balcony NE stair": 1.3,
}
SWAP_CHANCE = 0.5          # two stuck people facing each other squeeze past each other (swap) with this chance
QUEUE_RADIUS = 4.0         # people within 4 m of a doorway count as its queue
MAP_UPDATE_TIME = 5.0      # every 5 s people look at the queues again and update their route

PULL = 3.0                 # how strongly people want to get closer to their exit
CROWD = 0.3                # how strongly people avoid crowded squares
DOOR_FLOW = 0.28           # people one door square lets through per time step (= 0.66 per second)

# what each square can be
WALL, FLOOR, SEAT, EXIT = 0, 1, 2, 3

# the hall covers x from -25 m to 25 m (west to east) and y from -31 m to 3 m (foyer to stage)
X_MIN, X_MAX = -25.0, 25.0
Y_MIN, Y_MAX = -31.0, 3.0
COLUMNS = int((X_MAX - X_MIN) / CELL)     # 100 squares across
ROWS = int((Y_MAX - Y_MIN) / CELL)        # 68 squares from top to bottom

# sizes of the hall in metres (estimated from the CMCM drawings)
ORCHESTRA_RADIUS = 13.0      # orchestra seats: half-circle of 13 m
BARRIER_END = 13.5           # 0.5 m barrier between orchestra and mezzanine
MEZZANINE_END = 19.5         # mezzanine seats from 13.5 m to 19.5 m
MEZZ_WALL_END = 20.0         # 0.5 m wall with the 9 mezzanine doors
CORRIDOR_END = 22.4          # 2.4 m wide curved corridor
OUTER_WALL_END = 22.9        # outer wall of the corridor

# angles around the hall: 0 = west end, 90 = south (foyer side), 180 = east end
ACCESS_STAIRS = [12, 45, 78, 102, 135, 168]        # 6 openings orchestra -> mezzanine (1.2 m wide)
MEZZANINE_DOORS = [10, 30, 50, 70, 90, 110, 130, 150, 170]   # 9 doors into the corridor (1.6 m)
FOYER_DOORS = [80, 90, 100]                        # 3 doors corridor -> foyer (1.8 m)
PORTICO_DOORS = {"P1": 25, "P2": 50, "P3": 130, "P4": 155}   # 4 side exits (1.6 m)
FRONT_DOORS_X = [-6.8, -3.4, 0.0, 3.4, 6.8]        # 5 front doors of the foyer (1.8 m each)
BARRIER_THICKNESS = 1.0      # a temporary barrier is a 1 m thick wall across the corridor

EXIT_NAMES = ["FRONT", "NW", "NE", "P1", "P2", "P3", "P4"]

# number of people
ORCHESTRA_PEOPLE = 527
MEZZANINE_PEOPLE = 486
# balcony people come down 4 staircases: (name, number of people, angle or point where they arrive)
BALCONY_STAIRS = [("NW stair", 66, (-19.0, 1.0)),
                  ("W foyer stair", 92, 65),
                  ("E foyer stair", 92, 115),
                  ("NE stair", 66, (19.0, 1.0))]

# the 8 directions you can step in: (change in row, change in column)
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# ======================================================================== small helpers
def square_centre(row, col):
    """Middle of a square in metres (x to the east, y to the north)."""
    x = X_MIN + (col + 0.5) * CELL
    y = Y_MAX - (row + 0.5) * CELL
    return x, y


def square_at(x, y):
    """The square (row, col) that contains the point (x, y)."""
    col = int((x - X_MIN) / CELL)
    row = int((Y_MAX - y) / CELL)
    return row, col


def point_on_circle(radius, angle):
    """Point at `radius` metres from the hall centre, at `angle` degrees (0 = west, 90 = south)."""
    a = math.radians(angle)
    return -radius * math.cos(a), -radius * math.sin(a)


def near_opening(x, y, radius, angles, width):
    """True if (x, y) is within half a door width of one of the openings at these angles."""
    for angle in angles:
        door_x, door_y = point_on_circle(radius, angle)
        if math.hypot(x - door_x, y - door_y) <= width / 2:
            return True
    return False


# ======================================================================== 1. the map
def which_opening(x, y, radius, angles, width):
    """Number (1, 2, ...) of the opening that (x, y) belongs to, or None."""
    for number, angle in enumerate(angles, start=1):
        if near_opening(x, y, radius, [angle], width):
            return number
    return None


def make_hall(barriers=()):
    """Return (grid, exit_of_square, gate_of_square).
    barriers = angles (degrees) where a temporary barrier closes the corridor (see barriers.py).
    grid[row, col] is WALL, FLOOR, SEAT or EXIT.
    exit_of_square[(row, col)] is the name of the exit for every EXIT square.
    gate_of_square[(row, col)] is the name of the doorway ("gate") for every square of a
    stair opening, mezzanine door or foyer door; these are the places we have data for.
    stage_opening_of_square[(row, col)] marks the two openings from the orchestra to the stage passages."""
    grid = np.zeros((ROWS, COLUMNS), dtype=int)       # everything starts as wall
    exit_of_square = {}
    gate_of_square = {}
    stage_opening_of_square = {}

    for row in range(ROWS):
        for col in range(COLUMNS):
            x, y = square_centre(row, col)
            r = math.hypot(x, y)                      # distance from the centre of the hall
            kind = WALL
            exit_name = None

            if y < 0:                                 # the auditorium is the southern half-circle
                if r < ORCHESTRA_RADIUS:
                    kind = SEAT                       # orchestra
                elif r < BARRIER_END:
                    number = which_opening(x, y, ORCHESTRA_RADIUS, ACCESS_STAIRS, 1.2)
                    if number:
                        kind = FLOOR                  # opening in the barrier (stairs up)
                        gate_of_square[(row, col)] = f"stair {number}"
                elif r < MEZZANINE_END:
                    kind = SEAT                       # mezzanine
                elif r < MEZZ_WALL_END:
                    number = which_opening(x, y, MEZZANINE_END, MEZZANINE_DOORS, 1.6)
                    if number:
                        kind = FLOOR                  # mezzanine door
                        gate_of_square[(row, col)] = f"mezz door {number}"
                elif r < CORRIDOR_END:
                    kind = FLOOR                      # corridor
                    for angle in barriers:            # a temporary barrier across the corridor
                        bx, by = point_on_circle(r, angle)
                        if math.hypot(x - bx, y - by) <= BARRIER_THICKNESS / 2:
                            kind = WALL
                elif r < OUTER_WALL_END:
                    number = which_opening(x, y, CORRIDOR_END, FOYER_DOORS, 1.8)
                    if number:
                        kind = FLOOR                  # door into the foyer
                        gate_of_square[(row, col)] = f"foyer door {number}"
                    for name, angle in PORTICO_DOORS.items():
                        if near_opening(x, y, CORRIDOR_END, [angle], 1.6):
                            kind, exit_name = EXIT, name      # portico side exit
                if abs(x) <= 9.0 and r >= OUTER_WALL_END and y > -30.0:
                    kind = FLOOR                      # foyer (18 m x 8 m)
                if abs(x) <= 9.0 and -30.5 < y <= -30.0:
                    for door_x in FRONT_DOORS_X:
                        if abs(x - door_x) <= 0.9:
                            kind, exit_name = EXIT, "FRONT"   # front doors of the foyer

            else:                                     # just north of the auditorium (stage side)
                if 0.5 <= y < 2.0 and 11.0 <= abs(x) <= CORRIDOR_END:
                    kind = FLOOR                      # passage from the stage corners to the side exits
                if y < 0.5 and (11.0 <= abs(x) <= 12.5 or MEZZ_WALL_END <= abs(x) <= CORRIDOR_END):
                    kind = FLOOR                      # openings from the orchestra / corridor into it
                if y < 0.5 and 11.0 <= abs(x) <= 12.5:
                    stage_opening_of_square[(row, col)] = "stage W" if x < 0 else "stage E"
                if 0.5 <= y < 2.0 and CORRIDOR_END < abs(x) <= CORRIDOR_END + 0.5:
                    kind = EXIT
                    exit_name = "NW" if x < 0 else "NE"   # side exits next to the stage

            grid[row, col] = kind
            if kind == EXIT:
                exit_of_square[(row, col)] = exit_name
            if kind != FLOOR:
                gate_of_square.pop((row, col), None)  # a later rule turned it into something else
    return grid, exit_of_square, gate_of_square, stage_opening_of_square


def neighbours(grid, row, col):
    """Squares you can step to from (row, col). You can't step into a wall, and you can't
    cut a corner diagonally between two walls."""
    result = []
    for d_row, d_col in DIRECTIONS:
        r2, c2 = row + d_row, col + d_col
        if r2 < 0 or r2 >= ROWS or c2 < 0 or c2 >= COLUMNS:
            continue
        if grid[r2, c2] == WALL:
            continue
        if d_row != 0 and d_col != 0:                 # diagonal step
            if grid[row + d_row, col] == WALL or grid[row, col + d_col] == WALL:
                continue
        result.append((r2, c2))
    return result


def make_neighbour_lists(grid):
    """Work out the neighbours of every square once, so we don't repeat it every time step."""
    lists = {}
    for row in range(ROWS):
        for col in range(COLUMNS):
            if grid[row, col] != WALL:
                lists[(row, col)] = neighbours(grid, row, col)
    return lists


def door_widths(grid, exit_of_square):
    """How many squares wide the narrowest doors on the way to each exit are (used by the ushers).
    For the side exits that is the exit itself; for the front exit it is the 3 doors into the foyer."""
    widths = {name: 0 for name in EXIT_NAMES}
    for name in exit_of_square.values():
        if name != "FRONT":
            widths[name] += 1
    for row in range(ROWS):
        for col in range(COLUMNS):
            x, y = square_centre(row, col)
            in_outer_wall = y < 0 and CORRIDOR_END <= math.hypot(x, y) < OUTER_WALL_END
            if in_outer_wall and grid[row, col] == FLOOR:
                widths["FRONT"] += 1               # a square of one of the 3 foyer doors
    return widths


# ======================================================================== 2. distance maps
def distance_to_exit(grid, neighbour_lists, exit_of_square, exit_name):
    """Steps from every square to the exit `exit_name` (a breadth-first search: start at the exit
    and spread out one step at a time, like a wave). Squares that can't reach it get a huge number."""
    far_away = 10 ** 6
    distance = np.full((ROWS, COLUMNS), far_away)
    queue = deque()
    for square, name in exit_of_square.items():
        if name == exit_name:
            distance[square] = 0
            queue.append(square)
    while queue:
        square = queue.popleft()
        for nxt in neighbour_lists[square]:
            if distance[nxt] == far_away:            # not visited yet
                distance[nxt] = distance[square] + 1
                queue.append(nxt)
    return distance


def distance_with_queues(hall, exit_name, extra_cost):
    """Like distance_to_exit, but stepping into a doorway costs extra steps: the time you would wait
    in its queue. Because steps now cost different amounts we use Dijkstra's method: the same wave as
    before, but we always continue from the square with the smallest distance so far (a heap keeps
    them in order)."""
    neighbour_lists = hall["neighbours"]
    gate_of_square = hall["gate_of_square"]
    distance = np.full((ROWS, COLUMNS), float(10 ** 6))
    heap = []
    for square, name in hall["exit_of_square"].items():
        if name == exit_name:
            distance[square] = 0.0
            heap.append((0.0, square))
    heapq.heapify(heap)
    while heap:
        dist_here, square = heapq.heappop(heap)
        if dist_here > distance[square]:
            continue                                   # we already found a shorter way to this square
        for previous in neighbour_lists[square]:
            # a person on `previous` steps to `square`: 1 step, plus the queue if it means entering a doorway
            cost = 1.0
            gate = gate_of_square.get(square)
            if gate is not None and gate_of_square.get(previous) != gate:
                cost += extra_cost[gate]
            if dist_here + cost < distance[previous]:
                distance[previous] = dist_here + cost
                heapq.heappush(heap, (dist_here + cost, previous))
    return distance


def queue_costs(occupied, hall, rates):
    """For each doorway: how many time steps you would wait = people in its queue / tickets per step."""
    costs = {}
    for gate, nearby_squares in hall["gate_queue_area"].items():
        waiting = sum(occupied[square] for square in nearby_squares)
        costs[gate] = waiting / (gate_rate(gate, rates) * TIME_STEP)
    return costs


# ======================================================================== 3. people
def stair_landing(grid, where):
    """The floor square where a balcony staircase arrives."""
    if isinstance(where, tuple):
        x, y = where
    else:
        x, y = point_on_circle((MEZZ_WALL_END + CORRIDOR_END) / 2, where)   # middle of the corridor
    return square_at(x, y)


def create_people(grid, case):
    """Make a list of people for this case. Each person is a dictionary."""
    reaction = case["reaction"]
    orchestra_seats = []
    mezzanine_seats = []
    for row in range(ROWS):
        for col in range(COLUMNS):
            if grid[row, col] == SEAT:
                x, y = square_centre(row, col)
                if math.hypot(x, y) < ORCHESTRA_RADIUS:
                    orchestra_seats.append((row, col))
                else:
                    mezzanine_seats.append((row, col))

    people = []
    for seat in random.sample(orchestra_seats, ORCHESTRA_PEOPLE):
        people.append({"square": seat, "start": reaction, "stair": None, "start_square": seat})
    for seat in random.sample(mezzanine_seats, MEZZANINE_PEOPLE):
        people.append({"square": seat, "start": reaction, "stair": None, "start_square": seat})
    for name, how_many, where in BALCONY_STAIRS:
        landing = stair_landing(grid, where)
        for _ in range(how_many):
            # balcony people are not on the floor yet: they wait in the stair's queue
            people.append({"square": None, "start": reaction + STAIR_WALK_TIME, "stair": landing,
                           "stair_name": "balcony " + name, "start_square": landing})

    # give everyone a type: guided, habit or nearest
    for person in people:
        dice = random.random()
        if dice < case["guided"]:
            person["type"] = "guided"
        elif dice < case["guided"] + case["habit"]:
            person["type"] = "habit"
        else:
            person["type"] = "nearest"
        person["out_time"] = None
        person["exit_used"] = None
        person["path"] = []                  # the doorways this person goes through, in order
    return people


# ======================================================================== 4. choosing an exit
def choose_exits(people, distance, widths):
    """Give every person an exit to head for."""
    # where each person starts walking from: their seat, or the bottom of their staircase
    def start_square(person):
        return person["square"] if person["stair"] is None else person["stair"]

    sent_by_ushers = {name: 0 for name in EXIT_NAMES}
    guided = [p for p in people if p["type"] == "guided"]
    random.shuffle(guided)

    for person in people:
        here = start_square(person)
        if person["type"] == "habit":
            person["exit"] = "FRONT"
        elif person["type"] == "nearest":
            person["exit"] = min(EXIT_NAMES, key=lambda name: distance[name][here])

    # the ushers spread people out: each guided person goes where they would get out soonest,
    # counting the people the ushers have already sent there
    for person in guided:
        here = start_square(person)
        best_exit, best_time = None, None
        for name in EXIT_NAMES:
            walk = distance[name][here]
            wait = sent_by_ushers[name] / (widths[name] * DOOR_FLOW)
            if best_time is None or walk + wait < best_time:
                best_exit, best_time = name, walk + wait
        person["exit"] = best_exit
        sent_by_ushers[best_exit] += 1


# ======================================================================== 5. moving
def count_crowd(occupied):
    """For every square: how many of its 8 neighbours are occupied."""
    padded = np.pad(occupied, 1)
    crowd = np.zeros((ROWS, COLUMNS), dtype=int)
    for d_row, d_col in DIRECTIONS:
        crowd += padded[1 + d_row: 1 + d_row + ROWS, 1 + d_col: 1 + d_col + COLUMNS]
    return crowd


def choose_square(person, distance, neighbour_lists, occupied, crowd):
    """Pick the square this person wants to step to (or stay where they are).

    Every free option gets a weight:
        weight = exp( - PULL x (change in distance to the exit)  -  CROWD x (crowded neighbours) )
    A step 1 square closer gets about e^3 = 20 times the weight of a step that keeps the same distance;
    a step away gets about 1/20. Crowded squares get less weight. Then we pick one option at random,
    with chances in proportion to the weights (this is the probability rule)."""
    here = person["square"]
    dist = distance[person["exit"]]
    options = [here]                                       # staying put is always an option
    weights = [math.exp(-CROWD * crowd[here])]
    for square in neighbour_lists[here]:
        if occupied[square]:
            continue                                       # somebody is standing there
        change = dist[square] - dist[here]
        options.append(square)
        score = -PULL * change - CROWD * crowd[square]
        weights.append(math.exp(min(score, 50.0)))        # cap it: e^50 is already 'certainly this one'
    return random.choices(options, weights=weights)[0]


# ======================================================================== 6. our measured data
def load_flow_data():
    """Read the three data files into {place: {case number: [rows]}}, rows sorted by run number."""
    data = {}
    for place, file_name in DATA_FILES.items():
        data[place] = {}
        with open(os.path.join(DATA_FOLDER, file_name)) as f:
            for row in csv.DictReader(f):
                data[place].setdefault(int(row["Case"]), []).append(row)
        for rows in data[place].values():
            rows.sort(key=lambda row: int(row["Run"]))
    return data


def rates_for_run(data, case_number, run):
    """Flow rates (people per second) for one run, taken from row `run` of each data file.
    Every stair and every mezzanine door gets the measured rate. The foyer's total rate is split over
    its 3 doors in the same proportions as the data's door counts."""
    stairs = data["stairs"][case_number][run % len(data["stairs"][case_number])]
    corridor = data["corridor"][case_number][run % len(data["corridor"][case_number])]
    foyer = data["foyer"][case_number][run % len(data["foyer"][case_number])]
    door_counts = [int(foyer["Door_1_Count"]), int(foyer["Door_2_Count"]), int(foyer["Door_3_Count"])]
    total_flow = float(foyer["Flow_people_per_s"])
    return {
        "stair": float(stairs["Flow_people_per_s"]),
        "mezz door": float(corridor["Flow_people_per_s"]),
        "foyer door": [total_flow * count / sum(door_counts) for count in door_counts],
        "per metre": float(corridor["Specific_Flow_people_per_m_s"]),   # for doorways without data
    }


def gate_rate(gate, rates):
    """The flow rate of one gate, e.g. gate = "foyer door 2"."""
    if gate.startswith("foyer door"):
        number = int(gate.split()[-1])
        return rates["foyer door"][number - 1]
    if gate.startswith("mezz door"):
        return rates["mezz door"]
    return rates["stair"]


def doorway_rate(doorway, rates):
    """Flow rate (people per second) of any doorway: from our data if we have it,
    otherwise the mezzanine doors' rate per metre x the doorway's width."""
    if doorway in NO_DATA_WIDTHS:
        return rates["per metre"] * NO_DATA_WIDTHS[doorway]
    return gate_rate(doorway, rates)


def swap_stuck_pairs(ready, moved, route_maps, neighbour_lists, gate_of_square, tickets, gate_times, time):
    """People who could not move look at their favourite square (the neighbour closest to their exit,
    even if someone stands there). If two of them want each other's square, they swap with
    SWAP_CHANCE. Without this, two streams walking in opposite directions can block each other forever.
    Swapping INTO a doorway still needs a ticket, so the ticket machines still decide who goes through."""
    stuck_at = {}                                          # square -> stuck person standing there
    for person in ready:
        if id(person) not in moved and person["out_time"] is None:
            stuck_at[person["square"]] = person
    for square, person in list(stuck_at.items()):
        if stuck_at.get(square) is not person:
            continue                                       # already swapped this step
        dist = route_maps[person["exit"]]
        favourite = min(neighbour_lists[square], key=lambda nxt: dist[nxt])
        other = stuck_at.get(favourite)
        if other is None or other is person:
            continue
        other_dist = route_maps[other["exit"]]
        other_favourite = min(neighbour_lists[favourite], key=lambda nxt: other_dist[nxt])
        if other_favourite != square:
            continue                                       # they don't face each other
        # who would step INTO a doorway by swapping? each of them needs a ticket
        entering = [(p, gate_of_square[new]) for p, new in ((person, favourite), (other, square))
                    if new in gate_of_square and gate_of_square.get(p["square"]) != gate_of_square[new]]
        enough_tickets = all(tickets[gate] >= sum(1 for _, g in entering if g == gate) for _, gate in entering)
        if enough_tickets and random.random() < SWAP_CHANCE:
            for p, gate in entering:
                tickets[gate] -= 1
                gate_times[gate].append(time)
                p["path"].append(gate)
            person["square"], other["square"] = favourite, square
        # each pair is handled once
        del stuck_at[favourite]
        stuck_at.pop(square, None)


def run_once(seed, hall, case, rates):
    """Simulate one evacuation. `hall` comes from setup(), `case` from CASES, `rates` from rates_for_run().
    Returns (people, how many time steps each square was occupied, number of time steps, gate times).
    gate times = {gate: list of the times at which someone went through that doorway}"""
    grid = hall["grid"]
    exit_of_square = hall["exit_of_square"]
    gate_of_square = hall["gate_of_square"]
    stage_opening_of_square = hall["stage_opening_of_square"]
    neighbour_lists = hall["neighbours"]
    distance = hall["distance"]

    random.seed(seed)                                      # same seed -> exactly the same run
    people = create_people(grid, case)
    choose_exits(people, distance, hall["widths"])

    # one ticket machine per gate, plus one per balcony staircase
    gates = sorted(set(gate_of_square.values()))
    tickets = {gate: 0.0 for gate in gates}
    gate_times = {gate: [] for gate in gates}

    occupied = np.zeros((ROWS, COLUMNS), dtype=int)       # 1 where someone stands
    for person in people:
        if person["square"] is not None:
            occupied[person["square"]] = 1
    times_occupied = np.zeros((ROWS, COLUMNS))
    stair_queues = {}                                      # landing square -> people waiting on the stairs
    stair_name = {}                                        # landing square -> name of that staircase
    for person in people:
        if person["stair"] is not None:
            stair_queues.setdefault(person["stair"], []).append(person)
            stair_name[person["stair"]] = person["stair_name"]

    time = 0.0
    step = 0
    people_inside = len(people)
    stair_tickets = {landing: 0.0 for landing in stair_queues}
    exits_in_use = sorted(set(p["exit"] for p in people))
    route_maps = dict(distance)                            # the maps people walk by (updated with queues)
    next_map_update = 0.0
    while people_inside > 0 and time < MAX_TIME:
        # every few seconds: people look at the queues and a crowded doorway now counts as farther away
        if time >= next_map_update:
            extra = queue_costs(occupied, hall, rates)
            for name in exits_in_use:
                route_maps[name] = distance_with_queues(hall, name, extra)
            next_map_update = time + MAP_UPDATE_TIME

        # every ticket machine prints its share of a ticket (at most 1 ticket can wait)
        for gate in gates:
            tickets[gate] = min(1.0, tickets[gate] + gate_rate(gate, rates) * TIME_STEP)
        for landing in stair_queues:
            # balcony stairs have no data: rate per metre x stair width
            stair_rate = doorway_rate(stair_name[landing], rates)
            stair_tickets[landing] = min(1.0, stair_tickets[landing] + stair_rate * TIME_STEP)

        # balcony people step off their staircase when the landing square is free (and a ticket is there)
        for landing, waiting in stair_queues.items():
            if waiting and waiting[0]["start"] <= time and not occupied[landing] and stair_tickets[landing] >= 1:
                person = waiting.pop(0)
                person["square"] = landing
                person["path"].append(stair_name[landing])
                occupied[landing] = 1
                stair_tickets[landing] -= 1

        # everybody who is ready chooses a square (all at the same time)
        crowd = count_crowd(occupied)
        wanted = {}                                        # square -> people who want it
        ready = [p for p in people
                 if p["out_time"] is None and p["square"] is not None and p["start"] <= time]
        random.shuffle(ready)                              # fair: nobody always chooses first
        for person in ready:
            target = choose_square(person, route_maps, neighbour_lists, occupied, crowd)
            if target != person["square"]:
                wanted.setdefault(target, []).append(person)

        # if several people want the same square, a random one of them gets it
        moved = set()
        for target, candidates in wanted.items():
            winner = random.choice(candidates)
            # stepping INTO a doorway needs a ticket from its machine
            gate = gate_of_square.get(target)
            if gate is not None and gate_of_square.get(winner["square"]) != gate:
                if tickets[gate] < 1:
                    continue                               # no ticket: wait this step
                tickets[gate] -= 1
                gate_times[gate].append(time)              # remember when someone went through
                winner["path"].append(gate)
            opening = stage_opening_of_square.get(target)
            if opening is not None and stage_opening_of_square.get(winner["square"]) != opening:
                winner["path"].append(opening)             # went through a stage-side opening
            occupied[winner["square"]] = 0
            if grid[target] == EXIT:                       # stepped outside
                winner["square"] = None
                winner["out_time"] = time + TIME_STEP
                winner["exit_used"] = exit_of_square[target]
                winner["path"].append(exit_of_square[target])
                people_inside -= 1
            else:
                winner["square"] = target
                occupied[target] = 1
            moved.add(id(winner))

        # two people stuck face to face squeeze past each other (swap places)
        swap_stuck_pairs(ready, moved, route_maps, neighbour_lists, gate_of_square, tickets, gate_times, time)

        assert occupied.max() <= 1, "two people on one square"
        times_occupied += occupied
        time += TIME_STEP
        step += 1

    return people, times_occupied, step, gate_times


def setup(barriers=()):
    """Build everything that is the same in every run: the map, neighbours, door widths and
    distance maps. Returns them together in one dictionary.
    barriers = angles of temporary barriers in the corridor (none by default)."""
    grid, exit_of_square, gate_of_square, stage_opening_of_square = make_hall(barriers)
    neighbour_lists = make_neighbour_lists(grid)
    distance = {}
    for name in EXIT_NAMES:
        distance[name] = distance_to_exit(grid, neighbour_lists, exit_of_square, name)
    # the squares in front of each doorway where its queue forms: within QUEUE_RADIUS of its middle,
    # and only on the side people come from (closer to the hall centre, since everyone moves outwards)
    gate_queue_area = {}
    for gate in sorted(set(gate_of_square.values())):
        squares = [square for square, g in gate_of_square.items() if g == gate]
        middle_x = np.mean([square_centre(r, c)[0] for r, c in squares])
        middle_y = np.mean([square_centre(r, c)[1] for r, c in squares])
        gate_radius = math.hypot(middle_x, middle_y)
        area = []
        for square in neighbour_lists:
            x, y = square_centre(*square)
            close = math.hypot(x - middle_x, y - middle_y) <= QUEUE_RADIUS
            in_front = math.hypot(x, y) < gate_radius
            if close and in_front and square not in squares:
                area.append(square)
        gate_queue_area[gate] = area
    # every doorway (with or without data) and its squares, plus a distance map to it
    doorway_squares = {}
    for lookup in (gate_of_square, stage_opening_of_square, exit_of_square):
        for square, name in lookup.items():
            doorway_squares.setdefault(name, []).append(square)
    for name, _, where in BALCONY_STAIRS:
        doorway_squares["balcony " + name] = [stair_landing(grid, where)]
    doorway_distance = {}
    for name, squares in doorway_squares.items():
        doorway_distance[name] = distance_from_squares(neighbour_lists, squares)

    return {"grid": grid, "exit_of_square": exit_of_square, "gate_of_square": gate_of_square,
            "stage_opening_of_square": stage_opening_of_square,
            "neighbours": neighbour_lists, "distance": distance, "widths": door_widths(grid, exit_of_square),
            "gate_queue_area": gate_queue_area, "doorway_squares": doorway_squares,
            "doorway_distance": doorway_distance}


# ======================================================================== 8. calculated escape time
def distance_from_squares(neighbour_lists, start_squares):
    """Steps from every square to the nearest of `start_squares` (the same wave as distance_to_exit)."""
    far_away = 10 ** 6
    distance = np.full((ROWS, COLUMNS), far_away)
    queue = deque()
    for square in start_squares:
        distance[square] = 0
        queue.append(square)
    while queue:
        square = queue.popleft()
        for nxt in neighbour_lists[square]:
            if distance[nxt] == far_away:
                distance[nxt] = distance[square] + 1
                queue.append(nxt)
    return distance


def doorway_level(doorway):
    """How far out a doorway is: 1 = first doorway after the seats, ..., 4 = exits."""
    if doorway.startswith(("stair", "stage", "balcony")):
        return 1
    if doorway.startswith("mezz door"):
        return 2
    if doorway.startswith("foyer door"):
        return 3
    return 4                                              # the exits


def calculated_escape_time(people, hall, rates):
    """The time until the last person is completely out of the building, calculated from:
       * how many people went through each doorway (counted in the simulation)
       * each doorway's flow rate (our data, or rate per metre x width)
       * walking times between doorways
    We go from the seats outwards. For every doorway:
       earliest = when the first people can reach it
       finish   = the later of
                    earliest + people / rate          (everyone squeezing through at the doorway's rate)
                    when the last person reaches it   (they can't pass before they arrive)
       People reach a doorway either straight from their seat (start time + walk), or through the
       doorway before it (that doorway's times + the walk between the two).
    The answer is the latest finish of any exit.
    Returns (time in seconds, details per doorway)."""
    distance = hall["doorway_distance"]
    squares = hall["doorway_squares"]

    # ---- 1. clean paths, count people per doorway, and which doorway feeds which ----
    count = {}
    feeders = {}                     # doorway -> doorways people came through just before it
    first_arrivals = {}              # doorway -> arrival times of people who came straight from their seat
    for person in people:
        path = list(dict.fromkeys(person["path"]))        # each doorway once, in the order first used
        for doorway in path:
            count[doorway] = count.get(doorway, 0) + 1
        for before, after in zip(path[:-1], path[1:]):
            # only links that go outwards (a few people wander back and forth at the stage corners)
            if doorway_level(before) < doorway_level(after):
                feeders.setdefault(after, set()).add(before)
        if path:
            first = path[0]
            walk = distance[first][person["start_square"]] * TIME_STEP
            first_arrivals.setdefault(first, []).append(person["start"] + walk)

    # ---- 2. walking time between two doorways (shortest walk between them) ----
    def walk_between(before, after):
        return min(distance[after][square] for square in squares[before]) * TIME_STEP

    # ---- 3. earliest arrival and finish of every doorway, working outwards ----
    earliest = {}
    finish = {}

    def work_out(doorway):
        if doorway in finish:
            return
        earliest_times = []
        last_arrivals = []
        if doorway in first_arrivals:
            earliest_times.append(min(first_arrivals[doorway]))
            last_arrivals.append(max(first_arrivals[doorway]))
        for before in feeders.get(doorway, ()):
            work_out(before)
            walk = walk_between(before, doorway)
            earliest_times.append(earliest[before] + walk)
            last_arrivals.append(finish[before] + walk)
        earliest[doorway] = min(earliest_times)
        squeeze_time = count[doorway] / doorway_rate(doorway, rates)
        finish[doorway] = max(earliest[doorway] + squeeze_time, max(last_arrivals))

    for doorway in count:
        work_out(doorway)

    details = {}
    for doorway in count:
        details[doorway] = {"people": count[doorway], "rate": doorway_rate(doorway, rates),
                            "earliest": earliest[doorway], "finish": finish[doorway]}
    last_out = max(finish[name] for name in EXIT_NAMES if name in finish)
    return last_out, details
