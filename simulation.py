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
"""
import math
import random
from collections import deque

import numpy as np

# ---------------------------------------------------------------- settings you can change
CELL = 0.5                 # size of one square in metres
WALK_SPEED = 1.19          # normal walking speed in metres per second
TIME_STEP = CELL / WALK_SPEED   # time to walk one square: about 0.42 s

REACTION_TIME = 10.0       # seconds everyone waits before starting to move (concert)
STAIR_WALK_TIME = 16.0     # extra seconds for balcony people to walk down their stairs
MAX_TIME = 600.0           # stop the simulation after this many seconds

SHARE_GUIDED = 0.60        # concert with ushers: 60 % follow the ushers
SHARE_HABIT = 0.25         #                      25 % go back to the front foyer
SHARE_NEAREST = 0.15       #                      15 % go to the nearest exit

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
def make_hall():
    """Return (grid, exit_of_square).
    grid[row, col] is WALL, FLOOR, SEAT or EXIT.
    exit_of_square[(row, col)] is the name of the exit for every EXIT square."""
    grid = np.zeros((ROWS, COLUMNS), dtype=int)       # everything starts as wall
    exit_of_square = {}

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
                    if near_opening(x, y, ORCHESTRA_RADIUS, ACCESS_STAIRS, 1.2):
                        kind = FLOOR                  # opening in the barrier
                elif r < MEZZANINE_END:
                    kind = SEAT                       # mezzanine
                elif r < MEZZ_WALL_END:
                    if near_opening(x, y, MEZZANINE_END, MEZZANINE_DOORS, 1.6):
                        kind = FLOOR                  # mezzanine door
                elif r < CORRIDOR_END:
                    kind = FLOOR                      # corridor
                elif r < OUTER_WALL_END:
                    if near_opening(x, y, CORRIDOR_END, FOYER_DOORS, 1.8):
                        kind = FLOOR                  # door into the foyer
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
                if 0.5 <= y < 2.0 and CORRIDOR_END < abs(x) <= CORRIDOR_END + 0.5:
                    kind = EXIT
                    exit_name = "NW" if x < 0 else "NE"   # side exits next to the stage

            grid[row, col] = kind
            if kind == EXIT:
                exit_of_square[(row, col)] = exit_name
    return grid, exit_of_square


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


# ======================================================================== 3. people
def stair_landing(grid, where):
    """The floor square where a balcony staircase arrives."""
    if isinstance(where, tuple):
        x, y = where
    else:
        x, y = point_on_circle((MEZZ_WALL_END + CORRIDOR_END) / 2, where)   # middle of the corridor
    return square_at(x, y)


def create_people(grid):
    """Make a list of people. Each person is a dictionary."""
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
        people.append({"square": seat, "start": REACTION_TIME, "stair": None})
    for seat in random.sample(mezzanine_seats, MEZZANINE_PEOPLE):
        people.append({"square": seat, "start": REACTION_TIME, "stair": None})
    for name, how_many, where in BALCONY_STAIRS:
        landing = stair_landing(grid, where)
        for _ in range(how_many):
            # balcony people are not on the floor yet: they wait in the stair's queue
            people.append({"square": None, "start": REACTION_TIME + STAIR_WALK_TIME,
                           "stair": landing})

    # give everyone a type: guided, habit or nearest
    for person in people:
        dice = random.random()
        if dice < SHARE_GUIDED:
            person["type"] = "guided"
        elif dice < SHARE_GUIDED + SHARE_HABIT:
            person["type"] = "habit"
        else:
            person["type"] = "nearest"
        person["out_time"] = None
        person["exit_used"] = None
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
        weights.append(math.exp(-PULL * change - CROWD * crowd[square]))
    return random.choices(options, weights=weights)[0]


def run_once(seed, hall):
    """Simulate one evacuation. `hall` comes from setup().
    Returns (people, how many time steps each square was occupied, number of time steps)."""
    grid = hall["grid"]
    exit_of_square = hall["exit_of_square"]
    neighbour_lists = hall["neighbours"]
    distance = hall["distance"]

    random.seed(seed)                                      # same seed -> exactly the same run
    people = create_people(grid)
    choose_exits(people, distance, hall["widths"])

    occupied = np.zeros((ROWS, COLUMNS), dtype=int)       # 1 where someone stands
    for person in people:
        if person["square"] is not None:
            occupied[person["square"]] = 1
    times_occupied = np.zeros((ROWS, COLUMNS))
    stair_queues = {}                                      # landing square -> people waiting on the stairs
    for person in people:
        if person["stair"] is not None:
            stair_queues.setdefault(person["stair"], []).append(person)

    time = 0.0
    step = 0
    people_inside = len(people)
    while people_inside > 0 and time < MAX_TIME:
        # balcony people step off their staircase when the landing square is free
        for landing, waiting in stair_queues.items():
            if waiting and waiting[0]["start"] <= time and not occupied[landing]:
                person = waiting.pop(0)
                person["square"] = landing
                occupied[landing] = 1

        # everybody who is ready chooses a square (all at the same time)
        crowd = count_crowd(occupied)
        wanted = {}                                        # square -> people who want it
        ready = [p for p in people
                 if p["out_time"] is None and p["square"] is not None and p["start"] <= time]
        for person in ready:
            target = choose_square(person, distance, neighbour_lists, occupied, crowd)
            if target != person["square"]:
                wanted.setdefault(target, []).append(person)

        # if several people want the same square, a random one of them gets it
        for target, candidates in wanted.items():
            winner = random.choice(candidates)
            occupied[winner["square"]] = 0
            if grid[target] == EXIT:                       # stepped outside
                winner["square"] = None
                winner["out_time"] = time + TIME_STEP
                winner["exit_used"] = exit_of_square[target]
                people_inside -= 1
            else:
                winner["square"] = target
                occupied[target] = 1

        assert occupied.max() <= 1, "two people on one square"
        times_occupied += occupied
        time += TIME_STEP
        step += 1

    return people, times_occupied, step


def setup():
    """Build everything that is the same in every run: the map, neighbours, door widths and
    distance maps. Returns them together in one dictionary."""
    grid, exit_of_square = make_hall()
    neighbour_lists = make_neighbour_lists(grid)
    distance = {}
    for name in EXIT_NAMES:
        distance[name] = distance_to_exit(grid, neighbour_lists, exit_of_square, name)
    return {"grid": grid, "exit_of_square": exit_of_square, "neighbours": neighbour_lists,
            "distance": distance, "widths": door_widths(grid, exit_of_square)}
