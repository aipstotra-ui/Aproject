import csv
import heapq
import math
import os
import random
from collections import deque

import numpy as np

CELL = 0.5
WALK_SPEED = 1.19
TIME_STEP = CELL / WALK_SPEED

STAIR_WALK_TIME = 16.0
MAX_TIME = 600.0

CASES = [
    {"case": "3", "data_case": 3, "name": "Students, emergency",
     "reaction": 5.0, "guided": 0.0, "habit": 0.40, "nearest": 0.60},
    {"case": "4", "data_case": 4, "name": "Concert, emergency",
     "reaction": 10.0, "guided": 0.60, "habit": 0.25, "nearest": 0.15},
    {"case": "4 + CM", "data_case": 4, "name": "Concert, emergency, Crowd Managers",
     "reaction": 10.0, "guided": 0.80, "habit": 0.125, "nearest": 0.075},
]

DATA_FOLDER = "data"
DATA_FILES = {"stairs": "stairs.csv", "corridor": "corridor.csv", "foyer": "foyer_doors.csv"}
NO_DATA_WIDTHS = {
    "P1": 1.6, "P2": 1.6, "P3": 1.6, "P4": 1.6,
    "NW": 1.5, "NE": 1.5,
    "FRONT": 5 * 1.8,
    "stage W": 1.5, "stage E": 1.5,
    "balcony NW stair": 1.3, "balcony W foyer stair": 2.0,
    "balcony E foyer stair": 2.0, "balcony NE stair": 1.3,
}
SWAP_CHANCE = 0.5
QUEUE_RADIUS = 4.0
MAP_UPDATE_TIME = 5.0

PULL = 3.0
CROWD = 0.3
DOOR_FLOW = 0.28

WALL, FLOOR, SEAT, EXIT = 0, 1, 2, 3

X_MIN, X_MAX = -25.0, 25.0
Y_MIN, Y_MAX = -31.0, 3.0
COLUMNS = int((X_MAX - X_MIN) / CELL)
ROWS = int((Y_MAX - Y_MIN) / CELL)

ORCHESTRA_RADIUS = 13.0
BARRIER_END = 13.5
MEZZANINE_END = 19.5
MEZZ_WALL_END = 20.0
CORRIDOR_END = 22.4
OUTER_WALL_END = 22.9

ACCESS_STAIRS = [12, 45, 78, 102, 135, 168]
MEZZANINE_DOORS = [10, 30, 50, 70, 90, 110, 130, 150, 170]
FOYER_DOORS = [80, 90, 100]
PORTICO_DOORS = {"P1": 25, "P2": 50, "P3": 130, "P4": 155}
FRONT_DOORS_X = [-6.8, -3.4, 0.0, 3.4, 6.8]
BARRIER_THICKNESS = 1.0

EXIT_NAMES = ["FRONT", "NW", "NE", "P1", "P2", "P3", "P4"]

ORCHESTRA_PEOPLE = 527
MEZZANINE_PEOPLE = 486
BALCONY_STAIRS = [("NW stair", 66, (-19.0, 1.0)),
                  ("W foyer stair", 92, 65),
                  ("E foyer stair", 92, 115),
                  ("NE stair", 66, (19.0, 1.0))]

DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def square_centre(row, col):
    x = X_MIN + (col + 0.5) * CELL
    y = Y_MAX - (row + 0.5) * CELL
    return x, y


def square_at(x, y):
    col = int((x - X_MIN) / CELL)
    row = int((Y_MAX - y) / CELL)
    return row, col


def point_on_circle(radius, angle):
    a = math.radians(angle)
    return -radius * math.cos(a), -radius * math.sin(a)


def near_opening(x, y, radius, angles, width):
    for angle in angles:
        door_x, door_y = point_on_circle(radius, angle)
        if math.hypot(x - door_x, y - door_y) <= width / 2:
            return True
    return False


def which_opening(x, y, radius, angles, width):
    for number, angle in enumerate(angles, start=1):
        if near_opening(x, y, radius, [angle], width):
            return number
    return None


def make_hall(barriers=()):
    grid = np.zeros((ROWS, COLUMNS), dtype=int)
    exit_of_square = {}
    gate_of_square = {}
    stage_opening_of_square = {}

    for row in range(ROWS):
        for col in range(COLUMNS):
            x, y = square_centre(row, col)
            r = math.hypot(x, y)
            kind = WALL
            exit_name = None

            if y < 0:
                if r < ORCHESTRA_RADIUS:
                    kind = SEAT
                elif r < BARRIER_END:
                    number = which_opening(x, y, ORCHESTRA_RADIUS, ACCESS_STAIRS, 1.2)
                    if number:
                        kind = FLOOR
                        gate_of_square[(row, col)] = f"stair {number}"
                elif r < MEZZANINE_END:
                    kind = SEAT
                elif r < MEZZ_WALL_END:
                    number = which_opening(x, y, MEZZANINE_END, MEZZANINE_DOORS, 1.6)
                    if number:
                        kind = FLOOR
                        gate_of_square[(row, col)] = f"mezz door {number}"
                elif r < CORRIDOR_END:
                    kind = FLOOR
                    for angle in barriers:
                        bx, by = point_on_circle(r, angle)
                        if math.hypot(x - bx, y - by) <= BARRIER_THICKNESS / 2:
                            kind = WALL
                elif r < OUTER_WALL_END:
                    number = which_opening(x, y, CORRIDOR_END, FOYER_DOORS, 1.8)
                    if number:
                        kind = FLOOR
                        gate_of_square[(row, col)] = f"foyer door {number}"
                    for name, angle in PORTICO_DOORS.items():
                        if near_opening(x, y, CORRIDOR_END, [angle], 1.6):
                            kind, exit_name = EXIT, name
                if abs(x) <= 9.0 and r >= OUTER_WALL_END and y > -30.0:
                    kind = FLOOR
                if abs(x) <= 9.0 and -30.5 < y <= -30.0:
                    for door_x in FRONT_DOORS_X:
                        if abs(x - door_x) <= 0.9:
                            kind, exit_name = EXIT, "FRONT"

            else:
                if 0.5 <= y < 2.0 and 11.0 <= abs(x) <= CORRIDOR_END:
                    kind = FLOOR
                if y < 0.5 and (11.0 <= abs(x) <= 12.5 or MEZZ_WALL_END <= abs(x) <= CORRIDOR_END):
                    kind = FLOOR
                if y < 0.5 and 11.0 <= abs(x) <= 12.5:
                    stage_opening_of_square[(row, col)] = "stage W" if x < 0 else "stage E"
                if 0.5 <= y < 2.0 and CORRIDOR_END < abs(x) <= CORRIDOR_END + 0.5:
                    kind = EXIT
                    exit_name = "NW" if x < 0 else "NE"

            grid[row, col] = kind
            if kind == EXIT:
                exit_of_square[(row, col)] = exit_name
            if kind != FLOOR:
                gate_of_square.pop((row, col), None)
    return grid, exit_of_square, gate_of_square, stage_opening_of_square


def neighbours(grid, row, col):
    result = []
    for d_row, d_col in DIRECTIONS:
        r2, c2 = row + d_row, col + d_col
        if r2 < 0 or r2 >= ROWS or c2 < 0 or c2 >= COLUMNS:
            continue
        if grid[r2, c2] == WALL:
            continue
        if d_row != 0 and d_col != 0:
            if grid[row + d_row, col] == WALL or grid[row, col + d_col] == WALL:
                continue
        result.append((r2, c2))
    return result


def make_neighbour_lists(grid):
    lists = {}
    for row in range(ROWS):
        for col in range(COLUMNS):
            if grid[row, col] != WALL:
                lists[(row, col)] = neighbours(grid, row, col)
    return lists


def door_widths(grid, exit_of_square):
    widths = {name: 0 for name in EXIT_NAMES}
    for name in exit_of_square.values():
        if name != "FRONT":
            widths[name] += 1
    for row in range(ROWS):
        for col in range(COLUMNS):
            x, y = square_centre(row, col)
            in_outer_wall = y < 0 and CORRIDOR_END <= math.hypot(x, y) < OUTER_WALL_END
            if in_outer_wall and grid[row, col] == FLOOR:
                widths["FRONT"] += 1
    return widths


def distance_to_exit(grid, neighbour_lists, exit_of_square, exit_name):
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
            if distance[nxt] == far_away:
                distance[nxt] = distance[square] + 1
                queue.append(nxt)
    return distance


def distance_with_queues(hall, exit_name, extra_cost):
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
            continue
        for previous in neighbour_lists[square]:
            cost = 1.0
            gate = gate_of_square.get(square)
            if gate is not None and gate_of_square.get(previous) != gate:
                cost += extra_cost[gate]
            if dist_here + cost < distance[previous]:
                distance[previous] = dist_here + cost
                heapq.heappush(heap, (dist_here + cost, previous))
    return distance


def queue_costs(occupied, hall, rates):
    costs = {}
    for gate, nearby_squares in hall["gate_queue_area"].items():
        waiting = sum(occupied[square] for square in nearby_squares)
        costs[gate] = waiting / (gate_rate(gate, rates) * TIME_STEP)
    return costs


def stair_landing(grid, where):
    if isinstance(where, tuple):
        x, y = where
    else:
        x, y = point_on_circle((MEZZ_WALL_END + CORRIDOR_END) / 2, where)
    return square_at(x, y)


def create_people(grid, case):
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
            people.append({"square": None, "start": reaction + STAIR_WALK_TIME, "stair": landing,
                           "stair_name": "balcony " + name, "start_square": landing})

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
        person["path"] = []
    return people


def mezz_door_on_the_way(hall, exit_name, square):
    distance = hall["distance"][exit_name]
    gate_of_square = hall["gate_of_square"]
    while distance[square] > 0:
        square = min(hall["neighbours"][square], key=lambda nxt: distance[nxt])
        gate = gate_of_square.get(square, "")
        if gate.startswith("mezz door"):
            return gate
    return None


def choose_exits(people, hall, rates):
    distance = hall["distance"]
    widths = hall["widths"]

    def start_square(person):
        return person["square"] if person["stair"] is None else person["stair"]

    sent_by_ushers = {name: 0 for name in EXIT_NAMES}
    sent_through_door = {}
    guided = [p for p in people if p["type"] == "guided"]
    random.shuffle(guided)

    for person in people:
        here = start_square(person)
        if person["type"] == "habit":
            person["exit"] = "FRONT"
        elif person["type"] == "nearest":
            person["exit"] = min(EXIT_NAMES, key=lambda name: distance[name][here])

    door_tickets_per_step = rates["mezz door"] * TIME_STEP
    if hall["barriers"]:
        for person in people:
            if person["type"] != "guided":
                door = mezz_door_on_the_way(hall, person["exit"], start_square(person))
                if door is not None:
                    sent_through_door[door] = sent_through_door.get(door, 0) + 1
    for person in guided:
        here = start_square(person)
        best_exit, best_time, best_door = None, None, None
        for name in EXIT_NAMES:
            walk = distance[name][here]
            wait = sent_by_ushers[name] / (widths[name] * DOOR_FLOW)
            door = None
            if hall["barriers"]:
                door = mezz_door_on_the_way(hall, name, here)
                if door is not None:
                    wait = max(wait, sent_through_door.get(door, 0) / door_tickets_per_step)
            if best_time is None or walk + wait < best_time:
                best_exit, best_time, best_door = name, walk + wait, door
        person["exit"] = best_exit
        sent_by_ushers[best_exit] += 1
        if best_door is not None:
            sent_through_door[best_door] = sent_through_door.get(best_door, 0) + 1


def count_crowd(occupied):
    padded = np.pad(occupied, 1)
    crowd = np.zeros((ROWS, COLUMNS), dtype=int)
    for d_row, d_col in DIRECTIONS:
        crowd += padded[1 + d_row: 1 + d_row + ROWS, 1 + d_col: 1 + d_col + COLUMNS]
    return crowd


def choose_square(person, distance, neighbour_lists, occupied, crowd):
    here = person["square"]
    dist = distance[person["exit"]]
    options = [here]
    weights = [math.exp(-CROWD * crowd[here])]
    for square in neighbour_lists[here]:
        if occupied[square]:
            continue
        change = dist[square] - dist[here]
        options.append(square)
        score = -PULL * change - CROWD * crowd[square]
        weights.append(math.exp(min(score, 50.0)))
    return random.choices(options, weights=weights)[0]


def load_flow_data():
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
    stairs = data["stairs"][case_number][run % len(data["stairs"][case_number])]
    corridor = data["corridor"][case_number][run % len(data["corridor"][case_number])]
    foyer = data["foyer"][case_number][run % len(data["foyer"][case_number])]
    door_counts = [int(foyer["Door_1_Count"]), int(foyer["Door_2_Count"]), int(foyer["Door_3_Count"])]
    total_flow = float(foyer["Flow_people_per_s"])
    return {
        "stair": float(stairs["Flow_people_per_s"]),
        "mezz door": float(corridor["Flow_people_per_s"]),
        "foyer door": [total_flow * count / sum(door_counts) for count in door_counts],
        "per metre": float(corridor["Specific_Flow_people_per_m_s"]),
    }


def gate_rate(gate, rates):
    if gate.startswith("foyer door"):
        number = int(gate.split()[-1])
        return rates["foyer door"][number - 1]
    if gate.startswith("mezz door"):
        return rates["mezz door"]
    return rates["stair"]


def doorway_rate(doorway, rates):
    if doorway in NO_DATA_WIDTHS:
        return rates["per metre"] * NO_DATA_WIDTHS[doorway]
    return gate_rate(doorway, rates)


def swap_stuck_pairs(ready, moved, route_maps, neighbour_lists, gate_of_square, tickets, gate_times, time):
    stuck_at = {}
    for person in ready:
        if id(person) not in moved and person["out_time"] is None:
            stuck_at[person["square"]] = person
    for square, person in list(stuck_at.items()):
        if stuck_at.get(square) is not person:
            continue
        dist = route_maps[person["exit"]]
        favourite = min(neighbour_lists[square], key=lambda nxt: dist[nxt])
        other = stuck_at.get(favourite)
        if other is None or other is person:
            continue
        other_dist = route_maps[other["exit"]]
        other_favourite = min(neighbour_lists[favourite], key=lambda nxt: other_dist[nxt])
        if other_favourite != square:
            continue
        entering = [(p, gate_of_square[new]) for p, new in ((person, favourite), (other, square))
                    if new in gate_of_square and gate_of_square.get(p["square"]) != gate_of_square[new]]
        enough_tickets = all(tickets[gate] >= sum(1 for _, g in entering if g == gate) for _, gate in entering)
        if enough_tickets and random.random() < SWAP_CHANCE:
            for p, gate in entering:
                tickets[gate] -= 1
                gate_times[gate].append(time)
                p["path"].append(gate)
            person["square"], other["square"] = favourite, square
        del stuck_at[favourite]
        stuck_at.pop(square, None)


def run_once(seed, hall, case, rates):
    grid = hall["grid"]
    exit_of_square = hall["exit_of_square"]
    gate_of_square = hall["gate_of_square"]
    stage_opening_of_square = hall["stage_opening_of_square"]
    neighbour_lists = hall["neighbours"]
    distance = hall["distance"]

    random.seed(seed)
    people = create_people(grid, case)
    choose_exits(people, hall, rates)

    gates = sorted(set(gate_of_square.values()))
    tickets = {gate: 0.0 for gate in gates}
    gate_times = {gate: [] for gate in gates}

    occupied = np.zeros((ROWS, COLUMNS), dtype=int)
    for person in people:
        if person["square"] is not None:
            occupied[person["square"]] = 1
    times_occupied = np.zeros((ROWS, COLUMNS))
    stair_queues = {}
    stair_name = {}
    for person in people:
        if person["stair"] is not None:
            stair_queues.setdefault(person["stair"], []).append(person)
            stair_name[person["stair"]] = person["stair_name"]

    time = 0.0
    step = 0
    people_inside = len(people)
    stair_tickets = {landing: 0.0 for landing in stair_queues}
    exits_in_use = sorted(set(p["exit"] for p in people))
    route_maps = dict(distance)
    next_map_update = 0.0
    while people_inside > 0 and time < MAX_TIME:
        if time >= next_map_update:
            extra = queue_costs(occupied, hall, rates)
            for name in exits_in_use:
                route_maps[name] = distance_with_queues(hall, name, extra)
            next_map_update = time + MAP_UPDATE_TIME

        for gate in gates:
            tickets[gate] = min(1.0, tickets[gate] + gate_rate(gate, rates) * TIME_STEP)
        for landing in stair_queues:
            stair_rate = doorway_rate(stair_name[landing], rates)
            stair_tickets[landing] = min(1.0, stair_tickets[landing] + stair_rate * TIME_STEP)

        for landing, waiting in stair_queues.items():
            if waiting and waiting[0]["start"] <= time and not occupied[landing] and stair_tickets[landing] >= 1:
                person = waiting.pop(0)
                person["square"] = landing
                person["path"].append(stair_name[landing])
                occupied[landing] = 1
                stair_tickets[landing] -= 1

        crowd = count_crowd(occupied)
        wanted = {}
        ready = [p for p in people
                 if p["out_time"] is None and p["square"] is not None and p["start"] <= time]
        random.shuffle(ready)
        for person in ready:
            target = choose_square(person, route_maps, neighbour_lists, occupied, crowd)
            if target != person["square"]:
                wanted.setdefault(target, []).append(person)

        moved = set()
        for target, candidates in wanted.items():
            winner = random.choice(candidates)
            gate = gate_of_square.get(target)
            if gate is not None and gate_of_square.get(winner["square"]) != gate:
                if tickets[gate] < 1:
                    continue
                tickets[gate] -= 1
                gate_times[gate].append(time)
                winner["path"].append(gate)
            opening = stage_opening_of_square.get(target)
            if opening is not None and stage_opening_of_square.get(winner["square"]) != opening:
                winner["path"].append(opening)
            occupied[winner["square"]] = 0
            if grid[target] == EXIT:
                winner["square"] = None
                winner["out_time"] = time + TIME_STEP
                winner["exit_used"] = exit_of_square[target]
                winner["path"].append(exit_of_square[target])
                people_inside -= 1
            else:
                winner["square"] = target
                occupied[target] = 1
            moved.add(id(winner))

        swap_stuck_pairs(ready, moved, route_maps, neighbour_lists, gate_of_square, tickets, gate_times, time)

        assert occupied.max() <= 1, "two people on one square"
        times_occupied += occupied
        time += TIME_STEP
        step += 1

    return people, times_occupied, step, gate_times


def setup(barriers=()):
    grid, exit_of_square, gate_of_square, stage_opening_of_square = make_hall(barriers)
    neighbour_lists = make_neighbour_lists(grid)
    distance = {}
    for name in EXIT_NAMES:
        distance[name] = distance_to_exit(grid, neighbour_lists, exit_of_square, name)
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
    doorway_squares = {}
    for lookup in (gate_of_square, stage_opening_of_square, exit_of_square):
        for square, name in lookup.items():
            doorway_squares.setdefault(name, []).append(square)
    for name, _, where in BALCONY_STAIRS:
        doorway_squares["balcony " + name] = [stair_landing(grid, where)]
    doorway_distance = {}
    for name, squares in doorway_squares.items():
        doorway_distance[name] = distance_from_squares(neighbour_lists, squares)

    return {"grid": grid, "barriers": list(barriers), "exit_of_square": exit_of_square, "gate_of_square": gate_of_square,
            "stage_opening_of_square": stage_opening_of_square,
            "neighbours": neighbour_lists, "distance": distance, "widths": door_widths(grid, exit_of_square),
            "gate_queue_area": gate_queue_area, "doorway_squares": doorway_squares,
            "doorway_distance": doorway_distance}


def distance_from_squares(neighbour_lists, start_squares):
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
    if doorway.startswith(("stair", "stage", "balcony")):
        return 1
    if doorway.startswith("mezz door"):
        return 2
    if doorway.startswith("foyer door"):
        return 3
    return 4


def calculated_escape_time(people, hall, rates):
    distance = hall["doorway_distance"]
    squares = hall["doorway_squares"]

    count = {}
    feeders = {}
    first_arrivals = {}
    for person in people:
        path = list(dict.fromkeys(person["path"]))
        for doorway in path:
            count[doorway] = count.get(doorway, 0) + 1
        for before, after in zip(path[:-1], path[1:]):
            if doorway_level(before) < doorway_level(after):
                feeders.setdefault(after, set()).add(before)
        if path:
            first = path[0]
            walk = distance[first][person["start_square"]] * TIME_STEP
            first_arrivals.setdefault(first, []).append(person["start"] + walk)

    def walk_between(before, after):
        return min(distance[after][square] for square in squares[before]) * TIME_STEP

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
