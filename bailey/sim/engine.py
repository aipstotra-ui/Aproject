"""Model II, step 4: the simulation itself (a "cellular automaton").

Picture a board game:
  * the board is the grid of 0.5 m cells (grid.py)
  * every person stands on one cell; one cell holds at most one person
  * one turn = 0.42 s, the time to walk one cell at normal speed (1.19 m/s)

Every turn, in this order:
  1. every few seconds, redraw the "time to exit" maps, taking crowding into account (fields.py)
  2. people who just finished reacting choose an exit (choice.py)
  3. rational people reconsider their exit every 5 s
  4. Crowd Managers persuade people walking close to them
  5. people near a blocked door discover it (and pause 10 s); others may hear about it from neighbours
  6. everybody who is ready MOVES:
       a. score each free neighbouring cell: high if it is closer to your exit, a little extra if
          others walked there recently (following the crowd); staying put is also an option
       b. pick one option at random, with probability proportional to its score
       c. slow cells (seats, stairs) and diagonal steps sometimes fail
       d. if several people want the same cell: with probability mu (0.67) they bump and nobody gets
          it, otherwise one of them (at random) does
       e. stepping onto an exit cell means you are out
  7. two stuck people who face each other and want each other's cell swap places (50 %)
  8. footprints fade a little
"""
import numpy as np

from bailey.behavior import choice as C
from bailey.sim.fields import FieldSolver
from bailey.sim.grid import EXITS


class CAConfig:
    """Settings for one run."""

    def __init__(self, routing, premovement, blocked=None, known=True, crowd_managers=False,
                 guided_shares=None, guided_shares_informed=None, exit_capacity=None, snapshot_times=()):
        self.routing = routing                          # behaviour mix, e.g. {"habit": 0.4, "nearest": 0.6}
        self.premovement = premovement                  # {"delay": seconds}: reaction time, same for all
        self.blocked = set(blocked or set())            # names of blocked doors
        self.known = known                              # True = everybody knows about them at the start
        self.crowd_managers = crowd_managers            # put the 6 Crowd Managers in the hall?
        self.guided_shares = guided_shares              # Crowd Managers' plan {zone: {exit: share}}
        self.guided_shares_informed = guided_shares_informed  # their plan once the blockage is known
        self.exit_capacity = exit_capacity              # {exit: people per second}, for rational people
        self.snapshot_times = snapshot_times            # times at which to save everyone's position


class CAResult:
    """What one run produced."""

    def __init__(self, T, exit_time, exit_id, agent_type, agent_zone, start_time, dt, snapshots,
                 occupancy_time, remaining):
        self.T = T                          # time the last person got out (inf if someone is stuck)
        self.exit_time = exit_time          # per person: time they got out
        self.exit_id = exit_id              # per person: which exit (number in EXITS)
        self.agent_type = agent_type        # per person: HABIT / NEAREST / GUIDED / RATIONAL (at the end)
        self.agent_zone = agent_zone        # per person: seating zone number
        self.start_time = start_time        # per person: when they started moving
        self.dt = dt                        # length of one turn [s]
        self.snapshots = snapshots          # time -> (cells, types) of everybody still inside
        self.occupancy_time = occupancy_time  # per cell: seconds someone stood on it
        self.remaining = remaining          # people still inside at the end (should be 0)

    def curve(self, t_max, step=1.0):
        """People still inside at times 0, 1, 2, ... t_max."""
        times = np.arange(0, t_max + step, step)
        out_times = np.sort(self.exit_time[np.isfinite(self.exit_time)])
        inside = []
        for t in times:
            already_out = np.searchsorted(out_times, t, side="right")
            inside.append(len(self.exit_time) - already_out)
        return times, np.array(inside)

    def percentile(self, q):
        """Time by which q % of people are out."""
        times = np.where(np.isfinite(self.exit_time), self.exit_time, np.inf)
        return float(np.nanpercentile(times, q))


class Engine:
    """Things that stay the same for every run on the same building (with the same blocked doors)."""

    def __init__(self, grid, params, blocked=frozenset()):
        self.grid = grid
        self.P = params
        ca = params["ca"]
        # normal walking speed (1.19 m/s) and the length of one turn (0.42 s)
        self.v_free = params["hydraulic"]["free_speed_factor"] * params["hydraulic"]["k"]["level"]
        self.dt = ca["cell"] / self.v_free
        self.blocked = set(blocked)

        # two versions of the "time to exit" maps:
        #   actual: the building as it really is (blocked doors closed)
        #   plan:   the building as people think it is before they know about the blockage
        self.fs_actual = FieldSolver(grid, self.v_free, self.blocked)
        if self.blocked:
            self.fs_plan = FieldSolver(grid, self.v_free)
        else:
            self.fs_plan = self.fs_actual
        self.F_static = {"actual": self.fs_actual.solve(), "plan": self.fs_plan.solve()}

        # neighbour table: neigh[cell] = list of cells you can step to (-1 = no more neighbours),
        # nlen[cell] = their step lengths. People can never walk through a blocked door.
        pairs, lengths = grid.edges(self.blocked)
        neighbours = [[] for _ in range(grid.n)]
        for (a, b), length in zip(pairs, lengths):
            neighbours[a].append((b, length))
        for (a, b), length in zip(pairs, lengths):
            neighbours[b].append((a, length))
        most = max(len(n) for n in neighbours)
        self.neigh = -np.ones((grid.n, most), int)
        self.nlen = np.ones((grid.n, most))
        for cell in range(grid.n):
            for slot, (other, length) in enumerate(neighbours[cell]):
                self.neigh[cell, slot] = other
                self.nlen[cell, slot] = length

        self.exit_of_cell = grid.exit_index()
        self.zone_names = list(grid.zone_cells)
        # where the blocked doors are (people within 3 m of one discover it)
        self.block_centres = [grid.apertures[name].center for name in self.blocked if name in grid.apertures]
        # exits that can still be reached once you know about the blockage
        self.reach_actual = np.isfinite(self.F_static["actual"]).any(axis=1)

    # =================================================================================================
    def run(self, cfg, seed, seats):
        """Play one evacuation. `seats` = {zone: number of people}. `seed` fixes the random numbers."""
        ca = self.P["ca"]
        grid = self.grid
        rng = np.random.default_rng(seed)
        dt = self.dt
        num_exits = len(EXITS)

        # ---------------------------------------------------------------- 1. put people in their seats
        pos_list, zone_list = [], []
        for zone_number, zone in enumerate(self.zone_names):
            chosen_cells = rng.choice(grid.zone_cells[zone], size=seats[zone], replace=False)
            pos_list.append(chosen_cells)
            zone_list.append(np.full(seats[zone], zone_number))
        pos = np.concatenate(pos_list)            # pos[person] = the cell they stand on
        zone = np.concatenate(zone_list)          # zone[person] = their seating zone
        num_people = len(pos)

        # ---------------------------------------------------------------- 2. give everyone a type
        mix = {}
        for name, share in cfg.routing.items():
            if share > 0:
                mix[name] = share
        type_names = list(mix)
        probabilities = np.array([mix[name] for name in type_names]) / sum(mix.values())
        drawn = rng.choice(len(type_names), size=num_people, p=probabilities)
        person_type = np.array([C.TYPE_OF[type_names[i]] for i in drawn])

        # 10 % of people follow the crowd's footprints strongly
        herd_draw = rng.random(num_people)
        footprint_pull = np.where(herd_draw < ca["herding_share"], ca["k_d_herd"], ca["k_d"])

        # everyone starts moving after the same fixed reaction time
        start = np.full(num_people, float(cfg.premovement["delay"]))

        # ---------------------------------------------------------------- 3. everybody's state
        has_block = bool(self.blocked)
        informed = np.full(num_people, (not has_block) or cfg.known)   # knows about the blockage?
        cm_informed = (not has_block) or cfg.known                     # do the Crowd Managers know?
        inside = np.ones(num_people, bool)
        has_exit = np.zeros(num_people, bool)        # has already picked an exit
        target = np.zeros(num_people, int)           # the exit they are heading for
        wait_until = np.zeros(num_people)            # paused (after finding a blocked door) until then
        last_rethink = np.zeros(num_people)
        exit_time = np.full(num_people, np.nan)
        exit_id = -np.ones(num_people, int)

        occupied = np.zeros(grid.n, bool)
        occupied[pos] = True
        footprints = np.zeros(grid.n)
        occupied_seconds = np.zeros(grid.n)
        all_open = np.ones(num_exits, bool)
        open_for_real = self.reach_actual.copy()

        # the Crowd Managers' plan as a table [zone, exit]
        def plan_table(plan):
            table = np.zeros((len(self.zone_names), num_exits))
            if plan:
                for zone_number, zone_name in enumerate(self.zone_names):
                    for exit_name, share in plan.get(zone_name, {}).items():
                        table[zone_number, EXITS.index(exit_name)] = share
            return table

        plan_before = plan_table(cfg.guided_shares)
        plan_after = plan_table(cfg.guided_shares_informed or cfg.guided_shares)
        exit_capacity = np.array([(cfg.exit_capacity or {}).get(name, 2.0) for name in EXITS])

        # where the Crowd Managers stand
        cm_cells = []
        if cfg.crowd_managers:
            from bailey.geometry.schematic import polar
            for floor, radius, angle in ca["cm_positions"]:
                cm_cells.append(grid.nearest_cell(floor, *polar(radius, angle)))
        cm_cells = np.array(cm_cells, int)
        persuade_chance = 1 - np.exp(-ca["cm_switch_rate"] * dt)     # chance per turn within reach

        maps = {"actual": self.F_static["actual"], "plan": self.F_static["plan"]}
        next_map_update = 0.0
        snapshots = {}
        snapshot_times_left = sorted(cfg.snapshot_times)
        t = 0.0

        # ---------------------------------------------------------------- helper: choose exits
        def choose_exit(people):
            """Pick an exit for each person in `people`, using what they know."""
            if len(people) == 0:
                return
            knows = informed[people]
            for knows_blockage in (True, False):
                group = people[knows == knows_blockage]
                if len(group) == 0:
                    continue
                if knows_blockage:
                    which_map, allowed = "actual", open_for_real
                else:
                    which_map, allowed = "plan", all_open
                walk_time = self.F_static[which_map]
                cells = pos[group]
                nearest = C.nearest_exit(walk_time, cells, allowed)
                kind = person_type[group]
                choice = nearest.copy()                                   # NEAREST people

                is_habit = kind == C.HABIT
                choice[is_habit] = C.habit_exit(walk_time, cells[is_habit], allowed)

                is_guided = kind == C.GUIDED
                if is_guided.any():
                    if knows_blockage and has_block:
                        table = plan_after
                    else:
                        table = plan_before
                    choice[is_guided] = C.guided_exit(rng, table, zone[group][is_guided], allowed, nearest[is_guided])

                is_rational = kind == C.RATIONAL
                if is_rational.any():
                    crowd_map = maps[which_map]
                    ahead = C.people_ahead(crowd_map, pos, target, inside & has_exit, cells[is_rational])
                    cost = C.rational_cost(crowd_map, cells[is_rational], ahead, exit_capacity, allowed)
                    choice[is_rational] = np.argmin(cost, axis=0)
                target[group] = choice
            has_exit[people] = True

        # =============================================================================== THE TURNS
        while inside.any() and t <= ca["horizon"]:

            # save positions for the snapshot figure
            while snapshot_times_left and t >= snapshot_times_left[0]:
                snapshots[snapshot_times_left.pop(0)] = (pos[inside].copy(), person_type[inside].copy())

            # ---- 1. redraw the crowd-aware "time to exit" maps every few seconds ----
            if t >= next_map_update:
                beta = ca["congestion_beta"]
                maps["actual"] = self.fs_actual.solve(occupied, beta)
                if has_block and not informed.all():
                    maps["plan"] = self.fs_plan.solve(occupied, beta)
                else:
                    maps["plan"] = maps["actual"]
                next_map_update = t + ca["field_refresh"]

            ready = inside & (t >= start)

            # ---- 2. people who just started moving choose an exit ----
            choose_exit(np.where(ready & ~has_exit)[0])

            # ---- 3. rational people reconsider every few seconds ----
            rethink = np.where(ready & (person_type == C.RATIONAL) & (t - last_rethink >= ca["rational_reeval"]))[0]
            if len(rethink):
                last_rethink[rethink] = t
                for knows_blockage in (True, False):
                    group = rethink[informed[rethink] == knows_blockage]
                    if len(group) == 0:
                        continue
                    which_map = "actual" if knows_blockage else "plan"
                    allowed = open_for_real if knows_blockage else all_open
                    ahead = C.people_ahead(maps[which_map], pos, target, ready & has_exit, pos[group])
                    cost = C.rational_cost(maps[which_map], pos[group], ahead, exit_capacity, allowed)
                    best = np.argmin(cost, axis=0)
                    for i, person in enumerate(group):
                        # switch only if it saves at least 5 s
                        if cost[best[i], i] < cost[target[person], i] - ca["rational_switch_gain"]:
                            target[person] = best[i]

            # ---- 4. Crowd Managers ----
            if len(cm_cells):
                cm_knows = cm_informed or (t >= ca["cm_radio_delay"])
                candidates = np.where(ready)[0]
                if len(candidates):
                    # distance from each candidate to the closest Crowd Manager on the same floor
                    distance = np.full(len(candidates), np.inf)
                    for cm_cell in cm_cells:
                        same_floor = grid.layer[pos[candidates]] == grid.layer[cm_cell]
                        d = np.hypot(grid.x[pos[candidates]] - grid.x[cm_cell], grid.y[pos[candidates]] - grid.y[cm_cell])
                        distance = np.where(same_floor, np.minimum(distance, d), distance)
                    in_reach = candidates[np.nan_to_num(distance, nan=np.inf) <= ca["cm_radius"]]
                    # they pass on news about a blocked door
                    if cm_knows and has_block:
                        newly = in_reach[~informed[in_reach]]
                        informed[newly] = True
                        choose_exit(newly)
                    # and persuade people to follow their plan
                    roll = rng.random(len(in_reach))
                    persuaded = in_reach[(person_type[in_reach] != C.GUIDED) & (roll < persuade_chance)]
                    if len(persuaded):
                        person_type[persuaded] = C.GUIDED
                        choose_exit(persuaded)

            # ---- 5. finding a blocked door, and word of mouth ----
            if has_block and not informed.all():
                unaware = np.where(ready & ~informed)[0]
                if len(unaware):
                    sees_it = np.zeros(len(unaware), bool)
                    for floor, door_x, door_y in self.block_centres:
                        same_floor = grid.layer[pos[unaware]] == floor
                        d = np.hypot(grid.x[pos[unaware]] - door_x, grid.y[pos[unaware]] - door_y)
                        sees_it |= same_floor & (np.nan_to_num(d, nan=np.inf) <= ca["see_radius"])
                    found = unaware[sees_it]
                    informed[found] = True
                    wait_until[found] = t + self.P["blocked_discovery_delay"]   # pause to take it in
                    choose_exit(found)

                    others = unaware[~sees_it]
                    if len(others):
                        # cells where someone who knows is standing
                        knowing_cell = np.zeros(grid.n, bool)
                        knowing_cell[pos[inside & informed]] = True
                        their_neighbours = self.neigh[pos[others]]     # neighbours of the cells they stand on
                        next_to_knower = (knowing_cell[np.maximum(their_neighbours, 0)] & (their_neighbours >= 0)).any(axis=1)
                        roll = rng.random(len(others))
                        hears = others[next_to_knower & (roll < ca["info_spread"])]
                        informed[hears] = True
                        choose_exit(hears)

            # ---- 6. MOVE ----
            movers = np.where(ready & (t >= wait_until))[0]
            if len(movers):
                stepped = self._move(movers, pos, target, informed, maps, footprints, footprint_pull,
                                     occupied, inside, exit_time, exit_id, rng, t)

                # ---- 7. two stuck people facing each other swap places ----
                self._swap(movers, stepped, pos, target, informed, maps, inside, rng)

            # ---- 8. footprints fade; count how long each cell was occupied ----
            footprints *= (1.0 - ca["dyn_decay"])
            occupied_seconds += occupied * dt
            t += dt

        if inside.any():
            T = float("inf")
        else:
            T = float(np.nanmax(exit_time))
        return CAResult(T=T, exit_time=exit_time, exit_id=exit_id, agent_type=person_type, agent_zone=zone,
                        start_time=start, dt=dt, snapshots=snapshots, occupancy_time=occupied_seconds,
                        remaining=int(inside.sum()))

    # =================================================================================================
    def _move(self, movers, pos, target, informed, maps, footprints, footprint_pull, occupied, inside,
              exit_time, exit_id, rng, t):
        """Step 6 for everybody in `movers`. Returns the people who actually stepped."""
        ca = self.P["ca"]
        grid = self.grid
        dt = self.dt
        pull_to_exit = ca["k_s"]
        bump_chance = ca["friction"]

        # each mover uses the map of the building as they believe it is: 0 = actual, 1 = plan
        both_maps = np.stack([maps["actual"], maps["plan"]])
        which_map = np.where(informed[movers], 0, 1)
        cells = pos[movers]
        neighbours = self.neigh[cells]                     # [mover, neighbour slot], -1 = none
        neighbour_cell = np.maximum(neighbours, 0)
        free = (neighbours >= 0) & ~occupied[neighbour_cell]

        # a. score of each option: exp(- pull x (time gained) + footprint bonus)
        time_here = both_maps[which_map, target[movers], cells]
        time_there = both_maps[which_map[:, None], target[movers][:, None], neighbour_cell]
        with np.errstate(invalid="ignore", over="ignore"):
            score = -pull_to_exit * (time_there - time_here[:, None]) / dt + footprint_pull[movers, None] * footprints[neighbour_cell]
        score = np.where(free & np.isfinite(score), score, -np.inf)
        stay_score = footprint_pull[movers] * footprints[cells]
        all_scores = np.column_stack([stay_score, score])     # column 0 = stay
        all_scores -= all_scores.max(axis=1, keepdims=True)   # (keeps the numbers small)
        weights = np.exp(all_scores)
        weights /= weights.sum(axis=1, keepdims=True)

        # b. pick one option per mover at random, in proportion to its weight
        roll = rng.random(len(movers))
        running_total = np.cumsum(weights, axis=1)
        pick = (roll[:, None] > running_total).sum(axis=1)
        pick = np.minimum(pick, weights.shape[1] - 1)
        wants_to_step = pick > 0
        slot = np.maximum(pick - 1, 0)
        destination = np.where(wants_to_step, neighbours[np.arange(len(movers)), slot], -1)

        # c. slow cells and long (diagonal) steps succeed less often
        step_length = self.nlen[cells, slot]
        speed = 0.5 * (grid.speed[cells] + grid.speed[np.maximum(destination, 0)])
        success_roll = rng.random(len(movers))
        succeeds = wants_to_step & (success_roll < np.minimum(1.0, speed * grid.h / step_length))
        who = movers[succeeds]
        destination = destination[succeeds]

        if len(who):
            # d. bumping: shuffle the movers, the first one listed for each cell is the candidate winner
            order = rng.permutation(len(who))
            who = who[order]
            destination = destination[order]
            _, first_index, how_many = np.unique(destination, return_index=True, return_counts=True)
            allowed_to_step = np.zeros(len(who), bool)
            alone = how_many == 1
            allowed_to_step[first_index[alone]] = True                  # nobody else wanted that cell
            contested = first_index[~alone]
            bump_roll = rng.random(len(contested))
            allowed_to_step[contested[bump_roll >= bump_chance]] = True  # 33 %: one gets through
            who = who[allowed_to_step]
            destination = destination[allowed_to_step]

            # carry out the steps
            old_cells = pos[who]
            occupied[old_cells] = False
            footprints[old_cells] += 1.0
            pos[who] = destination

            # e. people who stepped onto an exit are out
            exit_number = self.exit_of_cell[destination]
            got_out = exit_number >= 0
            inside[who[got_out]] = False
            exit_time[who[got_out]] = t + dt
            exit_id[who[got_out]] = exit_number[got_out]
            occupied[destination[~got_out]] = True
        return who

    def _swap(self, movers, stepped, pos, target, informed, maps, inside, rng):
        """Step 7: two people who could not move and face each other swap cells (50 % chance)."""
        swap_chance = self.P["ca"]["swap_prob"]
        moved = np.zeros(len(pos), bool)
        moved[stepped] = True
        stuck = movers[~moved[movers] & inside[movers]]
        if len(stuck) <= 1 or swap_chance <= 0:
            return

        # the cell each stuck person would most like to step to (even if it is occupied)
        both_maps = np.stack([maps["actual"], maps["plan"]])
        cells = pos[stuck]
        neighbours = self.neigh[cells]
        which_map = np.where(informed[stuck], 0, 1)
        time_there = both_maps[which_map[:, None], target[stuck][:, None], np.maximum(neighbours, 0)]
        time_there = np.where(neighbours >= 0, time_there, np.inf)
        favourite = neighbours[np.arange(len(stuck)), np.argmin(time_there, axis=1)]

        # who is standing on that favourite cell (among the stuck people)?
        stuck_person_at = -np.ones(self.grid.n, int)
        stuck_person_at[cells] = np.arange(len(stuck))
        other = stuck_person_at[np.maximum(favourite, 0)]
        has_other = (favourite >= 0) & (other >= 0)
        first = np.where(has_other)[0]
        second = other[has_other]
        # they swap only if each wants the other's cell (each pair counted once)
        face_each_other = (favourite[second] == cells[first]) & (first < second)
        first = first[face_each_other]
        second = second[face_each_other]
        roll = rng.random(len(first))
        go = roll < swap_chance
        person_a = stuck[first[go]]
        person_b = stuck[second[go]]
        pos[person_a], pos[person_b] = pos[person_b].copy(), pos[person_a].copy()
