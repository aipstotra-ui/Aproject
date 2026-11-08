"""Model II, step 3: how each person picks an exit.

There are four kinds of people (the same idea as Model I):
  HABIT     go back the way you came in: the main foyer exit (unless you know it can't be reached)
  NEAREST   the exit with the shortest walk (ignores queues)
  GUIDED    the exit a Crowd Manager tells you (drawn from the plan for your seating zone)
  RATIONAL  the exit with the smallest (walking time + waiting time), where
            waiting time = people ahead of you going there / how many people per second it lets out.
            Rational people only count people who are already moving (they can't see who is still seated).

Each function gets a group of people at once (their cells) and returns one exit number per person.
Exit numbers are positions in grid.EXITS (0 = OUT_FRONT).
"""
import numpy as np

HABIT, NEAREST, GUIDED, RATIONAL = 0, 1, 2, 3
TYPE_OF = {"habit": HABIT, "nearest": NEAREST, "assigned": GUIDED, "plan": GUIDED, "ue": RATIONAL}
TYPE_NAMES = {HABIT: "habit", NEAREST: "nearest", GUIDED: "guided", RATIONAL: "rational"}
FRONT = 0  # OUT_FRONT is the first exit in grid.EXITS


def nearest_exit(walk_time, cells, allowed):
    """walk_time[exit, cell] = seconds to that exit; allowed[exit] = True if the person thinks it is open."""
    times = walk_time[:, cells].copy()          # [exit, person]
    for exit_number in range(len(allowed)):
        if not allowed[exit_number]:
            times[exit_number] = np.inf          # ignore exits believed closed
    return np.argmin(times, axis=0)              # quickest exit for each person


def habit_exit(walk_time, cells, allowed):
    """The foyer if it is open and reachable from the person's cell, otherwise the nearest exit."""
    nearest = nearest_exit(walk_time, cells, allowed)
    choice = nearest.copy()
    for i, cell in enumerate(cells):
        if allowed[FRONT] and np.isfinite(walk_time[FRONT, cell]):
            choice[i] = FRONT
    return choice


def guided_exit(rng, shares, zones, allowed, fallback):
    """Draw an exit from the Crowd Managers' plan.
    shares[zone, exit] = share of that zone the plan sends to that exit. Exits believed closed are
    dropped and the rest re-scaled to add up to 1. If nothing is left, use `fallback`."""
    zone_shares = shares[zones] * allowed[None, :]      # [person, exit]
    totals = zone_shares.sum(axis=1)
    choice = fallback.copy()
    has_plan = totals > 1e-12
    if has_plan.any():
        # one random number per person with a plan, compared with the running total of the shares
        probabilities = zone_shares[has_plan] / totals[has_plan, None]
        running_total = np.cumsum(probabilities, axis=1)
        random_numbers = rng.random(has_plan.sum())
        picked = []
        for person in range(len(random_numbers)):
            exit_number = 0
            while exit_number < len(allowed) - 1 and random_numbers[person] > running_total[person, exit_number]:
                exit_number += 1
            picked.append(exit_number)
        choice[has_plan] = picked
    return choice


def rational_cost(walk_time, cells, ahead, capacity, allowed):
    """Predicted time to get out through each exit: [exit, person]."""
    cost = walk_time[:, cells] + ahead / capacity[:, None]
    for exit_number in range(len(allowed)):
        if not allowed[exit_number]:
            cost[exit_number] = np.inf
    return cost


def people_ahead(walk_time, pos, target, moving, cells):
    """For each exit and each person (standing on `cells`): how many moving people are heading to
    that exit AND are closer to it than this person."""
    num_exits = walk_time.shape[0]
    counts = np.zeros((num_exits, len(cells)))
    for exit_number in range(num_exits):
        # distances of everyone already heading there, sorted from close to far
        others = np.sort(walk_time[exit_number, pos[moving & (target == exit_number)]])
        # for each person: how many of those are closer than they are
        counts[exit_number] = np.searchsorted(others, walk_time[exit_number, cells])
    return counts
