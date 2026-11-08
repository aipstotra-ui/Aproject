# Bailey Hall evacuation: a simple grid simulation

CMCM 2026, "Bail(ey)ing Out". How long does it take 1,329 people to leave Bailey Hall in an emergency?

We draw the hall as a grid of small squares and let every person take one step at a time.
The narrow doorways (the stairs, the mezzanine doors and the foyer doors) let people through at the rates we measured in our own bottleneck simulations.
Each step is partly random, so we repeat every evacuation 30 times.

We simulate the two **emergency** cases from our data:

| Case | Who | Reaction time | How people pick an exit |
|---|---|---|---|
| 3 | Students (lecture) | 5 s | 60% nearest exit, 40% habit (front foyer) |
| 4 | Concert audience | 10 s | 60% follow the ushers, 25% habit, 15% nearest exit |

## How to run

```bash
pip install -r requirements.txt
python main.py
```

It takes about 3–4 minutes. The results go into the `output/` folder.

## Files

| File | What it does |
|---|---|
| `simulation.py` | The model: the map, distances, people, choosing an exit, moving step by step, and the doorway limits from our data. |
| `main.py` | Runs each case 30 times, prints a summary and draws the figures. |
| `data/` | Our measured bottleneck data: `stairs.csv`, `corridor.csv`, `foyer_doors.csv`. |
| `output/` | `map.png`, `people_left.png`, `heatmap.png`, `flow_check.png`, `results.csv` |

## Our bottleneck data

Each data file holds 4 cases × 500 runs of a separate small simulation of one bottleneck:

- **Case 1:** students, no emergency
- **Case 2:** concert, no emergency
- **Case 3:** students, emergency
- **Case 4:** concert, emergency

We use cases 3 and 4.

| File | Where in the hall | Doorways in our map | Measured flow, case 3 | Measured flow, case 4 |
|---|---|---|---|---|
| `stairs.csv` | the stairs from the orchestra up to the mezzanine | the 6 openings in the barrier | 1.14 people/s | 0.99 people/s |
| `corridor.csv` | the exit from the mezzanine towards the foyer | the 9 mezzanine doors | 1.16 people/s | 0.90 people/s |
| `foyer_doors.csv` | the 3 doors from the corridor into the foyer | the 3 foyer doors | 5.83 people/s (all 3 together) | 5.04 people/s |

These are averages; every run uses its own row of the data (see rule 7).

## How the simulation works

### 1. The hall is a grid of 0.5 m × 0.5 m squares

One square holds one person. The hall is 50 m × 34 m, so the grid is 100 × 68 squares.
Each square is one of four things:

- **wall**
- **floor** (corridor, foyer, aisles)
- **seat**
- **exit**

The shapes are drawn at the hall's real size:

- orchestra seats in a half-circle of 13 m;
- a barrier with **6 stair openings** up to the mezzanine;
- mezzanine seats from 13.5 m to 19.5 m;
- a wall with **9 doors** into a 2.4 m curved corridor;
- **3 doors** into the foyer;
- **7 exits**:
  - **FRONT**: the 5 foyer doors;
  - **P1–P4**: side doors from the corridor;
  - **NW and NE**: doors next to the stage.

The orange squares on the map are the doorways that use our data.

![map](output/map.png)

### 2. Distance maps

For every exit, the computer counts how many steps each square is from it.
It starts at the exit and spreads out one step at a time, like a wave (a *breadth-first search*).
People can step in 8 directions, but never into a wall, and never diagonally between two walls.

### 3. People

- **Where they start:**
  - 527 people sit in random orchestra seats and 486 in random mezzanine seats.
  - The 316 balcony people come down 4 staircases. Each staircase is a waiting line, and people step off it at the measured stair rate.
- **Behaviour type:** each person gets one by a random draw, using the percentages for the case (see the table at the top).
- **Reaction time:** everyone starts moving after 5 s (students) or 10 s (concert). Balcony people need another 16 s to get down their stairs.

### 4. Choosing an exit

- **Habit:** FRONT.
- **Nearest:** the exit with the smallest distance.
- **Guided:** the ushers spread people out. One at a time, each guided person is sent to the exit where they would get out soonest:

  `walking distance + (people already sent there) ÷ (door width × 0.28 people per square per step)`

### 5. Moving, one time step at a time

One time step is **0.42 s**, the time it takes to walk one square at 1.19 m/s.
In each step, everyone who is ready does this:

**a. Score the options.**
The options are the free squares around you, plus staying where you are. Each one gets a weight:

```
weight = exp( −3 × (change in distance to my exit)  −  0.3 × (number of people around that square) )
```

- A step 1 square closer to your exit is about **20 times** more likely than a step that keeps the same distance.
- Squares surrounded by people are less attractive. This is how **congestion** works.

**b. Pick one option at random,** with chances in proportion to the weights. This is the **probability rule**.

**c. Resolve conflicts.**
If two or more people picked the same square, **one of them, chosen at random, gets it**, and the others wait.

**d. Leave the building.**
Anyone who steps onto an exit square is out, and we write down the time.

**e. Count crowding.**
We count how often every square is occupied. This gives the **density map**.

### 6. People see queues and pick a less busy doorway

Every 5 s, each doorway's queue is counted: the people within 4 m in front of it.
That doorway then counts as farther away, by the time you would wait in its queue:

`waiting steps = people in the queue ÷ (the doorway's measured rate × 0.42 s)`

The distance maps are redrawn with this extra cost.
Because squares now cost different amounts, the plain wave becomes **Dijkstra's method**: the same wave, but it always continues from the closest square found so far.

### 7. Doorways use our measured flow rates ("ticket machines")

Every stair opening, mezzanine door and foyer door has a ticket machine.

- **Printing tickets:** every time step, the machine prints `measured rate × 0.42 s` of a ticket. At most 1 ticket can wait.
- **Using tickets:** you may step **into** the doorway only if a ticket is ready, and stepping in uses it. With no ticket, you wait.
- **Where the rates come from:** run number *i* uses row *i* of each data file for the case. That way the run-to-run spread in our data carries into the simulation:
  - every stair opening gets the stair rate from `stairs.csv`;
  - every mezzanine door gets the rate from `corridor.csv`;
  - the foyer rate from `foyer_doors.csv` is split over the 3 doors in the same proportions as the data's door counts.

### 8. Two stuck people facing each other squeeze past

If two people can't move and each wants the other's square, they swap places half of the time.
Without this rule, two streams walking in opposite directions can block each other forever: this happened in one test run at the east end of the corridor.
Swaps never happen inside a doorway, so the ticket machines still decide who goes through.

### 9. Repeat 30 times per case

Each run uses a different random seed (0–29), and the same seed always gives exactly the same run.
The program also checks every step that no square ever holds two people.

## Results (30 runs per case)

| | Case 3: students, emergency | Case 4: concert, emergency |
|---|---|---|
| Average time until the last person is out | **191 s** | **243 s** |
| Spread between runs (standard deviation) | 6 s | 7 s |
| Fastest / slowest run | 180 s / 204 s | 231 s / 259 s |
| Runs within the 180 s limit | **1 of 30** | **0 of 30** |

Average number of people per exit:

| Case | FRONT | NW | NE | P1 | P2 | P3 | P4 |
|---|---|---|---|---|---|---|---|
| 3: students | 546 | 63 | 61 | 76 | 258 | 256 | 68 |
| 4: concert | 639 | 106 | 107 | 112 | 128 | 127 | 110 |

![people left](output/people_left.png)

**People still inside over time.**
The line is the average over the 30 runs; the band shows where 90% of the runs fall.

- **Students:** they start sooner and use the side exits more, so they finish about 50 s before the concert audience. They still miss 180 s in almost every run.
- **Concert:** the last 200 people take the longest, queueing for the foyer route.

![heat map](output/heatmap.png)

**Where people crowd.** The share of time each square was occupied, averaged over all runs.

- The dark red queues sit **in front of the mezzanine doors next to the foyer**, and in the corridor in front of the 3 foyer doors.
- **Concert:** 25% go back to the foyer by habit, and the ushers send many people there too. So the queues in front of the foyer-side mezzanine doors are the longest.
- **Students:** 60% head for the nearest exit, so there are also queues at the mezzanine doors near the P2 and P3 side exits.

![flow check](output/flow_check.png)

**Do the doorways stay within our data?**
For each kind of doorway the chart shows three bars:
- our measured flow;
- the busiest 20 seconds in the simulation;
- the simulation's average while the doorway was in use.

The simulation never goes above the measured rate, so the ticket machines work. It also never quite reaches it, for two reasons:
- **Supply:** the foyer doors only get the people that the mezzanine doors (about 1 per second each) and the balcony stairs pass on.
- **Room behind the door:** a doorway can only take the next person when the square behind it is free.

**Why it is slower than our earlier, simpler version (137 s for the concert):**
Our data says each mezzanine door and stair opening lets through only about 1 person per second.
In the full hall, people don't spread evenly over the 9 mezzanine doors. The doors closest to the foyer get about 150 people each, against about 125 in our data's setup, so they decide the time.
This points to where guidance (ushers or LED signs) could help most: spreading people over the mezzanine doors.

## Simplifications

- **Separate data:** the doorway rates come from separate small simulations of one bottleneck each. We use them as the most a doorway can let through.
- **Balcony stairs:** they also use the orchestra-stair rate. That's our assumption, because we have no data for them; set `USE_STAIR_DATA_FOR_BALCONY = False` to switch it off.
- **Doorways without data:** the side exits (P1–P4, NW, NE) and the stage-side openings have no data, so they have no ticket machine.
- **Map:** one floor, simplified shapes, door widths estimated from the CMCM drawings. The balcony itself isn't drawn.
- **Speed and reaction:** everyone walks at the same speed, and reaction time is fixed and the same for everyone.
- **Ushers:** modelled only by how they spread the guided 60%. They don't move.

All the settings are named constants at the top of `simulation.py`.
