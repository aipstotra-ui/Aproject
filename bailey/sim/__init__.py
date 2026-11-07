"""Model II: floor-field cellular automaton.

grid.py       schematic -> 0.5 m cell graph (regions joined only through apertures named like Model I edges)
fields.py     congestion-aware time-to-go fields (multi-source Dijkstra)
engine.py     parallel-update CA with friction, exchange rule, Crowd Managers, obstruction discovery
calibrate.py  test room used to calibrate friction against the SFPE maximum specific flow
model2.py     scenario driver and Monte Carlo
"""
