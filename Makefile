PY ?= python

.PHONY: building test model1 network baseline blocked model2 calibrate crossval baseline2 blocked2 all

building:            ## regenerate data/building.json from bailey/geometry/schematic.py
	$(PY) -m bailey.geometry.schematic

test:
	$(PY) -m pytest -q

network:
	$(PY) scripts/run_model1.py network

baseline:            ## Q2/Q3 baselines + policy comparison (~40 s)
	$(PY) scripts/run_model1.py baseline

blocked:             ## every single obstruction, known vs unknown (~10-15 min)
	$(PY) scripts/run_model1.py blocked

model1: building network baseline blocked

calibrate:           ## Model II: friction vs SFPE door flow (~15 s)
	$(PY) scripts/run_model2.py calibrate

crossval:            ## Model I vs Model II on identical behaviour (~2 min)
	$(PY) scripts/run_model2.py crossval

baseline2:           ## Model II Monte Carlo baselines + figures (~1 min)
	$(PY) scripts/run_model2.py baseline

blocked2:            ## Model II key obstructions, needs results/model1_blocked.csv for the overlay (~5 min)
	$(PY) scripts/run_model2.py blocked

model2: calibrate crossval baseline2 blocked2

all: model1 model2
