PY ?= python

.PHONY: building test model1 network baseline blocked figures

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
