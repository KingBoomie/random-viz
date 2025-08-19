# random-viz

this repo is a grab-bag of small sims and plots.

## projects

### rocket-simulation and viz (python + js)
jax/diffrax 3d rocket sim, parquet logging, and a tiny static web viz.

### human angle perception (r)
analysis/plots from an experiment on detecting sub-degree angular differences.

### sand heat storage (python)
back-of-the-envelope sand battery sim for seasonal heat storage.

### eu cyber plots (r)
simple quick analysis of EU cyber attacks data 

### dnd stat viz (julia): throwaway 3d scatter + glm on a d&d-ish dataset.

quickstart (very brief)

- python (uv/pip): see `pyproject.toml`. if you use uv: `uv sync` then run modules.
- r: install packages noted in each sub-readme; run the scripts from repo root so relative paths hit `data/`.
- julia: add `GLMakie`, `DataFrames`, `CSV`, `GLM`, `StatsModels` in your env; run `viz.jl` from repo root.

## docs

- rocket: `rocket/README.md`
- human angle: `human-degree-separation-recognition/README.md`
- sandbattery: `sandbattery/README.md`
- eu-cyber: `eu-cyber.md`
- dnd viz: `dnd-viz.md`
