# dnd-ish stat viz (julia)

- goal: scatter and trivial glm over a csv of character stats vs outcome.
- stack: julia + GLMakie, DataFrames, CSV, GLM, StatsModels, Statistics.

## run

- from repo root: `julia viz.jl` (expects `data/d_and_d_sci.csv`).

## notes

- the script currently builds a model `res ~ cha + con + dex + int + str + wis` after mapping `result` to 0/1.
- glmakie will pop an interactive window; ensure you have an opengl-capable backend.
