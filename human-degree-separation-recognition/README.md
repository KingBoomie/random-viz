human angle perception analysis

- goal: analyze whether tiny deviations from 90° are detectable; look at accuracy vs reaction time, learning/fatigue, etc.
- stack: r (ggplot2, dplyr, tidyr, zoo, jsonlite).

run

- from this folder (paths assume repo root as wd):
  - `Rscript analyze.R` (reads `results-0.2deg.json`, writes `processed_angle_data.csv` and pngs).

outputs

- `plot1_accuracy_over_time.png`: accuracy vs reaction time with ci.
- `plot2_moving_average.png`: moving avg accuracy over time.
- `plot3_accuracy_by_angle.png`: accuracy vs angle, loess fit.
- `plot4_reaction_time.png`: rt over experiment duration.
- `plot5_combined_rt_accuracy.png`: dual-axis moving averages.
- `plot6_accuracy_by_deviation.png`: accuracy vs |angle-90°|.
- console prints summary stats and hypotheses sanity checks.

caveats

- the script assumes stimulus/response alternation order in the json; if your export schema changes, the pairing will break.
- no multiple-comparison corrections; treat plots as exploratory.
