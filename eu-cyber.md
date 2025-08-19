# eu cyber incidents plots

- goal: quick and dirty plots summarizing initiator categories and top countries.
- stack: r (readxl, dplyr, tidyr, ggplot2, ggthemes, ggimage).

## run

- `Rscript EU-cyber.r` from repo root; it reads `data/data-cyber.xlsx`.

## notes

- the script calls `install.packages("ggimage", dependencies=TRUE)` inline; remove if you manage deps separately.
- network-embedded flag svgs are hot-linked from wikipedia; if offline, those plots won't render images.
