# import pakcage for xlsx files
library(readxl)

# import package for data manipulation
library(dplyr)
library(tidyr)
# import package for data visualization
library(ggplot2)
library(ggimage)

# install ggthemes
install.packages("ggimage",dependencies=TRUE)

# import package for data visualization

library(ggthemes)
library(ggExtra)
# read in xlsx from the data folder
cyber <- read_xlsx("data/data-cyber.xlsx")

# cyber table columns
colnames(cyber)


unique(cyber$initiator_category2)

# table for cateogiy mappings where key is new category and value is list of old categories

cyber %>%
 mutate(initiator_category2 = case_when(
    grepl("state-affiliation suggested", initiator_category) ~ "state-affiliation",
    grepl("State", initiator_category) ~ "state",
    grepl("Non-state-group", initiator_category) ~ "non-state",
    grepl("Individual hacker", initiator_category) ~ "individuals",
    TRUE ~ "other"
  )) |>
  select(initiator_category, initiator_category2) |>
  unique()

cyber <- cyber %>%
 mutate(initiator_category2 = case_when(
    grepl("state-affiliation suggested", initiator_category) ~ "state-affiliation",
    grepl("State", initiator_category) ~ "state",
    grepl("Non-state-group", initiator_category) ~ "non-state",
    grepl("Individual hacker", initiator_category) ~ "individuals",
    TRUE ~ "other"
  ))

# get top 4 countries in column initiator_country, skipping "not available" and "unknown"
top_countries <- cyber %>%
  filter(initiator_country != "Not available" & initiator_country != "Unknown") %>%
  count(initiator_country, sort = TRUE) %>%
  head(4)

# replace the country names with the top 4 countries or "other"
cyber <- cyber %>%
  mutate(initiator_country2 = case_when(
    initiator_country %in% top_countries$initiator_country ~ as.character(initiator_country),
    TRUE ~ "other"
  ))


# make sure ggplot figure is 16 by 9, with readable size text
# options(repr.plot.width = 16, repr.plot.height = 9, repr.plot.res = 100)

# using column initiator_category as a grouping variable and initiator_country as column, plot the number of cyber attacks as a stacked bar chart
cyber %>% 
  ggplot(aes(y = initiator_category2)) +
  geom_bar(aes(fill = initiator_country2)) + 
  # nice theme
    theme_minimal() +
    # add title
    ggtitle("EU andme-küberrünnakud riigiti")

# bar chart of cyber attacks by country top 10, colored
cyber %>%
  filter(initiator_country != "Not available" & initiator_country != "Unknown" & initiator_country != "Unknown; Unknown") %>%
  count(initiator_country, sort = TRUE) %>%
  head(5) %>%
  ggplot(aes(x = reorder(initiator_country, n), y = n, fill = initiator_country)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  ggtitle("EU andme-küberrünnakud riigiti") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  xlab("Riik") +
  ylab("Küberrünnakute arv") +
  scale_fill_tableau()


library(ggplot2)
library(dplyr)
library(ggimage)

# Assuming cyber is your dataset
cyber %>%
  filter(initiator_country != "Not available" & initiator_country != "Unknown" & initiator_country != "Unknown; Unknown") %>%
  count(initiator_country, sort = TRUE) %>%
  head(5) %>%
  mutate(flag_url = case_when(
    initiator_country == "Russia" ~ "https://upload.wikimedia.org/wikipedia/en/f/f3/Flag_of_Russia.svg",
    initiator_country == "China" ~ "https://upload.wikimedia.org/wikipedia/commons/f/fa/Flag_of_the_People%27s_Republic_of_China.svg",
    initiator_country == "Iran, Islamic Republic of" ~ "https://upload.wikimedia.org/wikipedia/commons/c/ca/Flag_of_Iran.svg",
    initiator_country == "Korea, Democratic People's Republic of" ~ "https://upload.wikimedia.org/wikipedia/commons/5/51/Flag_of_North_Korea.svg",
    TRUE ~ NA_character_ # Fallback in case of unknown country
  )) %>%
  ggplot(aes(x = reorder(initiator_country, n), y = n)) +
  geom_image(aes(image = flag_url), size = 0.05) + # Adjust size as needed
  theme_minimal() +
  ggtitle("EU andme-küberrünnakud riigiti") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  xlab("Riik") +
  ylab("Küberrünnakute arv")
