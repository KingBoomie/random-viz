# Analysis of Angle Perception Experiment Data
# Loading required libraries
library(jsonlite)
library(dplyr)
library(ggplot2)
library(tidyr)
library(stringr)
library(zoo)  # for moving averages
# %%
# Load the JSON data
data_raw <- fromJSON("results-0.2deg.json")

# Extract stimulus trials and their corresponding response trials
stimulus_data <- data_raw[data_raw$task == "stimulus" & !is.na(data_raw$task), ]
response_data <- data_raw[data_raw$task == "response" & !is.na(data_raw$task), ]

# Extract angle values from stimulus filenames
stimulus_data$angle <- as.numeric(str_extract(stimulus_data$stimulus, "\\d+\\.\\d+"))

# Combine stimulus and response data
# Each stimulus should be followed by a response
combined_data <- data.frame(
  trial_index = stimulus_data$trial_index,
  angle = stimulus_data$angle,
  stimulus_time = stimulus_data$time_elapsed,
  response_time = response_data$time_elapsed[1:nrow(stimulus_data)],
  reaction_time = response_data$rt[1:nrow(stimulus_data)],
  response = response_data$response[1:nrow(stimulus_data)],
  correct_response = response_data$correct_response[1:nrow(stimulus_data)],
  correct = response_data$correct[1:nrow(stimulus_data)]
)

# Calculate time elapsed from start of experiment to response
combined_data$time_from_start <- combined_data$response_time

# Remove any rows with missing data
combined_data <- combined_data[complete.cases(combined_data), ]

print(paste("Total valid trials:", nrow(combined_data)))
print(paste("Angle range:", min(combined_data$angle, na.rm = TRUE), "to", max(combined_data$angle, na.rm = TRUE)))
print(paste("Overall accuracy:", round(mean(combined_data$correct, na.rm = TRUE) * 100, 2), "%"))

# %%

# 1. Main Analysis: Percent Correct Over Reaction Time
# Create reaction time bins for averaging
n_bins <- 20
combined_data$rt_bin <- cut(combined_data$reaction_time, 
                           breaks = n_bins, 
                           labels = FALSE)

# Calculate bin centers, accuracy, and confidence intervals
rt_summary <- combined_data %>%
  group_by(rt_bin) %>%
  summarise(
    reaction_time = mean(reaction_time, na.rm = TRUE),
    correct = mean(correct, na.rm = TRUE),
    n = n(),
    se = sqrt(correct * (1 - correct) / n),  # Standard error for proportion
    ci_lower = pmax(0, correct - 1.96 * se),  # 95% CI lower bound
    ci_upper = pmin(1, correct + 1.96 * se),  # 95% CI upper bound
    .groups = 'drop'
  ) %>%
  filter(!is.na(rt_bin))

rt_summary$percent_correct <- rt_summary$correct * 100
rt_summary$ci_lower_pct <- rt_summary$ci_lower * 100
rt_summary$ci_upper_pct <- rt_summary$ci_upper * 100

# Main plot: Percent Correct Over Reaction Time with Confidence Intervals
p1 <- ggplot(rt_summary, aes(x = reaction_time, y = percent_correct)) +
  geom_ribbon(aes(ymin = ci_lower_pct, ymax = ci_upper_pct), 
              alpha = 0.3, fill = "lightblue") +
  geom_line(size = 1.2, color = "blue") +
  geom_point(aes(size = n), color = "darkblue", alpha = 0.8) +
  geom_hline(yintercept = 50, linetype = "dashed", color = "red", size = 1) +
  scale_size_continuous(name = "Sample Size", range = c(2, 6)) +
  labs(title = "Accuracy vs Reaction Time in Angle Perception Task",
       subtitle = "Percent correct responses by reaction time (with 95% confidence intervals)",
       x = "Reaction Time (ms)",
       y = "Percent Correct (%)") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, hjust = 0.5),
        plot.subtitle = element_text(size = 12, hjust = 0.5)) +
  ylim(0, 100)

# Find intersection with 50% line
intersection_rt <- approx(x = rt_summary$percent_correct, 
                         y = rt_summary$reaction_time, 
                         xout = 50)$y
if (!is.na(intersection_rt)) {
  p1 <- p1 + geom_vline(xintercept = intersection_rt, 
                        linetype = "dashed", color = "red", size = 1)
}

# Add text annotations for sample sizes at the bottom
p1 <- p1 + geom_text(data = rt_summary, 
                     aes(x = reaction_time, y = 5, label = paste("n =", n)),
                     size = 3, angle = 45, hjust = 0, vjust = 0, color = "gray60")

print(p1)

# %%
# 2. Moving Average Analysis
# Calculate moving average for smoother trend
window_size <- max(3, round(nrow(combined_data) / 20))  # Adaptive window size
combined_data_ordered <- combined_data[order(combined_data$time_from_start), ]
combined_data_ordered$moving_avg_correct <- rollmean(as.numeric(combined_data_ordered$correct), 
                                                    k = window_size, 
                                                    fill = NA, 
                                                    align = "center")

p2 <- ggplot(combined_data_ordered, aes(x = time_from_start)) +
  geom_point(aes(y = correct * 100), alpha = 0.3, color = "lightblue") +
  geom_line(aes(y = moving_avg_correct * 100), size = 1.2, color = "darkblue") +
  geom_hline(yintercept = 50, linetype = "dashed", color = "red", size = 1) +
  labs(title = "Moving Average: Accuracy Over Time",
       subtitle = paste("Moving average with window size =", window_size),
       x = "Time from Start (ms)",
       y = "Percent Correct (%)") +
  theme_minimal() +
  ylim(0, 100)

print(p2)

# %%
# 3. Analysis by Angle Category
# Categorize angles relative to 90 degrees
combined_data$angle_category <- ifelse(combined_data$angle < 90, "Less than 90°",
                                     ifelse(combined_data$angle > 90, "Greater than 90°", "Equal to 90°"))

combined_data$angle_deviation <- abs(combined_data$angle - 90)

p3 <- ggplot(combined_data, aes(x = angle, y = as.numeric(correct))) +
  geom_point(alpha = 0.5, position = position_jitter(height = 0.02, width=0.02)) +
  geom_smooth(method = "loess", se = TRUE, color = "red") +
  labs(title = "Accuracy by Angle Value",
       subtitle = "Relationship between angle deviation from 90° and accuracy",
       x = "Angle (degrees)",
       y = "Accuracy (0 = incorrect, 1 = correct)") +
  theme_minimal()

print(p3)

# %%

# 4. Original Reaction Time Analysis
p4 <- ggplot(combined_data, aes(x = time_from_start, y = reaction_time)) +
  geom_point(alpha = 0.6, color = "darkgreen") +
  geom_smooth(method = "loess", se = TRUE, color = "red") +
  labs(title = "Reaction Time Over Experiment Duration",
       subtitle = "Changes in response speed throughout the experiment",
       x = "Time from Start (ms)",
       y = "Reaction Time (ms)") +
  theme_minimal()

print(p4)

# %%
# 5. Combined Reaction Time and Accuracy Over Time Analysis (Moving Average)
# Order data by time and calculate moving averages
window_size <- max(5, round(nrow(combined_data) / 20))  # Adaptive window size
combined_data_ordered <- combined_data[order(combined_data$time_from_start), ]

# Calculate moving averages
combined_data_ordered$moving_avg_rt <- rollmean(combined_data_ordered$reaction_time, 
                                               k = window_size, 
                                               fill = NA, 
                                               align = "center")
combined_data_ordered$moving_avg_accuracy <- rollmean(as.numeric(combined_data_ordered$correct), 
                                                     k = window_size, 
                                                     fill = NA, 
                                                     align = "center") * 100

# Create the combined plot with dual y-axes using moving averages
p5 <- ggplot(combined_data_ordered, aes(x = time_from_start)) +
  # Reaction time moving average
  geom_line(aes(y = moving_avg_rt), color = "darkgreen", size = 1.2, na.rm = TRUE) +
  # Accuracy moving average (scaled to match RT axis)
  geom_line(aes(y = moving_avg_accuracy * 20), color = "darkblue", size = 1.2, na.rm = TRUE) +
  # Add horizontal reference line for 50% accuracy
  geom_hline(yintercept = 50 * 20, linetype = "dashed", color = "red", alpha = 0.7) +
  # Primary y-axis for reaction time
  scale_y_continuous(
    name = "Reaction Time (ms)",
    sec.axis = sec_axis(~ . / 20, name = "Accuracy (%)")
  ) +
  labs(title = "Reaction Time and Accuracy Over Experiment Duration (Moving Average)",
       subtitle = paste("Green: Reaction Time, Blue: Accuracy, Red dashed: 50% accuracy, Window size =", window_size),
       x = "Time from Start (ms)") +
  theme_minimal() +
  theme(
    axis.title.y.left = element_text(color = "darkgreen"),
    axis.title.y.right = element_text(color = "darkblue"),
    axis.text.y.left = element_text(color = "darkgreen"),
    axis.text.y.right = element_text(color = "darkblue")
  )

print(p5)

# %%
# 6. Accuracy by Angle Deviation
accuracy_by_deviation <- combined_data %>%
  group_by(angle_deviation) %>%
  summarise(
    percent_correct = mean(correct, na.rm = TRUE) * 100,
    count = n(),
    .groups = 'drop'
  ) %>%
  filter(count >= 3)  # Only include deviations with sufficient data

p6 <- ggplot(accuracy_by_deviation, aes(x = angle_deviation, y = percent_correct)) +
  geom_point(size = 3, color = "purple") +
  geom_line(color = "purple", size = 1) +
  geom_hline(yintercept = 50, linetype = "dashed", color = "red") +
  labs(title = "Accuracy by Angle Deviation from 90°",
       subtitle = "Performance decreases as angle gets closer to 90°",
       x = "Absolute Deviation from 90° (degrees)",
       y = "Percent Correct (%)") +
  theme_minimal()

print(p6)

# 7. Learning Effect Analysis - Summary Table Only
# Divide data into early, middle, and late phases
combined_data$phase <- cut(1:nrow(combined_data), 
                          breaks = 3, 
                          labels = c("Early", "Middle", "Late"))

phase_summary <- combined_data %>%
  group_by(phase) %>%
  summarise(
    accuracy = mean(correct, na.rm = TRUE) * 100,
    reaction_time = mean(reaction_time, na.rm = TRUE),
    count = n(),
    .groups = 'drop'
  )

# Statistical Summary and Hypotheses
cat("\n=== STATISTICAL SUMMARY ===\n")
cat("Overall Statistics:\n")
cat(sprintf("- Total trials: %d\n", nrow(combined_data)))
cat(sprintf("- Overall accuracy: %.2f%%\n", mean(combined_data$correct, na.rm = TRUE) * 100))
cat(sprintf("- Mean reaction time: %.0f ms\n", mean(combined_data$reaction_time, na.rm = TRUE)))
cat(sprintf("- Angle range: %.1f° to %.1f°\n", min(combined_data$angle), max(combined_data$angle)))

cat("\nAccuracy by Experimental Phase:\n")
for(i in 1:nrow(phase_summary)) {
  cat(sprintf("- %s phase: %.2f%% (n=%d)\n", 
              phase_summary$phase[i], 
              phase_summary$accuracy[i], 
              phase_summary$count[i]))
}

cat("\n=== HYPOTHESES AND FINDINGS ===\n")
cat("1. FATIGUE HYPOTHESIS: Performance decreases over time due to fatigue/attention decline\n")
early_late_diff <- phase_summary$accuracy[3] - phase_summary$accuracy[1]
cat(sprintf("   - Finding: Accuracy changed by %.2f%% from early to late phase\n", early_late_diff))
if(early_late_diff < -5) {
  cat("   - SUPPORTED: Significant decline suggests fatigue effect\n")
} else if(early_late_diff > 5) {
  cat("   - CONTRADICTED: Improvement suggests learning effect\n")
} else {
  cat("   - NEUTRAL: Minimal change suggests stable performance\n")
}

cat("2. DIFFICULTY HYPOTHESIS: Angles closer to 90° are harder to discriminate\n")
near_90_accuracy <- mean(combined_data$correct[combined_data$angle_deviation <= 0.5], na.rm = TRUE) * 100
far_90_accuracy <- mean(combined_data$correct[combined_data$angle_deviation > 0.5], na.rm = TRUE) * 100
cat(sprintf("   - Near 90° (±0.5°) accuracy: %.2f%%\n", near_90_accuracy))
cat(sprintf("   - Far from 90° (>0.5°) accuracy: %.2f%%\n", far_90_accuracy))
if(far_90_accuracy > near_90_accuracy + 10) {
  cat("   - SUPPORTED: Clear difficulty increase near 90°\n")
} else {
  cat("   - WEAK SUPPORT: Minimal difference in difficulty\n")
}

cat("3. REACTION TIME HYPOTHESIS: Reaction times increase over time due to fatigue\n")
rt_correlation <- cor(combined_data$time_from_start, combined_data$reaction_time, use = "complete.obs")
cat(sprintf("   - Correlation between time and RT: %.3f\n", rt_correlation))
if(rt_correlation > 0.2) {
  cat("   - SUPPORTED: Positive correlation suggests slowing over time\n")
} else if(rt_correlation < -0.2) {
  cat("   - CONTRADICTED: Negative correlation suggests speeding up\n")
} else {
  cat("   - NEUTRAL: Weak correlation suggests stable RT\n")
}

# Save processed data for further analysis
write.csv(combined_data, "processed_angle_data.csv", row.names = FALSE)
cat("\nProcessed data saved to 'processed_angle_data.csv'\n")

# Save plots as PNG files
png("plot1_accuracy_over_time.png", width = 800, height = 600)
print(p1)
dev.off()

png("plot2_moving_average.png", width = 800, height = 600)
print(p2)
dev.off()

png("plot3_accuracy_by_angle.png", width = 800, height = 600)
print(p3)
dev.off()

png("plot4_reaction_time.png", width = 800, height = 600)
print(p4)
dev.off()

png("plot5_combined_rt_accuracy.png", width = 800, height = 600)
print(p5)
dev.off()

png("plot6_accuracy_by_deviation.png", width = 800, height = 600)
print(p6)
dev.off()

cat("\nPlots saved as PNG files in the current directory\n")