import pint
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Setup Units and Physical Constants ---
ureg = pint.UnitRegistry()
ureg.formatter.default_format = '.2f~'

# Using values for dry sand.
SAND_DENSITY = 1600 * ureg.kg / ureg.m**3
SAND_SPECIFIC_HEAT = 830 * ureg.joule / (ureg.kg * ureg.delta_degC)

# --- Simulation Parameters ---
SAND_VOLUME = 32 * ureg.m**3
INITIAL_SAND_TEMP = ureg.Quantity(600, ureg.degC)
DAILY_HEAT_DEMAND_NET = 62 * ureg.kWh # Net energy delivered to the house per day
SIMULATION_DURATION = 90 * ureg.day

# --- System Model Assumptions ---
# Minimum temperature at which we can efficiently extract heat to make 100°C steam.
# This represents the boiling point of water in the heat exchanger.
BOILING_POINT_TEMP = ureg.Quantity(100, ureg.degC)

# The maximum power the heat exchanger is designed to deliver at the initial sand temperature.
# This is a design parameter of the system (e.g., size and number of pipes).
INITIAL_MAX_POWER = 200.0 * ureg.kW

def calculate_max_power(current_temp, initial_temp, boiling_temp, initial_max_power):
    """
    Calculates the maximum power (heat transfer rate) the system can deliver.
    Power is modeled as being directly proportional to the temperature difference (ΔT)
    between the sand and the boiling water in the heat exchanger.
    q = U * A * ΔT, so q is proportional to ΔT.
    """
    if current_temp <= boiling_temp:
        return 0.0 * ureg.kW

    # Calculate the temperature difference that drives heat transfer
    delta_t_current = (current_temp - boiling_temp).to('delta_degC')
    delta_t_initial = (initial_temp - boiling_temp).to('delta_degC')

    # Scale the power linearly with the change in ΔT
    current_max_power = initial_max_power * (delta_t_current / delta_t_initial)
    return current_max_power

def run_simulation():
    """
    Runs the day-by-day simulation of the sand battery based on power limits.
    """
    # --- Initial State Calculation ---
    sand_mass = SAND_VOLUME * SAND_DENSITY
    
    # The total energy stored above the boiling point temperature.
    initial_total_energy = sand_mass * SAND_SPECIFIC_HEAT * (INITIAL_SAND_TEMP - BOILING_POINT_TEMP)
    
    # The required average power to meet the daily energy demand
    required_power = (DAILY_HEAT_DEMAND_NET / (24 * ureg.hour)).to('kW')
    
    print("--- Initial System State ---")
    print(f"Sand Mass: {sand_mass.to('tonne')}")
    print(f"Initial Temperature: {INITIAL_SAND_TEMP}")
    print(f"Total Energy Stored (above 100°C): {initial_total_energy.to('MWh')}")
    print("-" * 30)
    print("--- System Requirements ---")
    print(f"Daily Net Energy Demand: {DAILY_HEAT_DEMAND_NET}")
    print(f"Required Average Power: {required_power}")
    print(f"Initial Max Power Capacity: {INITIAL_MAX_POWER}")
    print("-" * 30 + "\n")

    # --- Simulation Loop ---
    current_sand_temp = INITIAL_SAND_TEMP
    remaining_energy = initial_total_energy
    
    history = {
        'day': [],
        'temperature_C': [],
        'max_power_kW': [],
        'remaining_energy_MWh': []
    }

    print("--- Running 90-Day Winter Simulation ---")
    for day in tqdm(range(1, SIMULATION_DURATION.magnitude + 1), desc="Simulating Winter"):
        # 1. Calculate the system's current maximum power output
        current_max_power = calculate_max_power(current_sand_temp, INITIAL_SAND_TEMP, BOILING_POINT_TEMP, INITIAL_MAX_POWER)
        
        # 2. Check if the system can meet the required power demand
        if current_max_power < required_power:
            print(f"\n[!!] System Alert: On day {day}, max power capacity ({current_max_power:.2f~P}) dropped below required power ({required_power:.2f~P}).")
            print("System can no longer deliver heat fast enough.")
            break
            
        # 3. If power is sufficient, subtract the daily energy demand
        daily_energy_draw = DAILY_HEAT_DEMAND_NET.to('joule')
        remaining_energy -= daily_energy_draw
        
        # 4. Update sand temperature based on the new remaining energy
        delta_T_from_boiling = remaining_energy / (sand_mass * SAND_SPECIFIC_HEAT)
        current_sand_temp = BOILING_POINT_TEMP + delta_T_from_boiling
        
        # 5. Log data
        history['day'].append(day)
        history['temperature_C'].append(current_sand_temp.to('degC').magnitude)
        history['max_power_kW'].append(current_max_power.to('kW').magnitude)
        history['remaining_energy_MWh'].append(remaining_energy.to('MWh').magnitude)
        # Print results every 10 days, and on the last day or if simulation ends early
        if day % 10 == 0 or day == SIMULATION_DURATION.magnitude or current_max_power < required_power:
            print(f"Day {day}: Temp = {current_sand_temp.to('degC').magnitude:.2f} °C, Max Power = {current_max_power.to('kW').magnitude:.2f} kW, Remaining Energy = {remaining_energy.to('MWh').magnitude:.2f} MWh")
        
    print("--- Simulation Complete ---")
    return history, required_power.to('kW').magnitude

def plot_results(history, required_power_kw):
    """
    Plots the temperature and max power over the simulation duration.
    """
    if not history['day']:
        print("No data to plot.")
        return

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot Temperature
    color = 'tab:red'
    ax1.set_xlabel('Day of Winter')
    ax1.set_ylabel('Sand Temperature (°C)', color=color)
    ax1.plot(history['day'], history['temperature_C'], color=color, marker='o', linestyle='-', label='Sand Temperature')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot Power
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Max Power Capacity (kW)', color=color)
    ax2.plot(history['day'], history['max_power_kW'], color=color, marker='x', linestyle='--', label='Max Power Capacity')
    # Add a horizontal line for the required power
    ax2.axhline(y=required_power_kw, color='k', linestyle=':', linewidth=2, label=f'Required Power ({required_power_kw:.2f} kW)')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.title('Sand Battery Performance (Power-Limited Model)')
    fig.tight_layout()
    plt.show()
    plt.savefig('sand_battery_performance.png')
    
    # --- Final State Summary ---
    final_day = history['day'][-1]
    final_temp = history['temperature_C'][-1]
    final_energy = history['remaining_energy_MWh'][-1]
    
    print("\n--- Final System State ---")
    if final_day < SIMULATION_DURATION.magnitude:
        print(f"Battery became power-limited on Day {final_day}.")
    else:
        print("Battery successfully lasted the full 90-day winter period.")

    print(f"Final Sand Temperature: {final_temp:.2f} °C")
    print(f"Remaining Usable Energy: {final_energy:.2f} MWh")


if __name__ == '__main__':
    simulation_history, required_power = run_simulation()
    plot_results(simulation_history, required_power)

