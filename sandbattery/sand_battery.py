import pint

# Initialize unit registry
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

print("=== CORRECTED SIMULATION: AIR FIRST, THEN WATER ===")

# Physical properties with units
sand_density = Q_(1600, 'kg/m^3')
sand_specific_heat = Q_(0.835, 'kJ/kg/K')
initial_sand_temp = Q_(600, 'degC')
min_usable_temp = Q_(50, 'degC')
winter_days = 90
daily_heat_demand = Q_(85.6, 'kWh')  # Daily energy demand in kWh per day

# System temperature limits (SAFETY CRITICAL)
water_max_safe_temp = Q_(90, 'degC')
air_max_safe_temp = Q_(300, 'degC')
water_min_effective_temp = Q_(60, 'degC')

# Start with base sand volume and check if sufficient
test_sand_volume = Q_(45, 'm^3')
test_sand_mass = test_sand_volume * sand_density

print(f"Testing with {test_sand_volume:.0f} ({test_sand_mass.to('tonne'):.1f})")

# Check total energy available
total_temp_drop = initial_sand_temp - min_usable_temp
total_energy_stored = (test_sand_mass * sand_specific_heat * total_temp_drop).to('kWh')
total_winter_demand = daily_heat_demand * winter_days

print(f"Total energy stored: {total_energy_stored:.0f}")
print(f"Total winter demand: {total_winter_demand:.0f}")
print(f"Storage ratio: {(total_energy_stored / total_winter_demand):.2f}")

# Simulation with correct system priority
current_sand_temp = initial_sand_temp
cumulative_energy_extracted = Q_(0, 'kWh')
prev_system = "none"

print("\n=== SAFE TEMPERATURE-BASED EXTRACTION SIMULATION ===")
print("Day | Sand°C | System | Efficiency | Energy kWh | Cumulative")
print("----+--------+--------+------------+------------+-----------")

for day in range(1, winter_days + 1):
    system_type = "none"
    system_efficiency = 0
    energy_extracted_today = Q_(0, 'kWh')
    
    # System selection based on SAFE temperature ranges
    if current_sand_temp > water_max_safe_temp:
        # HIGH TEMPERATURE: Use AIR system only (SAFE)
        system_type = "air"
        max_extractable_temp = min(current_sand_temp, air_max_safe_temp)
        temp_gradient = (max_extractable_temp - Q_(20, 'degC')).magnitude
        max_gradient = (air_max_safe_temp - Q_(20, 'degC')).magnitude
        system_efficiency = min(temp_gradient / max_gradient, 1.0)
        
    elif current_sand_temp >= water_min_effective_temp:
        # MEDIUM TEMPERATURE: Use WATER system (SAFE and EFFICIENT)
        system_type = "water"
        temp_gradient = (current_sand_temp - Q_(40, 'degC')).magnitude
        max_gradient = (water_max_safe_temp - Q_(40, 'degC')).magnitude
        system_efficiency = min(temp_gradient / max_gradient, 1.0)
        
    elif current_sand_temp >= min_usable_temp:
        # LOW TEMPERATURE: Use AIR system (less efficient but still works)
        system_type = "air_low"
        temp_gradient = (current_sand_temp - Q_(20, 'degC')).magnitude
        max_gradient = (Q_(100, 'degC') - Q_(20, 'degC')).magnitude
        system_efficiency = max(temp_gradient / max_gradient, 0.2)
    
    # Extract energy based on system efficiency
    if system_efficiency > 0:
        # Extract energy in kWh 
        energy_extracted_today = daily_heat_demand * system_efficiency
        
        # Update sand temperature
        temp_drop = (energy_extracted_today.to('J') / (test_sand_mass * sand_specific_heat.to('J/kg/K'))).to('K')
        current_sand_temp -= temp_drop
        
        cumulative_energy_extracted += energy_extracted_today
    
    # Print key days and system transitions
    if (day % 15 == 0 or day <= 5 or day >= 85 or 
        (system_type == "water" and day > 1) or
        (day > 1 and prev_system != system_type)):  # Print on system change
        # Convert temperature properly (same method as test file)
        temp_display = current_sand_temp.to('kelvin').magnitude - 273.15
        print(f"{day:3d} | {temp_display:6.0f} | {system_type:>6s} | "
              f"{system_efficiency:10.2f} | {energy_extracted_today.magnitude:10.1f} | "
              f"{cumulative_energy_extracted.magnitude:9.0f}")
    
    prev_system = system_type
    
    # Check for system failure
    if current_sand_temp < min_usable_temp and day < winter_days:
        print(f"*** SYSTEM FAILURE on day {day} - insufficient temperature! ***")
        break

print(f"\nFinal results:")
final_temp_celsius = current_sand_temp.to('kelvin').magnitude - 273.15
print(f"Final sand temperature: {final_temp_celsius:.0f} °C")
print(f"Energy extracted: {cumulative_energy_extracted:.0f}")
print(f"Energy required: {total_winter_demand:.0f}")
performance_ratio = cumulative_energy_extracted / total_winter_demand 
print(f"System performance: {(performance_ratio * 100).magnitude:.1f}%")
