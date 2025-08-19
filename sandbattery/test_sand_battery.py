import pint

# Test with lower initial temperature to see water system activation
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

print("=== TESTING WATER SYSTEM ACTIVATION ===")

# Physical properties with units
sand_density = Q_(1600, 'kg/m^3')
sand_specific_heat = Q_(0.835, 'kJ/kg/K')
initial_sand_temp = Q_(80, 'degC')  # Start below water max safe temp
min_usable_temp = Q_(50, 'degC')
winter_days = 30  # Shorter test
daily_heat_demand = Q_(85.6, 'kWh')

# System temperature limits (SAFETY CRITICAL)
water_max_safe_temp = Q_(90, 'degC')
air_max_safe_temp = Q_(300, 'degC')
water_min_effective_temp = Q_(60, 'degC')

# Test sand volume
test_sand_volume = Q_(45, 'm^3')
test_sand_mass = test_sand_volume * sand_density

print(f"Testing with {test_sand_volume:.0f} ({test_sand_mass.to('tonne'):.1f})")

# Simulation
current_sand_temp = initial_sand_temp
cumulative_energy_extracted = Q_(0, 'kWh')
prev_system = "none"

print("\nDay | Sand°C | System | Efficiency | Energy kWh")
print("----+--------+--------+------------+-----------")

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
        energy_extracted_today = daily_heat_demand * system_efficiency
        
        # Update sand temperature
        temp_drop = (energy_extracted_today.to('J') / (test_sand_mass * sand_specific_heat.to('J/kg/K'))).to('K')
        current_sand_temp -= temp_drop
        
        cumulative_energy_extracted += energy_extracted_today
    
    # Print every day for this test
    temp_display = current_sand_temp.to('kelvin').magnitude - 273.15
    print(f"{day:3d} | {temp_display:6.0f} | {system_type:>6s} | "
          f"{system_efficiency:10.2f} | {energy_extracted_today.magnitude:10.1f}")

print(f"\nFinal sand temperature: {current_sand_temp.to('kelvin').magnitude - 273.15:.0f} °C")
print(f"Total energy extracted: {cumulative_energy_extracted:.0f}")
