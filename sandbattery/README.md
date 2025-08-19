sand heat storage toy sim

- goal: crude energy budget for a sand-based seasonal store; choose air or water heat extraction based on safe temps.
- stack: python + pint for unit safety.

run

- baseline winter sim (prints daily snippets and totals): `python sandbattery/sand_battery.py`
- small activation test (water vs air switching): `python sandbattery/test_sand_battery.py`

model sketch

- mass: volume × density; energy: m·c·ΔT, tracked in kwh.
- controllers: 
  - if sand > 90°C, use AIR up to 300°C; 
  - if 60–90°C, use WATER; 
  - if 50–60°C, AIR_LOW with floor efficiency.
- daily demand is fixed; effective delivery scales with temp gradient proxy.

caveats

- zero spatial resolution, no heat exchanger modeling, no losses, no pump/blower limits. it's a sanity check, not a design tool.
- change `test_sand_volume`, efficiencies, and thresholds to explore regimes.
