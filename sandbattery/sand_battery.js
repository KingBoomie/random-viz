// Corrected simulation: AIR FIRST (high temp), then WATER (lower temp)
// Water systems are limited to ~90°C maximum for residential safety

console.log("=== CORRECTED SIMULATION: AIR FIRST, THEN WATER ===");

const sandDensity = 1600; // kg/m³
const sandSpecificHeat = 0.835; // kJ/kg·K
const initialSandTemp = 600; // °C
const minUsableTemp = 50; // °C (lower minimum since we have both systems)
const winterDays = 90;
const dailyHeatDemand = 85.6; // kWh/day

// System temperature limits (SAFETY CRITICAL)
const waterMaxSafeTemp = 90; // °C (safe for residential water systems)
const airMaxSafeTemp = 300; // °C (air can handle much higher temps)
const waterMinEffectiveTemp = 60; // °C (minimum for useful water heating)

// Start with base sand volume and check if sufficient
let testSandVolume = 45; // m³ (starting estimate)
let testSandMass = testSandVolume * sandDensity;

console.log(`Testing with ${testSandVolume} m³ sand (${(testSandMass/1000).toFixed(1)} tonnes)`);

// Check total energy available
const totalTempDrop = initialSandTemp - minUsableTemp;
const totalEnergyStored = (testSandMass * sandSpecificHeat * totalTempDrop) / 3600;
const totalWinterDemand = dailyHeatDemand * winterDays;

console.log(`Total energy stored: ${totalEnergyStored.toFixed(0)} kWh`);
console.log(`Total winter demand: ${totalWinterDemand.toFixed(0)} kWh`);
console.log(`Storage ratio: ${(totalEnergyStored / totalWinterDemand).toFixed(2)}`);

// Simulation with correct system priority
let currentSandTemp = initialSandTemp;
let cumulativeEnergyExtracted = 0;

console.log("\n=== SAFE TEMPERATURE-BASED EXTRACTION SIMULATION ===");
console.log("Day | Sand°C | System | Efficiency | Energy kWh | Cumulative");
console.log("----+--------+--------+------------+------------+-----------");

for (let day = 1; day <= winterDays; day++) {
    let systemType = "none";
    let systemEfficiency = 0;
    let energyExtractedToday = 0;
    
    // System selection based on SAFE temperature ranges
    if (currentSandTemp > waterMaxSafeTemp) {
        // HIGH TEMPERATURE: Use AIR system only (SAFE)
        systemType = "air";
        const maxExtractableTemp = Math.min(currentSandTemp, airMaxSafeTemp);
        const tempGradient = maxExtractableTemp - 20; // air inlet temp
        const maxGradient = airMaxSafeTemp - 20;
        systemEfficiency = Math.min(tempGradient / maxGradient, 1.0);
        
    } else if (currentSandTemp >= waterMinEffectiveTemp) {
        // MEDIUM TEMPERATURE: Use WATER system (SAFE and EFFICIENT)
        systemType = "water";
        const tempGradient = currentSandTemp - 40; // water return temp
        const maxGradient = waterMaxSafeTemp - 40;
        systemEfficiency = Math.min(tempGradient / maxGradient, 1.0);
        
    } else if (currentSandTemp >= minUsableTemp) {
        // LOW TEMPERATURE: Use AIR system (less efficient but still works)
        systemType = "air_low";
        const tempGradient = currentSandTemp - 20;
        const maxGradient = 100 - 20; // lower temp air operation
        systemEfficiency = Math.max(tempGradient / maxGradient, 0.2); // minimum 20%
    }
    
    // Extract energy based on system efficiency
    if (systemEfficiency > 0) {
        energyExtractedToday = dailyHeatDemand * systemEfficiency;
        
        // Update sand temperature
        const tempDrop = (energyExtractedToday * 3600) / (testSandMass * sandSpecificHeat);
        currentSandTemp -= tempDrop;
        
        cumulativeEnergyExtracted += energyExtractedToday;
    }
    
    // Print key days
    if (day % 15 === 0 || day <= 5 || day >= 85 || systemType === "water" && day > 1 && systemType !== "water") {
        console.log(`${day.toString().padStart(3)} | ${currentSandTemp.toFixed(0).padStart(6)} | ${systemType.padStart(6)} | ${systemEfficiency.toFixed(2).padStart(10)} | ${energyExtractedToday.toFixed(1).padStart(10)} | ${cumulativeEnergyExtracted.toFixed(0).padStart(9)}`);
    }
    
    // Check for system failure
    if (currentSandTemp < minUsableTemp && day < winterDays) {
        console.log(`*** SYSTEM FAILURE on day ${day} - insufficient temperature! ***`);
        break;
    }
}

console.log(`\nFinal results:`);
console.log(`Final sand temperature: ${currentSandTemp.toFixed(0)}°C`);
console.log(`Energy extracted: ${cumulativeEnergyExtracted.toFixed(0)} kWh`);
console.log(`Energy required: ${totalWinterDemand.toFixed(0)} kWh`);
console.log(`System performance: ${(cumulativeEnergyExtracted / totalWinterDemand * 100).toFixed(1)}%`);