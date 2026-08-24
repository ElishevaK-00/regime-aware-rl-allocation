import time
import os
import math
import random

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def render_simulation_frame(stages, tax_friction):
    # 1. Math Model: Compute Exponential Tree Nodes vs. Truncated Hybrid Steps
    # Classical MSP Nodes = Branching_Factor^Stages (Assuming a branch factor of 3)
    msp_nodes = int(math.pow(3, stages))
    # Hybrid RL nodes stay tightly bound because the Critic truncates the horizon
    hybrid_nodes = stages * 3 

    # 2. Map Constraint Frictions to Allocation Vectors
    # Higher taxes "lock" capital, causing classical systems to miss rebalancing targets
    classical_efficiency = max(10, int(100 - (tax_friction * 180)))
    hybrid_efficiency = max(45, int(98 - (tax_friction * 35))) # RL learns smooth hedging loops

    print("=" * 70)
    print(f"📡 DYNAMIC MSP vs. HYBRID RL VISUALIZATION WORKSPACE (Horizon: {stages} Periods)")
    print("=" * 70)
    
    # 3. Render the Visual Graph of the State Space
    print("\n[STATE SPACE COMPUTATIONAL FOOTPRINT]")
    
    # Render Classical Tree Depth
    msp_bar = "🌲" * min(15, max(1, msp_nodes // 5))
    print(f"  Classical MSP Tree Space:  {msp_bar:<35} ({msp_nodes} Active Scenario Nodes)")
    
    # Render Hybrid Truncated Depth
    hybrid_bar = "🧠" * min(15, max(1, hybrid_nodes // 2))
    print(f"  Hybrid RL Network Space:   {hybrid_bar:<35} ({hybrid_nodes} Bounded State Variables)")
    
    print("\n" + "-" * 70)
    print("[PORTFOLIO POLICY TRAJECTORY EFFICIENCY UNDER HARVESTING DRAG]")
    
    # Render Path Frictions Metrics
    print(f"  Path Tax Drag Penalty:     [{'💥' * int(tax_friction * 10)}{'·' * (10 - int(tax_friction * 10))}] Matrix Drag: {tax_friction:.2f}")
    print(f"  Classical Optimizer Path:  [{'▓' * (classical_efficiency // 5)}{' ' * (20 - (classical_efficiency // 5))}] {classical_efficiency}% Tracking Accuracy")
    print(f"  Learning Value Network:    [{'█' * (hybrid_efficiency // 5)}{' ' * (20 - (hybrid_efficiency // 5))}] {hybrid_efficiency}% Tracking Accuracy")
    print("-" * 70)
    
    # 4. Physical Vector Attractor Field Simulation
    print("\n[ACTIVE REBALANCING AGENT ATTRACTOR FIELD]")
    assets = ["Equities (BMGA)", "Fixed Income", "Defensive Cash", "Commodities", "Defensive Hedges (BMDA)"]
    
    for i, asset in enumerate(assets):
        # Calculate shifts in asset weights based on tax friction parameters
        base_noise = random.randint(-2, 2)
        if "BMDA" in asset and tax_friction > 0.4:
            weight = min(40, int(15 + (tax_friction * 30) + base_noise))
        else:
            weight = max(5, int(20 - (tax_friction * 15) + base_noise))
            
        dots = "·" * (weight // 2)
        print(f"  -> Allocation Vector {i+1} [{asset:<24}]: {dots}▷ {weight}% Target Allocation")
        
    print("\n" + "=" * 70)
    print(" Press Ctrl+C to terminate live optimization visualization sequence...")

if __name__ == "__main__":
    # Simulate a dynamic scenario shift where lookahead horizons extend and tax regulations contract
    try:
        current_horizon = 2
        current_tax_rate = 0.10
        direction = 1
        
        while True:
            clear_screen()
            render_simulation_frame(current_horizon, current_tax_rate)
            
            # Oscillate parameters to simulate dynamic changing horizons and market environments
            current_tax_rate += 0.05 * direction
            if current_tax_rate >= 0.50 or current_tax_rate <= 0.05:
                direction *= -1
                current_horizon = 4 if current_horizon == 2 else 2
                
            time.sleep(1.2)
    except KeyboardInterrupt:
        print("\nOptimization visualizer successfully halted.")
