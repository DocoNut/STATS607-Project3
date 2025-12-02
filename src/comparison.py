import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import os
from src.methods_opt import kde as kde_opt, adaptive_kde as akde_opt
# Assuming your old slow versions are in src/methods.py (or wherever you kept them)
from src.methods import kde as kde_slow, adaptive_kde as akde_slow

def compare_versions():
    # Ensure output directory exists
    output_dir = 'results/raw/'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'performance_table.txt')

    N_values = [100, 500, 1000, 2000, 5000]
    results = []
    
    # Test Data
    h = 0.5
    eval_grid = np.linspace(-3, 3, 100)

    # Open file for writing
    with open(output_file, 'w') as f:
        # Helper to print to both console and file
        def log(msg):
            print(msg)
            f.write(msg + '\n')

        header = f"{'N':<6} | {'Method':<10} | {'Slow (s)':<10} | {'Opt (s)':<10} | {'Speedup':<8}"
        log(header)
        log("-" * 55)

        for N in N_values:
            data = np.random.randn(N)
            
            # --- Standard KDE ---
            # Time Slow
            t0 = time.perf_counter()
            _ = kde_slow(h, data)(eval_grid)
            t_slow_kde = time.perf_counter() - t0
            
            # Time Optimized
            t0 = time.perf_counter()
            _ = kde_opt(h, data)(eval_grid)
            t_opt_kde = time.perf_counter() - t0
            
            # --- Adaptive KDE ---
            # Time Slow
            t0 = time.perf_counter()
            _ = akde_slow(h, data)(eval_grid)
            t_slow_akde = time.perf_counter() - t0
            
            # Time Optimized
            t0 = time.perf_counter()
            _ = akde_opt(h, data)(eval_grid)
            t_opt_akde = time.perf_counter() - t0

            results.append({
                "N": N,
                "KDE_Speedup": t_slow_kde / t_opt_kde,
                "AKDE_Speedup": t_slow_akde / t_opt_akde
            })
            
            log(f"{N:<6} | KDE        | {t_slow_kde:.4f}     | {t_opt_kde:.4f}     | {t_slow_kde/t_opt_kde:.1f}x")
            log(f"{N:<6} | AKDE       | {t_slow_akde:.4f}     | {t_opt_akde:.4f}     | {t_slow_akde/t_opt_akde:.1f}x")

    # Plotting
    df = pd.DataFrame(results)
    
    # Also save the raw CSV data for future reference
    df.to_csv(os.path.join(output_dir, 'performance_data.csv'), index=False)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['N'], df['KDE_Speedup'], marker='o', label='Standard KDE Speedup')
    plt.plot(df['N'], df['AKDE_Speedup'], marker='s', label='Adaptive KDE Speedup')
    plt.xlabel('Sample Size (N)')
    plt.ylabel('Speedup Factor (x times faster)')
    plt.title('Optimization Impact: Vectorization vs Loops')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Ensure figure directory exists
    fig_dir = 'results/figures/'
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, 'performance_speedup.png')
    
    plt.savefig(fig_path)
    print(f"\nVisualization saved to {fig_path}")
    print(f"Table saved to {output_file}")

if __name__ == "__main__":
    compare_versions()