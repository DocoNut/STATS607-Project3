import numpy as np
import matplotlib.pyplot as plt
import time
import json
import pandas as pd
from tqdm.auto import tqdm
from src.dgps import DistributionSampler
from src.methods import kde, multi_kde_n0 as multi_kde, adaptive_kde, plugin_kde

def measure_runtime():
    # 1. Load configuration for fixed parameters (like bandwidths)
    with open('src/config.json', 'r') as fp:
        cfg = json.load(fp)

    # 2. Define the sample sizes to test
    # We use a logarithmic spread or linear spread depending on preference.
    # Here we go from 100 to 5000 (adjust based on your machine's speed)
    sample_sizes = [100, 500, 1000, 2000, 3000, 4000, 5000]
    
    # 3. Setup fixed parameters
    h = cfg['base_bandwidth']
    seed = cfg['seed']
    xii = np.array(cfg['bandwidth_coefficients'])
    d = len(xii)
    
    # Use a simple standard normal for runtime benchmarking
    dist_name = "Normal"
    params = [0, 1] 
    
    # Evaluation grid (fixed size to ensure we measure build+eval time fairly)
    x_grid = np.linspace(-4, 4, 100)

    results = []

    print(f"Benchmarking Runtime on {dist_name} distribution...")
    
    for N in tqdm(sample_sizes, desc="Sample Sizes"):
        # Generate data once per sample size
        sampler = DistributionSampler(dist_name, params, seed=seed)
        data = sampler.generate_samples(N)

        # Dictionary to hold methods to test
        # Key: Label, Value: Function wrapper
        methods = {
            "KDE": lambda: kde(h, data)(x_grid),
            "DDE": lambda: plugin_kde(h, data)(x_grid),
            "AKDE": lambda: adaptive_kde(h, data)(x_grid),
            "MKDE": lambda: multi_kde(h, data, xii, d=d)(x_grid)
        }

        for method_name, func in methods.items():
            # Measure time
            # We run it 3 times and take the average to reduce noise
            n_repeats = 3
            times = []
            
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                _ = func() # Run the construction + evaluation
                t1 = time.perf_counter()
                times.append(t1 - t0)
            
            avg_time = np.mean(times)
            
            results.append({
                "N": N,
                "Method": method_name,
                "Time": avg_time
            })

    # 4. Save and Plot Data
    df = pd.DataFrame(results)
    
    # Save raw timing data
    df.to_csv(cfg['raw_dir'] + 'runtime_benchmark.csv', index=False)

    # Plotting
    methods_unique = df['Method'].unique()
    n_methods=len(methods_unique)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5), constrained_layout=True)
    
    # If there is only 1 method, axes is not a list, so wrap it

    for ax, method in zip(axes, methods_unique):
        subset = df[df['Method'] == method]
        
        ax.plot(subset['N'], subset['Time'], marker='o', color='teal')
        ax.set_title(method)
        ax.set_xlabel('Sample Size (N)')
        ax.set_ylabel('Time (s)')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Optional: Force start at 0 to avoid misleading scales
        ax.set_ylim(bottom=0)

    fig.suptitle(f'Runtime Analysis by Method ({dist_name})', fontsize=16)
    
    output_path = cfg['fig_dir'] + 'runtime_subplots.png'
    plt.savefig(output_path)
    print(f"\nPlot saved to {output_path}")
    print(f"Raw data saved to {cfg['raw_dir']}runtime_benchmark.csv")

if __name__ == "__main__":
    measure_runtime()