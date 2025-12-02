import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from src.dgps import DistributionSampler
from src.methods import kde, multi_kde_n0 as multi_kde, adaptive_kde, plugin_kde
from tqdm.auto import tqdm
import time
import sys

class FileLog:
    def __init__(self, filename):
        self.f = open(filename, 'w')
    
    def write(self, message):
        self.f.write(message + '\n') # Add newline for readability
        self.f.flush() # Ensure it writes immediately

# 1. create your logging object
log_file = FileLog('numpy_errors.log')

# 2. Tell numpy to send errors to this object
np.seterrcall(log_file)

# 3. Tell numpy to use the 'log' mode
# Note: We use 'log' instead of 'warn' here. 
# 'warn' prints to stderr; 'log' sends it to the object set in seterrcall.
np.seterr(all='log')

def main():
    """main function"""
    with open('src/config.json', 'r') as fp:
        cfg = json.load(fp)

    if '--profile_mode' in sys.argv:
        profiling= True
    else:
        profiling = False

    N = cfg['sample_size']
    h = cfg['base_bandwidth']
    seed = cfg['seed']
    xii = np.array(cfg['bandwidth_coefficients'])
    d = len(xii)
    raw_dir = cfg['raw_dir']
    fig_dir = cfg['fig_dir']

    # timers for fine-grained profiling
    if profiling:
        t_total_start = time.perf_counter()
        method_times = {
            "sampler_init": 0.0,
            "data_gen": 0.0,
            "write_data": 0.0,
            "get_pdf": 0.0,
            "kde_build": 0.0,
            "dde_build": 0.0,
            "akde_build": 0.0,
            "mkde_build": 0.0,
            "eval_true": 0.0,
            "eval_kde": 0.0,
            "eval_dde": 0.0,
            "eval_akde": 0.0,
            "eval_mkde": 0.0,
            "imse": 0.0,
            "plot": 0.0,
        }
        n_experiments = 0

    rows = []
    for exp in tqdm(cfg['experiments'], desc="Experiments", unit="exp"):
        dist = exp['dist']
        params = exp['params']

        if profiling:
            n_experiments += 1
            t0 = time.perf_counter()
        sampler = DistributionSampler(dist, params, seed=seed)
        if profiling:
            method_times["sampler_init"] += time.perf_counter() - t0

        # ---- data generation ----
        if profiling:
            t0 = time.perf_counter()
        data = sampler.generate_samples(N)
        if profiling:
            method_times["data_gen"] += time.perf_counter() - t0

        if profiling:
            t0 = time.perf_counter()
        pd.DataFrame(data).to_csv('data/simulated.csv', index=False)
        if profiling:
            method_times["write_data"] += time.perf_counter() - t0

        if profiling:
            t0 = time.perf_counter()
        f = sampler.get_pdf()
        if profiling:
            method_times["get_pdf"] += time.perf_counter() - t0

        # ---- method constructions ----
        if profiling:
            t0 = time.perf_counter()
        f_es = kde(h, data)
        if profiling:
            method_times["kde_build"] += time.perf_counter() - t0

        if profiling:
            t0 = time.perf_counter()
        f_ps = plugin_kde(h, data)
        if profiling:
            method_times["dde_build"] += time.perf_counter() - t0

        if profiling:
            t0 = time.perf_counter()
        f_as = adaptive_kde(h, data)
        if profiling:
            method_times["akde_build"] += time.perf_counter() - t0

        if profiling:
            t0 = time.perf_counter()
        f_xxi = multi_kde(h, data, xii, d=d)
        if profiling:
            method_times["mkde_build"] += time.perf_counter() - t0

        # ---- evaluation grid + IMSE ----
        x = np.linspace(-4, 4, 100)

        if profiling:
            t0 = time.perf_counter()
        y = f(x)
        if profiling:
            method_times["eval_true"] += time.perf_counter() - t0

        if profiling:
            t0 = time.perf_counter()
        y_es = f_es(x)
        if profiling:
            method_times["eval_kde"] += time.perf_counter() - t0

        if profiling:
            t0 = time.perf_counter()
        y_ps = f_ps(x)
        if profiling:
            method_times["eval_dde"] += time.perf_counter() - t0

        if profiling:
            t0 = time.perf_counter()
        y_as = f_as(x)
        if profiling:
            method_times["eval_akde"] += time.perf_counter() - t0

        if profiling:
            t0 = time.perf_counter()
        y_xxi = f_xxi(x)
        if profiling:
            method_times["eval_mkde"] += time.perf_counter() - t0

        if profiling:
            t0 = time.perf_counter()
        dx = x[1] - x[0]
        imse_mke = np.sum((y_xxi - y) ** 2) * dx
        imse_adp = np.sum((y_as - y) ** 2) * dx
        imse_plugin = np.sum((y_ps - y) ** 2) * dx
        if profiling:
            method_times["imse"] += time.perf_counter() - t0

        rows += [
            {"Distribution": dist, "Parameters": params, "Method": "MKDE", "IMSE": imse_mke},
            {"Distribution": dist, "Parameters": params, "Method": "DDE",  "IMSE": imse_plugin},
            {"Distribution": dist, "Parameters": params, "Method": "AKDE", "IMSE": imse_adp},
        ]

        # ---- plotting ----
        if profiling:
            t0 = time.perf_counter()
        plt.figure()
        plt.plot(x, y,      label='Real',  color='blue')
        plt.plot(x, y_es,   label='KDE',   color='red')
        plt.plot(x, y_ps,   label='DDE',   color='purple')
        plt.plot(x, y_as,   label='AKDE',  color='yellow')
        plt.plot(x, y_xxi,  label='MKDE',  color='green')
        plt.legend()
        fig_name = dist + '(' + str(params) + ')_Comparison.png'
        plt.savefig(fig_dir + fig_name)
        plt.close()
        if profiling:
            method_times["plot"] += time.perf_counter() - t0

    # save the raw results
    raw_results = pd.DataFrame(rows)
    raw_results.to_csv(raw_dir + 'raw.csv', index=False)

    if profiling:
        t_total_end = time.perf_counter()
        total_runtime = t_total_end - t_total_start

        # build summary lines, sorted by total time desc
        lines = []
        lines.append(f"[PROFILE] Total runtime: {total_runtime:.6f} seconds\n")
        lines.append("[PROFILE] Per-component times (total / per-experiment):\n")

        # sort keys by total time descending
        for name, ttot in sorted(method_times.items(), key=lambda kv: kv[1], reverse=True):
            avg = ttot / n_experiments if n_experiments > 0 else float('nan')
            lines.append(f"  {name:12s}: total = {ttot:.6f}s, avg/exp = {avg:.6f}s\n")

        # print to terminal
        print("".join(lines))

        # also write to file
        profile_path = raw_dir + "profile_summary.txt"
        with open(profile_path, "w") as f:
            f.writelines(lines)


if __name__ == '__main__':
    main()
