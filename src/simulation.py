import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from src.dgps import DistributionSampler
from src.methods import kde,multi_kde_n0 as multi_kde, adaptive_kde, plugin_kde
from tqdm.auto import tqdm

def main():
    """main function"""
    with open('src/config.json','r') as fp:
        cfg=json.load(fp)
    N = cfg['sample_size']
    h = cfg['base_bandwidth']
    seed = cfg['seed']
    xii = np.array(cfg['bandwidth_coefficients'])
    d=len(xii)
    raw_dir = cfg['raw_dir']
    fig_dir = cfg['fig_dir']
    
    #store raw results into rows
    rows=[]
    for exp in tqdm(cfg['experiments'], desc="Experiments", unit="exp"):
        dist = exp['dist']
        params = exp['params']
        sampler = DistributionSampler(dist,params,seed=seed)

        # generate data and true pdf
        data = sampler.generate_samples(N)
        pd.DataFrame(data).to_csv('data/simulated.csv',index=False)
        f = sampler.get_pdf()
            
        # several estimated pdf  
        f_es = kde(h,data)
        f_ps = plugin_kde(h,data)
        f_as = adaptive_kde(h,data)
        f_xxi = multi_kde(h,data,xii,d=d)

        # generate points for plotting
        x = np.linspace(-4,4,100)
        y = f(x)
        y_es = f_es(x)
        y_ps = f_ps(x)
        y_as = f_as(x)
        y_xxi = f_xxi(x)

        # compute and compare IMSE
        dx = x[1]-x[0]
        imse_mke = np.sum((y_xxi-y)**2)*dx
        imse_adp = np.sum((y_as-y)**2)*dx
        imse_plugin = np.sum((y_ps-y)**2)*dx
        rows += [
            {"Distribution": dist, "Parameters": params, "Method": "MKDE", "IMSE": imse_mke},
            {"Distribution": dist, "Parameters": params, "Method": "DDE",  "IMSE": imse_plugin},
            {"Distribution": dist, "Parameters": params, "Method": "AKDE", "IMSE": imse_adp},
        ]

        # plot these methods and save them
        plt.figure()
        plt.plot(x,y,label='Real',color='blue')
        plt.plot(x,y_es,label='KDE',color='red')
        plt.plot(x,y_ps,label='DDE',color='purple')
        plt.plot(x,y_as,label='AKDE',color='yellow')
        plt.plot(x,y_xxi,label=f'MKDE',color='green')
        plt.legend()
        fig_name = dist + '(' + str(params) + ')_Comparison.png'
        plt.savefig(fig_dir + fig_name)
        
    # save the raw results
    raw_results = pd.DataFrame(rows)
    raw_results.to_csv(raw_dir + 'raw.csv')

if __name__ == '__main__':
    main()