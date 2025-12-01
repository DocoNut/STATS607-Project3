# from src.simulation import main
import json
import numpy as np
from src.dgps import DistributionSampler
from src.methods import multi_kde_n0 as multi_kde

with open('src/config.json','r') as fp:
    cfg=json.load(fp)
N = cfg['sample_size']
h = cfg['base_bandwidth']
seed = 1234
xii = np.array(cfg['bandwidth_coefficients'])
d=len(xii)

#tested values(let's test on f distribution)
dist = 'bimodal'
params = [-1.0, 1.0, 0.5, 0.5, 0.7]
test_x = np.linspace(-2,2,50)

#first round
sampler1 = DistributionSampler(dist,params,seed=seed)
data1 = sampler1.generate_samples(N)
f_1 = multi_kde(h,data1,xii,d=d) 
y_1= f_1(test_x)

#second round
sampler2 = DistributionSampler(dist,params,seed=seed)
data2 = sampler2.generate_samples(N)
f_2 = multi_kde(h,data2,xii,d=d) 
y_2= f_1(test_x)

# let's see if our methods give the same result
if (y_1 != y_2).any():
    raise ValueError('Reproducibity failed!')

print("✅ Reproducibility tests all passed!")