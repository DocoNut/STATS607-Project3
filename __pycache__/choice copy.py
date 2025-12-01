from mulkde import multi_kde,kde,kernel_secondorder_derevative as f2,kernel as f
from mulkde_coef import var,b
from turtle import color
import matplotlib.pyplot as plt
import numpy as np

for i in range(2,20):
    xi = np.linspace(1,2,i)
    for j in range(2,j):
        print(i,j,var(xi,d=j))