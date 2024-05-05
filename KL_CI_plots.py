'''KL and Confidence Interval Plots for IAF with p=2 ReLU activation, single hidden layer and one hidden node'''

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import torch.nn as nn
import math
import scipy.stats as ss
import random
from torch.distributions import MultivariateNormal, Normal
import argparse
sys.path.append('/mnt/ufs18/home-104/premchan/FAVI_for_Bayesian_Regression/')
from utils import set_all_seeds 
from output import generate_plots
from itertools import product
import random
import plotly.graph_objects as go

parser = argparse.ArgumentParser(description='inputs_kl_iaf')
parser.add_argument('--out',default = '/mnt/home/premchan/FAVI_for_Bayesian_Regression/Out/', type=str, help='path to results')
args = parser.parse_args()

global p
p=2

'''Function to Simulate Data-sets across correlation and sample size'''

#Inputs
#Rho - correlation between predictors. Float
#sigma - variance of response y. Float
#seed - random seed. Int
#n_data - sample size. Int
def simulate_data(rho,sigma,seed,n_data):
    set_all_seeds(seed)
    beta0=np.random.uniform(0.5,2,p)
    mean=np.zeros((p,))
    cov=(1-rho)*np.identity(p)+rho*np.ones((p,p))
    X= np.random.multivariate_normal(mean, cov, n_data)
    e=np.random.normal(0,1,n_data)
    sigma = sigma
    y=np.dot(X,beta0)+sigma*e

    return torch.Tensor(X), torch.Tensor(y), beta0

'''Functions to aid KL and alpha_pi coverage calculation'''

#Calculates j(t) and l(t). See SI Lemma 2.4 and 2.5. 
def fn(tt,xx):
    normal_d = Normal(0, 1)  
    num = (1-normal_d.cdf(torch.sgn(xx)*tt))
    den = (1+tt**2)* (1-normal_d.cdf(torch.sgn(xx)*tt)) -torch.sgn(xx)*tt* normal_d.log_prob(tt).exp()
    den+= -(tt*(1-normal_d.cdf(torch.sgn(xx)*tt)) - torch.sgn(xx)*(normal_d.log_prob(tt).exp()))**2
    return num, den

#Optimal value for a1. See SI Eqn 18 proof of Theorem 2.6 
def out_a1(tt,X):

    x1_tilda = (X[:,0]**2).sum()+1
    x2_tilda = (X[:,1]**2).sum()+1
    x1x2 = ((X[:,0]*X[:,1]).sum())
    
    num, den =fn(torch.Tensor(tt),x1x2)
    a1 = x2_tilda*(den)/(x1_tilda*x2_tilda*den - (x1x2**2)*(num**2))
    
    return a1

#Kl divergence as function of \tilde{t}. See SI proof of Theorem 2.6.
#X - torch tensor of size n x 2
#y - torch tensor of size n
def kld(tt,X,y):

    kl_sum_l=[-1]*tt.size()[0]
    x1_tilda = (X[:,0]**2).sum()+1
    x2_tilda = (X[:,1]**2).sum()+1
    x1x2 = ((X[:,0]*X[:,1]).sum())
    pi2 = torch.Tensor([2*math.pi])
    y_y=torch.sum(y**2)
    X_y=torch.sum(X.T*y,1)
    X_X=torch.mm(X.T,X)
    b1 = (x2_tilda*X_y[0] - x1x2*X_y[1])/(x1_tilda*x2_tilda - x1x2**2)

    num, den =fn(torch.Tensor(tt),x1x2)
    a1_l = out_a1(tt,X)
     
    #KL divergence
    kl_sum0=  -0.5*torch.log(x1_tilda*x2_tilda - x1x2**2) + .5*torch.log(x2_tilda)
    kl_sum0+= 0.5*(X_y*((torch.inverse(X_X+torch.eye(p))*X_y).sum(1))).sum()
    kl_sum0+= 0.5*(b1**2)*x1_tilda - X_y[0]*b1 - 0.5*(X_y[1] - x1x2*b1)**2/x2_tilda
    
    for i in range(tt.shape[0]): 
        if a1_l[i]>0 and den[i]>0:
            kl_sum1 = kl_sum0 - 0.5*torch.log(x2_tilda)-0.5*torch.log((den[i])) + 0.5*torch.log(x1_tilda*x2_tilda*den[i] - (x1x2**2)*(num[i]**2))
            kl_sum_l[i]=kl_sum1.numpy()

    return kl_sum_l


##############################Generate Figure 1b) in paper#######################################
'''Simulate data-sets and calculate kl'''

seed=random.sample(range(100), 100)
rho=[0.6,-0.6]
n_data=[200]
kl_favi={}
kl_mfvi={}
iters=product(seed,rho,n_data)
tt_array=np.linspace(-4,4,1000) 
for i in iters:
    #Simulate data
    X,y,beta0 = simulate_data(rho=i[1],sigma=1.0,seed=i[0],n_data=i[2])
    #Calculate KL for IAF and store to dict   
    kl_favi[i] = np.stack(kld(torch.Tensor(tt_array),X,y))
    kl_mfvi[i] = np.stack([(-0.5*torch.log(((X[:,0]**2).sum()+1)*((X[:,1]**2).sum()+1) - ((X[:,0]*X[:,1]).sum())**2) + 0.5*torch.log((X[:,0]**2).sum()+1) + 0.5*torch.log((X[:,1]**2).sum()+1)).numpy()]*len(tt_array))
    

'''Functions to aid plotting'''

#Inputs:
#dict_in: dictionary of arrays where keys are tuple of simulation params
#params: dictionary {'rho':0.8,'n_data':100}
def extract_kl(dict_in,params):
    kl_out=np.zeros((len(seed),len(tt_array)))
    j=0
    for ind, key in enumerate(dict_in):
        if (key[1]==params['rho'] and key[2]==params['n_data']):
            kl_out[j,:]=dict_in[key]
            j+=1
    return kl_out

'''Generate Plots'''

#KL as function of \tilde{t}
kl_mfvi_out=extract_kl(kl_mfvi,{'rho':rho[0],'n_data':200})
kl_mfvi_mean=np.mean(kl_mfvi_out,0)
kl_mfvi_sd=np.std(kl_mfvi_out,0)
#rho >0
kl_favi_out1=extract_kl(kl_favi,{'rho':rho[0],'n_data':200})
kl_favi_mean1=np.mean(kl_favi_out1,0)
kl_favi_sd1=np.std(kl_favi_out1,0)
#rho <0
kl_favi_out2=extract_kl(kl_favi,{'rho':rho[1],'n_data':200})
kl_favi_mean2=np.mean(kl_favi_out2,0)
kl_favi_sd2=np.std(kl_favi_out2,0)


plt.plot(tt_array,kl_mfvi_mean,label=r'MF-VI $|\rho|\ =\ $'+str(rho[0]),color='blue')
plt.fill_between(tt_array, kl_mfvi_mean - kl_mfvi_sd, kl_mfvi_mean + kl_mfvi_sd, color='blue', alpha=0.2)
#rho >0
plt.plot(tt_array,kl_favi_mean1,label=r'FAVI $\rho\ =\ $'+str(rho[0]),color='green')
plt.fill_between(tt_array, kl_favi_mean1 - kl_favi_sd1, kl_favi_mean1 + kl_favi_sd1, color='green', alpha=0.2)
#rho <0
plt.plot(tt_array,kl_favi_mean2,label=r'FAVI $\rho\ =\ $'+str(rho[1]),color='purple')
plt.fill_between(tt_array, kl_favi_mean2 - kl_favi_sd2, kl_favi_mean2 + kl_favi_sd2, color='purple', alpha=0.2)
plt.ylabel('KL',fontsize=18)
plt.xlabel(r'$\tilde{t}$',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='center right',bbox_to_anchor=(1, 0.3),fontsize=13)
plt.tight_layout()
plt.savefig(args.out+'kl.pdf')
plt.clf()
###################################Generate Fig 1c) in paper######################################


'''Coverage calculations'''

rho=np.linspace(0.0,0.9,100)
tt_array=np.linspace(-4,4,100)
rho, tt_array = np.meshgrid(rho,tt_array) 
alpha=0.05
alpha_pi_mfvi = 2*(1-ss.norm.cdf(ss.norm.ppf(1-alpha/2)*np.sqrt(1-rho**2))) 
num, den =fn(torch.Tensor(tt_array),torch.Tensor(rho))
num=num.numpy()
den=den.numpy()
C_rho_tt = ((1 - rho**2)*num)/(num - (rho**2)*(den**2))
alpha_pi_favi1 = 2*(1 - ss.norm.cdf(ss.norm.ppf(1-alpha/2)*np.sqrt(C_rho_tt)))

#Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf1=ax.plot_surface(rho, tt_array, 1-alpha_pi_favi1, cmap='Greens', alpha=0.7,label='FAVI')
surf2=ax.plot_surface(rho, tt_array, 1-alpha_pi_mfvi, cmap='Blues', alpha=0.7,label='MF-VI') 
ax.set_xlabel(r'$\mathbf{\rho}$',fontsize=18,labelpad=10)
ax.set_ylabel(r'$\mathbf{\tilde{t}}$',fontsize=18,labelpad=10)
ax.set_zlabel(r'$\mathbf{1-\alpha_\Pi}$',fontsize=18,labelpad=10)
ax.grid(False)
plt.xticks([0.2,0.5,0.8],fontsize=17)
plt.yticks([-2,0,2],fontsize=17)
ax.set_zticks([0.7,0.8,0.9])
ax.tick_params(axis='z', labelsize=17)  
cbar1=fig.colorbar(surf1, ax=ax, shrink=0.5, aspect=5,location='left')
cbar1.ax.set_title('FAVI',fontsize=17)
cbar1.set_ticks([0.7,0.8,0.9])
cbar1.ax.tick_params(labelsize=18)
cbar2=fig.colorbar(surf2, ax=ax, shrink=0.5, aspect=5,location='left')
cbar2.ax.set_title('MF-VI',fontsize=17)
cbar2.set_ticks([0.7,0.8,0.9])
cbar2.ax.tick_params(labelsize=18)
fig.subplots_adjust(left=0.05, right=0.85, bottom=-0.2, top=1)
plt.savefig(args.out+'alpha.pdf')




