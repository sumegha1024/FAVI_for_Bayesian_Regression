'''Script to generate Plots and Other Metrics based on results in Regression.py'''

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn import metrics
from utils import pr_or_roc_curve
import os


####Note: beta and mu are used interchangeably here to denote the regression parameters. 

#Inputs
#mu_post_smps: dictionary of posterior samples for mean parameters as numpy arrays with method name as the key
#color_dict: dictionary of colours to be used in plots with method name as the key. 
#hyper_params: dictionary of integer hyper parameters containing keys: seed, n, p, dsfdim, dsl, cdim, cl, bsize, rho
#path: output path where results and plots are stored
#mu0: true value of mean parameters if known. Numpy array.
#sigma_post_smps: dictionary of posterior samples for variance parameters as numpy arrays with method name as the key 
#mcmc_diag: list of MCMC methods for which to include diagnostic plots. e.g ['NUTS','MCMC']
def generate_plots(mu_post_smps, color_dict, hyper_params, path, mu0, sigma_post_smps=None, mcmc_diag=None):

    out_path2='trial'+str(hyper_params['seed'])+'_ndata_'+str(hyper_params['n'])+'_p_'+str(hyper_params['p'])+'_dsdim_'+str(hyper_params['dsfdim'])+'_dsl_'+str(hyper_params['dsl'])+'_cdim_'+str(hyper_params['cdim'])+'_cl_'+str(hyper_params['cl'])+'_bsize_'+str(hyper_params['bsize'])+'_rho_'+str(hyper_params['rho'])+'.pdf'

    #Beta0, beta1 plots 
    fig, axes = plt.subplots(1, 2, figsize=(6,3))
    for key, val in enumerate(mu_post_smps):
        sns.kdeplot(mu_post_smps[val][:,0],color=color_dict[val],label=val,ax=axes[0])
        sns.kdeplot(mu_post_smps[val][:,1],color=color_dict[val],label=val,ax=axes[1])
    if hyper_params['p']>2: 
        axes[0].axvline(x=0, color='purple',ls='--')
        axes[1].axvline(x=0, color='purple',ls='--')
    axes[0].axvline(x=mu0[0], color='gray',ls='--', label='Truth')
    axes[1].axvline(x=mu0[1], color='gray',ls='--', label='Truth')
    plt.legend(bbox_to_anchor = (2.50, 0.6), loc='center right')   
    fig.tight_layout()
    fig.savefig(path+'beta_'+out_path2)
    plt.clf()

    #Sigma squared plots
    if not (sigma_post_smps is None):
        plt.figure(figsize=(6.4,4.8))
        for key, val in enumerate(sigma_post_smps):
            sns.kdeplot(sigma_post_smps[val],color=color_dict[val],label=val)
        plt.axvline(x=1, color='gray',ls='--', label='Truth')
        plt.xlabel(r'$\sigma^{2}$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(path +'sigmasq'+out_path2)
        plt.clf()

    #Contour plots for beta0, beta1
    fig, ax = plt.subplots()
    handles =[]
    for key, val in enumerate(mu_post_smps):
        sns.kdeplot(mu_post_smps[val][:,0],mu_post_smps[val][:,1],color=color_dict[val],label=val,ax=ax)
        handles.append(mlines.Line2D([], [], color=color_dict[val], label=val))
    ax.legend(handles = handles,fontsize=16)
    plt.xlabel(r'$\theta_{1}$',fontsize=18)
    plt.ylabel(r'$\theta_{2}$',fontsize=18)
    plt.xticks([0.9,1.1,1.3,1.5,1.7],fontsize=18)
    plt.yticks([1.2,1.4,1.6,1.8],fontsize=18)
    plt.tight_layout()
    plt.savefig(path+'contour'+out_path2)
    plt.clf()


    #SSE Plots ||mu-mu0||^2_2
    for key, val in enumerate(mu_post_smps):
        print(val)
        sse_mu=np.sum((mu_post_smps[val]- mu0)**2,1)
        sns.kdeplot(sse_mu,color=color_dict[val],label=val)
    plt.legend()
    plt.ylabel('Density of '+r'$||\beta-\beta_{0}||_{2}^{2}$')
    plt.tight_layout()
    plt.savefig(path+'sse_beta_dist'+out_path2)
    plt.clf()

    #MCMC Diagnostic plots
    if not (mcmc_diag is None):
        for val in mcmc_diag:
            #Trace plots for mu
            fig = plt.figure(figsize=(6,3))       
            ax = fig.add_subplot(1,2,1)
            ax.plot(mu_post_smps[val][:,0])
            ax = fig.add_subplot(1,2,2)
            ax.plot(mu_post_smps[val][:,1])
            plt.savefig(path+'MH_Diagnostics/'+val+'_beta_trace_seed'+str(hyper_params['seed'])+'_ndata_'+str(hyper_params['n'])+'_p_'+str(hyper_params['p'])+'.png')
            plt.clf()
            #Auto-corr plots for mu
            fig, ax = plt.subplots(1,2) 
            sm.graphics.tsa.plot_acf(pd.DataFrame(mu_post_smps[val][:,0]), lags=10, ax = ax[0])
            sm.graphics.tsa.plot_acf(pd.DataFrame(mu_post_smps[val][:,1]), lags=10, ax = ax[1])
            plt.savefig(path+'MH_Diagnostics/'+val+'_beta_autocorr_seed'+str(hyper_params['seed'])+'ndata_'+str(hyper_params['n'])+'_p_'+str(hyper_params['p'])+'.png')
            plt.clf()

        if not (sigma_post_smps is None):
            for val in mcmc_diag:
                #trace plots for sigma
                plt.plot(sigma_post_smps[val])
                plt.savefig(path+'MH_Diagnostics/'+val+'_sigma_trace_seed'+str(hyper_params['seed'])+'ndata_'+str(hyper_params['n'])+'_p_'+str(hyper_params['p'])+'.png')
                plt.clf()
                #Auto-corr plots for sigma
                sm.graphics.tsa.plot_acf(pd.DataFrame(sigma_post_smps[val]), lags=10)
                plt.savefig(path+'MH_Diagnostics/'+val+'_sigma_autocorr_seed'+str(hyper_params['seed'])+'ndata_'+str(hyper_params['n'])+'_p_'+str(hyper_params['p'])+'.png')
                plt.clf()




#Inputs
#mu_post_smps: dictionary of posterior samples for mean parameters with method name as the key
#color_dict: dictionary of colours to be used in plots with method as the key 
#hyper_params: dictionary of hyper parameters containing keys: seed, n, p, dsfdim, dsl, cdim, cl, bsize, rho
#rmse: Model predictive rmse
#time: training time taken by each method
#path: output path where results and plots are stored
#mu0: true value of mean parameters if known
#pr_roc: Whether to generate pr_roc curves or not. boolean true or false.
#write_csv: Whether to write metrics to csv files. boolean true or false
#kl: dictionary of kl div for each method with method name as key and numeric (floating) kl value. 

def generate_metrics(mu_post_smps, color_dict, hyper_params, rmse, Time, path, mu0, pr_roc=False, writecsv=False,kl=None):

    out_path2='trial'+str(hyper_params['seed'])+'_ndata_'+str(hyper_params['n'])+'_p_'+str(hyper_params['p'])+'_dsdim_'+str(hyper_params['dsfdim'])+'_dsl_'+str(hyper_params['dsl'])+'_cdim_'+str(hyper_params['cdim'])+'_cl_'+str(hyper_params['cl'])+'_bsize_'+str(hyper_params['bsize'])+'_rho_'+str(hyper_params['rho'])+'.png'
    se_beta_avg = {}
    se_beta_min = {}
    se_beta_max = {}
    confusion = {}
    acc_betas = {}
    fdr = {}
    recall = {}
    prec = {}
    fscore = {}
    pr_auc_dict = {}
    roc_auc_dict = {}

    ht_mu0 = mu0!=0
    
    for key, val in enumerate(mu_post_smps):
        #Standard erros for beta posterior
        se_beta_avg[val] =np.mean(np.std(mu_post_smps[val],axis=0,ddof=1))
        se_beta_min[val] =np.min(np.std(mu_post_smps[val],axis=0,ddof=1))
        se_beta_max[val] =np.max(np.std(mu_post_smps[val],axis=0,ddof=1))
        
        #Credible interval for betas 
        q_lower = np.quantile(mu_post_smps[val],0.025,axis=0)
        q_upper = np.quantile(mu_post_smps[val],0.975,axis=0)
        cint = (q_lower<= mu0) & (mu0 <= q_upper)
        acc_betas[val] = sum(cint)/(cint).shape[0]
        
        #Hypothesis Testing
        ht_est = (q_lower > np.zeros(hyper_params['p'])) | (np.zeros(hyper_params['p']) > q_upper)
        confusion[val] = metrics.confusion_matrix(ht_mu0,ht_est)

        #Fscore 
        prec[val] = sum(ht_mu0*ht_est)/sum(ht_est)
        recall[val] = sum(ht_mu0*ht_est)/sum(ht_mu0)
        fscore[val] = (2*prec[val]*recall[val])/(prec[val]+recall[val])

        #FDR
        fdr[val]=sum((ht_est.astype(int)-ht_mu0.astype(int))==np.ones(hyper_params['p']))/sum(ht_est)

        #AUC Metrics 
        prec_scores, recall_scores, fpr_scores, _ = pr_or_roc_curve(ht_mu0.astype(int), mu_post_smps[val].mean(0)/mu_post_smps[val].std(0))
        pr_auc_dict[val] = metrics.auc(recall_scores, prec_scores)
        roc_auc_dict[val] = metrics.auc(fpr_scores,recall_scores)

    
    #ROC Plots
    if pr_roc==True:
        fig, ax = plt.subplots()
        for key, val in enumerate(mu_post_smps):
             prec_scores, recall_scores, _, _ = pr_or_roc_curve(ht_mu0.astype(int), mu_post_smps[val].mean(0)/mu_post_smps[val].std(0))
             plt.plot(recall_scores, prec_scores,color=color_dict[val],label=val)  
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(path+'pr_curve_'+out_path2)
        plt.clf()

        fig, ax = plt.subplots()
        for key, val in enumerate(mu_post_smps):
             _, recall_scores, fpr_scores, _ = pr_or_roc_curve(ht_mu0.astype(int), mu_post_smps[val].mean(0)/mu_post_smps[val].std(0))
             plt.plot(fpr_scores, recall_scores,color=color_dict[val],label=val)  
        plt.xlabel('FPR')
        plt.ylabel('Sensitivity')
        plt.title('ROC Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(path+'roc_curve_'+out_path2)
        plt.clf()

    #printing results
    print("Rmse",rmse)
    print("Time in s",Time)
    print("Correct prop",acc_betas)
    print("Fscore",fscore)
    print("SE_min",se_beta_min)
    print("SE_avg",se_beta_avg)
    print("SE_max",se_beta_max)
    print("FDR",fdr)
    print("Recall",recall)
    print("AUC PR",pr_auc_dict)
    print("AUC ROC",roc_auc_dict)
    print("kl",kl)

    #converting results to pandas array to write to csv
    if writecsv==True:
        isExist = os.path.exists(path+'Metrics/')
        if not isExist:
            os.makedirs(path+'Metrics/')
        dict_csv={'Corr':[],'n':[],'p':[],'Method':[],'cdim':[],'Seed':[],'Time':[],'Rmse':[]
        ,'Acc_betas':[],'Fscore':[],'SE_betas':[],'FDR':[],'Recall':[],'Prec':[],
        'AUC PR':[],'AUC ROC':[],'KL':[]}
        for key, val in enumerate(mu_post_smps):
            dict_csv['Corr'].append(hyper_params['rho'])
            dict_csv['n'].append(hyper_params['n'])
            dict_csv['p'].append(hyper_params['p'])
            dict_csv['Method'].append(val)
            dict_csv['Seed'].append(hyper_params['seed'])
            dict_csv['Time'].append(Time[val])
            dict_csv['Rmse'].append(rmse[val])
            dict_csv['Acc_betas'].append(acc_betas[val])
            dict_csv['Fscore'].append(fscore[val])
            dict_csv['SE_betas'].append(se_beta_avg[val])
            dict_csv['FDR'].append(fdr[val])
            dict_csv['Recall'].append(recall[val])
            dict_csv['Prec'].append(prec[val])
            dict_csv['AUC PR'].append(pr_auc_dict[val])
            dict_csv['AUC ROC'].append(roc_auc_dict[val])
            dict_csv['KL'].append(kl[val])
            dict_csv['cdim'].append(hyper_params['cdim'])
            
        df=pd.DataFrame.from_dict(dict_csv)
        df.to_csv(path+'Metrics/trial'+str(hyper_params['seed'])+'_ndata_'+str(hyper_params['n'])+'_p_'+str(hyper_params['p'])+'_rho_'+str(hyper_params['rho'])+'_cdim_'+str(hyper_params['cdim'])+'.csv')