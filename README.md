# FAVI_for_Bayesian_Regression
Code that both implements and visualizes how Inverse Auto-regressive Flows can be used for uncertainty quantification in Bayesian Linear Regression.

More specifically, this code can be used to reproduce the results in the paper "Normalizing Flows for Bayesian Inference: A case study with Linear Regression" submitted to Proceedings of the National Academy of Sciences - Brief Reports.

1. Regression.py with defaults can be used to reproduce Fig 1(a).
2. KL_CI_plots.py can be used to reproduce Figures 1(b) and 1(c) in the paper.
3. KL_IAF.py is merely an empirical validation for the KL divergence derivation in the Supplementary Information for the paper. It is not required to reproduce any results. 

The scripts in the folder "Flows_scripts" have been re-purposed from the repository <https://github.com/CW-Huang/NAF>.


# Requirements

Python 3.0+ PyTorch 1.13.0+cpu + numpy 1.22.1      
For more details on package versions see Conda_env_info1.png, Conda_env_info2.png. 