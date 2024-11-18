The code is broadly organized along the following lines:
•config.py: contains the composition of DJI and NIFTY 50 on different rebalance dates.
•data_processing_utils.py: contains the data processing utils common to both SMC and LASSO algorithms (generate the roll schedule and effective composition used for regression)
•LASSO_utils.py: contains utility functions specific to LASSO algorithm
•LASSO.py: used for generating results pertaining to LASSO algorithm for different indices
•SMC_utils.py: contains utility functions specific to SMC algorithm
•SMC.py: used for generating results pertaining to SMC algorithm for different indices
•performance_utils.py: contains utils for calculating performance metrics and analytics
•comparison_LASSO_vs_SMC.py: contains additional utils for comparing the two methods

