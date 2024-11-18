## MscFE CapstoneProject

Code repository of WQU Capstone Project from group 7119:
* Abhijeet Aanand – abhijeetaanand0@gmail.com
* Amparo Garcia Garcia - garciamparo14@gmail.com


## Project Track
The current project is included within Stock Indices and Their Components track, specifically within the Equity Index Replication. This track is a combination of research and practical track. The research component of the track involves understanding the key drivers of broader market indices and researching methodologies to do a sparse replication of indices of interest. The practical component of the track involves implementing some of these methodologies on indices of interest and comparing their strengths and potential weaknesses.

## Problem statement
Index replication is a popular passive portfolio management strategy which aims to reproduce the performance of a market index. Full replication of an index with many constituents involves trading in potentially hundreds of assets which drives up execution costs due to illiquidity and higher fixed costs. Hence, the natural way for fund managers to replicate a broader market index involves selecting a smaller set of assets and using those to sparsely track the index.
'Sparse Index Replication' is an active area of research owing to its direct application in asset management industry. Several methods have been proposed for efficient replication of broader market indices using a limited set of constituents. We aim to perform a thorough literature survey of popular methods used for solving the sparse index tracking problem and compare their strengths and potential weaknesses after implementing them on popular market indices. 

## Goals and Objectives
Goal:  The broader objective of the project is to research and implement sparse index replication methodologies on some popular market indices of interest and compare their strengths and potential weaknesses. 
Objectives:
* Obtaining clean and reliable data for market indices and their constituents.
* Understanding the composition of market indexes and the importance of individual components in driving returns.
* Analyzing data, aiming to understand correlations between individual assets within the index. Aiming to find key uncorrelated drivers of index returns.
* Researching popular methods used for sparse replication of indexes.
* Implementing popular sparse method replication methodologies on selected indices.
* Comparing and evaluating strengths and potential weaknesses of these methods on different kinds of indices (market cap weighted, price weighted etc.)


## Code Design

The code is supposed to be tentatively structured in four phases:
(this may be subject to change as we progress through literature survey and implementation process)
* Data preparation - We would be taking a well represented sample of data from different sources. This would be cleaned and preprocessed for later usage. The data would most likely include corporate action adjusted close prices of global market indices and their respective constituents.
* Data analysis.- Then evolution of the market indexes and their stocks would be analyzed. The correlation among various index constituents and other summary statistics would also be calculated.
* Index Replication.- In this section, several methods for replication would be coded and applied to the data chosen in the second step.
* Evaluation of Results- In this section, some metrics for accuracy and error would be calculated such as the tracking error, mean square error and mean absolute error to learn how well the sparse replication approach performs. Besides, graphs of replicated and original data for market indexes would be plotted so as to see the difference between them.

  The central theme of the project revolves around comparing the performance of the SMC algorithm against the benchmark LASSO algorithm for sparse index replication. A relatively thorough analysis has been done for Dow Jones Industrial Average Index (price-weighted index) and NIFTY 50 Index (free-float market capitalization weighted index). These two indices were selected for their different mode of calculation and for having a fewer number of stocks which makes the analysis relatively simpler and less time consuming. To analyse the performance through time, an extended period of 8 years has been taken into consideration for generating all the results. 
The code for this segment can be found under the folder ‘SMC_vs_LASSO’ and is broadly organized along the following lines:
*	config.py: contains the composition of DJI and NIFTY 50 on different rebalance dates.
*	data_processing_utils.py: contains the data processing utils common to both SMC and LASSO algorithms (generate the roll schedule and effective composition used for regression)
*	LASSO_utils.py: contains utility functions specific to LASSO algorithm
*	LASSO.py: used for generating results pertaining to LASSO algorithm for different indices
*	SMC_utils.py: contains utility functions specific to SMC algorithm
*	SMC.py: used for generating results pertaining to SMC algorithm for different indices
*	performance_utils.py: contains utils for calculating performance metrics and analytics
*	comparison_LASSO_vs_SMC.py: contains additional utils for comparing the two methods

The alternative theme of the project involved using risk-folio, principal component analysis, sparse index replication without Monte Carlo and non-negative least squares replication (NNLS) can be found in the file: ‘MscFE_Capstone_Project_alternative_methods.ipynb.’ The main reason for this code to be in formatted as a notebook and not as a python file with extension .py is due to large number of graphs generated in the process (notebook provides a more comprehensive format for code with images than the files with extension .py).  This file is organized in sections according first the index that is being replicated and second the method applied. The indexes considered in this file were DAX30 from Germany, IBEX35 from Spain, BVSP from Brazil, Dow Jones (DJI) from USA and NIFTY50 from India. At the end of the replication of each index there is a section with a comparative analysis of the different methods, based on metrics such as tracking differences, tracking errors, total performance, sharpe index, estimation of costs and liquidity.

