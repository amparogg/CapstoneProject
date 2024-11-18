## utils for calculating performance metrics and analytics

import matplotlib.pyplot as plt


def getTrackingError( rollPeriod, rollInfo, subset, beta):
  ''' calculates tracking error based on subset selection '''
  ''' 
  Inputs: rollPeriod: tuple containg the (startDate, endDate) for rollperiod
          rollInfo: dictionary containing info about the rollPeriod
          subset: list containing the selected subset
          beta: dictionary containing the respective betas
    Outputs: baseIndexLevels: index levels for the base index
             subsetIndexLevels: index levels for the subset (sparse replication portfolio)
             trackingError: daily average tracking error in bps       
    '''
  
  baseIndexLevels = rollInfo['df']['y']
  baseIndexReturns = baseIndexLevels.pct_change(1)*100

  ## generating the levels for selected subset portfolio
  subsetIndexLevels = (rollInfo['df'][subset]* list(beta.values())).sum(axis=1)
  subsetIndexReturns = subsetIndexLevels.pct_change(1)*100

  absDiffReturns =  abs( subsetIndexReturns - baseIndexReturns )
  absDiffReturns = absDiffReturns.dropna()
  trackingError = sum(absDiffReturns)/len(absDiffReturns) * 100  ## in bps
  return baseIndexLevels, subsetIndexLevels, trackingError

def getAverageTrackingError( rollPeriodInfoDict ):
    ''' returns preCovid and postCovid average tracking error (averaged across all period)'''
    ''' Inputs: rollPeriodInfoDict: dictionary containing rollPeriod information including trackingErrors
        Outputs: preCovidTrackingError: average tracking error (preCovid)
                postCovidTrackingErorr: average tracking error (postCovid)
    '''
    preCovidTrackingError = []
    postCovidTrackingError = []
    
    for rollPeriod, rollInfo in rollPeriodInfoDict.items():
      if rollInfo['period'] == 'preCovid':
        preCovidTrackingError.append( rollInfo['trackingError'])
      else:
        postCovidTrackingError.append( rollInfo['trackingError'])
    
    print( 'PreCovid Tracking Error:', sum(preCovidTrackingError)/len(preCovidTrackingError))
    print( 'PostCovid Tracking Error:', sum(postCovidTrackingError)/len(postCovidTrackingError))

    return preCovidTrackingError, postCovidTrackingError


def getStockSelectionFrequency( df, rollPeriodInfoDict, method = 'LASSO', plot=True ):
    ''' returns the stock selection frequency (preCovid and postCovid)'''
    ''' Inputs: df: initial dataset (for index) (loaded from excel)
                rollPeriodInfoDict:dictionary containing rollPeriod information including trackingErrors
                method: either "SMC" or "LASSO"
                plot: boolean flag (plots results if True)
        Output: stockSelectionFrequency: dictionary containing preCovid and postCovid selection frequency
    '''
    allStocks = list( df.columns )[1:]
    stockSelectionFrequency = {'preCovid': {stock: 0 for stock in allStocks}, 'postCovid': {stock: 0 for stock in allStocks}}
    
    for rollPeriod, rollInfo in rollPeriodInfoDict.items():
       if method == 'LASSO':
           betaSubset = list(rollInfo['betaLasso'].keys())
       else:
           betaSubset = list(rollInfo['betaSMC'].keys())
       commonStockVariableMapping = rollInfo['commonStockVariableMapping']
       selectedStocks = [commonStockVariableMapping[stock] for stock in betaSubset]
    
       for stock in selectedStocks:
         if rollInfo['period'] == 'preCovid':
            stockSelectionFrequency['preCovid'][stock] = stockSelectionFrequency['preCovid'].get(stock,0) + 1
         else:
            stockSelectionFrequency['postCovid'][stock] = stockSelectionFrequency['postCovid'].get(stock,0) + 1
     
    if plot:

        keys = list(stockSelectionFrequency['preCovid'].keys())
        values = list(stockSelectionFrequency['preCovid'].values())
        
        plt.figure(figsize=(15, 6))
        
        plt.bar(keys, values, color='skyblue')
        plt.title('Frequency of Stocks Selected in {} preCovid'.format(method))
        plt.xlabel('Stocks')
        plt.ylabel('Frequency')
        
        # Show the plot
        plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
        plt.tight_layout()  # Adjust layout to make room for rotated labels
        plt.show()
    
        keys = list(stockSelectionFrequency['postCovid'].keys())
        values = list(stockSelectionFrequency['postCovid'].values())
        
        plt.figure(figsize=(15, 6))

        plt.bar(keys, values, color='skyblue')
        plt.title('Frequency of Stocks Selected in {} postCovid'.format(method))
        plt.xlabel('Stocks')
        plt.ylabel('Frequency')
        
        # Show the plot
        plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
        plt.tight_layout()  # Adjust layout to make room for rotated labels
        plt.show()

    return stockSelectionFrequency
 
    
def plotSubsetBetas( rollPeriodInfoDict, relevantPeriods, method='LASSO' ):
    ''' plots subsetBetas for selected relevant rollPeriods'''
    ''' Inputs: rollPeriodInfoDict:dictionary containing rollPeriod information including trackingErrors
                relevantPeriods: list containing relevant rollPeriods
                method: either "SMC" or "LASSO"
        Outputs: plots containing selected subsetBetas 
    '''
    for rollPeriod in relevantPeriods:
        rollInfo = rollPeriodInfoDict[rollPeriod]
        if method == 'LASSO':
            betaSubset = rollInfo['betaLasso']
        else:
            betaSubset = rollInfo['betaSMC']
        
        subsetStocks = {}
        commonStockVariableMapping = rollInfo['commonStockVariableMapping']
        for var in betaSubset:
          subsetStocks[ commonStockVariableMapping[var] ] = betaSubset[var]
      
        stocks =  list(subsetStocks.keys())
        betas = list(subsetStocks.values())
        plt.bar(stocks, betas, color='skyblue', width = 0.5)
        plt.title('Betas for {} Subset Selected from {} to {}'.format(method, rollPeriod[0], rollPeriod[1]))
        plt.xlabel('Stocks')
        plt.ylabel('Betas')
      
        # Show the plot
        plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
        plt.tight_layout()  # Adjust layout to make room for rotated labels
        plt.show()
  
    return