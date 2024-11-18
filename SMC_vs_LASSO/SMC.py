## generating SMC results for different indices

import datetime
from data_processing_utils import loadData, generateRollPeriodInfoDict, generateRebalDates, generateComposition, generateDataFrame
from LASSO_utils import linearRegBeta
from performance_utils import getTrackingError, getAverageTrackingError, getStockSelectionFrequency, plotSubsetBetas

from SMC_utils import selectSubsetSMC


def getInitialResultsSMC(df, indexName, numStocks):
    ''' generates initial results (trackingError) for an index'''
    ''' Inputs: df: initial dataset (loaded from excel sheet)
                indexName: indexName 
                numStocks: number of stocks in the sparse replication portfolio
        Output: rollPeriodInfoDict: dictionary containing initial set of results including trackingError for selected subset
        used for sparse replication '''
    rollPeriodInfoDict = generateRollPeriodInfoDict( df, indexName )
    
    ## params specific to SMC
    p = numStocks
    nSamples = 100
    maxStepSize = 0.05
    cumulativeAcceptanceRate = 500
    
    ## master loop for all rollPeriods
    ## storing results inside the rollPeriodInfoDict
    for rollPeriod, rollInfo in rollPeriodInfoDict.items():
      print( rollPeriod )
      rebalDates = generateRebalDates( indexName, rollInfo['start'], rollInfo['end'] )
      rollPeriodInfoDict[ rollPeriod ]['rebalDates'] = rebalDates
      commonStocks, diffStocksDict, commonStockVariableMapping =  generateComposition( indexName, rebalDates, rollInfo['start'], rollInfo['end'])
      rollPeriodInfoDict[ rollPeriod ]['commonStocks'] = commonStocks
      rollPeriodInfoDict[ rollPeriod ]['diffStocksDict'] = diffStocksDict
      rollPeriodInfoDict[ rollPeriod ]['commonStockVariableMapping'] = commonStockVariableMapping
      rollPeriodInfoDict[ rollPeriod ]['df'] = generateDataFrame( df, rollPeriod, rollInfo, indexName)
      subset, X, y = selectSubsetSMC( rollPeriod, rollInfo , p, nSamples, maxStepSize, cumulativeAcceptanceRate )
      betaSMC = linearRegBeta(subset, X, y)
      baseIndexLevels, smcIndexLevels, trackingError = getTrackingError( rollPeriod, rollInfo, subset, betaSMC )
      rollInfo['betaSMC'] = betaSMC
      rollInfo['smcIndexLevels'] = smcIndexLevels
      rollInfo['baseIndexLevels'] = baseIndexLevels
      rollInfo['trackingError'] = trackingError

    return rollPeriodInfoDict


def getTrackingErrorVsNumberOfStocks( df, indexName, numStocksList ):
    ''' generates trackingErorr (averaged) for varying number of stocks in the subset'''
    '''
    Inputs:  df: initial dataset (loaded from excel sheet)
             indexName: indexName 
             numStocksList: list containing number of stocks in the sparse replication portfolio
    Output: preCovidTrackingErrorDict: dictionary containing preCovid average tracking erorr for different number of stocks in the subset
            postCovidTrackingErrorDict: dictionary containing postCovid average tracking erorr for different number of stocks in the subset'''
    
    
    ## params specific to LASSO
    nSamples = 100
    maxStepSize = 0.05
    cumulativeAcceptanceRate = 500
    
    preCovidTrackingErorrDict, postCovidTrackingErorrDict = {}, {}
    
    for numStocks in numStocksList:
        rollPeriodInfoDict = generateRollPeriodInfoDict( df, indexName  )
        p = numStocks
        ## master loop for all rollPeriods
        ## storing results inside the rollPeriodInfoDict
        for rollPeriod, rollInfo in rollPeriodInfoDict.items():
            print(rollPeriod)
            rebalDates = generateRebalDates( indexName, rollInfo['start'], rollInfo['end'] )
            rollPeriodInfoDict[ rollPeriod ]['rebalDates'] = rebalDates
            commonStocks, diffStocksDict, commonStockVariableMapping =  generateComposition( indexName, rebalDates, rollInfo['start'], rollInfo['end'])
            rollPeriodInfoDict[ rollPeriod ]['commonStocks'] = commonStocks
            rollPeriodInfoDict[ rollPeriod ]['diffStocksDict'] = diffStocksDict
            rollPeriodInfoDict[ rollPeriod ]['commonStockVariableMapping'] = commonStockVariableMapping
            rollPeriodInfoDict[ rollPeriod ]['df'] = generateDataFrame( df, rollPeriod, rollInfo, indexName)

            subset, X, y = selectSubsetSMC( rollPeriod, rollInfo , p, nSamples, maxStepSize, cumulativeAcceptanceRate)
            betaSMC = linearRegBeta(subset, X, y)
            baseIndexLevels, smcIndexLevels, trackingError = getTrackingError( rollPeriod, rollInfo, subset, betaSMC )
            rollInfo['betaSMC'] = betaSMC
            rollInfo['smcIndexLevels'] = smcIndexLevels
            rollInfo['baseIndexLevels'] = baseIndexLevels
            rollInfo['trackingError'] = trackingError
            
        preCovidTrackingError = []
        postCovidTrackingError = []

        for rollPeriod, rollInfo in rollPeriodInfoDict.items():
          if rollInfo['period'] == 'preCovid':
            preCovidTrackingError.append( rollInfo['trackingError'])
          else:
            postCovidTrackingError.append( rollInfo['trackingError'])
        print( 'Number of Stocks Selected', numStocks)
        print( 'PreCovid Tracking Error:', sum(preCovidTrackingError)/len(preCovidTrackingError))
        print( 'PostCovid Tracking Error:', sum(postCovidTrackingError)/len(postCovidTrackingError))
        
        preCovidTrackingErorrDict[numStocks] = sum(preCovidTrackingError)/len(preCovidTrackingError)
        postCovidTrackingErorrDict[numStocks] = sum(postCovidTrackingError)/len(postCovidTrackingError)
    
    return preCovidTrackingErorrDict, postCovidTrackingErorrDict


def main():
    
    indexName = 'NIFTY'  ## choices for indexName : ['NIFTY', 'DJI']
    
    if indexName == 'NIFTY':
      numStocks = 10
    elif indexName == 'DJI':
      numStocks = 6
    
    ### runs all the utils and generates the initial set of results
    df = loadData(indexName)
    rollPeriodInfoDict = getInitialResultsSMC(df, indexName, numStocks)
    
    ## generates preCovid and postCovid trackingErrors
    preCovidTrackingError, postCovidTrackingError = getAverageTrackingError( rollPeriodInfoDict )
        
    ## generating stockSelectionFrequency
    getStockSelectionFrequency(df, rollPeriodInfoDict, "SMC")
    
    ## plotting selected subsetBetas for selected rollPeriods
    ## plotting selected subsetBetas for selected rollPeriods
    if indexName == 'DJI':
        relevantPeriods = [(datetime.date(2015, 10, 1), datetime.date(2015, 11, 30)),
                    (datetime.date(2018, 4, 2), datetime.date(2018, 5, 31)),
                     (datetime.date(2020, 2, 3), datetime.date(2020,2,28)),
                    (datetime.date(2024, 9, 2), datetime.date(2024,9,30))]
    elif indexName == 'NIFTY':
        relevantPeriods = [(datetime.date(2015, 7, 1), datetime.date(2015, 7, 31)),
                      (datetime.date(2018, 4, 2), datetime.date(2018, 4, 30)),
                      (datetime.date(2020, 2, 3), datetime.date(2020,2,28)),
                     (datetime.date(2024, 9, 2), datetime.date(2024,9,30))]
    
   
    plotSubsetBetas(rollPeriodInfoDict, relevantPeriods, "SMC")
    
    ## average tracking errors for varying number of stocks in the selected subset
    ## any calls to getTrackingErrorVsNumberOfStocks( df, indexName, numStocksList ) take a lot of time especially if computations are being done for SMC method
    if indexName == 'DJI':
       numStocksList = [6,8,10,12,14,16,18]
       preCovidTrackingErrorDict, postCovidTrackingErrorDict = getTrackingErrorVsNumberOfStocks( df, indexName, numStocksList )
    
    return