## utils for comparing some metrics for SMC and LASSO

import six

from SMC import getInitialResultsSMC
from LASSO import getInitialResultsLASSO
from data_processing_utils import loadData

import matplotlib.pyplot as plt



def getIndividualPeriodTrackingErros( rollPeriodInfoDictLASSO, rollPeriodInfoDictSMC , indexName):
    ''' compares and plots tracking errors for individual rollPeriods for both methods '''
    '''
    Inputs: rollPeriodInfoDictLASSO: dictionary containing info for LASSO
            rollPeriodInfoDictSMC: dictionary containing info for SMC
    Outputs: trackingErrorsDict: dictionary containing trackingErrors for periods
    '''
    
    trackingErrorsDict = {'LASSO':{}, 'SMC':{}}
    
    if indexName == 'DJI':
        p=6
    elif indexName == 'NIFTY':
        p=10
    
    for rollPeriod, rollInfo in six.iteritems(rollPeriodInfoDictLASSO):
        trackingErrorsDict['LASSO'][rollPeriod] = rollInfo['trackingError']
    
    for rollPeriod, rollInfo in six.iteritems(rollPeriodInfoDictSMC):
        trackingErrorsDict['SMC'][rollPeriod] = rollInfo['trackingError']    
    
    lassoTrackingErrors = list(trackingErrorsDict['LASSO'].values())
    smcTrackingErrors = list(trackingErrorsDict['SMC'].values())
    keys = [i for i in range(1, len(lassoTrackingErrors)+1)]
    
    plt.plot( keys, lassoTrackingErrors, color='g', marker='o', label='LASSO Tracking Errors' )
    plt.plot(keys, lassoTrackingErrors, color='g', linestyle='-', marker='o')
    
    
    plt.plot( keys, smcTrackingErrors, color='b', marker='o', label='SMC Tracking Errors' )
    plt.plot(keys, smcTrackingErrors, color='b', linestyle='-', marker='o')
    
    plt.title( 'Tracking Errors for individual roll periods (p={})'.format(p) )
    plt.xlabel( 'Roll Period Number' )
    plt.ylabel( 'Tracking Errors (in bps)')
    plt.legend(title="Tracking Errors (in bps)", loc='upper right', fontsize=12)
    plt.show()   
    
    return trackingErrorsDict

def main():
    indexName = 'NIFTY'    ## choices for indexName : ['NIFTY', 'DJI']
    
    if indexName == 'NIFTY':
      numStocks = 10
    elif indexName == 'DJI':
      numStocks = 6
    
    ### runs all the utils and generates the initial set of results
    df = loadData(indexName)
    
    rollPeriodInfoDictLASSO = getInitialResultsLASSO(df, indexName, numStocks)
    
    rollPeriodInfoDictSMC = getInitialResultsSMC(df, indexName, numStocks)
    
    trackingErrorsDict = getIndividualPeriodTrackingErros( rollPeriodInfoDictLASSO, rollPeriodInfoDictSMC, indexName )
    
    count1=0
    count2=0
    for rollPeriod, rollInfo in six.iteritems( rollPeriodInfoDictLASSO ):
        if trackingErrorsDict['LASSO'][rollPeriod] > trackingErrorsDict['SMC'][rollPeriod]:
            count1+=1
        else:
            count2+=1
    
    return trackingErrorsDict, count1, count2