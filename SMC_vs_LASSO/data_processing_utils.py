
## data processing utils common to both SMC and LASSO

import datetime
import pandas as pd
import six

from config import indexVarMapping, DJI, NIFTY

def loadData( indexName = 'DJI' ):
  ''' loads initial index data from the csv file as a dataframe'''
  '''
  Input: indexName whose initial dataset needs to be loaded
  Output: dataframe with index and its constituent closing levels/adjusted levels
  '''
  ##df = pd.read_csv(url,index_col=0)
  df = pd.read_csv('./Data/{}.csv'.format(indexName), index_col=0)

  ## converting index into datetime 
  strIndex = list(df.index)
  dateIndex = []
  for dateString in strIndex:
    dateInd = datetime.datetime.strptime(dateString, '%d-%m-%Y').date()
    dateIndex.append( dateInd )

  df.index = dateIndex
  df.head()
  return df


def generateRollPeriodInfoDict( df , indexName = 'DJI' ):
  ''' generates rollPeriods along with relevant lookback periods '''
  
  '''
  Input: df: dataframe with closing levels/adjsuted levels for an index and its constituents
          indexName: name of the index (for index specific adjustments)
  Output:rollPeriodInfoDict: dictionary with roll period information with keys as (rollPeriodStartDate, rollPeriodEndDate) and 
          values as (lookbackStartDate, lookbackEndDate, period)
  period can be either of the following ['preCovid', 'postCovid']
  '''
  
  '''rollPeriodDict contains info data for a particular roll period
     pre-COVID period, rolling every two month and lookback is 6M
     COVID period (post cutoffDate), rolling every month and lookback is 3M '''

  
  ### finding first business day and last business day of the month
  ### these dates would serve as roll period dates later

  dates = list(df.index)
  dateDf = pd.DataFrame( dates, columns = ['date'] )

  ## convert to datetime (if not already)
  dateDf['date'] = pd.to_datetime( dateDf['date'] )
  dateDf.set_index('date', inplace=True)

  startOfMonth = dateDf.resample('BMS').first()  ## Business Month Start
  endOfMonth = dateDf.resample('BM').first()    ## Business Month End
  ## endOfMonth = dateDf.resample('BME').first() 

  dateList =  pd.concat([startOfMonth, endOfMonth]).drop_duplicates().sort_index()
  startEndDates = dateList.index.date.tolist()  ## converting into dateList

  cutOffDate = datetime.date(2020,2,3)  ### marks the onset of COVID regime
  cutOffIndex = startEndDates.index( cutOffDate )  ## index of cutOffDate

  rollPeriodInfoDict = {}
  if indexName == 'DJI':
      preCovidLength = 2   ## 2 months length of period (preCovid)
      preCovidOffset = 12  ## 6 months lookback (preCovid)
  elif indexName == 'NIFTY':
      preCovidLength = 1
      preCovidOffset = 6 
      
  postCovidLength = 1  ## 1 month length of period (postCovid)
  postCovidOffset = 6  ## 3 months lookback (postCovid)

  ## preCovid
  for i in range(preCovidOffset,cutOffIndex,2*preCovidLength):
    rollPeriodInfoDict[(startEndDates[i], startEndDates[i+2*preCovidLength-1])] = {'start': startEndDates[i-preCovidOffset], 'end': startEndDates[i-1], 'period' : 'preCovid'}

  ## postCovid
  for i in range(cutOffIndex,len(startEndDates)-1,2*postCovidLength):
    rollPeriodInfoDict[(startEndDates[i], startEndDates[i+2*postCovidLength-1])] = {'start': startEndDates[i-postCovidOffset], 'end': startEndDates[i-1], 'period' : 'postCovid'}

  return rollPeriodInfoDict


def generateRebalDates( indexName, startDate, endDate ):
  ''' returns the actual index rebalance dates between the specified startDate and endDate '''
  '''
  Inputs:  index : name of the index in consideration
           startDate: startDate of the lookback period
           endDate: endDate of the lookback period
  Output: list of rebalDates in consideration for generating indec comsposition
  '''
  
  index = indexVarMapping[indexName]
  indexRebalDates = list(index.keys())

  def nearestSmallerDate( indexRebalDates, refStartDate ):
    '''returns nearest SmallerDate for a given refStartDate among given rebalDates '''
    ''' 
    Inputs: indexRebalDates: list containing rebalDates
            refStartDate: reference startDate
    Output: nearest smallerDate for a given refStartDate among given indexRebalDates
    '''
    nearestSmallerDate = None
    for date in indexRebalDates:
      if date < refStartDate:
        nearestSmallerDate = date
    return nearestSmallerDate

  baseRebalDate = [nearestSmallerDate( indexRebalDates, startDate )] if nearestSmallerDate( indexRebalDates, startDate ) is not None else []
  rebalDatesInRange = [date for date in indexRebalDates if startDate <= date <= endDate]
  rebalDates =  baseRebalDate + rebalDatesInRange
  return rebalDates


def generateComposition( indexName, rebalDates, startDate, endDate ):
  ''' generates the composition dataframe for specified startDate and endDate for the lookback period'''
  ''' 
  Inputs: index: name of the index
          rebalDates: list containing actual index rebalance dates in the lookback period
          startDate: startDate of the lookback period
          endDate: endDate of the lookback period
    Outputs: commonStocks: a list containing stocks which were present in the index throughout the composition
             diffStocksDict: a dictionary containing stocks which were added/removed from the index in the lookback period
             commonStockVariableMapping: a dictionary containing mapping of common stocks with variable names
    '''
  
  index = indexVarMapping[indexName]
  
  if len(rebalDates) == 1: ## only single actual index rebalance date during the lookback
    commonStocks = index[rebalDates[0]]
    unionStocks = commonStocks
    diffStocksDict = {}
    
  else: ## multipke actual index rebalance dates during the lookback
    commonStocks = []
    unionStocks = []
    for i in range(len(rebalDates)-1):
      commonStocks.append( set(index[rebalDates[i]]) & set(index[rebalDates[i+1]]) )
      unionStocks.append(  set(index[rebalDates[i]]) | set(index[rebalDates[i+1]]) )
    commonStocks = list(commonStocks[0])
    unionStocks = list(unionStocks[0])
    diffStocks = [ stock for stock in unionStocks if stock not in commonStocks]
    diffStocksDict = {}
    for stock in diffStocks:
      diffStocksDict[stock] = [date for date in rebalDates if stock in index[date]]

  commonStockVariableMapping = {}
  for k in range(1,len(commonStocks)+1):
    commonStockVariableMapping['x'+str(k)] = commonStocks[k-1]
  return commonStocks, diffStocksDict, commonStockVariableMapping

def generateDataFrame( origDF, rollPeriod, rollInfo, index ):
  ''' returns the dataframe which goes into the calculation engine '''
  ''' 
  Inputs: origDF: initial dataframe containing index and composition closing
          rollPeriod: tuple containg the (startDate, endDate) for rollperiod
          rollInfo: dictionary containing info about the rollPeriod
          index: indexName (to be set as 'y' in the output dataframe)          
    Outputs: df: dataframe which feeds into the calculation engine
    '''
  #rebalDates = rollInfo['rebalDates']
  start = rollInfo['start']
  end = rollInfo['end']
  
  commonStocks = rollInfo['commonStocks']
  #diffStocksDict = rollInfo['diffStocksDict']
  commonStockVariableMapping = rollInfo['commonStockVariableMapping']
  
  reverseMapping = {val:key for key,val in six.iteritems(commonStockVariableMapping)}

  columns = commonStocks
  df = origDF[start:end][commonStocks]
  mapping = []
  for stock in columns:
    mapping.append( reverseMapping[stock] )
  df.columns = mapping
  df['y'] = origDF[index].loc[start:end] ## setting the index as the 'y'
  return df
