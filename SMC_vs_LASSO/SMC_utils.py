## utils for implementing SMC

### basic utility functions based on linear regression
from __future__ import division, print_function
from builtins import range
import numpy.random
import numpy as np
import six
import math

from sklearn import linear_model
import copy

#------------------------------------------------------------------------

def calculateNewSampleProbability( probabilityDict, features, toBeRetainedList, replacementList ):
   ''' calculates P(replacementList/toBeReplacedList) based on provided sampler'''
   '''
   Inputs: probabilityDict:  numpy array with weights/probabilities assigned to each variable/feature based on some metric
            features: list containing 'x1','x2','x3' .. as independent variables
            toBeRetainedList: list of features to be retained in the new sample
            replacementList: list of features incoming in the new sample
   Output: newSampleProbability : a scalar representing the P(replacementList/toBeReplacedList)
    '''
    
   possibleReplacements = [feature for feature in features if feature not in toBeRetainedList]
   updatedProbabilityDict = {}
   for k in possibleReplacements:
       updatedProbabilityDict[k] = probabilityDict[k]
   sumUpdatedProbDict = sum(updatedProbabilityDict.values())
   updatedProbabilityDict = {k:v/sumUpdatedProbDict for k, v in six.iteritems( updatedProbabilityDict ) }
   newSampleProbability = calculateSampleProbability( updatedProbabilityDict, np.array( replacementList ) )
   return newSampleProbability


def calculateCorrectionFactor( oldSample, newSampleData, relativeFreqDict, individualProbDict, features ):
  ''' calculates correction factor used while calculating acceptance probability'''
  '''
  Inputs: oldSample: an individual permutation (old permutation)
          newSamplesData: list containing info (replacedElements, retainedElements, replacementElements) for given sample
          relativeFreqDict: dictionary containing relative frequencies of occurrence of features
          individualProbDict: dictionary with weights/probabilities assigned to each variable/feature based on its RSquare
          features: list containing 'x1','x2','x3' .. as independent variables
  Outputs: correctionFactor: a scalar representing the correctionFactor (as per the SMC paper)  
  Proposal(currentSample|newSample)/Proposal(newSample|currentSample)
  '''
  
  toBeReplacedList = newSampleData[0]
  toBeRetainedList = newSampleData[1]
  replacementsList= newSampleData[2]
  
  newSample = np.array( toBeReplacedList + replacementsList )
  
  ### calculating proposalOldGivenNew  
  proposalIndividualProbabilities = calculateNewSampleProbability(individualProbDict, features, toBeRetainedList, toBeReplacedList)
  proposalRelativeFreq = calculateNewSampleProbability(relativeFreqDict, features, toBeRetainedList, toBeReplacedList)
  proposalMix = 0.5*proposalIndividualProbabilities + 0.5*proposalRelativeFreq
  proposalCurrentGivenNew = 1/math.comb( len(newSample), len(toBeReplacedList) ) * proposalMix
  
  
  ### calculating proposalNewGivenOld 
  proposalIndividualProbabilities = calculateNewSampleProbability(individualProbDict, features, toBeRetainedList, replacementsList)
  proposalRelativeFreq = calculateNewSampleProbability(relativeFreqDict, features, toBeRetainedList, replacementsList)
  proposalMix = 0.5*proposalIndividualProbabilities + 0.5*proposalRelativeFreq
  proposalNewGivenCurrent = 1/math.comb( len(newSample), len(replacementsList) ) * proposalMix

  return proposalCurrentGivenNew/proposalNewGivenCurrent


def calculateAcceptanceProbability( gamma, oldSample, newSampleData, relativeFreqDict, individualProbDict, features, X, y, nMove, useSeed=True, eta=1 ):
  ''' calculates the acceptance probability for the new sample '''
  '''
  Inputs:  gamma: current value of gamma as per the SMC model
           oldSample: an individual permutation (old permutation)
           newSamplesData: list containing info (replacedElements, retainedElements, replacementElements) for given sample
           relativeFreqDict: dictionary containing relative frequencies of occurrence of features
           individualProbDict: dictionary with weights/probabilities assigned to each variable/feature based on its RSquare
           features: list containing 'x1','x2','x3' .. as independent variables
           X: pandas dataframe containing 'x1','x2','x3' .. as independent variables
           y: pandas series representing dependent variable
           nMove: number of MH move we are at
           useSeed = a boolean flag for fixing seed of random number generator
           eta: constant as per the SMC model
  Outputs: acceptanceProb: (as obtained by Metropolis-Hastings Algorithm)
  '''
  ### calculating Target Ratio
  
  newSample = np.array( newSampleData[1] + newSampleData[2] )

  initialProposalTerm = np.power( calculateSampleProbability(individualProbDict, newSample)/calculateSampleProbability(individualProbDict, oldSample), 1-gamma)
  
  sumSquaredErrorsOldSamples = getLinearRegressionMetric( X[oldSample], y, 'SSE' ) 
  sumSquaredErrorsNewSamples = getLinearRegressionMetric( X[newSample], y, 'SSE' ) 
  
  targetProposalTerm = np.exp( -gamma * ( sumSquaredErrorsNewSamples - sumSquaredErrorsOldSamples)/eta )
  targetRatioTerm = initialProposalTerm * targetProposalTerm
  correctionFactor = calculateCorrectionFactor( oldSample, newSampleData, relativeFreqDict, individualProbDict, features)
  
  return min(1, targetRatioTerm * correctionFactor)
    

def performSupportBoosting( gamma, samples, individualProbDict, features, X, y, nMove, useSeed=True, eta=1):
  '''generates the new set of particles after implementing one round of Metropolis Hastings algorithm'''
  '''
  Inputs: gamma: current value of gamma as per the SMC model
          samples: list of numpy arrays representing individual permutations
          individualProbDict : a dictionary with weights/probabilities assigned to each variable/feature based on its RSquare
          features: list containing 'x1','x2','x3' .. as independent variables
          X: pandas dataframe containing 'x1','x2','x3' .. as independent variables
          y: pandas series representing dependent variable
          nMove: number of MH move we are at
          useSeed: a boolean flag for fixing seed of random number generator
          eta: constant as per the SMC model
  Output: newSamples: list of numpy arrays representing individual permutations after the MH(Metropolis Hastings) move
  '''
    
  ## getting weights of features/varNames based on frequency of occurrence
  relativeFreqDict = getFeatureWeightsBasedOnFrequency( features, samples)
  tSamples = copy.deepcopy( samples )
  
  newSamplesData = replaceElementsInSamples( relativeFreqDict, individualProbDict, tSamples, features, nMove )

  newSamples = []
  for i in range(len(tSamples)):
      acceptanceProbability = calculateAcceptanceProbability(gamma, tSamples[i], newSamplesData[i], relativeFreqDict, individualProbDict, features, X, y,nMove, useSeed=True, eta=1 )
      newSample = np.array( newSamplesData[i][1] + newSamplesData[i][2] )

      if acceptanceProbability >= newSamplesData[i][3]: ## uniformRandomNumber
          newSamples.append( newSample )
      else:
          newSamples.append( tSamples[i] )
          
  return newSamples


def replaceElementsInSamples( relativeFreqDict , individualProbDict, tSamples, features, nMove, useSeed=True):
  ''' generates newSamples based on given samples and relativeFrequencies '''
  '''
  Inputs: relativeFreqDict: dictionary containing relative frequencies of occurrence of features
         individualProbDict: dictionary with weights/probabilities assigned to each variable/feature based on its RSquare
         tSamples: list of numpy arrays representing individual permutations
         features: list containing 'x1','x2','x3' .. as independent variables
         nMove: number of MH move we are at
         useSeed = a boolean flag for fixing seed of random number generator
  Output: newSamplesData: a list of list containing info about individual samples
          (toBeReplaced elements, toBeRetained elements, replacements)
  '''
       
  newSamplesData = []      
  for i in range(0, len(tSamples) ):
      sample = tSamples[i]
      
      np.random.seed( i+nMove )
      nReplaced = np.random.randint( 1, len(sample) )  ## number of features to be replaced
      
      np.random.seed( i+nMove )
      toBeReplacedList = list(np.random.choice( sample, nReplaced, replace = False))
      
      toBeRetainedList = [ feature for feature in sample if feature not in toBeReplacedList ]
      possibleReplacements = [feature for feature in features if feature not in toBeRetainedList]
      
      sumProb1 = sum([relativeFreqDict[feature] for feature in possibleReplacements ])  ## relativeFreqDict (scanning)
      sumProb2 = sum([individualProbDict[feature] for feature in possibleReplacements ]) ## individualProbDict (scanning)
      
      prob1 =  [relativeFreqDict[feature]/sumProb1 for feature in possibleReplacements ]
      prob2 =  [individualProbDict[feature]/sumProb2 for feature in possibleReplacements ]
       
      np.random.seed( i+nMove )
      choice = np.random.choice( ['A', 'B'], 1, True, [0.5,0.5] )
      
      if choice == 'A':
          ## sampling based on relativeFreqDict
          np.random.seed( i+nMove )
          replacementsList = list( np.random.choice( possibleReplacements, len(toBeReplacedList), False, prob2 ))
      else:
          ## sampling based on individualProbDict (RSquare)
          np.random.seed( i+nMove )
          replacementsList = list( np.random.choice( possibleReplacements, len(toBeReplacedList), False, prob1 ))
      
      np.random.seed( i+nMove )
      randomNumber = np.random.uniform()
      newSamplesData.append( [toBeReplacedList, toBeRetainedList, replacementsList, randomNumber] )
  return newSamplesData



##-----------------------------------------------------------------------------
def calculateSampleProbability( individualProbDict, sample ):
  ''' calculates the probability of picking the sample (without replacement) based on individual probability '''
  '''Inputs: individualProbDict : a dictionary with weights/probabilities assigned to each variable/feature based on its RSquare
          sample: a numpy array representing the permutation/sample selected
  Output: sampleProbability: represents the required probability of picking up the sample/permutation
  '''
  sampleSize = len(sample)

  pIndividual = np.ones( sampleSize )
  probabilities = np.ones( sampleSize )
  sumProbability = 0.0
  
  for i in range( sampleSize ):
    ## extracting the variable after removing 'x'
    pIndividual[i] = individualProbDict[ sample[i] ]  
    Numerator = pIndividual[i]
    if i==0:
      Denominator = 1.0
    else:
      sumProbability = sumProbability + pIndividual[i-1]## updating the Numerator and Denominator
      Denominator = 1 - sumProbability
    probabilities[i] = Numerator/Denominator
  sampleProbability = np.prod(probabilities)
  assert(sampleProbability >= 0), 'Negative Probability Detected based on provided weights'
  return sampleProbability


def getNewWeightsAndGamma( gamma, samples, individualProbDict, importanceWeights, X, y, stepSize, eta=1):
  ''' returns the new weights of permutations/samples and next gamma '''
  '''Inputs: gamma: prevGamma as per  per SMC model
          samples: list of numpy arrays representing individual permutations
          individualProbDict : a dictionary with weights/probabilities assigned to each variable/feature based on its RSquare
          importanceWeights: a numpy array assigning importance weights assigned to each sample
          X: pandas dataframe containing 'x1','x2','x3' .. as independent variables
          y: pandas series representing dependent variable
          stepSize: for adding increments in gamma
          eta: constant defined in the formulation
  Outputs: newGamma: newGamma after the increment
           updatedImportanceWeights: new importance weights
      tuple with newgamma and updated weights array
  '''

  newGamma = min( stepSize + gamma, 1) ## limit on gamma is 1
  n = len(samples)  
  newImportanceWeights = np.zeros( len(importanceWeights) )
  for i in range(n):
    sample = samples[i]
    sumSquaredErrors = getLinearRegressionMetric( X[sample], y, 'SSE' )
    proposalProbability = calculateSampleProbability(individualProbDict, sample) ## I(P) term
    newImportanceWeights[i]=importanceWeights[i]*np.power( np.exp(-sumSquaredErrors)/proposalProbability, (newGamma-gamma))
      
  return newGamma, newImportanceWeights/newImportanceWeights.sum()


##-----------------------------------------------------------------------------
def getLinearRegressionMetric( X, y, metric ):
  ''' returns the specified metric pertaining to linear regression '''
  ''' Inputs: X: independent variables/features
              y: dependent variable
              metric: can be either of the following [SSE, RSquare, Betas]
      Output: specified metric for the linear regression 
  '''
  regressionModel = linear_model.LinearRegression( )
  regressionModel.fit(X, y)
  
  assert metric in ['RSquare', 'Betas', 'SSE'], 'metric not implemented, please implement it'
  if metric == 'RSquare':
      return regressionModel.score(X, y)
  elif metric == 'Betas':
      return regressionModel.coef_
  elif metric == 'SSE':
      predictions = regressionModel.predict(X)
      squaredErrors = np.square( y - predictions )
      return np.sum( squaredErrors )
  

def getIndividualRegressorWeightsByRSquare( X,y ):
  ''' calculates the probability of sampling individual variables/regressor based on RSquare values '''
  '''Input : X: pandas dataframe containing 'x1','x2','x3' .. as independent variables
             y: pandas series representing dependent variable
     Output : weights: a dictionary with weights assigned to each variable/feature based on its RSquare
  '''

  weights = {}
  features = X.columns.tolist()
  
  for i in range(len(features)):
      variable = features[i]
      Xind = X[variable]  ## individual feature isolated 
      rSquare = getLinearRegressionMetric( Xind.values.reshape(-1,1), y , 'RSquare')
      assert( rSquare > 0 ), 'negative RSquare detected for {}'.format( variable )
      weights[variable] = rSquare
  
  ## normalizing weights
  weightsSum = sum(list(weights.values()))
  weights = {k:v/weightsSum for k,v in six.iteritems(weights)}
  return weights

##----------------------------------------------------------------------------------

def getInitialSamples( X , weightsDict, p , nSamples ):
  ''' returns samples on the basis of R**2 values '''
  '''
  Inputs: X: a pandas dataframe with featurees as columns  'x1','x2'...
         weightsDict: individual probabilities based on R**2 values
          p : number of stocks in the subset portfolio
         nSamples= number of samples to be used in approximating the sampler
                (sampling without replacement)
  Outputs: samples: a list of numpy array of strings representing some permutation 
  '''
  
  varNames = np.array( ['x'+str(i) for i in range(1,X.shape[1]+1)] )
  weights = np.array(list(weightsDict.values()))
  samples = []
  for i in range(0, nSamples):
      numpy.random.seed( i )
      samples.append( np.random.choice( varNames, p, False, weights ))
  return samples

##------------------------------------------------------------------------------

def getFeatureWeightsBasedOnFrequency( features, samples ):
  ''' returns a list containing individual frequencies of different variables/features inside samples '''
  '''Inputs: features: list of features/variables
             samples: list of numpy arrays representing individual permutations
     Output: relativeFreqDict: dictionary represnting relative normalized frequency of each feature/variable '''
  
  from collections import Counter
  counts = Counter(x for sublist in samples for x in sublist)
  
  sumValues = sum(list(counts.values()))
  relativeFreqDict = {i:j/sumValues for i,j in six.iteritems(counts)}
  keys = list(relativeFreqDict.keys())
  
  zeroFreq = list(set(features) - set(keys))  ## elements with zeroFreq
  for i in zeroFreq:
      relativeFreqDict[i] = 0
  
  return relativeFreqDict

########-----------------------------------------------------------------------

def performReSampling( samples, nSamples, importanceWeights ):
  ''' performs resapmling and returns newSamples'''
  '''
  Inputs: samples: list of numpy arrays representing individual permutations
            nSamples: number of samples
            importanceWeights: importanceWeights used for resampling
  Output: reSamples: list of numpy arrays representing individual permutations (after resampling)
  '''
  reSampledOrder = np.random.choice( [i for i in range(0,nSamples)], nSamples, True, importanceWeights )
  reSamples = np.array( samples )
  reSamples = list( reSamples[reSampledOrder] )
  return reSamples


def calculatePercentageAcceptanceRate( newSamples, oldSamples ):
  ''' calculates the percentage of samples accepted for the Metropolis Hastings move '''
  '''
  Inputs: newSamples: numpy array containing new samples after MH move
          oldSamples: numpy array containing samples from the previous iteration
  Output: percentageAcceptanceRate: scalar containing the percentage acceptance rate
  '''
  count = 0
  for i in range(0,len(oldSamples)):
      if not np.array_equal( oldSamples[i], newSamples[i] ):
          count += 1
    
  percentageAcceptanceRate = 100*(count)/len(newSamples)
  return percentageAcceptanceRate


def selectSubsetSMC( rollPeriod, rollInfo, p, nSamples, maxStepSize, cumulativeAcceptanceRate, minStepSize=0.01, eta=1, seed=1):
  """ returns the optimal permutation/subset for a fixed number of regressors """

  ## fetching the initialDf
  initDf = rollInfo['df']                                                                    
  initDf = initDf.dropna(axis=1) ## removing NaNs (columns with NaNs)

  ## calculating returnsDf
  returnsDf = initDf.pct_change(1) * 100
  returnsDf = returnsDf.dropna()
  X = returnsDf[['x'+str(i) for i in range(1,returnsDf.shape[1])]]
  y = returnsDf['y']
  
  
  ## gets individual feature weights by Rsquare (linear regression : y vs individual feature)
  individualProbDict = getIndividualRegressorWeightsByRSquare( X,y ) 
  
  if any(weight<0 for feature, weight in six.iteritems(individualProbDict)):
      ### if negative RSquare detected for some variables/features, assigning equal weight to all the features (1/n)
      n = len(individualProbDict)
      individualProbDict = np.array([1/n]*n)

   
  ## generating initial samples 
  ## initial sample of particles #wt is the weights as per initialization for each of the regressor
  samples = getInitialSamples( X, individualProbDict, p, nSamples )
  features = X.columns.tolist()
  
  importanceWeights = np.array( [1/nSamples]* nSamples )  ## setting equal weights to all the samples
  stepSize = minStepSize  ## 0.01

  sampleCountList = []
  gamma = 0

  while (gamma<1) :

    sampleCountList.append( getFeatureWeightsBasedOnFrequency( features, samples ) )
    newGamma, newImportanceWeights = getNewWeightsAndGamma( gamma, samples, individualProbDict, importanceWeights, X, y, stepSize, eta=1 )
    ESS = np.power( np.sum(newImportanceWeights),2 )/numpy.sum( np.power(newImportanceWeights,2 ) )


    if ( ESS <= 0.5*nSamples ):

      stepSize = minStepSize

      ## Resampling---------
      np.random.seed( seed )
      samples = performReSampling( samples, nSamples, newImportanceWeights ) ## resampling step
      ## Resampling--------------------------------
     
      ## Support Boosting -------------------------------
      nMove = 0
      acceptanceRatesList = []
      
      nMove = 0
      while( sum(acceptanceRatesList) < cumulativeAcceptanceRate ):
        newSamples = performSupportBoosting(gamma, samples, individualProbDict, features, X, y, nMove)
        acceptanceRate = calculatePercentageAcceptanceRate(newSamples, samples)
        acceptanceRatesList.append( acceptanceRate )
        nMove = nMove+1
        samples =  copy.deepcopy( newSamples )
           
        if ( nMove >=3) and ( sum(acceptanceRatesList[-3:] ) < 50):  
            break
      ##-------------------------------------------------------
      importanceWeights = np.array( [1/nSamples]* nSamples )  
      
    else:
      ## stepSize = max_stepSize
      stepSize = min( stepSize*2, maxStepSize )
      importanceWeights = copy.deepcopy( newImportanceWeights )
    gamma = newGamma

  ### calculating permutation with min SSE
  SSEList = []
  for sample in samples:
      SSEList.append( getLinearRegressionMetric( X[sample], y, 'SSE' ) )
  return samples[ np.argmin(SSEList) ], X, y