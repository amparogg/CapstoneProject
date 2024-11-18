
## utils for implementing LASSO


def selectSubsetLASSO( rollPeriod, rollInfo, numStocks ):
  ''' generates the initial subset via LARS (Least Angle Regression)'''
  ''' 
  Inputs: rollPeriod: tuple containg the (startDate, endDate) for rollperiod
          rollInfo: dictionary containing info about the rollPeriod
          numStocks: number of stocks in the subset
    Outputs: subSet: list containing the selected stocks
             alpha: final alpha inside the LARS algorithm
             X: X to be provided to the linear regressor 
             y: y to be provided to the linear regressor
    '''  
  
    
  initDf = rollInfo['df']                                                                    
  initDf = initDf.dropna(axis=1) ## removing NaNs (columns with NaNs)
  
  ## calculating returnsDf
  returnsDf = initDf.pct_change(1) * 100
  returnsDf = returnsDf.dropna()
  X = returnsDf[['x'+str(i) for i in range(1,returnsDf.shape[1])]]
  y = returnsDf['y']

  ## performing LARS regression
  from sklearn import linear_model

  alpha = 0.1
  dAlpha = 0.01
  reg = linear_model.LassoLars(alpha=alpha)  ## doing initial fit
  reg.fit(X,y)
  
  ## if initial fit results in a subset containing less than or equal to numStocks
  if len(list(filter(lambda x:x!=0, reg.coef_))) <= numStocks:
    subSet = [ 'x'+str(i+1) for i in range(len(reg.coef_)) if reg.coef_[i]!=0]

  ## otherwise, increase alpha (alpha --> alpha + dAlpha)
  else:
    while len(list(filter(lambda x:x!=0, reg.coef_))) > numStocks:
      alpha += dAlpha
      reg = linear_model.LassoLars(alpha=alpha)
      reg.fit(X,y)

  subSet = [ 'x'+str(i+1) for i in range(len(reg.coef_)) if reg.coef_[i]!=0]
  return subSet, alpha, X, y

def linearRegBeta( subset, X, y):
  """ performs linear regression on the selected dataset and returns the selected betas"""
  ''' 
  Inputs: subset: list containing the selected variables (stocks)
          X: original X (contains constituents returns)
          y: contains index returns
    Outputs: betas: dictionary containing the linear regression betas for the selected subset
    '''  
  from sklearn import linear_model
  reg = linear_model.LinearRegression()
  reg.fit(X[subset],y)
  coeff = reg.coef_
  betas = dict(zip(subset, coeff))

  return betas


