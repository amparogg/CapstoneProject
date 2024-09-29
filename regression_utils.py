# -*- coding: utf-8 -*-


from __future__ import division, print_function
import numpy
from sklearn import linear_model



def GetRegressionSSE( x, y, fit_intercept ):
    """
    gets Sum of Squares of Error in Regression
    """
    reg = linear_model.LinearRegression( fit_intercept=fit_intercept )
    reg.fit(x, y)
    yhat = reg.predict(x)
    return numpy.sum( numpy.square( y - yhat ))


def GetRegressionSquare(x, y, fit_intercept ):
    """
    function to get Linear Regression RSquare 
    """
    reg = linear_model.LinearRegression( fit_intercept=fit_intercept )
    reg.fit(x, y)
    return reg.score(x,y)


def findRegCoef(x, y, fit_intercept=True):
    """ function to find the regression coefficients of y with a permutation """
    ## we assume that the target variable is defined by 'y'
    reg = linear_model.LinearRegression( fit_intercept=fit_intercept )
    reg.fit(x, y)
    return reg.coef_


