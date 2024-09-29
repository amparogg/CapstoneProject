# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy

from builtins import range
import multiprocessing as mp
import regression_utils as RU
from functools import partial
import lib.logging
import platform


def ComputeLogIncrementalWeight( i, particles, df, indprob, newgamma, oldgamma, eta, fit_intercept ):
    """
    function to update weights using the formula for incremental weights 
    Inputs: particles is a list of numpy arrays. each array is of type string array(['x1','x2','x6'])
    df is the dataframe with some 'y' and other columns as 'x1','x2' etc
    indprob is a numpy array of probabilities of selecting individual regressors
    oldgamma, newgamma as per the model
    i is the index of the particles for which we find the result
    output is a scalar
    we compute log incremental weights instead of incremental weights
    due to the numerical precision issues faced by the inc weight expression
    we use same tricks commonly used in log likelihood calculations
    the expression of incremental weight (and its log) is as per the paper
    
    """
    Y = df['y']
    perm = particles[i]
    SSE = RU.GetRegressionSSE( df[perm], Y, fit_intercept )
    expnt = -1*(SSE/eta)
    initprob = findInitPermProb( perm, indprob )
    logincwt = (newgamma-oldgamma)* (expnt - numpy.log(initprob))  ## current
    # lib.logging.info( ' log inc wt' )
    # lib.loggin.info( logincwt )
    return (i,logincwt) ## current


def findInitPermProb(perm, wt):  
    ## this finds the probabilities of a permutation as per initialization phase weights.
    ## we assume date is of the form y for target and x1,x2,..for the regressors
    
    """
    function to find the initial permutation's probability
    
    Inputs: perm is a numpy array of strings. eg. array(['x1','x2','x5'])
    df is the dataframe with some 'y' and other columns as 'x1','x2' etc.
    output is a scalar which is the required probability of picking the permutation 
    """
    
    """
    if the individual vars have probability  p1,p2,p3 ..etc
    suppose ith var is selected first
    the probability of selection of ith var = pi
    suppose jth var is selected 2nd
    the probability of selection of jth var =  pj/(1-pi)
    suppose kth var is selected 3rd
    the probability of selection of kth var = pk/(1- (pi+pj))
    """
    
    perm_size = perm.size
    p_individual = numpy.zeros( perm_size )
    prob = numpy.ones( perm_size )
    sum_prob = 0.0
    
    for i in range( perm_size ):
        tempvar = perm[i].lstrip('x')  ## if regressor name is x11, we want the number hence we strip 'x'
        tempvar = int(tempvar)
        p_individual[i] = wt[(tempvar)-1] ## we subtract one since the features start from x1 and indexing  starts from 0
        Nr = p_individual[i]
        if i==0:
            Dr = 1.0
        else:
            sum_prob =  sum_prob + p_individual[i-1]
            Dr = 1-sum_prob
        prob[i] = Nr/Dr
        ans = numpy.product(prob)
        assert(ans>0), 'negative probabilty'
        return ans
    

def UpdateWeightsAndGamma( oldgamma, df, particles, indprob, weightarr, stepSize, eta=1, fit_intercept=True):
    """
    function to find the updated weights of particles and next gamma
    Inputs: oldgamma as per model
    df is the dataframe with some 'y' and other columns as 'x1','x2'..
    particles is a list of numpy arrays. each array is of type string .array(['x1','x2','x6']) 
    indprob is a numpy array of probabilities of selecting individual regressors
    weightarr = weights corresponding to the particles
    stepSize = stepsize to increment gamma
    numThreads: threads to be specified in forked apply
    output is a tuple with newgamma and updated weight array. updated weight = weight*incremental weight
    
    """
    n = len(particles)
    newgamma = min(oldgamma + stepSize, 1)
    
    os_type = platform.system()
    
    if os_type != 'Windows':
        i = [j for j in range(n)]
        cpu_count = mp.cpu_count()
        lib.logging.info('inside UpdateWeightsAndGamma')
        lib.logging.info('cou cores available')
        lib.logging.info(cpu_count)
        pool = mp.Pool( cpu_count )
        ans = pool.map( partial( ComputeLogIncrementalWeight, particles = particles, df = df, indprob = indprob, 
                                newgamma = newgamma, oldgamma = oldgamma, eta = eta, fit_intercept = fit_intercept ), i )
        pool.close()
        pool.join()
    else:
        ans = []
        for i in range(n):
            temp = ComputeLogIncrementalWeight(i, particles, df, indprob, newgamma, oldgamma, eta, fit_intercept)
            ans.append(temp)
        
        '''
        jobs = []
        temp = [0, particles, df, indprob, newgamma, oldgamma, fit_intercept]
        for i in range(len(particles)):
            temp[0] = i
            jobs.append(list(temp))
        ans = forkedApply( ComputeLogIncrementalWeight, jobs, numThreads=20, killChildrenOnParentDead=False, verbose=False, logging =False)
        '''
        
        ans = sorted( ans, key=lambda x:x[0])
        logincweightlist = [item[1] for item in ans]
        logincweightarr = numpy.array(logincweightlist)
        logweightarr = numpy.log(weightarr) + logincweightarr
        maxlogweightarr = numpy.max(logweightarr)
        logweightarr = logweightarr - maxlogweightarr
        weightarr = numpy.exp(logweightarr)
        
        
        ## normalize weights
        weightarr = weightarr/weightarr.sum()
        
        ## lib.logging.info('weights')
        ## lib.logging.info(weightarr)
        
        return newgamma, weightarr
        
        
        
        