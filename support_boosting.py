# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy
import numpy.random
import scipy.special
from builtins import zip, range
import multiprocessing as mp
import regression_utils as RU
import copy
import lib.logging
from functools import partial
import platform

## import forkedApply
## import lib.dist
## import cbb


class RandomData:
    def __init__(self):
        self.RNlol = None
        return
    

def GetWeightBasedOnCount( particles, cols ):
    """
    function to find weight corresponding to proposal based on count
    Inputs: particles in a list of numpy arrays . each array is a type string. array(['x1','x2','x5'])
    cols is column list of dataframe without 'y'
    output is a list of weights of size len(cols)
    """
    
    a,b = numpy.unique( particles, return_counts=True)
    b = b/b.sum()
    d = dict(zip(a,b))
    
    reg_neverSelected = set(cols).difference(set(a))
    for i in reg_neverSelected:
        d[i] = 0.0
        
    weights = []
    for i in cols:
        weights.append(d[i])
    return weights


def findPermProb( perm, probdict ):
    """
    function to get probability of sampling a permutation without replacement (as given in paper)
    Inputs: perm is a numpy array of type string. eg. array(['x1','x2','x5'])
    probdict is a dictionary of probabilities corresponding to regressors
    output is a scalar which is the probability of sampling the permutation
    """
    perm_size = perm.size
    p_individual = numpy.zeros( perm_size )
    prob = numpy.ones( perm_size )
    sum_prob = 0.0
    
    for i,val in enumerate(perm):
        p_individual[i] = probdict[val]
        Nr = p_individual[i]
        if i==0:
            Dr = 1.0
        else:
            sum_prob = sum_prob + p_individual[i-1]
            Dr = 1 - sum_prob
        prob[i]  = Nr/Dr
    ans = numpy.product(prob)
    assert ans >= 0, 'negative probability'
    return ans


def GetNewParticlesFromMHMove( i, temppart, df, gamma, indprobdict, countprobdict, cols, round_sb,
                              use_seed, eta, fit_intercept ):
    """
    wrapper function to get new particles after MH move 
    """
    perm = temppart[i]
    RNlol_local = RandomData.RNlol
    Alist = RNlol_local[i][0]
    Aretainlist = RNlol_local[i][1]
    Areplacementlist = RNlol_local[i][2]
    urnd = RNlol_local[i][3]
    newperm = numpy.array( Aretainlist + Areplacementlist )
    
    Acceptance_prob = GetMHAcceptanceProbability(df, gamma, Alist, Aretainlist, Areplacementlist, 
                                                 perm, newperm, indprobdict, countprobdict, cols,
                                                 eta, fit_intercept, i)
    
    if Acceptance_prob >= urnd:
        return i, newperm
    else:
        return i, perm
    
    
    
# def GetNewParticlesFromMHMove_fa( args ):
#     """ test """
    
#     i = args[0][0]
#     temppart = args[0][1]
#     df = args[0][2]
#     gamma = args[0][3]
#     indprobdict = args[0][4]
#     countprobdict = args[0][5]
#     cols = args[0][6]
#     fit_intercept = args[0][7]
#     RNlol_local = RandomData().RNlol
#     Alist = RNlol_local[i][1]
#     Aretainlist = RNlol_local[i][2]
#     Areplacementlist = RNlol_local[i][3]
#     urnd = RNlol_local[i][4]
    
#     perm = temppart[i]
    
#     newperm = numpy.array( Aretainlist + Areplacementlist )
#     Acceptance_prob = GetMHAcceptanceProbability(df, gamma, Alist, Aretainlist, Areplacementlist, 
#                                                  perm, newperm, indprobdict, countprobdict, cols,
#                                                  eta, fit_intercept, i)

#     if Acceptance_prob >= urnd:
#         return i, newperm
#     else:
#         return i, perm
    

def MHMove( gamma, df, particles, indprob, round_sb, use_seed=True, eta=1, fit_intercept=True ):
    """
    function to get new set of particles based on MH move on each particle
    
    Inputs: gamma is the currnet value of gamma for which MH move is to be applied 
    df is the dataframe
    cols is the list of columns in the dataframe without 'y'
    particles is a list of numpy arrays . each array is a type string. array(['x1','x2','x5'])
    indprob is an array of probabilities of individual regressors of size len(cols)
    output is a new list of particles after the MH move
    
    """
    
    cols = df.columns.tolist()
    cols.remove('y')
    
    # get weights of regressors based on count 
    countprob = GetWeightBasedOnCount(particles, cols)
    countprobdict = dict( zip( cols, list(countprob)) )
    indprobdict = dict( zip(cols, list(indprob)) )
    temppart = copy.deepcopy( particles )
    
    RandomData.RNlol = GetSubsetAndReplacement( temppart, indprobdict, countprobdict, cols, use_seed, round_sb)
    
    
    ## parallelize
    os_type = platform.system()
    if os_type != 'Windows':
        i= [j for j in range(len(temppart))]
        cpu_count = mp.cpu_count()
        pool = mp.Pool(cpu_count)
        lib.logging.info( "cpu cores avaiable = {0}".format( cpu_count ))
        ans = pool.map(
            partial( GetNewParticlesFromMHMove, 
                     tempart = temppart,
                     df = df,
                     gamma = gamma,
                     indprobdict = indprobdict,
                     countprobdict = countprobdict,
                     cols = cols,
                     round_sb = round_sb,
                     use_seed = use_seed,
                     eta = eta,
                     fit_intercept = fit_intercept ),
                     i) 
        pool.close()
        pool.join()
        
    else:
        ans = []
        for i in range(len(temppart)):
            temp = GetNewParticlesFromMHMove(i, 
                                             temppart = temppart,
                                             df = df,
                                             gamma = gamma, 
                                             indprobdict = indprobdict,
                                             countprobdict = countprobdict,
                                             cols = cols,
                                             round_sb = round_sb,
                                             use_seed = use_seed,
                                             eta = eta,
                                             fit_intercept = fit_intercept )
            ans.append( temp )
    
    ans = sorted(ans, key=lambda x:x[0])
    newparticles = [item[1] for item in ans]
    return newparticles


def SetSeed(j):
    """ set seed of a random number """
    numpy.random.seed(j)
    return


def GetModifiedWeights( replacement_superlist, probdict ):
    """ 
    function to get weights so that the superset of regressors is part of the dictionary
    Input: takes superset and dictionary
    output : returns a list of weights by normalizing them wrt the items present in the dictionary 
    """
    
    p = numpy.zeros( len(replacement_superlist) )
    for i,item in enumerate( replacement_superlist ):
        p[i] = probdict[item]
    p = p/p.sum()
    return p


def GetSubsetAndReplacement( temppart, indprobdict, countprobdict, cols, use_seed, round_sb):
    """
    function to get a new permutation based on Initial and Count Sampler
    this function returns Alist, Aretainlist, Areplacementlist
    this function is written to separate the random number generation part from getting the new
    because of parallel processsing issues, its better to design this way
    
    Inputs: temppart is a list of numpy array of strings. eg.array(['x1','x2','x5'])
    indprobdict is a dictionary of individual probabilities of regressors based on their R^2
    countprobdict is a dictionary of individual probabilities of regressors based on their count
    cols is a list of columns of dataframe without 'y'
    
    output is a list of lists [[L1,L2,L3,urnd]]
    L1 is the random list of elements to be replaced  from the permutation
    L2 is the random list of elements to be retained from the permutation
    L3 is the new list of elements to replace L1
    urnd is a unifrom random number (0,1)
    
    """
    RNList = []
    for i,perm in enumerate( temppart ):
        if use_seed == True:
            SetSeed( i+round_sb )
        size = numpy.random.randint( 1, perm.size )
        if use_seed == True:
            SetSeed( i+round_sb )
            
        Alist = list(numpy.random.choice( perm, size=size, replace = False))   ## maxA is perm.size-1
        if use_seed == True:
            SetSeed(i+round_sb)
        ## choice = numpy.random.choice(numpy.array(['I, 'H']), size=1, replace=True, p=[0.75,0.25] 
        choice = numpy.random.choice( numpy.array(['I', 'H']), size=1, replace=True, p = [0.5,0.5] ) 
        Aretainlist = [ j for j in list(perm) if j not in Alist ]
        size_replacement = len(Alist)
        replacement_superlist = [j for j in cols if j not in Aretainlist]
        replacement_superarr =  numpy.array( replacement_superlist )
        
        p_initial = GetModifiedWeights(replacement_superlist, indprobdict)
        p_count = GetModifiedWeights(replacement_superlist, countprobdict)
        
        if use_seed == True:
            SetSeed( i+round_sb )
        if choice == "I":
            Areplacementlist = list( numpy.random.choice( replacement_superarr, 
                                                         size=size_replacement, 
                                                         replace = False, 
                                                         p = p_initial) )
        else:
            Areplacementlist = list( numpy.random.choice( replacement_superarr,
                                                         size=size_replacement, 
                                                         replace=False, 
                                                         p=p_count) )
        if use_seed == True:
            SetSeed( i + round_sb )
        rnd = numpy.random.uniform()
        templist = [Alist, Aretainlist, Areplacementlist, rnd]
        RNList.append(templist)
    return RNList



def GetNewSampleProb( Areplacementlist, Aretainlist, probdict, cols ):
    """
    function to get P(Areplacementlist|Alist) based on some sampler
    Inputs: Areplacementlist is the list of elements from the new permutation that replaces A in the old permutation
    Aretainlist is the list of elements to be retained in the old permutation
    
    e.g. if oldperm = ['x4', 'x3', 'x1', 'x9', 'x11']
            newperm = ['x4', 'x1', 'x11', 'x6', 'x7']
            Aretain = ['x4', 'x1', 'x11']
            A = ['x3', 'x9']
            Areplacement = ['x6', 'x7']
        probdict is a dictionary of probabilities corresponding to regressors
        output is a scaler which is the probability of choosing the replacement list from the modified dictionary
        
    """
    perm = numpy.array( Areplacementlist )
    
    ## make new dict to get dictionary with only items in cols-Aretain
    replacement_superlist = [ i for i in cols if i not in Aretainlist ]
    newprobdict = {}
    for item in replacement_superlist:
        newprobdict[item] = probdict[item]
    
    ## normalize probabilities
    total = sum(newprobdict.values())
    for key in newprobdict:
        newprobdict[key] = newprobdict[key]/total
    
    return findPermProb( perm, newprobdict )



def GetMHAcceptanceProbability( df, 
                               gamma, 
                               Alist,
                               Aretainlist, 
                               Areplacementlist,
                               oldperm,
                               newperm, 
                               indprobdict,
                               countprobdict,
                               cols,
                               eta,
                               fit_intercept,
                               i):
    """
    function to return the MH Acceptance probability
    Inputs: df is the dataframe 
    gamma is the current value of the gamma
    Areplacmentlist is the list of elements from the new permutation that replaces A in the 
    old permutation
    
    Aretainlist is the list of elements to be retained in the old permutation
    e.g. if oldperm = ['x4', 'x3', 'x1', 'x9', 'x11']
            newperm = ['x4', 'x1', 'x11', 'x6', 'x7']
            Aretain = ['x4', 'x1', 'x11']
            A = ['x3', 'x9']
            Areplacement = ['x6', 'x7']
        indprobdict is a dictionary of individual probabilities of regressors based on their R**2
        countprobdict is a dictionary of individual probabilities of regressors based on their count until
        that step
        cols is the list of columns in the dataframe without 'y'

        output is the Acceptance probability of a standard Metropolis Hastings algo
        which is Target(newperm)/Target(oldperm)* Proposal(oldperm|newperm)/Proposal(newperm|oldperm)
        
    """
    ProbNewGivenOld = GetProposalProbability( len(Alist), Aretainlist, Areplacementlist, oldperm.size, indprobdict, 
                                             countprobdict, cols )
    
    ProbOldGivenNew = GetProposalProbability( len(Areplacementlist), Aretainlist, Alist, newperm.size, 
                                             indprobdict, countprobdict, cols )
    
    TargetRatio = GetTargetProbabilityRatio( gamma, oldperm, newperm, df, indprobdict, eta, fit_intercept )
    return min(1, TargetRatio* ProbOldGivenNew/ProbNewGivenOld )


def GetProposalProbability( k, Aretainlist, Areplacementlist, n, indprobdict, countprobdict, cols ):
    """
    function to get a P(newperm|oldperm) based on I or H sampler
    Inputs: k is the number of elements in A
    n is the number of elements in the perm
    Areplacementlist os the list of elements from the new permutation that replaces A in the old permutation
    Aretainlist  is the list of elements to be retained in  the old permutation
    e.g. if oldperm = ['x4', 'x3', 'x1', 'x9', 'x11']
            newperm = ['x4', 'x1', 'x11', 'x6', 'x7']
            Aretain = ['x4', 'x1', 'x11']
            A = ['x3', 'x9']
            Areplacement = ['x6', 'x7']
            
    indprobdict is a dictionary of individual probabilities of regressors based on their R**2
    countprobdict is a dictionary of individual probabilities of regressors based on their count until
    that step
    cols is the list of columns in the dataframe without 'y'
    
    output is the probability of proposing a replacement 
      = probability of choosing a sample for replacement( =1/nck)*probability of replacing that with a new sample
    probability of replacing that with a new sample is based on a mixture sapmler based on intial probabilities  and
    probabilities based on count
      
    """
    
    OSRProb = 1/scipy.special.comb(n,k)
    Proposal_IProb = GetNewSampleProb(Areplacementlist, Aretainlist, indprobdict, cols)    
    Proposal_HProb = GetNewSampleProb(Areplacementlist, Aretainlist, countprobdict, cols)
    Proposal_Prob = 0.5*Proposal_IProb + 0.5*Proposal_HProb
    ##  Proposal_Prob = 0.75*Proposal_IProb + 0.25*Proposal_HProb
    return OSRProb + Proposal_Prob


def GetTargetProbabilityRatio( gamma, oldperm , newperm, df, indprobdict, eta, fit_intercept ):
    """
    function returns the ratio: TargetDensity(newperm,gamma)/TargetDensity(oldperm, gamma)
    Inputs: gamma is the current value of gamma
    oldperm is the existing permutation
    newperm is the proposed new permutation
    df is the dataframe
    indprobdict is a dictionary corresponding to probabilities of individual regressors based on their initial R**2
    
    output is the MH probability as given in the equation  6 in the paper
    """
    
    Y = df['y']
    
    l2norm_newperm = RU.GetRegressionSquare(df[newperm], Y, fit_intercept)
    l2norm_oldperm = RU.GetRegressionSSE(df[oldperm], Y, fit_intercept)

    l2diff = (l2norm_newperm - l2norm_oldperm)/eta
    
    Term1 = numpy.exp( -1*gamma*l2diff)

    initprob_newperm = findPermProb( newperm, indprobdict )
    initprob_oldperm = findPermProb( oldperm, indprobdict )
    
    Term2 = numpy.powe(initprob_newperm/initprob_oldperm, 1-gamma)
    
    return Term1*Term2

    

