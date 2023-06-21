import numpy as np
import itertools
from itertools import chain
import networkx as nx
import pandas as pd
from random import randint, randrange
import string
import os
from numpy import random
from numpy.linalg import matrix_rank
from sklearn.covariance import GraphicalLasso
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.mode.chained_assignment = None


import warnings
warnings.filterwarnings('ignore')


def flatten(listOfLists):
    return chain.from_iterable(listOfLists)


# Given a number of dimensions, alpha value, and number of samples, take numSamples samples from
# standard multivariate normal, run graphLasso and get covariance estimate
def getGLCovariance(nDim, alpha, numSamples):
    zeroMean = np.zeros(nDim)
    oneVariance = np.identity(nDim)
    normalSample = np.random.multivariate_normal(mean = zeroMean, cov = oneVariance, size = numSamples)
    # Use lars solver, otherwise bumps into convergence issues
    graphLasso = GraphicalLasso(alpha = alpha, mode = 'lars', max_iter = 100, assume_centered = True)
    graphLasso.fit(normalSample)
    covarianceEstimate = graphLasso.covariance_
    return(covarianceEstimate)


def adjacencyJacobian(graph_input):
    graph = graph_input.copy()
    edges = graph.edges
    nodes = graph.nodes
    nodeweights = {n:random.uniform(-1000,1000) for n in graph.nodes}
    jacobian = np.empty((len(edges), len(nodes)))
    for e in enumerate(edges):
        for n in enumerate(nodes):
            nodeat = n[1]
            edgeat = e[1]
            
            if nodeat == edgeat[0]:
                weight = nodeweights[edgeat[0]] - nodeweights[edgeat[1]]
            elif nodeat == edgeat[1]:
                weight = nodeweights[edgeat[1]] - nodeweights[edgeat[0]]
            else:
                weight = 0
            
            jacobian[e[0], n[0]] = weight
    return(jacobian)
            
    
def MLTBig(graph_input):
    jacobian = adjacencyJacobian(graph_input)
    at = 1
    num_edges = len(graph_input.edges)

    if num_edges == 0:
        return(1)

    rank = matrix_rank(jacobian)

    
    while rank < num_edges:
        at += 1
        temp = adjacencyJacobian(graph_input)
        
        jacobian = np.hstack((temp, jacobian))
        rank = matrix_rank(jacobian)
        #print(jacobian, "\n")
    return(at + 1)