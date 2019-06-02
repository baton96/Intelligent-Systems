#cov
from numpy import empty, argsort, sqrt, cov, reshape, floor, repeat, newaxis, argmin, array
from numpy.random import seed, uniform, multivariate_normal, normal, randint
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from numpy.linalg import norm
from time import time
seed(0)

nBombs = 3
kMeans = KMeans(n_clusters=nBombs)
populationSize = 150
nParents = 20
nOffspring = 150
nGenerations = 10
dMax = sqrt(20000)
nNests = 12
wasps = randint(low=100, high=1000, size=nNests)
allWasps = sum(wasps)
positions = uniform(low=0, high=100, size=(nNests,2))
origin = uniform(100, size=(populationSize,nBombs,2))
population = origin[:]
clusters = empty((nBombs,nOffspring,2))
fitness = empty(populationSize)
start = time()

'''
#verbose Simple Evolution strategy
for _ in range(nGenerations):
    #all bomb co-ordinates of currently processed solution  
    for i, bombs in enumerate(population):
        #temporary numbers of wasps in each nest
        tmpWasps = wasps[:]
        #co-ordinates of currently processed bomb
        for coords in bombs:
            #distances between each nest and currently processed bomb
            distances = norm(positions-coords, axis=1)
            #information about what part of wasps in each nest will remain after the bomb
            parts = distances/dMax
            #updating numbers of wasps in each nest
            tmpWasps = floor(parts*tmpWasps)
        #fitness of currently proccessed solution
        fitness[i] = sum(tmpWasps)
    #index of the single best solution
    maxId = argmin(fitness)
    #single best solution
    parents = population[maxId]
    #nOffspring clones
    clones = repeat(parents[newaxis], nOffspring, axis=0)
    #offspring created by adding noise to clones
    population = clones + normal(size=(nOffspring,nBombs,2))
'''

'''
#compact Simple Evolution strategy
for _ in range(nGenerations):
    for i, bombs in enumerate(population):
        tmpWasps = wasps[:]
        for coords in bombs:
            tmpWasps = floor(tmpWasps*norm(positions-coords, axis=1)/dMax)
        fitness[i] = sum(tmpWasps)
    population = repeat(population[argmin(fitness)][newaxis], nOffspring, axis=0) + normal(size=(nOffspring,nBombs,2))
'''

'''
#verbose Simple Genetic strategy
for _ in range(nGenerations):
    #all bomb co-ordinates of currently processed solution  
    for i, bombs in enumerate(population):
        #temporary numbers of wasps in each nest
        tmpWasps = wasps[:]
        #co-ordinates of currently processed bomb
        for coords in bombs:
            #distances between each nest and currently processed bomb
            distances = norm(positions-coords, axis=1)
            #information about what part of wasps in each nest will remain after the bomb
            parts = distances/dMax
            #updating numbers of wasps in each nest
            tmpWasps = floor(parts*tmpWasps)
        #fitness of currently proccessed solution
        fitness[i] = sum(tmpWasps)
    #indices that would sort an array
    indices = argsort(fitness)
    #indices of the nParents best solutions
    parentIds = indices[:nParents]
    #nParents best solutions indicated by parentIndices
    parents = population[parentIds]
    #container for nOffspring solutions derived from-parents
    offspring = empty((nOffspring,nBombs,2))    
    for offspringId in range(nOffspring):
        #parents used to create new offspring solution
        mom = parents[offspringId%nParents]
        dad = parents[(offspringId+1)%nParents]     
        #temporary container for offspring solution
        tmpOffspring = empty((3,2))
        #co-ordinates of currently processed mom bomb
        for i, coords in enumerate(mom):
            #distances between each dad bomb and currently processed mom bomb
            distances = norm(coords-dad, axis=1)
            #index of the dad bomb nearest to the currently processed mom bomb
            nearestId = argmin(distances)
            #dad bomb nearest to the currently processed mom bomb
            nearest = dad[nearestId]
            #child bomb derived from crossover between parent bombs
            childBomb = ( nearest + coords)/2
            #child bomb mutated by adding sample from a normal (Gaussian) distribution
            mutated = childBomb + normal(size=2)
            #filling temporary offspring container with bombs
            tmpOffspring[i] = mutated
        #filling offspring container with ready solutions
        offspring[offspringId] = tmpOffspring       
    population = offspring
    #population = concatenate((parents,offspring))
'''

'''
#compact Simple Genetic strategy
for _ in range(nGenerations):
    for i, bombs in enumerate(population):
        tmpWasps = wasps[:]
        for coords in bombs:
            tmpWasps = floor(tmpWasps*norm(positions-coords, axis=1)/dMax)
        fitness[i] = sum(tmpWasps)
    parents = population[argsort(fitness)[:nParents]]
    population = array([[(parents[(i+1)%nParents][argmin(norm(coords-parents[(i+1)%nParents], axis=1))]+coords)/2.0+normal(size=2) for coords in parents[i%nParents]] for i in range(nOffspring)])
'''

'''
#verbose Covariance-Matrix Adaptation Evolution strategy (CMA-ES)
for _ in range(nGenerations):
    #all bomb co-ordinates of currently processed solution  
    for i, bombs in enumerate(population):
        #temporary numbers of wasps in each nest
        tmpWasps = wasps[:]
        #co-ordinates of currently processed bomb
        for coords in bombs:
            #distances between each nest and currently processed bomb
            distances = norm(positions-coords, axis=1)
            #information about what part of wasps in each nest will remain after the bomb
            parts = distances/dMax
            #updating numbers of wasps in each nest
            tmpWasps = floor(parts*tmpWasps)
        #fitness of currently proccessed solution
        fitness[i] = sum(tmpWasps)
    #indices that would sort an array
    indices = argsort(fitness)
    #indices of the nParents best solutions
    parentIds = indices[:nParents]
    #nParents best solutions indicated by parentIndices
    parents = population[parentIds]
    #flattened population, co-ordinates of all bombs
    flatPopulation = reshape(parents,(-1,2))
    #tool used to group bombs according to their co-ordinates
    kMeans = KMeans(n_clusters=nBombs).fit(flatPopulation)
    #centers of each group
    centers = kMeans.cluster_centers_
    #labels of bombs
    labels = pairwise_distances_argmin(flatPopulation, centers)
    for i in range(nBombs):
        #all bomb belonging to the group i
        cluster = flatPopulation[labels == i]
        #covariance matrix within cluster
        covMatrix = cov(cluster,rowvar=False)
        #new bomb co-ordinates sampled from a multivariate normal distribution
        clusters[i] = multivariate_normal(centers[i], covMatrix, nOffspring)
    #container for nOffspring solutions from combining bomb groups
    population = empty((nOffspring,nBombs,2))
    #each new individual takes one bomb from each group
    for i, bombs in enumerate(zip(*clusters)):
        population[i] = bombs
'''

'''
#compacted Covariance-Matrix Adaptation Evolution strategy (CMA-ES)
for _ in range(nGenerations):
    fitness = empty(len(population))
    for i, bombs in enumerate(population):
        tmpWasps = wasps[:]
        for coords in bombs:
            tmpWasps = floor(tmpWasps*norm(positions-coords, axis=1)/dMax)
        fitness[i] = sum(tmpWasps)
    flatPopulation = reshape(population[argsort(fitness)[:nParents]],(-1,2))
    kMeans = KMeans(n_clusters=nBombs).fit(flatPopulation)
    labels = pairwise_distances_argmin(flatPopulation, kMeans.cluster_centers_)
    clusters = [multivariate_normal(kMeans.cluster_centers_[i], cov(flatPopulation[labels==i],rowvar=False), nOffspring) for i in range(nBombs)]
    population = array([array(bombs) for bombs in zip(*clusters)])
'''

print("Excecution took:",round(time()-start,3),"s")
print("Number of wasps killed:", allWasps-int(min(fitness)))
