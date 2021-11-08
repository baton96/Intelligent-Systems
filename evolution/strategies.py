from time import time

import numpy as np
import numpy.random as rnd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin

rnd.seed(0)

nBombs = 3
kMeans = KMeans(n_clusters=nBombs)
populationSize = 150
nParents = 20
nOffspring = 150
nGenerations = 10
dMax = np.sqrt(20000)
nNests = 12
wasps = rnd.randint(low=100, high=1000, size=nNests)
allWasps = sum(wasps)
positions = rnd.uniform(low=0, high=100, size=(nNests, 2))
origin = rnd.uniform(100, size=(populationSize, nBombs, 2))
population = origin[:]
clusters = np.empty((nBombs, nOffspring, 2))
fitness = np.empty(populationSize)
start = time()

# Simple Evolution strategy
for _ in range(nGenerations):
    # all bomb co-ordinates of currently processed solution  
    for i, bombs in enumerate(population):
        # temporary numbers of wasps in each nest
        tmpWasps = wasps[:]
        # co-ordinates of currently processed bomb
        for coords in bombs:
            # distances between each nest and currently processed bomb
            distances = np.linalg.norm(positions - coords, axis=1)
            # information about what part of wasps in each nest will remain after the bomb
            parts = distances / dMax
            # updating numbers of wasps in each nest
            tmpWasps = np.floor(parts * tmpWasps)
        # fitness of currently proccessed solution
        fitness[i] = sum(tmpWasps)
    # index of the single best solution
    maxId = np.argmin(fitness)
    # single best solution
    parents = population[maxId]
    # nOffspring clones
    clones = np.repeat(parents[np.newaxis], nOffspring, axis=0)
    # offspring created by adding noise to clones
    population = clones + rnd.normal(size=(nOffspring, nBombs, 2))

# Simple Genetic strategy
for _ in range(nGenerations):
    # all bomb co-ordinates of currently processed solution  
    for i, bombs in enumerate(population):
        # temporary numbers of wasps in each nest
        tmpWasps = wasps[:]
        # co-ordinates of currently processed bomb
        for coords in bombs:
            # distances between each nest and currently processed bomb
            distances = np.norm(positions - coords, axis=1)
            # information about what part of wasps in each nest will remain after the bomb
            parts = distances / dMax
            # updating numbers of wasps in each nest
            tmpWasps = np.floor(parts * tmpWasps)
        # fitness of currently proccessed solution
        fitness[i] = sum(tmpWasps)
    # indices that would sort an array
    indices = np.argsort(fitness)
    # indices of the nParents best solutions
    parentIds = indices[:nParents]
    # nParents best solutions indicated by parentIndices
    parents = population[parentIds]
    # container for nOffspring solutions derived from-parents
    offspring = np.empty((nOffspring, nBombs, 2))
    for offspringId in range(nOffspring):
        # parents used to create new offspring solution
        mom = parents[offspringId % nParents]
        dad = parents[(offspringId + 1) % nParents]
        # temporary container for offspring solution
        tmpOffspring = np.empty((3, 2))
        # co-ordinates of currently processed mom bomb
        for i, coords in enumerate(mom):
            # distances between each dad bomb and currently processed mom bomb
            distances = np.norm(coords - dad, axis=1)
            # index of the dad bomb nearest to the currently processed mom bomb
            nearestId = np.argmin(distances)
            # dad bomb nearest to the currently processed mom bomb
            nearest = dad[nearestId]
            # child bomb derived from crossover between parent bombs
            childBomb = (nearest + coords) / 2
            # child bomb mutated by adding sample from a normal (Gaussian) distribution
            mutated = childBomb + rnd.normal(size=2)
            # filling temporary offspring container with bombs
            tmpOffspring[i] = mutated
        # filling offspring container with ready solutions
        offspring[offspringId] = tmpOffspring
    population = offspring
    # population = concatenate((parents,offspring))

# Covariance-Matrix Adaptation Evolution strategy (CMA-ES)
for _ in range(nGenerations):
    # all bomb co-ordinates of currently processed solution  
    for i, bombs in enumerate(population):
        # temporary numbers of wasps in each nest
        tmpWasps = wasps[:]
        # co-ordinates of currently processed bomb
        for coords in bombs:
            # distances between each nest and currently processed bomb
            distances = np.norm(positions - coords, axis=1)
            # information about what part of wasps in each nest will remain after the bomb
            parts = distances / dMax
            # updating numbers of wasps in each nest
            tmpWasps = np.floor(parts * tmpWasps)
        # fitness of currently proccessed solution
        fitness[i] = sum(tmpWasps)
    # indices that would sort an array
    indices = np.argsort(fitness)
    # indices of the nParents best solutions
    parentIds = indices[:nParents]
    # nParents best solutions indicated by parentIndices
    parents = population[parentIds]
    # flattened population, co-ordinates of all bombs
    flatPopulation = np.reshape(parents, (-1, 2))
    # tool used to group bombs according to their co-ordinates
    kMeans = KMeans(n_clusters=nBombs).fit(flatPopulation)
    # centers of each group
    centers = kMeans.cluster_centers_
    # labels of bombs
    labels = pairwise_distances_argmin(flatPopulation, centers)
    for i in range(nBombs):
        # all bomb belonging to the group i
        cluster = flatPopulation[labels == i]
        # covariance matrix within cluster
        covMatrix = np.cov(cluster, rowvar=False)
        # new bomb co-ordinates sampled from a multivariate normal distribution
        clusters[i] = rnd.multivariate_normal(centers[i], covMatrix, nOffspring)
    # container for nOffspring solutions from combining bomb groups
    population = np.empty((nOffspring, nBombs, 2))
    # each new individual takes one bomb from each group
    for i, bombs in enumerate(zip(*clusters)):
        population[i] = bombs

print("Excecution took:", round(time() - start, 3), "s")
print("Number of wasps killed:", allWasps - int(min(fitness)))
