#!/usr/bin/env python
# coding: utf-8

__author__ = "Himanshu Dutta"



import numpy as np
import pandas as pd


data = pd.read_csv("u_c_hihi.csv")
ETMat = data.to_numpy().reshape(1024,32)


def min_min_scheduler(ETMat,readyTime = 0):
    #Runs with a default ready time of 0, for all the machine in the initial state
    if readyTime == 0: 
        readyTime = np.zeros(ETMat.shape[1])
    #Output format: {Task ID, Machine ID, Execution Time}
    out = np.empty((ETMat.shape[0],3))
    #Completion Time = ExecutionTime + ReadyTime
    CTMat = ETMat + readyTime
    CTMat = np.hstack((CTMat, np.arange(CTMat.shape[0]).reshape(CTMat.shape[0],-1)))
    for i in range(ETMat.shape[0]):
        TId = np.where(CTMat == CTMat[:,:-1].min())[0][0]
        MId = np.where(CTMat == CTMat[:,:-1].min())[1][0]
        TId_ET = int(CTMat[TId,-1])
        print("TId : %d MId : %d Time: %f" %(TId,MId, ETMat[TId_ET,MId]))
        #Updating the Ready Time for the current machine
        readyTime[MId] = readyTime[MId] + ETMat[TId_ET,MId]
        #Output the Task-Machine array to an output array
        #Format: {Task ID, Machine ID, Execution Time}
        out[i,0] = CTMat[TId,-1]; out[i,1] = MId; out[i,2] = ETMat[TId_ET,MId]
        #Updating the value with current ready time for the given machine in the Completion Time Matrix
        CTMat[:,MId] = CTMat[:,MId] + ETMat[TId_ET,MId]
        #Removal of the mapped task from the CTMatrix
        CTMat = np.delete(CTMat,TId, 0)
    print("Final Ready Time \n", readyTime)
    return (out, readyTime)

print("Running Min-Min Algorithm....")
out,readyTime = min_min_scheduler(ETMat)


print("Ready Time with Min-Min Algorithm:", readyTime.max())


out.view('i8,i8,i8').sort(order=['f0'], axis=0)



# Phases:
#     1. Initial Population
#     2. Fitness Function
#     3. Selection
#     4. Crossover
#     5. Mutation



def initialize_population(size,tasks,machines):
    #Size: Number of Chromosomes in the population
    #Tasks: Number of tasks in the MetaTask Matrix
    #Machines: Total Number of virtual resources.
    return np.random.choice(machines,(size,tasks))


def calc_fitness(population,ETMat):
    #Calculation of fitness based on total readyTime of all the solutions/chromosomes in the population.
    #Ready Time of all the solutions are calculated, and the chromosomes with lesser readyTime are better.
    machines = len(np.unique(population))
    readyTime = np.zeros((population.shape[0],machines))
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            readyTime[i,population[i,j]] += ETMat[j,population[i,j]]
        readyTime[i] = readyTime[i].max()
    fitness = readyTime[:,0]
    return fitness
    
def selection(population,fitness,survivalRate):
    #Selection of the parents to crossover.
    #Selection is done on the basis of fitness value calculated over the current population
    parent = np.hstack((population,fitness.reshape(population.shape[0],-1)))
    parent = parent[parent[:,-1].argsort()]
    cutoff = int(survivalRate*population.shape[0])
    return parent[:cutoff,:-1]

def crossover(parents,offspring_size):
    #Crossing is done in a single-point manner, so as to breed the best of the current generation
    offspring_size = (int(parents.shape[0]/(1-offspring_size))-parents.shape[0],parents.shape[1])
    print("Offspring_Size: ")
    print(offspring_size)
    children = np.empty(offspring_size)
    crossover_point = int(offspring_size[1]/2)
    
    for i in range(offspring_size[0]):
        parent1 = i%parents.shape[0]
        parent2 = (i+1)%parents.shape[0]
        
        children[i,:crossover_point] = parents[parent1,:crossover_point]
        children[i,crossover_point:] = parents[parent2,crossover_point:]
    return children

def mutation(children,mutation_ratio):
    machines = len(np.unique(children))
    mutation_size = int(mutation_ratio * children.shape[1])
    
    for child in range(children.shape[0]):
        genes = np.random.choice(children.shape[1],mutation_size)
        for gene_pos in genes: children[child,gene_pos] = np.random.choice(machines)
    return children






"""
The algorithm works on the principle of makespan minimization. 
For the same, at each generation,The individuals/chromosomes which give the minimum makespan survive, 
so as to breed further to transfer their best genes onto the next generation. 
"""

#Parameters
pop_size = 100
tasks = 1024
machines = 32
survivalRate = 0.5
mutationRatio = 0.01
total_generations = 500
    




population = initialize_population(pop_size,tasks,machines)

#Adding Output of min-min algorith as one of the parents
population[0,:] = out[:,1].reshape(1,-1)



print("Initial Population: ", population)
overall_fitness = []
best_results = []


for generation in range(total_generations):
    print("*****************************************************")
    print("Generation: ", generation)
    #Calculation of fitness of each generation
    fitness = calc_fitness(population,ETMat)
    print("Fitness: \n", fitness)
    
    #Adding the best result of each generation to a list
    best_results.append(fitness.min())
    
    #Selection of Parents to crossover
    parents = selection(population,fitness,survivalRate)
    print("Parents: \n", parents)
    
    #Crossing-over to get children
    children = crossover(parents,1-survivalRate)
    
    #Adding variations/mutations to the current children
    children = mutation(children,mutationRatio)
    
    #Updating the population with the parent and children
    population[:parents.shape[0],:] = parents
    population[parents.shape[0]:,:] = children
    print("*****************************************************")
    
#After all the generations


print ("Best fitness :", fitness[0])



import matplotlib.pyplot
matplotlib.pyplot.plot(best_results)
matplotlib.pyplot.xlabel("Generation")
matplotlib.pyplot.ylabel("readyTime")
matplotlib.pyplot.show()
