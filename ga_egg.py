import numpy as np, random, operator, pandas as pd

#, matplotlib.pyplot as plt

#Create necessary classes and functions

#Create a fitness function

class Fitness:
    def __init__(self, coord):
        self.coord = coord
        self.funcValue = 0.0
        self.fitness= 0.0
    
    def functionValue(self):
        if self.funcValue == 0.0:
            x = self.coord[0]
            y = self.coord[1]
            a = np.sqrt(np.fabs(y+x/2+47))
            b = np.sqrt(np.fabs(x-(y+47)))
            self.funcValue = -(y+47)*np.sin(a)-x*np.sin(b)
        return self.funcValue
    
    def coordFitness(self):
        if self.fitness == 0.0:
            self.fitness = 1 / (self.functionValue() + 960.0)
        return self.fitness

#Create our initial population

#Coordinate generator

def createCoord():
    coord = [0.0, 0.0]
    for i in range(2):
        coord[i] = random.uniform(-512, 512.0000001)
    return coord

#Create first "population" (list of coordinates)

def initialPopulation(popSize):
    population = []

    for i in range(0, popSize):
        population.append(createCoord())
    return population

#Create the genetic algorithm

#Rank individuals

def rankCoords(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).coordFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

#Create a selection function that will be used to make the list of parent coordinates

def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

#Create mating pool

def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

#Create a crossover function for two parents to create one child
# BLX-alpha algorithm for chromosome crossover

def breed(parent1, parent2, crossRate):
    child = []
    if(random.random() < crossRate):
        aux = 0.0
        alpha = 0.5
        for i in range(2):
            d = alpha * abs(parent1[i] - parent2[i])
            min_value = min(parent1[i], parent2[i]) - d
            max_value = max(parent1[i], parent2[i]) + d
            aux = random.uniform(min_value, max_value)
            child.append(max(min(aux, 512), -512))
    else:
        if(Fitness(parent1).functionValue() < Fitness(parent2).functionValue()):
            child = parent1
        else:
            child = parent2
    return child

#Create function to run crossover over full mating pool

def breedPopulation(matingpool, eliteSize, crossRate):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1], crossRate)
        children.append(child)
    return children

#Create function to mutate a single coordinate

def mutate(individual, mutationRate):
    if(random.random() < mutationRate):
        mutation = createCoord()
        aux = 0.0
        alpha = 0.5
        for i in range(2):
            d = alpha * abs(individual[i] - mutation[i])
            min_value = min(individual[i], mutation[i]) - d
            max_value = max(individual[i], mutation[i]) + d
            aux = random.uniform(min_value, max_value)
            individual[i] = max(min(aux, 512), -512)
    return individual

#Create function to run mutation over entire population

def mutatePopulation(population, mutationRate):
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

#Put all steps together to create the next generation

def nextGeneration(currentGen, eliteSize, mutationRate, crossRate):
    popRanked = rankCoords(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize, crossRate)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

#Final step: create the genetic algorithm

def geneticAlgorithm(popSize, eliteSize, mutationRate, generations, crossRate):
    pop = initialPopulation(popSize)
    print("Initial minimum function value: " + str((1 / rankCoords(pop)[0][1]) - 960.0))
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate, crossRate)
    
    print("Final minimum function value: " + str((1 / rankCoords(pop)[0][1]) - 960.0))
    bestCoordIndex = rankCoords(pop)[0][0]
    bestCoord = pop[bestCoordIndex]
    return bestCoord

#Running the genetic algorithm

#Run the genetic algorithm

bestCoordinate = geneticAlgorithm(popSize=50, eliteSize=10, mutationRate=0.01, generations=250, crossRate=1.001)

print("Best solution found: " + str(bestCoordinate))

#Plot the progress

#Note, this will win run a separate GA

"""""
def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])
    
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()
"""""

#Run the function with our assumptions to see how distance has improved in each generation

#geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)





