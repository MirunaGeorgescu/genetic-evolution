import random
import sys
from math import log2, ceil

def updateFitness(chromosomes, domain, a, b, c): 
    fitness = []
    for i in range(len(chromosomes)):
        value = int(str(chromosomes[i]), 2)
        value = domain[0] + value * (domain[1] - domain[0]) / (2 ** len(str(chromosomes[i])))
        fitness.append(calculateFitness(a, b, c, value))
    return fitness

"""
Population generation process:
    - generate a real number within the domain
    - encode the real number into binary
    - return the population of chromosomes (bit strings)
"""

def encode(u, domain, precision): 
    numBits = ceil(log2((domain[1] - domain[0]) * (10 ** precision)))
    discretizationStep = (domain[1] - domain[0]) / (2 ** numBits)
    
    index = (u - domain[0]) // discretizationStep
    chromosome = bin(int(index)).replace('0b', '')
    chromosome = f"{chromosome:>0{numBits}}"
    return chromosome

def generatePopulation(populationSize, domain, precision):
    population = []
    for i in range(populationSize):
        u = random.uniform(domain[0], domain[1])
        population.append(encode(u, domain, precision))
    return population

"""
Selection process:
    - generate a random number between 0 and 1
    - find the interval to which the generated number belongs (binary search), determining an interval [qi, qi+1)
    - return the index i+1 corresponding to the selected chromosome
"""

def calculateFitness(a, b, c, x):
    return a * (x ** 2) + b * x + c

# Selection probabilities for each chromosome
def selectionProbabilities(fitness):
    probabilities = []
    for i in range(len(fitness)):
        ps = fitness[i] / sum(fitness)
        probabilities.append(ps)
    return probabilities

def cumulativeProbabilities(probabilities):
    intervals = []
    intervals.append(0)
    for i in range(len(probabilities)):
        intervals.append(sum(probabilities[:i + 1]))
    return intervals

def binarySearch(intervals, u):
    left = 0
    right = len(intervals) - 1
    while left <= right:
        mid = (left + right) // 2
        if intervals[mid] < u:
            left = mid + 1
        elif intervals[mid] > u:
            right = mid - 1
        else:
            return mid
    return left - 1

def selection(intervals, populationSize, population):
    selectedChromosomes = []
    for i in range(populationSize):
        u = random.uniform(0, 1)
        index = binarySearch(intervals, u)
        selectedChromosomes.append(population[index])
    return selectedChromosomes

"""
Crossover process:
    - generate a natural number between 0 and numBits-1 (chromosome length)
    - perform gene exchange at the generated position
    - return the two resulting chromosomes
"""

def participates(crossoverProbability):
    u = random.uniform(0, 1)
    if u < crossoverProbability:
        return 1
    return 0

def crossover(chromosome1, chromosome2):
    numBits = len(chromosome1)
    i = random.randint(0, numBits - 1)
    child1 = chromosome1[:i] + chromosome2[i:]
    child2 = chromosome2[:i] + chromosome1[i:]
    return i, child1, child2

def populationAfterCrossover(initialPopulation, crossoverProbability):
    p = []
    for i in range(len(initialPopulation)):
        p.append(participates(crossoverProbability))
    
    crossoverChromosomes = []
    for i in range(len(initialPopulation)):
        if p[i] == 0:
            crossoverChromosomes.append(initialPopulation[i])
        else:
            crossoverChromosomes.append(0)
    
    for i in range(p.count(1) // 2):
        index1 = p.index(1)
        index2 = random.randint(index1 + 1, len(initialPopulation) - 1)
        while p[index2] == 0:
            index2 = random.randint(index1 + 1, len(initialPopulation) - 1)
        
        crossoverIndex, crossoverChromosomes[index1], crossoverChromosomes[index2] = crossover(initialPopulation[index1], initialPopulation[index2])
        p[index1] = 2
        p[index2] = 2
    
    return crossoverChromosomes

"""
Mutation process:
    - generate a natural number between 0 and numBits-1 (chromosome length)
    - perform gene mutation at the generated position
    - return the resulting chromosome
"""  
def undergoesMutation(mutationProbability):
    u = random.uniform(0, 1)
    if u < mutationProbability:
        return 1
    return 0

def mutation(chromosome, index):
    chromosomeStr = str(chromosome)
    if chromosomeStr[index] == '0':
        chromosomeStr = chromosomeStr[:index] + '1' + chromosomeStr[index + 1:]
    else: 
        chromosomeStr = chromosomeStr[:index] + '0' + chromosomeStr[index + 1:]
    return bin(int(chromosomeStr)).replace('0b', '')  

def populationAfterMutation(chromosomes, mutationProbability):
    mutations = []
    for i in range(len(chromosomes)): 
        mutations.append(undergoesMutation(mutationProbability))
        
    for i in range(len(chromosomes)):
        if mutations[i] == 1:
            index = random.randint(0, len(str(chromosomes[i])) - 1)
            chromosomes[i] = mutation(chromosomes[i], index)
    return chromosomes

def main(): 
    """
    Reading from the input file
    """ 
    with open("C:\\Users\mirub\OneDrive\Documente\FMI\Anul II\Semestrul II\Algoritmi avansati\Algoritmi aproximativi si algoritmi genetici\Algoritmi Genetici\\input.txt", "r") as input_file:
        content = input_file.readlines()

    """
    Processing input data
    """
    # Population size (number of chromosomes)
    populationSize = int(content[0])

    # Domain of the function (interval in which x is located) (closed interval endpoints)
    domain = content[1].split()
    domain[0] = float(domain[0])
    domain[1] = float(domain[1])

    # a, b, c = parameters for the maximized function (coefficients of the quadratic polynomial)
    a, b, c = content[2].split()
    a = int(a)
    b = int(b)
    c = int(c)

    # Precision = precision used in computations (interval discretization)
    precision = float(content[3])

    # Crossover probability
    crossoverProbability = float(content[4])
    crossoverProbability = crossoverProbability / 100

    # Mutation probability
    mutationProbability = float(content[5])
    mutationProbability = mutationProbability / 100

    # Number of generations (algorithm steps)
    numGenerations = int(content[6])

    with open("C:\\Users\mirub\OneDrive\Documente\FMI\Anul II\Semestrul II\Algoritmi avansati\Algoritmi aproximativi si algoritmi genetici\Algoritmi Genetici\\output.txt", "w") as output_file: 
        # Redirect standard output to the output file
        sys.stdout = output_file 
        print("######################################################## STAGE 1 #######################################################")
        
        populationBinary = generatePopulation(populationSize, domain, precision)
        fitness = []
        population = []
        print("-------------------------------------------------- INITIAL POPULATION --------------------------------------------------")
        for i in range(populationSize):
            gene = populationBinary[i]
            value = int(gene, 2)
            
            x = domain[0] + value * (domain[1] - domain[0]) / (2 ** len(gene))
            population.append(x)
            
            fx = calculateFitness(a, b, c, x)
            fitness.append(fx)
            
            print(f"Individual {i + 1}:\n  value: {str(x):30s}       gene: {gene:30s}       function value: {str(fx):30s}")
        print()
        
        print("----------------------------------------------- SELECTION PROBABILITIES ------------------------------------------------")
        ps = selectionProbabilities(fitness)
        for i in range(populationSize):
            print(f"Individual {i + 1}:\n  selection probability: {str(ps[i]):30s}")
        print()
        
        print("-------------------------------------------------- SELECTION INTERVALS ---------------------------------------------------")
        intervals = cumulativeProbabilities(ps)
        for i in range(len(intervals)):
            print(f"                                                   {str(intervals[i])}")
        print()
        
        print("-------------------------------------------------------- SELECTION ----------------------------------------------------------")
        selectedChromosomes = []
        for i in range(populationSize):
            u = random.uniform(0, 1)
            index = binarySearch(intervals, u)
            selectedChromosomes.append(populationBinary[index])
            print(f"{i + 1}. For randomly generated u: " + str(u) + f", the selected chromosome is: {index}")
        print()
        
        print("------------------------------------------------ SELECTED POPULATION ----------------------------------------------------")
        for i in range(populationSize):
            gene = selectedChromosomes[i]
            value = int(gene, 2)
            
            x = domain[0] + value * (domain[1] - domain[0]) / (2 ** len(gene))
            fx = calculateFitness(a, b, c, x)
            print(f"Individual {i + 1}:\n  value: {str(x):30s}       gene: {gene:30s}       function value: {str(fx):30s}")
        print()
        
        populationBinary = selectedChromosomes
        
        print("-------------------------------- CROMOSOMES PARTICIPATING IN CROSSOVER ------------------------------------------")
        print(f"Crossover probability: {crossoverProbability}")
        crossoverList = [] 
        for i in range(populationSize):
            u = random.uniform(0, 1)
            string = "For randomly generated u: " + str(u) + " chromosome " + str(i + 1) + ": " + str(populationBinary[i])
            if u < crossoverProbability:
                crossoverList.append(1)
                string += " participates in crossover"
            else:
                crossoverList.append(0)
                string += " does not participate in crossover"
            print(string)
        print()
        
        print("---------------------------------------------------- CROSSOVER -----------------------------------------------------------")
        crossoverChromosomes = []
        
        for i in range(len(populationBinary)):
            if crossoverList[i] == 0:
                crossoverChromosomes.append(populationBinary[i])
            else:
                crossoverChromosomes.append(bin(0).replace('0b', ''))
        
        for i in range(crossoverList.count(1) // 2):
            index1 = crossoverList.index(1)
            index2 = random.randint(index1 + 1, len(populationBinary) - 1)
            
            while crossoverList[index2] == 0:
                index2 = random.randint(index1 + 1, len(populationBinary) - 1)
                
            string = f"Individuals {index1 + 1} and {index2 + 1} undergo crossover, the crossover point being "
            
            crossoverIndex, crossoverChromosomes[index1], crossoverChromosomes[index2] = crossover(populationBinary[index1], populationBinary[index2])
            crossoverList[index1] = 1
            crossoverList[index2] = 1
        
            string = f"{crossoverIndex} and the resulting chromosomes are:\n  {crossoverChromosomes[index1]}\n    {crossoverChromosomes[index2]}"
            print(string)
            
        print()
        
        print("--------------------------------------------- POPULATION AFTER CROSSOVER --------------------------------------------------")
        for i in range(populationSize):
            gene = crossoverChromosomes[i]
            value = int(gene, 2)
            
            x = domain[0] + value * (domain[1] - domain[0]) / (2 ** len(gene))
            fx = calculateFitness(a, b, c, x)
            print(f"Individual {i + 1}:\n  value: {str(x):30s}       gene: {gene:30s}       function value: {str(fx):30s}")
        print()
        
        populationBinary = crossoverChromosomes
        
        print("--------------------------------- CHROMOSOMES THAT WILL UNDERGO MUTATIONS ---------------------------------------------")
        print(f"Mutation probability: {mutationProbability}")
        mutations = []
        for i in range(populationSize):
            u = random.uniform(0, 1)
            string = "For randomly generated u: " + str(u) + " chromosome " + str(i + 1) + ": " + str(crossoverChromosomes[i])
            if u < mutationProbability:
                mutations.append(1)
                string += " will undergo mutation"
            else:
                mutations.append(0)
                string += " will not undergo mutation"
            print(string)
        print()
        
        print("-------------------------------------------------- POPULATION AFTER MUTATIONS ------------------------------------------------")
        for i in range(len(populationBinary)):
            if mutations[i] == 1:
                index = random.randint(0, len(populationBinary[i]) - 1)
                populationBinary[i] = mutation(populationBinary[i], index)
        
        fitness = []
        
        for i in range(populationSize):
            gene = populationBinary[i]
            value = int(gene, 2)
            
            x = domain[0] + value * (domain[1] - domain[0]) / (2 ** len(gene))
            fx = calculateFitness(a, b, c, x)
            fitness.append(fx)
            print(f"Individual {i + 1}:\n  value: {str(x):30s}       gene: {gene:30s}       function value: {str(fx):30s}")
        print()
        
        intervals = cumulativeProbabilities(selectionProbabilities(fitness))
        
        chromosomes = crossoverChromosomes
        for generation in range(2, numGenerations + 1):
            print(f"######################################################## STAGE {generation} #######################################################")
            
            # Keep the chromosome with the maximum fitness from the previous generation
            indexMax = fitness.index(max(fitness))
            chromosome = chromosomes[indexMax]
            
            chromosomes = selection(intervals, populationSize, chromosomes)
            
            # Calculate the new fitness and eliminate the chromosome with the minimum fitness, replacing it with the one with the maximum fitness from the previous generation
            fitness = updateFitness(chromosomes, domain, a, b, c)
            indexMin = fitness.index(min(fitness))
            chromosomes[indexMin] = chromosome
            
            # Crossover and mutation process
            chromosomes = populationAfterCrossover(chromosomes, crossoverProbability)
            chromosomes = populationAfterMutation(chromosomes, mutationProbability)
            
            fitness = updateFitness(chromosomes, domain, a, b, c)
            selectionProb = selectionProbabilities(fitness)
            intervals = cumulativeProbabilities(selectionProb)
            
            print("MaxFitness: " + str(max(fitness)))
            print("meanFitness: " + str(sum(fitness) / len(fitness)))
        
main()
