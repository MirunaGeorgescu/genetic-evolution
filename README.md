# 🧬 Genetic Algorithm for Function Maximization

## 📑 Overview
This project implements a genetic algorithm to find the maximum of a given positive function within a specified domain. The target function is a quadratic polynomial with user-defined coefficients.

## 📊 Algorithm Steps

### Initialization
Generate an initial population of chromosomes using binary encoding.
Evaluate the fitness of each chromosome based on the target function.

### Selection
Calculate selection probabilities for each chromosome based on their fitness.
Use cumulative probabilities to create selection intervals.
Randomly select chromosomes for the next generation based on these intervals.

### Crossover
Determine crossover participation based on a user-defined probability.
Randomly select pairs of chromosomes for crossover.
Apply single-point crossover to create new chromosomes.

### Mutation
Determine mutation participation based on a user-defined probability.
Introduce random mutations to selected chromosomes.

### Update Population
Replace the old population with the new one obtained after crossover and mutation.
Repeat the process for a specified number of generations.

## 📋 Input Parameters

- Population size
- Domain of the function
- Coefficients of the quadratic polynomial
- Precision for discretization
- Crossover probability
- Mutation probability
- Number of generations
