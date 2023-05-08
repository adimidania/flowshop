import numpy as np
import pandas as pd
import math
import random
from utils import evaluate_sequence

'''Neighborhood functions'''
def swap(solution, i, k):
    temp = solution[k]
    solution[k] = solution[i]
    solution[i] = temp
    return solution

def random_swap(solution, processing_times):
    i = np.random.choice(list(solution))
    k = np.random.choice(list(solution))
    # Generating two different random positions
    while (i == k):
        k = np.random.choice(list(solution))
    # Switch between job i and job k in the given sequence
    neighbor = solution.copy()
    return swap(neighbor, i, k), evaluate_sequence(neighbor, processing_times)

def best_first_swap(solution, processing_times):
    # This function will take a solution, and return the first best solution.
    # The first solution that is better then the current one 'solution' in args.
    num_jobs = len(solution)
    best_cmax = evaluate_sequence(solution, processing_times)
    best_neighbor = solution.copy()
    for k1 in range(num_jobs):
        for k2 in range(k1+1, num_jobs):
            neighbor = solution.copy()
            neighbor = swap(neighbor,k1,k2)
            cmax = evaluate_sequence(neighbor, processing_times)
            if cmax < best_cmax:
                best_neighbor = neighbor
                best_cmax = cmax
                return best_neighbor, best_cmax
    return best_neighbor, best_cmax

def best_swap(solution, processing_times):
    # This function will take a solution, and return its best neighbor solution.
    num_jobs = len(solution)
    best_cmax = np.Infinity
    for k1 in range(num_jobs):
        for k2 in range(k1+1, num_jobs):
            neighbor = solution.copy()
            neighbor = swap(neighbor,k1,k2)
            cmax = evaluate_sequence(neighbor, processing_times)
            if cmax < best_cmax:
                best_neighbor = neighbor
                best_cmax = cmax
    return best_neighbor, best_cmax

def best_swaps(solution, processing_times):
    # This function will take a solution, and return a list that contains all solutions that are better than it.
    num_jobs = len(solution)
    cmax = evaluate_sequence(solution, processing_times)
    bests = []
    for k1 in range(num_jobs):
        for k2 in range(k1+1, num_jobs):
            neighbor = solution.copy()
            swap(neighbor,k1,k2)
            neighbor_cmax = evaluate_sequence(neighbor, processing_times)
            if neighbor_cmax < cmax:
                bests.append((neighbor_cmax, neighbor))
    bests.sort(key=lambda x: x[0])
    return bests

def random_insertion(solution, processing_times):
    # This function consists of choosing random two indices, i and k.
    # Remove the element at indice i, and insert it in the position k.
    i = np.random.choice(list(solution))
    k = np.random.choice(list(solution))
    while (i == k):
        k = np.random.choice(list(solution))
    neighbor = solution.copy()
    neighbor.remove(solution[i])
    neighbor.insert(k, solution[i])
    return neighbor, evaluate_sequence(neighbor, processing_times)

def best_insertion(solution, processing_times):
    # This function consists of trying all different insertions.
    # Then it returns the best one among them
    num_jobs = len(solution)
    best_cmax = np.Infinity
    for k1 in range(num_jobs):
        s = solution.copy()
        s_job = solution[k1]
        s.remove(s_job)
        for k2 in range(num_jobs):
            if k1 != k2:
                neighbor = s.copy()
                neighbor.insert(k2, s_job)
                cmax = evaluate_sequence(neighbor, processing_times)
                if cmax < best_cmax:
                    best_neighbor = neighbor
                    best_cmax = cmax
    return best_neighbor, best_cmax

def best_edge_insertion(solution, processing_times):
    num_jobs = len(solution)
    best_cmax = np.Infinity
    for k1 in range(num_jobs-1):
        s = solution.copy()
        s_job1 = s[k1] 
        s_job2 = s[k1+1]
        s.remove(s_job1)
        s.remove(s_job2)
        for k2 in range(num_jobs-1):
            if(k1 != k2):
                neighbor = s.copy()
                neighbor.insert(k2, s_job1)
                neighbor.insert(k2+1, s_job2)
                cmax = evaluate_sequence(neighbor, processing_times)
                if cmax < best_cmax:
                    best_neighbor = neighbor
                    best_cmax = cmax
    return best_neighbor, best_cmax

def get_neighbor(solution, processing_times, method="random_swap"):
    # Swapping methods
    if method == "random_swap":
        neighbor, cost = random_swap(solution, processing_times)
    elif method == "best_swap":
        neighbor, cost = best_swap(solution, processing_times)
    elif method == "best_first_swap":
        neighbor, cost = best_first_swap(solution, processing_times)
    # Insertion methods
    elif method == "random_insertion":
        neighbor, cost = random_insertion(solution, processing_times)
    elif method == "best_edge_insertion":
        neighbor, cost = best_edge_insertion(solution, processing_times)
    elif method == "best_insertion":
        neighbor, cost = best_insertion(solution, processing_times)
    # Randomly pick a method of generating neighbors.
    else:     
        i = random.randint(0, 5)
        if i == 0:
            neighbor, cost = random_swap(solution, processing_times)
        elif i == 1:
            neighbor, cost = best_swap(solution, processing_times)
        elif i == 2:
            neighbor, cost = best_first_swap(solution, processing_times)
        elif i == 3:
            neighbor, cost = random_insertion(solution, processing_times)
        elif i == 4:
            neighbor, cost = best_edge_insertion(solution, processing_times)
        else:
            neighbor, cost = best_insertion(solution, processing_times)
    return neighbor, cost

'''Random Walk'''
def random_walk(solution, processing_times, nb_iter=1000, threshold=None):
    x = solution
    cmax = evaluate_sequence(solution, processing_times)
    iterations = 0
    while iterations < nb_iter:
        x, cmax = random_swap(x, processing_times)
        if threshold is not None and cmax < threshold:
            return x, cmax, iterations
        iterations += 1
    return x, cmax, iterations

'''Simple Hill climbing'''
def simple_hill_climbing(solution, processing_times, nb_iter=10000):
    x = solution
    cmax = evaluate_sequence(solution, processing_times)
    iterations = 0
    while iterations < nb_iter:
        best_neighbor, best_cmax  = best_first_swap(x, processing_times)
        if best_cmax == cmax:
            return best_neighbor, best_cmax, iterations
        x = best_neighbor
        cmax = best_cmax
        iterations += 1
    return x, cmax, iterations

'''Steepest Ascent Hill climbing'''
def steepest_ascent_hill_climbing(solution, processing_times, nb_iter=1000):
    x = solution
    cmax = evaluate_sequence(solution, processing_times)
    iterations = 0
    while iterations < nb_iter:
        best_neighbor, best_cmax = best_swap(solution, processing_times)
        if best_cmax > cmax:
            return x, cmax
        else:
            x = best_neighbor
            cmax = best_cmax
            iterations += 1
    return best_neighbor, best_cmax, iterations

'''Stochastic Hill climbing'''
def stochastic_hill_climbing(solution, processing_times, nb_iter=1000):
    x = solution
    cmax = evaluate_sequence(solution, processing_times)
    iterations = 0
    while iterations < nb_iter:
        best_neighbours  = best_swaps(x, processing_times)
        if len(best_neighbours) == 0:
            return x, cmax, iterations
        i = random.randint(0,len(best_neighbours) - 1)
        best_cmax, best_neighbor = best_neighbours[i]
        if best_cmax > cmax:
            return x, cmax, iterations
        x = best_neighbor
        cmax = best_cmax
        iterations += 1
    return best_neighbor, best_cmax, iterations

'''Simulated annealing'''
def simulated_annealing(initial_solution, processing_times, method="random", initial_temp=100, final_temp=1, alpha=0.1):
    current_temp = initial_temp
    current_solution = initial_solution.copy()
    current_cost = evaluate_sequence(initial_solution, processing_times)
    while current_temp > final_temp:
        neighbor, neighbor_cost  = get_neighbor(current_solution, processing_times, method)
        cost_diff = current_cost - neighbor_cost
        if cost_diff > 0:
            current_solution = neighbor
            current_cost = neighbor_cost
        else:
            if random.uniform(0, 1) < math.exp(-cost_diff / current_temp):
                solution = neighbor
        current_temp -= alpha
    return current_solution, current_cost

'''Tabu Search'''
def tabu_search(initial_solution, processing_times, nb_iter=1000):
    tabu_list = deque(maxlen=15)
    best_solution = initial_solution.copy()
    best_cmax = evaluate_sequence(initial_solution, processing_times)
    iterations = 0
    while iterations < nb_iter:
        neighbours = best_swaps(best_solution, processing_times)
        # If we don't have any best neighboors, we generate one randomly
        if len(neighbours):
            best_solution = random_swap(best_solution, processing_times)
        # We check for neighbours
        for neighbour in neighbours:
            if neighbour[1] not in tabu_list:
                best_solution = neighbour[1]
                best_cmax = neighbour[0]
                tabu_list.append(neighbour[1])
                break
            # If It is in the tabu list we would look for the next neighbor
            else:
                continue
        iterations += 1
    return best_solution, best_cmax, iterations

'''VNS'''
def shake(solution, k):
    n = len(solution)
    # If k is greater than the length of the list, then we create perturbations on all elements.
    indices = random.sample(range(n), min(k, n-1))
    indices.sort()
    neighbor = solution.copy()
    for i in indices:
        j = (i+k) % n
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor

def vns(sol_init, processing_times, max_iterations, k_max):
    num_jobs = len(sol_init)
    current_solution = sol_init
    current_cost = evaluate_sequence(current_solution, processing_times)
    k = 1
    iteration = 0
    while iteration < max_iterations:       
        best_neighbor_solution = shake(current_solution, k)
        best_neighbor_cost = evaluate_sequence(best_neighbor_solution, processing_times)
        for l in range(1, k_max+1):
            neighbor, neighbor_cost  = get_neighbor(current_solution, processing_times)
            if (neighbor_cost < best_neighbor_cost):
                best_neighbor_solution = neighbor
                best_neighbor_cost = neighbor_cost            
        if ( best_neighbor_cost < current_cost ):
            current_solution = best_neighbor_solution
            current_cost = best_neighbor_cost
            k = 1
        else:
            k += 1
        iteration += 1
    return current_solution, current_cost
