import numpy as np
import pandas as pd
import math
import random
from utils import evaluate_sequence
from heuristics import *
from local_search import *

'''Genetic Algorithm'''
def selection_GA(population, processing_times, n_selected, strategie):
    # case "roulette":
    if strategie == "roulette":
        fitness = [evaluate_sequence(seq, processing_times) for seq in population]
        fitness_sum = sum(fitness)
        selection_probs = [fitness[i]/fitness_sum for i in range(len(population))]
        cum_probs = [sum(selection_probs[:i+1]) for i in range(len(population))]
        selected = []
        for i in range(n_selected):
            while True:
                rand = random.random()
                for j, cum_prob in enumerate(cum_probs):
                    if rand < cum_prob:
                        break
                if population[j] not in selected:
                    selected.append(population[j])
                    break
    # case "Elitism":
    elif strategie == "Elitism":
        fitness = [evaluate_sequence(seq, processing_times) for seq in population]
        sorted_population = [x for x, _ in sorted(zip(population, fitness), key=lambda pair: pair[1], reverse=False)]
        selected = sorted_population[:n_selected]

    # case "rank":
    elif strategie == "rank":
        fitness = [evaluate_sequence(seq, processing_times) for seq in population]
        sorted_population = sorted(population, key = lambda x: fitness[population.index(x)])
        fitness_sum = sum(i+1 for i in range(len(sorted_population)))
        selection_probs = [(len(sorted_population)-i)/fitness_sum for i in range(len(sorted_population))]
        selected = []
        for i in range(n_selected):
            selected_index = random.choices(range(len(sorted_population)), weights=selection_probs)[0]
            selected.append(sorted_population[selected_index])
            sorted_population.pop(selected_index)
            selection_probs.pop(selected_index)
            
    # case "tournament":
    elif strategie == "tournament":
        k = 2
        selected = []
        for i in range(n_selected):
            while True:
                tournament = random.sample(population, k)
                tournament = [seq for seq in tournament if seq not in selected]
                if tournament:
                    break
            fitness = [evaluate_sequence(seq, processing_times) for seq in tournament]
            selected.append(tournament[fitness.index(min(fitness))])

    return selected

def crossover_GA(p1, p2, points):
    jobs = len(p1) - 1
    # One points crossover
    if points == 'ONE':
        point = random.randint(0, jobs)
        offspring1 = p1[:point] + p2[point:]
        offspring2 = p2[:point] + p1[point:]
        points = [point]
    else: # Two Points crossover
        point_1 = random.randint(0, jobs)
        point_2 = random.randint(0, jobs)
        if point_1 > point_2:
            point_1, point_2 = point_2, point_1
        offspring1 = p1[:point_1] + p2[point_1:point_2] + p1[point_2:]
        offspring2 = p2[:point_1] + p1[point_1:point_2] + p2[point_2:]
        points = [point_1, point_2]
    # Remove duplicates and replace with genes from the other offspring
    offspring1 = remove_duplicates_GA(offspring1, offspring2, points)
    offspring2 = remove_duplicates_GA(offspring2, offspring1, points)
    return offspring1, offspring2

def remove_duplicates_GA(offspring, other_offspring, points):
    jobs = len(offspring) - 1
    check_points = len(points) > 1
    while True:
        duplicates = set([job for job in offspring if offspring.count(job) > 1])
        if not duplicates:
            break
        for job in duplicates:
            pos = [i for i, x in enumerate(offspring) if x == job]
            if (check_points and ((pos[0] < points[0]) or (pos[0] >= points[1])) ) or  ( (pos[0] < points[0]) and not check_points):
                dup = pos[0]
                index = pos[1]
            else:
                dup = pos[1]
                index = pos[0]

            offspring[dup] = other_offspring[index]
    return offspring

def mutation_GA(sequence, mutation_rate):
    num_jobs = len(sequence)
    for i in range(num_jobs):
        r = random.random()
        if r < mutation_rate:
            available_jobs = [j for j in range(num_jobs) if j != sequence[i]]
            newjob = random.sample(available_jobs, 1)[0]
            sequence[sequence.index(newjob)] = sequence[i]
            sequence[i] = newjob
    return sequence

def genetic_algorithm(processing_times, init_pop, pop_size, select_pop_size, selection_method, cossover, mutation_probability, num_iterations):
    # Init population generation
    population = init_pop
    best_seq = selection_GA(population, processing_times, 1, "Elitism")[0]
    best_cost = evaluate_sequence(best_seq, processing_times)
    for i in range(num_iterations):
        # Selection
        s = int(select_pop_size * pop_size) # number of selected individus to be parents (%)
        parents = selection_GA(population, processing_times, s, selection_method)
        # Crossover
        new_generation = []
        for _ in range(0, pop_size, 2):
            parent1 = random.choice(parents)
            parent2 = random.choice([p for p in parents if p != parent1])
            child1, child2 = crossover_GA(parent1, parent2, cossover)
            new_generation.append(child1)
            new_generation.append(child2)

        new_generation = new_generation[:pop_size]
        # Mutation
        for i in range(pop_size):
            if random.uniform(0, 1) < mutation_probability:
                new_generation[i] = mutation_GA(new_generation[i], mutation_probability)
        # Replacement
        population = new_generation

        # checking for best seq in current population
        best_seq_pop = selection_GA(population, processing_times, 1, "Elitism")[0]
        best_cost_pop = evaluate_sequence(best_seq_pop, processing_times)
        if best_cost_pop < best_cost:
            best_seq = best_seq_pop.copy()
            best_cost = best_cost_pop

    return best_seq, best_cost   

'''Ant colony'''
def distance(i, j, processing_times):
    m = processing_times.shape[1]
    max_delay = 0
    for k in range(2, m):
        delay = np.sum(processing_times[i,1:k]) - np.sum(processing_times[j,1:k-1])
        if delay > max_delay:
            max_delay = delay
    return (processing_times[i, 0] + max(0, max_delay))

def ant_colony_optimization(num_ants, num_iterations, alpha, beta, evaporation_rate, Q, tau0, q0, processing_times):
    num_jobs = processing_times.shape[0]
    tau = np.ones((num_jobs, num_jobs)) * tau0
    best_schedule = None
    best_makespan = np.inf
    
    # visibility
    heuristic_values = np.array([ [1/distance(i, j, processing_times) if j != i else 0 for j in range(num_jobs)] for i in range(num_jobs)])
    for iteration in range(num_iterations): 
        sequences = []
        for ant in range(num_ants):
            current_job = np.random.randint(num_jobs)
            current_sequence = [current_job] 
            for job in range(num_jobs-1):
                unscheduled = [j for j in range(num_jobs) if j not in current_sequence]
                probabilities = np.zeros(len(unscheduled))
                
                for i, unscheduled_job in enumerate(unscheduled):
                    probabilities[i] = (tau[current_job, unscheduled_job]**alpha) * (heuristic_values[current_job, unscheduled_job]**beta)
                
                probabilities /= np.sum(probabilities)
                
                # pseudo-random-proportional
                q = random.random()
                if q < q0: # exploitation : choose the best
                    next_job = unscheduled[np.where( probabilities == np.max(probabilities))[0][0]]
                else: # exploration
                    next_job = np.random.choice(unscheduled, p=probabilities)
                    
                current_sequence.append(next_job)
                current_point = next_job
            
            makespan = evaluate_sequence(current_sequence, processing_times)
            sequences.append((current_sequence, makespan))
            
            if makespan < best_makespan:
                best_makespan = makespan
                best_schedule = current_sequence.copy()
        
        pheromone_delta =  np.zeros((num_jobs, num_jobs))
        for ant in range(num_ants):
            seq, ev = sequences[ant]
            for j in range(num_jobs-1):
                pheromone_delta[seq[j], seq[j+1]] += Q / ev
                             
        # Update pheromone  
        tau = tau * evaporation_rate + pheromone_delta
    
    return best_schedule, best_makespan

'''Greedy NEH'''
def greedy_neh_algorithm(processing_times, num_candidates, num_iterations):
    n = len(processing_times)
    ordered_sequence = order_jobs_in_descending_order_of_total_completion_time(processing_times)
    best_sequence = []
    best_cmax = float('inf')
    for i in range(num_iterations):
        partial_sequence = [ordered_sequence[0]]
        for k in range(1, n):
            candidates = []
            for job in ordered_sequence:
                if job not in partial_sequence:
                    for i in range(k+1):
                        candidate_sequence = insertion(partial_sequence, i, job)
                        candidate_cmax = evaluate_sequence(candidate_sequence, processing_times)
                        candidates.append((candidate_sequence, candidate_cmax))
            candidates.sort(key=lambda x: x[1])
            partial_sequence, cmax = random.choice(candidates[:num_candidates])
        if cmax < best_cmax:
                best_sequence = partial_sequence
                best_cmax = cmax
        ordered_sequence.append(ordered_sequence.pop(0))
    return best_sequence, best_cmax

def GRNEH(processing_times, num_candidates):
    n = len(processing_times)
    ordered_sequence = order_jobs_in_descending_order_of_total_completion_time(processing_times)
    partial_sequence = [ordered_sequence[0]]
    for k in range(1, n):
        candidates = []
        for job in ordered_sequence:
            if job not in partial_sequence:
                for i in range(k+1):
                    candidate_sequence = insertion(partial_sequence, i, job)
                    candidate_cmax = evaluate_sequence(candidate_sequence, processing_times)
                    candidates.append((candidate_sequence, candidate_cmax))
        candidates.sort(key=lambda x: x[1])
        partial_sequence, cmax = random.choice(candidates[:num_candidates])
    return partial_sequence, cmax


'''Genetic/VNS hybrid'''
def init_pop(processing_times, pop_size, a):
    
    #Les individus stockeront la séquence ( solution ), le makespan et un compteur c qui va déterminé si une solution est agée ou non
    population = []
    #Heuristics
    sol, cmax = neh_algorithm(processing_times)
    population.append(( sol, cmax ,0))
    sol, cmax = CDS_heuristic(processing_times)
    population.append(( sol, cmax ,0))
    sol=palmer_heuristic(processing_times)
    population.append((sol, evaluate_sequence(sol, processing_times),0))
    sol, cmax = artificial_heuristic(processing_times)
    population.append(( sol, cmax ,0))
    sol = gupta_heuristic(processing_times)
    population.append((sol, evaluate_sequence(sol, processing_times),0))
    
    #GRNEH
    for i in range(int(a*pop_size) - 5 ):
        sol, cmax = GRNEH(processing_times, 5)
        population.append(( sol, cmax ,0))
    
    #Random
    for i in range (int((1-a)*pop_size)):
        sol = np.random.permutation(processing_times.shape[0]).tolist() 
        population.append((sol, evaluate_sequence(sol, processing_times),0))
    
    return population

def selection(population, processing_times, n_selected, strategie):
    # case "roulette":
    if strategie == "roulette":
        fitness = [evaluate_sequence(seq, processing_times) for seq in population]
        fitness_sum = sum(fitness)
        selection_probs = [fitness[i]/fitness_sum for i in range(len(population))]
        cum_probs = [sum(selection_probs[:i+1]) for i in range(len(population))]
        selected = []
        for i in range(n_selected):
            while True:
                rand = random.random()
                for j, cum_prob in enumerate(cum_probs):
                    if rand < cum_prob:
                        break
                if population[j] not in selected:
                    selected.append(population[j])
                    break
    # case "Elitism":
    elif strategie == "Elitism":
        fitness = [evaluate_sequence(seq, processing_times) for seq in population]
        sorted_population = [x for x, _ in sorted(zip(population, fitness), key=lambda pair: pair[1], reverse=False)]
        selected = sorted_population[:n_selected]

    # case "rank":
    elif strategie == "rank":
        fitness = [evaluate_sequence(seq, processing_times) for seq in population]
        sorted_population = sorted(population, key = lambda x: fitness[population.index(x)])
        fitness_sum = sum(i+1 for i in range(len(sorted_population)))
        selection_probs = [(len(sorted_population)-i)/fitness_sum for i in range(len(sorted_population))]
        selected = []
        for i in range(n_selected):
            selected_index = random.choices(range(len(sorted_population)), weights=selection_probs)[0]
            selected.append(sorted_population[selected_index])
            sorted_population.pop(selected_index)
            selection_probs.pop(selected_index)
            
    # case "tournament":
    elif strategie == "tournament":
        #a random number of solutions is selected at first, ranging from 3 to 10.
        k = random.randint(3, 10)
        selected = []
        for i in range(n_selected):
            while True:
                tournament = random.sample(population, k)
                tournament = [seq for seq in tournament if seq not in selected]
                if tournament:
                    break
            selected.append(min(tournament, key=lambda x: x[1]))

    return selected

def tp_crossover(parent1, parent2, processing_times):
    jobs = len(processing_times) - 1
    point1 = random.randint(0, jobs)
    point2 = random.randint(0, jobs)
    
    while point1 == point2 or point1 == (point2-1) or (point1 == point2+1) : 
        point2 = random.randint(0, jobs)
        
    if point1 > point2:
        point1, point2 = point2, point1

    # Create offspring as copies of parents
    offspring1 = parent1[0].copy()
    offspring2 = parent2[0].copy()

     # Set the segment between point1 and point2 to empty in offspring1
    offspring1[point1:point2] = [None] * (point2 - point1)

     # Set the segment between point1 and point2 to empty in offspring2
    offspring2[point1:point2] = [None] * (point2 - point1)
    
    # Remove duplicates from offspring1 and copy remaining genes from parent2
    for gene in parent2[0]:
         if gene not in offspring1:
            offspring1[offspring1.index(None)] = gene
            
    # Remove duplicates from offspring2 and copy remaining genes from parent1
    for gene in parent1[0]:
         if gene not in offspring2:
            offspring2[offspring2.index(None)] = gene
    
    return offspring1, offspring2

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


def sol_unique(population, solution): 
    for p in population:
        if p[0] == solution:
            return False
    return True

def replace_sol(population, old_sol, new_sol):
    for i in range(len(population)):
        if (population[i] == old_sol):
            population[i] = new_sol
            return
    return  

def replace_with_offspring(population, offspring):
    #Vérifier que l'offspring n'existe pas déjà dans la population
    if (sol_unique(population, offspring[0])==False): 
        return

    #Si il est unique, on prend le pire individu de la population courante
    worst_ind = max(population, key=lambda x: x[1])
    if worst_ind[1] > offspring[1]:
        replace_sol(population, worst_ind, offspring)
        
    return  

def AG(population, processing_times):
    children = []
    #génération des enfants
    for i in range(int(len(population)/2)):
        mating_pool = selection(population, processing_times, 2, "tournament")
        os1, os2 = tp_crossover(mating_pool[0], mating_pool[1], processing_times)
        os1, fit1 = random_insertion(os1, processing_times)
        os2, fit2 = random_insertion(os2, processing_times)
        children.append((os1, fit1, 0))
        children.append((os2, fit2, 0))
   
    #remplacement dans la population
    for child in children: 
        replace_with_offspring(population, child)
    return

def or_opt(solution, lenInterv, processing_times): 
    num_jobs = len(processing_times)
    
    # [i,j] les positions respectives des éléments à déplacer dans la séquence de la solution
    i = random.randint(0, num_jobs - lenInterv)
    j = i + lenInterv
        
    #sélectionner k, la position où insérer la séquence déplacée
    k = random.randint(0, num_jobs-1)

    neighbor = solution.copy()
    # Extraire la séquence entre i et j
    seq = neighbor[i:j+1]
    
    # Retirer la séquence de la solution
    neighbor = neighbor[:i] + neighbor[j+1:]
    
    # Insérer la séquence après k
    neighbor = neighbor[:k+1] + seq + neighbor[k+1:]
    
    return neighbor, evaluate_sequence(neighbor, processing_times)

def two_opt(solution, lenInterv, processing_times):
    
    num_jobs = len(processing_times)
    # [i,j] les positions respectives des éléments à déplacer dans la séquence de la solution
    i = random.randint(0, num_jobs - lenInterv)
    j = i + lenInterv
    
    #sélectionner k, la position où insérer la séquence déplacée
    k = random.randint(0, num_jobs-1)
    
    neighbor = solution.copy()
    # Extraire la séquence entre i et j et l'inverser
    seq = list(reversed(neighbor[i:j+1]))
    
    # Retirer la séquence de la solution
    neighbor = neighbor[:i] + neighbor[j+1:]

    # Insérer la séquence après k
    neighbor = neighbor[:k+1] + seq + neighbor[k+1:]
    
    return neighbor, evaluate_sequence(neighbor, processing_times)

def shake(solution, k):
    n = len(solution)
    # Si k supérieur à la longueur de la liste, alors on crée des perturbations sur tous les éléments.
    indices = random.sample(range(n), min(k, n-1))
    indices.sort()
    neighbor = solution.copy()
    for i in indices:
        j = (i+k) % n
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor

def intensification(population, processing_times):
    
    #On commence par sélectionner la sous-population sur laquelle on appliquera l'intensification    
    #La taille de la sous_pop est égale au nombre de jobs. 
    num_jobs = len(processing_times)
    
    #La moitié de la sous-population est représentée par les meilleurs solutions.
    population.sort(key=lambda x: x[1])
    sous_pop = population[:int(num_jobs/2)]
    
    #l'autre moitié sera choisie au hasard
    choix = population[int(num_jobs/2):]
    seq = random.sample(choix, int(num_jobs/2))
    sous_pop = sous_pop + seq

    
    #Maintenant nous allons appliquer un VNS sur chacune des populations résultantes
    for S in sous_pop:
        #Si la solution est considérée comme ancienne, nous la remplaçons par une solution gérérée par GRNEH sans itérations
        #Sauf pour la meilleure solution courrante
        if (S[2]>=8): 
            if (S != min(population, key=lambda x: x[1])):
                sol, cmax = GRNEH(processing_times, 5)
                B = (sol, cmax, 0)
                while (sol_unique(population, B[0])==False): 
                    B = (shake(B[0], 1), evaluate_sequence(B[0], processing_times), 0)
                replace_sol ( population, S, B)
        
        else:
            #B va stocker l'optimum local trouvé par VNS
            B = S 
                
            if ( 4 <= S[2] < 8 ):
                B = ( shake(B[0], 2), evaluate_sequence(B[0], processing_times), B[2])
        
            for loop in range ( (num_jobs)*(num_jobs-1) ):
                if (S[2]== 0 or S[2]== 4): 
                    sol, cmax = random_swap(B[0], processing_times)
                    ST = (sol, cmax, 0)
                if (S[2]== 1 or S[2]== 5):
                    sol, cmax = random_insertion(B[0], processing_times)
                    ST = (sol, cmax, 0)
                if (S[2]== 2 or S[2]== 6): 
                    sol, cmax = or_opt(B[0], 3, processing_times)
                    ST = (sol, cmax, 0)
                if (S[2]== 3 or S[2]== 7):
                    sol, cmax = two_opt(B[0], 3, processing_times)
                    ST = (sol, cmax, 0)
                if ( ST[1] <= B[1] ): 
                    B = ST
         
        #Si l'optimum local retournée par VNS améliore la solution i, alors on vérifie qu'il est unique puis on remplace
        #OBSERVATION : Si on met <=, on aura toujours des remplacement et bcp de solution semblable
            if ( B[1] < S[1] ): 
                if (sol_unique(population, B[0])): 
                    replace_sol ( population, S, B)

                else:
                    replace_sol ( population, S, (S[0], S[1], S[2]+1))        

            else: 
                replace_sol ( population, S, (S[0], S[1], S[2]+1))        
    
    best_indiv = min(population, key=lambda x: x[1])
    
    return best_indiv[0], best_indiv[1]

def NEGA(population, processing_times, num_iterations):
    for i in range(num_iterations):
        AG(population, processing_times)
        best_sol, best_fitness = intensification(population, processing_times)
    return best_sol, best_fitness