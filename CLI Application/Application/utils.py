import numpy as np
import itertools
''' Evaluate sequence '''
def evaluate_sequence(sequence, processing_times):
    _, num_machines = processing_times.shape
    num_jobs = len(sequence)
    completion_times = np.zeros((num_jobs, num_machines))
    
    # Calculate the completion times for the first machine
    completion_times[0][0] = processing_times[sequence[0]][0]
    for i in range(1, num_jobs):
        completion_times[i][0] = completion_times[i-1][0] + processing_times[sequence[i]][0]
    
    # Calculate the completion times for the remaining machines
    for j in range(1, num_machines):
        completion_times[0][j] = completion_times[0][j-1] + processing_times[sequence[0]][j]
        for i in range(1, num_jobs):
            completion_times[i][j] = max(completion_times[i-1][j], completion_times[i][j-1]) + processing_times[sequence[i]][j]
    
    # Return the total completion time, which is the completion time of the last job in the last machine
    return completion_times[num_jobs-1][num_machines-1]

''' Brute Force algorithm '''
def all_permutations(iterable):
    permutations = list(itertools.permutations(iterable))
    permutations_as_lists = [list(p) for p in permutations]
    return permutations_as_lists

def brute_force(processing_times, permutations):
    M = float('inf')
    sol = []
    for permutation in permutations:
        m = evaluate_sequence(permutation, processing_times)
        if m < M:
            M = m
            sol = permutation
    return sol, M

''' Generate random data with n jobs and m machines '''
def generate_data(n, m):
    rnd_data = np.random.randint(size=(n,m), low=5, high=120)
    return rnd_data

''' Generate random solution of n jobs '''
def generate_seq(n):
    permutation = np.random.permutation(n).tolist()
    return permutation
