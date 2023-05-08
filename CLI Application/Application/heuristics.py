import numpy as np
from utils import evaluate_sequence

'''NEH'''
def order_jobs_in_descending_order_of_total_completion_time(processing_times):
    total_completion_time = processing_times.sum(axis=1)
    return np.argsort(total_completion_time, axis=0).tolist()

def insertion(sequence, position, value):
    new_seq = sequence[:]
    new_seq.insert(position, value)
    return new_seq

def neh_algorithm(processing_times):
    ordered_sequence = order_jobs_in_descending_order_of_total_completion_time(processing_times)
    # Define the initial order
    J1, J2 = ordered_sequence[:2]
    sequence = [J1, J2] if evaluate_sequence([J1, J2], processing_times) < evaluate_sequence([J2, J1], processing_times) else [J2, J1]
    del ordered_sequence[:2]
    # Add remaining jobs
    for job in ordered_sequence:
        Cmax = float('inf')
        best_sequence = []
        for i in range(len(sequence)+1):
            new_sequence = insertion(sequence, i, job)
            Cmax_eval = evaluate_sequence(new_sequence, processing_times)
            if Cmax_eval < Cmax:
                Cmax = Cmax_eval
                best_sequence = new_sequence
        sequence = best_sequence
    return sequence, Cmax

'''Johnson'''
def johnson_method(processing_times):
    jobs, machines = processing_times.shape
    copy_processing_times = processing_times.copy()
    maximum = processing_times.max() + 1
    m1 = []
    m2 = []
    
    if machines != 2:
        print("Johson method only works with two machines")
        return []
        
    for i in range(jobs):
        minimum = copy_processing_times.min()
        position = np.where(copy_processing_times == minimum)
        
        if position[1][0] == 0:
            m1.append(position[0][0])
        else:
            m2.insert(0, position[0][0])
        
        copy_processing_times[position[0][0]] = maximum
        
    return m1+m2

'''Ham'''
def ham_heuristic(processing_time):
    jobs, machines = processing_time.shape
    sequence = list(range(jobs))
    # Calculating the first summation
    P1 = processing_time[:,:machines//2].sum(axis=1)
    # Calculating the second summation
    P2 = processing_time[:,machines//2:].sum(axis=1)
    # Calculating the first solution, ordered by P2 - P1
    P2_P1 = P2 - P1
    solution_1 = [job for _ , job in sorted(zip(P2_P1, sequence), reverse=True)]
    # Calculating the second solution
    positives = np.argwhere(P2_P1 >= 0).flatten()
    negatives = np.argwhere(P2_P1 < 0).flatten()
    positive_indices = [job for _ , job in sorted(zip(P1[positives], positives))]
    negative_indices = [job for _ , job in sorted(zip(P2[negatives], negatives), reverse=True)]
    positive_indices.extend(negative_indices)
    # Calculating Cmax for both solutions
    Cmax1 = evaluate_sequence(solution_1, processing_time)
    Cmax2 = evaluate_sequence(positive_indices, processing_time)
    # Returning the best solution among them
    if Cmax1 < Cmax2:
        return solution_1, Cmax1
    else:
        return positive_indices, Cmax2

'''Palmer'''
def palmer_heuristic(processing_times):
    jobs, machines = processing_times.shape
    slope_indices = []
    for i in range(jobs):
        processing_time_sum = np.sum(processing_times[i])
        fi = 0
        for j in range(machines):
            fi += (machines - 2*j + 1) * processing_times[i][j] / processing_time_sum
        slope_indices.append(fi)
    order = sorted(range(jobs), key=lambda k: slope_indices[k])
    return order

'''CDS'''
def CDS_heuristic(processing_times):
    jobs, machines = processing_times.shape
    m = machines-1
    johnson_proc_times = np.zeros((jobs,2))
    best_cost = np.inf
    best_seq = []
    for k in range(m):
        johnson_proc_times[:,0] += processing_times[:,k]
        johnson_proc_times[:,1] += processing_times[:,-k-1]
        seq = johnson_method(johnson_proc_times)
        cost = evaluate_sequence(seq,processing_times)
        if cost < best_cost:
            best_cost = cost
            best_seq = seq
    return best_seq, best_cost

'''Gupta'''
def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def min_gupta(job, processing_times):
    m = np.inf
    _, machines = processing_times.shape
    for i in range(machines-1):
        k = processing_times[job][i] + processing_times[job][i+1]
        if (k < m):
            m = k
    return m

def gupta_heuristic(processing_times):
    jobs, machines = processing_times.shape
    f = []
    total_times = []
    for i in range(jobs):
        fi = sign(processing_times[i][0] - processing_times[i][machines-1]) / min_gupta(i,processing_times)
        f.append(fi)
        total_time = sum(processing_times[i])
        total_times.append(total_time)
    order = sorted(range(jobs), key=lambda k: (f[k], total_times[k]))
    return order

'''PRSKE'''
def skewness(processing_times):
    jobs, machines = processing_times.shape
    skewnesses = []
    # Calculate the skewness for each job 
    for i in range(jobs):
        avg = np.mean(processing_times[i,:])
        numerator = 0
        denominator = 0
        for j in range(machines):
            m = (processing_times[i,j] - avg)
            numerator += m**3
            denominator += m**2
        # Actually calculating the skewness    
        numerator = numerator*(1/machines)
        denominator = (np.sqrt(denominator*(1/machines)))**3
        skewnesses.append(numerator/denominator)
    return np.array(skewnesses)

def PRSKE_heuristic(processing_times):
    avg = np.mean(processing_times, axis=1)
    std = np.std(processing_times, axis=1, ddof=1)
    skw = skewness(processing_times)
    order = skw + std + avg
    sequence = [job for _ , job in sorted(zip(order, list(range(processing_times.shape[0]))),reverse=True)]
    return sequence, evaluate_sequence(sequence, processing_times)

'''Artificial Heuristic'''
def artificial_heuristic(processing_times):
    jobs, machines = processing_times.shape
    r = 1
    best_cost = np.inf
    best_seq = []
    while r != machines :
        wi = np.zeros((jobs, machines - r))
        for i in range(jobs):
            for j in range(0, machines - r):
                wi[i, j] = (machines - r) - (j)
       
        am = np.zeros((jobs, 2))    
        am[:, 0] = np.sum(wi[:, :machines - r] * processing_times[:, :machines - r], axis=1)
        for i in range(jobs):
            for j in range(0, machines - r):
                am[i, 1] += wi[i, j ] * processing_times[i, machines - j - 1]
        seq = johnson_method(am)
        cost = evaluate_sequence(seq, processing_times)
        if cost < best_cost:
            best_cost = cost
            best_seq = seq
        r += 1
       
    return best_seq, best_cost