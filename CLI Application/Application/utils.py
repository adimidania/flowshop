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

''' Generate data '''
def generate_data(n, m):
    rnd_data = np.random.randint(size=(n,m), low=5, high=120)
    return rnd_data

def readfile(filename):
    # Open the file that contains the instances
    path = f'files/{filename}.txt'
    file = open(path, "r")
    # Read the file line by line to retrieve the instances
    data = []
    line = file.readline()
    while line:
        if line != '\n':
            line = line.strip(' ')
            line = line[:-1]
            line = line.split()
            line = [int(num) for num in line]
            data.append(line)
        line = file.readline()
    return np.array(data)

''' Generate random solution of n jobs '''
def generate_seq(n):
    permutation = np.random.permutation(n).tolist()
    return permutation

''' Generate data '''

def processing_times():
    print('1. Generate random data.')
    print('2. Read data from a text file.')
    choice = int(input("Enter your choice: "))
    while True:
        if choice == 1:
            num_jobs = int(input("Enter the number of jobs: "))
            num_machines = int(input("Enter the number of machines: "))
            data = generate_data(num_jobs, num_machines)
            return data
        elif choice == 2:
            print("Follow the instructions below.")
            print("> Go to the [files] folder.")
            print("> Create a text file that contains your processing times matrix.")
            print("> Each row represents a job, each column reprensents a machine")
            print("REMARK: Please make sure to not include any chars. Seperate the times with blanks")
            filename = int(input("Enter the name of the file that you created: "))
            data = readfile(filename)
            return data
        else:
            print('Invalid choice')
            choice = int(input("Enter your choice: ")) 