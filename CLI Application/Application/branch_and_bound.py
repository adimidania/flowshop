import numpy as np
from utils import evaluate_sequence

class Node:
    def __init__(self, jobs, remaining_jobs, parent=None, lower_bound=1e100):
        self.jobs = jobs
        self.remaining_jobs = remaining_jobs
        self.parent = parent
        self.lower_bound = lower_bound
    def __str__(self):
        return f"Node(jobs={self.jobs}, remaining_jobs={self.remaining_jobs}, lower_bound={self.lower_bound})"

'''Branch and Bound'''   
def branch_and_bound(processing_times, initial_solution, initial_cost):
    jobs, machines = processing_times.shape
    # Initialize the nodes list to the `root_node`
    root_node = Node([], set(range(jobs)))
    best_solution = initial_solution.copy()
    best_solution_cost = initial_cost
    nodes = [root_node]
    i = 1
    while nodes:
        node = nodes.pop()
        # Explore neighbours of the node `node`
        for job in node.remaining_jobs:
            child_jobs = node.jobs + [job]
            child_remaining_jobs = node.remaining_jobs - {job}
            child_lower_bound = evaluate_sequence(child_jobs, processing_times)
            # If the child node is a leaf node (i.e., all jobs have been assigned) then calculate its cost
            if not child_remaining_jobs:
                if child_lower_bound < best_solution_cost:
                    best_solution = child_jobs
                    best_solution_cost = child_lower_bound   
                    continue
            # If the child node is not a leaf then calculate its lower bound and compare it with current `best_solution_cost`
            if child_lower_bound < best_solution_cost:
                child_node = Node(child_jobs, child_remaining_jobs, parent=node, lower_bound=child_lower_bound)
                nodes.append(child_node)
        i += 1
    return best_solution, best_solution_cost, i

'''Branch and Bound pure'''
def branch_and_bound_pure(processing_times,initial_solution, initial_cost):
    jobs, machines = processing_times.shape
    # Initialize the nodes list to the `root_node`
    root_node = Node([], set(range(jobs)))
    best_solution = initial_solution.copy()
    best_solution_cost = initial_cost
    nodes = [root_node]
    i = 1
    while nodes:
        node = nodes.pop()
        # Explore neighbours of the node `node`
        for job in node.remaining_jobs:
            child_jobs = node.jobs + [job]
            child_remaining_jobs = node.remaining_jobs - {job}
            # If the child node is a leaf node (i.e., all jobs have been assigned) then calculate its cost
            if not child_remaining_jobs:
                child_lower_bound = evaluate_sequence(child_jobs, processing_times)
                if child_lower_bound < best_solution_cost:
                    best_solution = child_jobs
                    best_solution_cost = child_lower_bound   
                    continue
            else:
                # If the child node is not a leaf then calculate its lower bound and compare it with current `best_solution_cost`
                child_lower_bound = evaluate_sequence(child_jobs, processing_times)
                child_node = Node(child_jobs, child_remaining_jobs, parent=node, lower_bound=child_lower_bound)
                nodes.append(child_node)
        i += 1
    return best_solution, best_solution_cost, i