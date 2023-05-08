import time
from colorama import Fore, Style
from tqdm import tqdm
from rich.console import Console
from rich import print
from rich.text import Text
from rich.emoji import Emoji
from branch_and_bound import *
from heuristics import *
from local_search import *
from utils import *

def initialization(data):
    print('Possible initializations.')
    print('1- Random initialization')
    print('2- With NEH heuristic')
    print('3- With Johnson heuristic')
    print('4- With Ham heuristic')
    print('5- With Palmer heuristic')
    print('6- With CDS heuristic')
    print('7- With Gupta heuristic')
    print('8- With PRSKE heuristic')
    print('9- With Artificial heuristic')
    choice = int(input("Enter the number of the chosen initialization: "))
    while True:
        if choice == 1:
            sequence = generate_seq(data.shape[0])
            break
        elif choice == 2:
            sequence, _ = neh_algorithm(data)
            break
        elif choice == 3:
            sequence = johnson_method(data)
            if len(sequence) != 0:
                break
            else:
                choice = int(input("Enter the number of the chosen initialization: "))
        elif choice == 4:
            sequence, _ = ham_heuristic(data)
            break
        elif choice == 5:
            sequence = palmer_heuristic(data)
            break
        elif choice == 6:
            sequence, _ = CDS_heuristic(data)
            break
        elif choice == 7:
            sequence = gupta_heuristic(data)
            break
        elif choice == 8:
            sequence, _ = PRSKE_heuristic(data)
            break
        elif choice == 9:
            sequence, _ = artificial_heuristic(data)
            break
        else:
            print("Invalid choice, please enter a valid choice.")
            choice = int(input("Enter the number of the chosen initialization: "))
    return sequence

def choose_heuristic(data):
    print('Possible heuristics.')
    print('1- NEH heuristic')
    print('2- Johnson heuristic')
    print('3- Ham heuristic')
    print('4- Palmer heuristic')
    print('5- CDS heuristic')
    print('6- Gupta heuristic')
    print('7- PRSKE heuristic')
    print('8- Artificial heuristic')
    choice = int(input("Enter the number of the chosen heuristic: "))
    if choice == 1:
        start_time = time.time()
        sequence, cmax = neh_algorithm(data)
        elapsed_time = time.time() - start_time
    elif choice == 2:
        start_time = time.time()
        sequence = johnson_method(data)
        elapsed_time = time.time() - start_time
        if len(sequence) == 0:
            return None
        else:
            cmax = evaluate_sequence(sequence, data)
    elif choice == 3:
        start_time = time.time()
        sequence, cmax = ham_heuristic(data)
        elapsed_time = time.time() - start_time
    elif choice == 4:
        start_time = time.time()
        sequence = palmer_heuristic(data)
        cmax = evaluate_sequence(sequence, data)
        elapsed_time = time.time() - start_time
    elif choice == 5:
        start_time = time.time()
        sequence, cmax = CDS_heuristic(data)
        elapsed_time = time.time() - start_time
    elif choice == 6:
        start_time = time.time()
        sequence = gupta_heuristic(data)
        cmax = evaluate_sequence(sequence, data)
        elapsed_time = time.time() - start_time
    elif choice == 7:
        start_time = time.time()
        sequence, cmax = PRSKE_heuristic(data)
        elapsed_time = time.time() - start_time
    elif choice == 8:
        start_time = time.time()
        sequence, cmax = artificial_heuristic(data)
        elapsed_time = time.time() - start_time
    else:
        print('Invalid choice.')
        return None
    print('\n**Results**')
    print(f'Generated sequence is {sequence} with a makespan of {cmax}.')
    print(f'Elapsed time of {elapsed_time} seconds.\n')   
        
menu_items = [
    "1. Branch and Bound",
    "2. Heuristics",
    "3. Metaheuristics",
    "4. Exit"
]

menu_color = Fore.LIGHTBLUE_EX
menu_highlight_color = Fore.WHITE + Style.BRIGHT
input_color = Fore.YELLOW
input_prompt = "> "

def display_welcome():
    print(Fore.YELLOW + r"""

 ▄█     █▄     ▄████████  ▄█        ▄████████  ▄██████▄    ▄▄▄▄███▄▄▄▄      ▄████████ 
███     ███   ███    ███ ███       ███    ███ ███    ███ ▄██▀▀▀███▀▀▀██▄   ███    ███ 
███     ███   ███    █▀  ███       ███    █▀  ███    ███ ███   ███   ███   ███    █▀  
███     ███  ▄███▄▄▄     ███       ███        ███    ███ ███   ███   ███  ▄███▄▄▄     
███     ███ ▀▀███▀▀▀     ███       ███        ███    ███ ███   ███   ███ ▀▀███▀▀▀     
███     ███   ███    █▄  ███       ███    █▄  ███    ███ ███   ███   ███   ███    █▄  
███ ▄█▄ ███   ███    ███ ███▌    ▄ ███    ███ ███    ███ ███   ███   ███   ███    ███ 
 ▀███▀███▀    ██████████ █████▄▄██ ████████▀   ▀██████▀   ▀█   ███   █▀    ██████████ 
                         ▀                                                         """)
def display_TEAM():
    print(Fore.RED + r"""  _____                 _     _           _     
  / ____|               | |   (_)         (_)    
 | (___  _   _ _ __ ___ | |__  _  ___  ___ _ ___ 
  \___ \| | | | '_ ` _ \| '_ \| |/ _ \/ __| / __|
  ____) | |_| | | | | | | |_) | | (_) \__ \ \__ \
 |_____/ \__, |_| |_| |_|_.__/|_|\___/|___/_|___/
          __/ |                                  
         |___/                                   
""")
def display_DEMO():
    print(Fore.CYAN + r""" 

██████  ███████ ███    ███  ██████      ████████  ██████  ██  █████  
██   ██ ██      ████  ████ ██    ██        ██    ██    ██ ██ ██   ██ 
██   ██ █████   ██ ████ ██ ██    ██        ██    ██    ██ ██ ███████ 
██   ██ ██      ██  ██  ██ ██    ██        ██    ██    ██ ██ ██   ██ 
██████  ███████ ██      ██  ██████         ██     ██████  ██ ██   ██ 

 """)   
def display_goodbye():
    print(Fore.BLUE + r"""

 ██████╗  ██████╗  ██████╗ ██████╗     ██████╗ ██╗   ██╗███████╗
██╔════╝ ██╔═══██╗██╔═══██╗██╔══██╗    ██╔══██╗╚██╗ ██╔╝██╔════╝
██║  ███╗██║   ██║██║   ██║██║  ██║    ██████╔╝ ╚████╔╝ █████╗  
██║   ██║██║   ██║██║   ██║██║  ██║    ██╔══██╗  ╚██╔╝  ██╔══╝  
╚██████╔╝╚██████╔╝╚██████╔╝██████╔╝    ██████╔╝   ██║   ███████╗
 ╚═════╝  ╚═════╝  ╚═════╝ ╚═════╝     ╚═════╝    ╚═╝   ╚══════╝
""")
def display_menu():
    print(menu_color + "MENU")
    print("-" * 20)
    for item in menu_items:
        print(menu_color + item)
    print()
def get_user_choice():
    while True:
        try:
            choice = int(input(input_color + input_prompt))
            if choice < 1 or choice > len(menu_items):
                raise ValueError
            return choice
        except ValueError:
            print("Invalid choice. Please try again.")
def handle_choice(choice):
    if choice == 1:
        num_jobs = int(input("Enter the number of jobs: "))
        num_machines = int(input("Enter the number of machines: "))
        data = generate_data(num_jobs, num_machines)
        sequence = initialization(data)
        cmax = evaluate_sequence(sequence, data)
        print(f'Generated sequence is {sequence} with a makespan of {cmax}.')
        print('\n**Results**')
        start_time = time.time()
        best_solution, best_cost, i = branch_and_bound(data, sequence, cmax)
        elapsed_time = time.time() - start_time
        print(f'Best sequence is {best_solution} with a makespan of {best_cost}.')
        print(f'No. Nodes visited is {i}.')
        print(f'Elapsed time of {elapsed_time} seconds.\n')
    elif choice == 2:
        num_jobs = int(input("Enter the number of jobs: "))
        num_machines = int(input("Enter the number of machines: "))
        data = generate_data(num_jobs, num_machines)
        choose_heuristic(data)
    elif choice == 3:
        print("You chose Option 3.")
    elif choice == 4:
        print("Exiting program...")
        time.sleep(1)
        display_goodbye()
        exit()
def main():
    display_welcome()
    display_TEAM()
    display_DEMO()
    time.sleep(1)
    print("-----------------------------------")
    print("Hang tight! Good things take time!",":heart_eyes:")

    with tqdm(total=50) as pbar:
        for i in range(10):
            pbar.update(10)
            time.sleep(0.5)
    while True:
        display_menu()
        choice = get_user_choice()
        handle_choice(choice)

if __name__ == "__main__":
    main()

