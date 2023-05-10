from colorama import Fore, Style
from tqdm import tqdm
from rich.console import Console
from rich import print
from rich.text import Text
from rich.emoji import Emoji
from calls import *

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
    # Branch and Bound
    if choice == 1:
        num_jobs = int(input("Enter the number of jobs: "))
        num_machines = int(input("Enter the number of machines: "))
        data = generate_data(num_jobs, num_machines)

        sequence, cmax = initialization()
        print('\n**Results**')
        start_time = time.time()
        best_solution, best_cost, i = branch_and_bound(data, sequence, cmax)
        elapsed_time = time.time() - start_time
        print(f'Best sequence is {best_solution} with a makespan of {best_cost}.')
        print(f'No. Nodes visited is {i}.')
        print(f'Elapsed time of {elapsed_time} seconds.\n')
    # Heuristics
    elif choice == 2:
        num_jobs = int(input("Enter the number of jobs: "))
        num_machines = int(input("Enter the number of machines: "))
        data = generate_data(num_jobs, num_machines)
        sequence, cmax, elapsed_time = heuristics(data)
        print('\n**Results**')
        print(f'Best sequence is {sequence} with a makespan of {cmax}.')
        print(f'Elapsed time of {elapsed_time} seconds.\n')
    # Metaheuristics
    elif choice == 3:
        num_jobs = int(input("Enter the number of jobs: "))
        num_machines = int(input("Enter the number of machines: "))
        data = generate_data(num_jobs, num_machines)
        print('Choose the metaheuristic type.')
        print('1- Local search based metaheuristics.')
        print('2- Population based metaheuristics.')
        choice = int(input("Enter the number of the chosen type: "))
        sequence = []
        cmax = 0
        elapsed_time = 0
        while True:
            if choice == 1:
                sequence, cmax, elapsed_time = localsearch(data)
                break
            elif choice == 2:
                sequence, cmax, elapsed_time = heuristics(data)
                break
            else:
                print("Invalid choice, please enter a valid choice.")
                choice = int(input("Enter the number of the chosen type: "))
        
        print('\n**Results**')
        print(f'Best sequence is {sequence} with a makespan of {cmax}.')
        print(f'Elapsed time of {elapsed_time} seconds.\n')
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

