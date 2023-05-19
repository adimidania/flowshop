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
input_color = Fore.GREEN
input_prompt = "> "

def display_TEAM():
    print(Fore.LIGHTMAGENTA_EX + r"""
╔═╗┬ ┬┌┬┐┌┐ ┬┌─┐┌─┐┬┌─┐
╚═╗└┬┘│││├┴┐││ │└─┐│└─┐
╚═╝ ┴ ┴ ┴└─┘┴└─┘└─┘┴└─┘  
  ______________________
 |  __________________  |
 | |Rezkellah Rania   | |
 | |Adimi Dania       | |
 | |Irmouli Maissa    | |
 | |Hamzaoui Imane    | |
 | |Benazzoug Houda   | |
 | |Hamitouche Thanina| |
 |______________________|
     _[_______]_      
 ___[___________]___          
|         [_____] []|
|         [_____] []|
|___________________|                                         
""")
def display_DEMO():
    print(Fore.LIGHTMAGENTA_EX + r""" 

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
    print('\n \n \n')
    print(menu_color + "-------------------------------- MENU ----------------------------------")
   
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
        data = processing_times()
        sequence, cmax = initialization(data)
        print('-------------------------------------  RESULTS --------------------------------------------------')
        start_time = time.time()
        best_solution, best_cost, i = branch_and_bound(data, sequence, cmax)
        elapsed_time = time.time() - start_time
        print(Fore.WHITE+f'Best sequence is {best_solution} with a makespan of {best_cost}.')
        print('------------------------------------------------------------------------------------')
        print(Fore.GREEN+f'No. Nodes visited is {i}.')
        print(Fore.WHITE+'-------------------------------------------------------------------------')
        print(Fore.GREEN+f'Elapsed time of {elapsed_time} seconds.\n')
        print('-------------------------------------- End of B&B ------------------------------------')
    # Heuristics
    
    elif choice == 2:
        data = processing_times()
        sequence, cmax, elapsed_time = heuristics(data)
        print('------------------------------------- RESULTS -----------------------------------------------')
        
        print(Fore.BLUE+f'Best sequence is {sequence} with a makespan of {cmax}.')
        print('---------------------------------------------------------------------------------')
        print(Fore.BLUE+f'Elapsed time of {elapsed_time} seconds.\n')
        print('---------------------------------- End of heuristics ------------------------------------------')
    # Metaheuristics
    elif choice == 3:
        data = processing_times()
        print(Fore.YELLOW+'Choose the metaheuristic type.')
        print(Fore.CYAN+'1- Local search based metaheuristics.')
        print(Fore.CYAN+'2- Population based metaheuristics.')
        choice = int(input(Fore.YELLOW+"Enter the number of the chosen type: "))
        sequence = []
        cmax = 0
        elapsed_time = 0
        while True:
            if choice == 1:
                sequence, cmax, elapsed_time = localsearch(data)
                break
            elif choice == 2:
                sequence, cmax, elapsed_time = population(data)
                break
            else:
                print(menu_color=Fore.RED+"Invalid choice, please enter a valid choice.")
                choice = int(input(Fore.WHITE+"Enter the number of the chosen type: "))
        
        print('------------------------------------- RESULTS -----------------------------------------------')
        print(Fore.CYAN+f'Best sequence is {sequence} with a makespan of {cmax}.')
        print('-'*100)
        print(menu_color=Fore.CYAN+f'Elapsed time of {elapsed_time} seconds.\n')
        print('---------------------------------- End of Methaheuristics -----------------------')
    elif choice == 4:
        print('-'*100)
        print(Fore.RED+"Exiting program, take care! ")
        time.sleep(1)
        display_goodbye()
        exit()

def main():
    display_TEAM()
    display_DEMO()
    time.sleep(1)
    print("-"*95)
    print('\n')
    print('\n')
    print('\n')
    print(Fore.LIGHTGREEN_EX+"Hang tight! Good things take time!",":heart_eyes:")
    with tqdm(total=100) as pbar:
        for i in range(10):
            pbar.update(10)
            time.sleep(0.5)
    while True:
        display_menu()
        choice = get_user_choice()
        handle_choice(choice)

if __name__ == "__main__":
    main()

