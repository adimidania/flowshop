import time
from colorama import Fore, Style
from tqdm import tqdm
from rich.console import Console
from rich import print
from rich.text import Text
from rich.emoji import Emoji

menu_items = [
    "1. Option 1",
    "2. Option 2",
    "3. Option 3",
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

    # display menu
    print("Your complete guide menu for the FlowShop problem!")
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
        print("You chose Option 1.")
    elif choice == 2:
        print("You chose Option 2.")
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
