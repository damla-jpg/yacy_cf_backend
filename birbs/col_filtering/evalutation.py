"""
This file contains functions to evaluate the performance of the col_filtering module.
"""

# Default imports
import os
import json

def get_number_of_peers_in_list():
    """
    This function returns the number of peers in the list.
    """

    # if the whitelist file does not exist, return 0
    if not os.path.exists("resources/whitelist/whitelist.json"):
        return 0
    
    # If the whitelist file is empty, return 0
    if os.stat("resources/whitelist/whitelist.json").st_size == 0:
        return 0

    # Opening whitelist which is a json
    with open("resources/whitelist/whitelist.json", "r", encoding="utf-8") as file:
        whitelist = json.load(file)
    
    return len(whitelist["whitelist"])

def evaluate_sending(start_time: float, message_size: int, end_time: float, delta: float, from_: str, to_: str, message_type: str):
    """
    This function saves the sending message time and size in a csv file.
    """

    elapsed_time = end_time - start_time

    if not os.path.exists("resources/data/evaluation.csv"):
        with open("resources/data/evaluation.csv", "w", encoding="utf-8") as file:
            file.write("start_time,end_time,delta,elapsed_time,message_size,sending,from,to,num_peers,message_type\n")

    with open("resources/data/evaluation.csv", "a", encoding="utf-8") as file:
        file.write(f"{start_time},{end_time},{delta},{elapsed_time},{message_size},{True},{from_},{to_},{get_number_of_peers_in_list()},{message_type}\n")
    


def evaluate_receiving(start_time: float, message_size: int, end_time: float, from_: str, to_: str, message_type: str):
    """
    This function saves the receiving message time and size in a csv file.
    """

    elapsed_time = end_time - start_time

    if not os.path.exists("resources/data/evaluation.csv"):
        with open("resources/data/evaluation.csv", "w", encoding="utf-8") as file:
            file.write("start_time,end_time,delta,elapsed_time,message_size,sending,from,to,num_peers,message_type\n")
    
    with open("resources/data/evaluation.csv", "a", encoding="utf-8") as file:
        file.write(f"{start_time},{end_time}, ,{elapsed_time},{message_size},{False},{from_},{to_},{get_number_of_peers_in_list()},{message_type}\n")

def evaluate_startup(start_time: float, end_time: float, port_num: str, success: bool):
    """
    This function saves the startup time in a csv file.
    """

    elapsed_time = end_time - start_time

    if not os.path.exists("resources/data/startup.csv"):
        with open("resources/data/startup.csv", "w", encoding="utf-8") as file:
            file.write("start_time,end_time,elapsed_time,portnum,success\n")
    
    with open("resources/data/startup.csv", "a", encoding="utf-8") as file:
        file.write(f"{start_time},{end_time},{elapsed_time},{port_num},{success}\n")