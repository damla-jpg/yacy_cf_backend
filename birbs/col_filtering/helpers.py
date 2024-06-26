'''
This module contains helper functions for the collaborative filtering algorithm.
'''

# Default imports
import hashlib
import random

# Third-party imports
import numpy as np
from sklearn.linear_model import LinearRegression

b = 4
N = 50
hash_size = 16
hex_map = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'a': 10, 'b': 11, 'c': 12, 'd': 13,
    'e': 14, 'f': 15
}

def cosine_similarity(user_i, user_j):
    '''
    This function calculates the cosine similarity between two users.
    '''

    dot_product = np.dot(user_i, user_j)
    norm_i = np.linalg.norm(user_i)
    norm_j = np.linalg.norm(user_j)

    similarity = dot_product / (norm_i * norm_j)
    return similarity

def hex_distance(id1, id2):
    '''
    This function calculates the distance between two hex strings.
    '''

    for i in range(hash_size):
        if id1[i] != id2[i]:
            return i, abs(hex_map[id1[i]] - hex_map[id2[i]])
    return hash_size, 0

def hex_different_index(id1, id2):
    '''
    This function calculates the index of the first different character between two hex strings.
    '''

    for i in range(hash_size):
        if id1[i] != id2[i]:
            return i
    return -1

def hex_compare(id1, id2, equality=True):
    '''
    This function compares two hex strings.
    '''

    if id1 == id2:
        if equality:
            return True
        return False

    for i in range(hash_size):
        if id1[i] != id2[i]:
            d = hex_map[id1[i]] - hex_map[id2[i]]
            if d > 0:
                return True
            else:
                return False

def distance_metric(point1, point2):
    '''
    This function calculates the distance between two points.
    '''

    point = (point2[0] - point1[0], point2[1] - point1[1])
    return max(abs(point[0]), abs(point[1]))

def distance_compare(origin, point1, point2):
    '''
    This function compares two points based on their distance from the origin.
    Returns True if point1 is farther from the origin than point2.
    '''

    d1 = distance_metric(origin, point1)
    d2 = distance_metric(origin, point2)

    if d1 >= d2:
        return True
    return False


def generate_node_id_from_ip(ip_address : str):
    '''
    This function generates a node ID from an IP address.
    '''

    return hashlib.md5(ip_address.encode()).hexdigest()[:hash_size]

def generate_link_hashes(link: str):
    '''
    This function generates hashes the links for private search.
    '''

    return hashlib.md5(link.encode("utf-8")).hexdigest()


def normalize_ratings_maxmin(ai: dict):
    """
    Function to normalize the ratings between -1 and 1.
    Input: ai: url hash to ratings
    Output: ai: url hash to normalized ratings
    """

    ratings = list(ai.values())

    min_value = min(ai.values())
    max_value = max(ai.values())
    ai_ = {}
    for index, (key, value) in enumerate(ai.items()):
        search_hash = key

        normalized_rating = 2 * (ratings[index] - min_value) / (max_value - min_value) - 1
        
        ai_[search_hash] = normalized_rating

    return ai_

