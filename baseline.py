#!/usr/bin/env python3

'''
This is the baseline solution. The goal is to beat this with better NLP
solutions.

Current method: Random responces OR choose a random respose / comment
'''

import sys, subreddit as s

class Random_Response:
    '''
    A random response model: it will randomly select a respose from the data
    '''
    def __init__(self, data):
        pass


if __name__ == "__main__":
    _, input_file = sys.argv

    comments = []

    sub = s.Subreddit(input_file)

    print(sub.posts[0]["comments"][1])
    print(sub)
