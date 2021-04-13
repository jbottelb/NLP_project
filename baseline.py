#!/usr/bin/env python3

'''
This is the baseline solution. The goal is to beat this with better NLP
solutions.

Current method: Random responces OR choose a random respose / comment
'''

import sys
import subreddit as s

class MostCommonScore:
    '''
    Returns the most common score/karma in the given subreddit
    '''
    def __init__(self, sub):
        '''
        Reads data and
        '''
        self.scores = {}
        for post in sub.posts:
            for comment in post["comments"]:
                if comment.karma in self.scores:
                    self.scores[comment.karma] += 1
                else:
                    self.scores[comment.karma] = 1

        self.best_guess = self.score()

    def score(self):
        '''
        Takes a line and finds the probable score
        '''
        max_val = None
        max_count = 0
        for key, value in self.scores.items():
            if value > max_count:
                max_count, max_val = value, key

        return max_val

    def best(self):
        '''
        Gives best guess of trained model
        '''
        return self.best_guess


def test_baseline(model, data):
    '''
    Returns the percentage of correctness of the baseline
    '''
    total_guesses = 0
    total_correct = 0

    test_sub = s.Subreddit(data)

    for post in test_sub.posts:
        for comment in post["comments"]:
            if comment.karma == model.best():
                total_correct += 1
            total_guesses += 1

    return total_correct / total_guesses

if __name__ == "__main__":
    _, input_file, test_file = sys.argv

    comments = []

    sub = s.Subreddit(input_file)
    model = MostCommonScore(sub)

    percent_correct = test_baseline(model, test_file)

    print(percent_correct)
