#!/usr/bin/env python3

# much of this is from http://api.intelligentonlinetools.com/diy/reddit/

# used to get data
import requests, praw, pandas as pd, datetime as dt
import json
import os
from praw.models import MoreComments

from tqdm import tqdm

# oeKzoNS0AZe4wv27ldJAuI_4YeZKUQ

# constants
URL_BASE = 'https://www.reddit.com/'

reddit = praw.Reddit(username='data_getting',     \
        password='data_gettingPassword',          \
        user_agent='comment_NLP',                 \
        client_id='HtfZhak8mPNxNw',               \
        client_secret='oeKzoNS0AZe4wv27ldJAuI_4YeZKUQ')


subreddit = reddit.subreddit('memes')

top_subreddit  = subreddit.hot(limit=3)

output = ''

for submission in tqdm(top_subreddit):
    # output += "<TITLE>" + submission.title + "<\TITLE>\n"
    s = str(submission.score)
    for top_level_comment in submission.comments:
        if isinstance(top_level_comment, MoreComments):
            continue
        output += s + " " + str(top_level_comment.created_utc - submission.created_utc) + " "
        output += str(int(top_level_comment.saved)) + " <COMMENT> "
        output +=  top_level_comment.body + " <POST> " + submission.title
        output += "<SPLIT>" + str(top_level_comment.score) + '\n' # tokenization

f = open("testing.test", "w")
f.write(output)
f.close()
