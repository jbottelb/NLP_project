#!/usr/bin/env python3

'''
This is guidance code from Systems programming.
We use the reddit API here
'''

import os
import sys
import requests
import json

# Constants

ISGD_URL = 'http://is.gd/create.php'
URL_BASE = 'https://www.reddit.com/'

# Functions

def usage(status=0):
    ''' Display usage information and exit with specified status '''
    print('''Usage: {} [options] URL_OR_SUBREDDIT

    -s          Shorten URLs using (default: False)
    -n LIMIT    Number of articles to display (default: 10)
    -o ORDERBY  Field to sort articles by (default: score)
    -t TITLELEN Truncate title to specified length (default: 60)
    '''.format(os.path.basename(sys.argv[0])))
    sys.exit(status)#!/usr/bin/env python3

import os
import sys
import requests
import json

# Constants

ISGD_URL = 'http://is.gd/create.php'
URL_BASE = 'https://www.reddit.com/'

# Functions

def usage(status=0):
    ''' Display usage information and exit with specified status '''
    print('''Usage: {} [options] URL_OR_SUBREDDIT

    -s          Shorten URLs using (default: False)
    -n LIMIT    Number of articles to display (default: 10)
    -o ORDERBY  Field to sort articles by (default: score)
    -t TITLELEN Truncate title to specified length (default: 60)
    '''.format(os.path.basename(sys.argv[0])))
    sys.exit(status)

def load_reddit_data(url):
    ''' Load reddit data from specified URL into dictionary

    >>> len(load_reddit_data('https://reddit.com/r/nba/.json'))
    27

    >>> load_reddit_data('linux')[0]['data']['subreddit']
    'linux'
    '''

    # look, i am sorry it has to be this way but your doc tests are overcomplicating things
    if url[0] != 'h':
        url = URL_BASE + 'r/' + url + '/' + '.json'

    headers  = {'user-agent': 'reddit-{}'.format(os.environ.get('USER', 'cse-20289-sp20'))}
    response = requests.get(url, headers=headers)
    response = response.json()
    return response['data']['children']

def shorten_url(url):
    ''' Shorten URL using is.gd service

    >>> shorten_url('https://reddit.com/r/aoe2')
    'https://is.gd/dL5bBZ'

    >>> shorten_url('https://cse.nd.edu')
    'https://is.gd/3gwUc8'
    '''
    toParse = requests.get(ISGD_URL, params={'format': 'json', 'url': url})
    return toParse.json()['shorturl']

def print_reddit_data(data, limit=10, orderby='score', titlelen=60, shorten=False):
    ''' Dump reddit data based on specified attributes '''

    # not needed but reducing the information down to what is needed
    articles = []
    for i in range(0, len(data)):
        tp = dict()
        tp['title'] = data[i]['data']['title'][:int(titlelen)]
        tp['score'] = data[i]['data']['score']
        if shorten:
            tp['url'] = shorten_url(data[i]['data']['url'])
        else:
            tp['url'] = data[i]['data']['url']
        articles.append(tp)

        # articles now contains the relivant information about the found posts

    # details about sorting, then sort
    rev = True
    if orderby != 'score':
        rev = False

    articles = sorted(articles, key=lambda k: k[orderby], reverse=rev)

    # display infortmation
    limit = min(int(limit), len(articles))      # cant give more than we have

    row = ''
    for i in range(0, limit):
        row += f'{(str(i+1)):>4}'
        row += '.\t'
        row += articles[i]['title'] + " (Score: "
        row += str(articles[i]['score']) + ")"
        print(row)
        print("\t" + articles[i]['url'])
        if not i == limit -1:
            print("")
        row = ''

def main():
    arguments = sys.argv[1:]
    url       = 'todayilearned'
    limit     = 10
    orderby   = 'score'
    titlelen  = 60
    shorten   = False

    if len(arguments) < 1:
        usage()

    url = arguments.pop(-1)
    if '-' in url and len(url) == 2:    # dont give me unclear expectations and not expect this
        usage()                         # if it were up to me, all bad input would give usage()

    # manage flags
    i = 0
    while i < len(arguments):
        if arguments[i] == '-s':
            shorten = True
        elif arguments[i] == '-n':
            limit = arguments[i + 1]
            i += 1
        elif arguments[i] == '-o':
            orderby = arguments[i + 1]
            i += 1
        elif arguments[i] == '-t':
            titlelen = arguments[i + 1]
            i += 1
        else:
            sys.exit(1)
        i += 1

        '''
    for i in range(0, len(arguments)):
        arg = arguments[i]
        if arg == '-s':
            shorten = True
        elif arg == '-n':
            limit = arguments[i+1]
        elif arg == '-o':
            orderby = arguments[i+1]1
        elif arg == '-t':
            titlelen = arguments[i+1]
        elif '-' not in arg:
            url = arg
            '''

    # the subreddit data has a map ['data'] to the acutal data
    # this then has children (the indiviual posts) which is a list
    # each list is then a dictionary

    data = load_reddit_data(url)

    # now data holds a list of all the posts on that page

    print_reddit_data(data, limit, orderby, titlelen, shorten)

# Main Execution

if __name__ == '__main__':
    main()


def load_reddit_data(url):
    ''' Load reddit data from specified URL into dictionary

    >>> len(load_reddit_data('https://reddit.com/r/nba/.json'))
    27

    >>> load_reddit_data('linux')[0]['data']['subreddit']
    'linux'
    '''

    # look, i am sorry it has to be this way but your doc tests are overcomplicating things
    if url[0] != 'h':
        url = URL_BASE + 'r/' + url + '/' + '.json'

    headers  = {'user-agent': 'reddit-{}'.format(os.environ.get('USER', 'cse-20289-sp20'))}
    response = requests.get(url, headers=headers)
    response = response.json()
    return response['data']['children']

def shorten_url(url):
    ''' Shorten URL using is.gd service

    >>> shorten_url('https://reddit.com/r/aoe2')
    'https://is.gd/dL5bBZ'

    >>> shorten_url('https://cse.nd.edu')
    'https://is.gd/3gwUc8'
    '''
    toParse = requests.get(ISGD_URL, params={'format': 'json', 'url': url})
    return toParse.json()['shorturl']

def print_reddit_data(data, limit=10, orderby='score', titlelen=60, shorten=False):
    ''' Dump reddit data based on specified attributes '''

    # not needed but reducing the information down to what is needed
    articles = []
    for i in range(0, len(data)):
        tp = dict()
        tp['title'] = data[i]['data']['title'][:int(titlelen)]
        tp['score'] = data[i]['data']['score']
        if shorten:
            tp['url'] = shorten_url(data[i]['data']['url'])
        else:
            tp['url'] = data[i]['data']['url']
        articles.append(tp)

        # articles now contains the relivant information about the found posts

    # details about sorting, then sort
    rev = True
    if orderby != 'score':
        rev = False

    articles = sorted(articles, key=lambda k: k[orderby], reverse=rev)

    # display infortmation
    limit = min(int(limit), len(articles))      # cant give more than we have

    row = ''
    for i in range(0, limit):
        row += f'{(str(i+1)):>4}'
        row += '.\t'
        row += articles[i]['title'] + " (Score: "
        row += str(articles[i]['score']) + ")"
        print(row)
        print("\t" + articles[i]['url'])
        if not i == limit -1:
            print("")
        row = ''

def main():
    arguments = sys.argv[1:]
    url       = 'todayilearned'
    limit     = 10
    orderby   = 'score'
    titlelen  = 60
    shorten   = False

    if len(arguments) < 1:
        usage()

    url = arguments.pop(-1)
    if '-' in url and len(url) == 2:    # dont give me unclear expectations and not expect this
        usage()                         # if it were up to me, all bad input would give usage()

    # manage flags
    i = 0
    while i < len(arguments):
        if arguments[i] == '-s':
            shorten = True
        elif arguments[i] == '-n':
            limit = arguments[i + 1]
            i += 1
        elif arguments[i] == '-o':
            orderby = arguments[i + 1]
            i += 1
        elif arguments[i] == '-t':
            titlelen = arguments[i + 1]
            i += 1
        else:
            sys.exit(1)
        i += 1

        '''
    for i in range(0, len(arguments)):
        arg = arguments[i]
        if arg == '-s':
            shorten = True
        elif arg == '-n':
            limit = arguments[i+1]
        elif arg == '-o':
            orderby = arguments[i+1]1
        elif arg == '-t':
            titlelen = arguments[i+1]
        elif '-' not in arg:
            url = arg
            '''

    # the subreddit data has a map ['data'] to the acutal data
    # this then has children (the indiviual posts) which is a list
    # each list is then a dictionary

    data = load_reddit_data(url)

    # now data holds a list of all the posts on that page

    print_reddit_data(data, limit, orderby, titlelen, shorten)

# Main Execution

if __name__ == '__main__':
    main()
