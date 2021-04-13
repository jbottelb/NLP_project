'''
Post Module:
creates data structure for each post from a text file to this format

[
    {title : "Title text",
    comments :
        [
            [comment, karma],
            [comment, karma]
        ]
    }
    {title : "Title text",
    comments :
        [
            [comment, karma],
            [comment, karma]
        ]
    }
]

'''

import re

class Subreddit:
    '''
    Subreddit class
    '''
    def __init__(self, file):
        self.posts = [] # list of posts, dict objs leading to file

        title = None
        comments = []

        with open(file) as file_reader:
            # this is to get multiline comments
            multi_comment = ''
            for line in file_reader:
                line = line.strip()
                if line[:7] == "<TITLE>":
                    if title:
                        # Save post and continue
                        self.posts.append(  \
                            {"title" : title, "comments" : comments} \
                        )
                        comments = []

                    title = Title(line)
                elif line[:9] == "<COMMENT>":
                    # if we have not reached the end of the comment, keep reading it
                    if "<\COMMENT" not in line:
                        multi_comment += line
                    else:
                        if multi_comment != '':
                            line = multi_comment + line
                        comments.append(Comment(line))

            self.posts.append(  \
                {"title" : title, "comments" : comments} \
            )

    def __str__(self):
        '''
        Just returns the titles of the posts
        '''
        titles = ''
        for post in self.posts:
            titles += str(post["title"]) + '\n'

        return titles

class Comment:
    '''
    Comment class
    '''

    def __init__(self, line):
        self.text, self.karma = "", None
        self._extract(line)

    def __str__(self):
        return self.text

    def __repr__(self):
        pass

    def _extract(self, line):
        """
        Use the Godly power of regex
        """
        self.karma = re.compile(r'=&=-?\d+=&=').findall(line)[0].replace('=&=','')

        text       = re.sub("<COMMMENT>", "", line)
        self.text  = re.sub("<COMMMENT>", "", text)


class Title:
    '''
    Title class
    Could just be a tuple but may want more functions later
    '''
    def __init__(self, line):
        self.text = None
        self._extract(line)

    def __str__(self):
        return self.text

    def __repr__(self):
        pass

    def _extract(self, line):
        self.text = line[7:len(line)-9]
