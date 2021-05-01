import pandas as pd
import numpy as np
import sys
from scipy import stats
import math

from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from part1 import Vocab

# Keep track of the guesses so we export them to a file
GUESSES = {}

# Translate to log values
LOG = False

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

class Models:
    def __init__(self, train, test):
        self.train_data = self.create_df(train)
        self.test_data  = self.create_df(test)
        self.train_prediction = None
        self.test_predicition = None

    def neural(self):
        '''
        an attempt to get a neural network going

        This never ended up working.
        My best attempts rewarded almost always gussing 1 and 2

        I tried a number of loss functions as well
        '''
        x_train = self.train_data['Comment']
        x_test = self.test_data['Comment']
        y_train = self.train_data['Score']
        y_test = self.test_data['Score']

        max = 0
        vocab = Vocab()
        for sentence in x_train:
            if len(sentence.strip().split(" ")) > max:
                max = len(sentence.strip().split(" "))
            for word in sentence.strip().split(" "):
                vocab.add(word)
        for sentence in x_test:
            if len(sentence.strip().split(" ")) > max:
                max = len(sentence.strip().split(" "))

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(max + 1, 64)
                self.fc2 = nn.Linear(64, 64)
                self.fc3 = nn.Linear(64, 64)
                self.fc4 = nn.Linear(64, 1)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = self.fc4(x)

                return x

            def clean_data(self, data):
                cleaned = []
                for sentence in data:
                    l = len(sentence.strip().split())
                    for i in range(max - len(sentence.strip().split(" "))):
                        sentence += " <FILLER>"
                    nums = [ vocab.numberize(w) for w in sentence.strip().split(" ") ]
                    nums.append(l)
                    sentence = torch.tensor(nums,dtype=torch.float32)
                    cleaned.append(sentence)
                return cleaned

        net = Net()
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        # Normalize and clean data
        clean_train_x = net.clean_data(x_train)
        clean_test_x = net.clean_data(x_test)

        # We need to group target scores together for shuffle
        for i, e in enumerate(clean_train_x):
            e = e, y_train[i]
            clean_train_x[i] = e
        for i, e in enumerate(clean_test_x):
            e = e, y_test[i]
            clean_test_x[i] = e

        trainset = clean_train_x
        testset  = clean_test_x

        for epoch in range(10):
            for i, data in enumerate(trainset):
                x, y = data

                net.zero_grad()

                output = net(x)

                loss = abs(output - y) # or abs(output ** 2 - y ** 2)

                loss.backward()
                optimizer.step()

        results = []
        with torch.no_grad():
            for data in testset:
                X, y = data
                results.append(net(X))

        f_results = []
        for r in results:
            f_results.append(float(r))
        del results

        self.model_diagnostics(y_test, f_results)

        GUESSES['Neural Net'] = f_results

        return net

    def combined_model_1(self):
        '''
        This takes the Forrest Classifier, and makes it learn with the svm values
        '''

        if not self.train_prediction.any() or not self.test_predicition.any():
            print("Run an svm regrerssor model first!")
            return

        print("Running Combined Model")

        self.train_data['Predicted'] = self.train_prediction
        self.test_data['Predicted'] = self.test_predicition

        x_train = self.train_data[['Parent_Score', 'Time', 'Saved', 'Predicted']]#, 'Edited', 'Author_Karma', 'Author_Age',]]
        x_test = self.test_data[['Parent_Score', 'Time', 'Saved', 'Predicted']]#, 'Edited', 'Author_Karma', 'Author_Age',]]
        y_train = self.train_data['Score']
        y_test = self.test_data['Score']

        rf = RandomForestRegressor(n_jobs=-1, n_estimators=70, min_samples_leaf=10, random_state = 10)
        rf.fit(x_train, y_train)

        y_pred = rf.predict(x_test)

        self.model_diagnostics(y_test, y_pred)

        GUESSES['Combined Model 1'] = y_pred

        return rf

    def combined_model_2(self):
        '''
        This takes the GradientBoostingRegressor, and makes it learn with the svm values
        '''

        if not self.train_prediction.any() or not self.test_predicition.any():
            print("Run an svm regrerssor model first!")
            return

        print("Running Combined Model")

        self.train_data['Predicted'] = self.train_prediction
        self.test_data['Predicted'] = self.test_predicition

        x_train = self.train_data[['Parent_Score', 'Time', 'Saved', 'Predicted']] #, 'Edited', 'Author_Karma', 'Author_Age',]]
        x_test = self.test_data[['Parent_Score', 'Time', 'Saved', 'Predicted']] #, 'Edited', 'Author_Karma', 'Author_Age',]]
        y_train = self.train_data['Score']
        y_test = self.test_data['Score']

        gbr = GradientBoostingRegressor(n_estimators=70, max_depth=5)
        gbr.fit(x_train, y_train)

        y_pred = gbr.predict(x_test)
        self.model_diagnostics(y_test, y_pred)

        GUESSES['Combined Model 2'] = y_pred

        return gbr

    def LR_classifier(self):
        '''
        This takes attributes that are not words and learns the upvotes

        The goal is to improve this with the text
        '''


        x_train = self.train_data[['Parent_Score', 'Time', 'Saved', 'Edited']] #, 'Author_Karma', 'Author_Age',]]
        x_test = self.test_data[['Parent_Score', 'Time', 'Saved', 'Edited']] # , 'Author_Karma', 'Author_Age',]]
        y_train = self.train_data['Score']
        y_test = self.test_data['Score']


        linear = LinearRegression()

        LR = linear.fit(x_train, y_train)
        y_pred = LR.predict(x_test)

        GUESSES['LR Classifier'] = y_pred

        self.model_diagnostics(y_test, y_pred)

    def RandomForesetClassifier(self):
        '''
        This takes attributes that are not words and learns the upvotes

        The goal is to improve this with the text
        '''

        x_train = self.train_data[['Parent_Score', 'Time', 'Saved', 'Edited']] #, 'Author_Karma', 'Author_Age',]]
        x_test = self.test_data[['Parent_Score', 'Time', 'Saved', 'Edited']] # , 'Author_Karma', 'Author_Age',]]
        y_train = self.train_data['Score']
        y_test = self.test_data['Score']

        rf = RandomForestRegressor(n_jobs=-1, n_estimators=70, min_samples_leaf=10, random_state = 10)
        rf.fit(x_train, y_train)

        y_pred = rf.predict(x_test)
        self.model_diagnostics(y_test, y_pred)

        GUESSES['Random Forest Classifier'] = y_pred

        return rf

    def GradiantBoostingRegression(self):

        x_train = self.train_data[['Parent_Score', 'Time', 'Saved', 'Edited']] #, 'Author_Karma', 'Author_Age',]]
        x_test = self.test_data[['Parent_Score', 'Time', 'Saved', 'Edited']] #, 'Author_Karma', 'Author_Age',]]
        y_train = self.train_data['Score']
        y_test = self.test_data['Score']

        gbr = GradientBoostingRegressor(n_estimators=70, max_depth=5)
        gbr.fit(x_train, y_train)

        y_pred = gbr.predict(x_test)
        self.model_diagnostics(y_test, y_pred)

        GUESSES['Gradiant Boosting Regressor'] = y_pred

        return gbr

    def svm_classifier(self):
        '''
        Classifier, not very good. Basically comes up with 1 every time.
        '''
        x_train = self.train_data['Comment']
        x_test = self.test_data['Comment']
        y_train = self.train_data['Score']
        y_test = self.test_data['Score']

        vectorizer = CountVectorizer(min_df=1, stop_words='english')
        vectorizer.fit(list(x_train) + list(x_test))

        x_train_vec = vectorizer.transform(x_train)
        x_test_vec = vectorizer.transform(x_test)

        s = svm.SVC(kernel = 'linear', probability=True)

        # fit the SVC model based on the given training data
        prob = s.fit(x_train_vec, y_train).predict_proba(x_test_vec)

        # perform classification and prediction on samples in x_test
        y_pred_svm = s.predict(x_test_vec)
        print("Accuracy score for SVC is: ", accuracy_score(y_test, y_pred_svm) * 100, '%')

    def svm_regressor(self):
        x_train = self.train_data['Comment']
        x_test = self.test_data['Comment']
        y_train = self.train_data['Score']
        y_test = self.test_data['Score']

        vectorizer = CountVectorizer(min_df=1, stop_words='english')
        vectorizer.fit(list(x_train) + list(x_test))

        x_train_vec = vectorizer.transform(x_train)
        x_test_vec = vectorizer.transform(x_test)

        regr = svm.SVR()

        # fit the SVC model based on the given training data
        regr.fit(x_train_vec, y_train)

        # perform classification and prediction on samples in x_test
        y_pred_svm = regr.predict(x_test_vec)

        self.model_diagnostics(y_test, y_pred_svm)

        y_pred_train_svm = regr.predict(x_train_vec)
        self.train_prediction = y_pred_train_svm # this is a cheated matrix but fdp
        self.test_predicition = y_pred_svm

        GUESSES['SVM Regressor'] = y_pred_svm

    def baseline_mode(self):
        '''
        this baseline uses mode
        '''

        mode = int(stats.mode(self.train_data['Score'])[0])

        pred = []
        actual = []
        for guess in self.test_data['Score']:
            pred.append(mode)
            actual.append(guess)

        self.model_diagnostics(pred, actual)

        GUESSES['Mode Baseline'] = pred

        return mode

    def baseline_avg(self):
        '''
        this baseline uses average
        '''
        ave = np.average(self.train_data['Score'])

        pred = []
        actual = []
        for guess in self.test_data['Score']:
            pred.append(ave)
            actual.append(guess)

        self.model_diagnostics(pred, actual)

        GUESSES['Average Baseline'] = pred

        return ave

    def create_df(self, file):
        sentences = []
        scores    = []
        parent_scores = []
        times = []
        saved = []
        edited = []
        # author_karma = []
        # author_age = []

        running = ''
        with open(file) as tf:
            for line in tf:
                if "<SPLIT>" not in line:
                    running += line
                    continue
                line = running + line

                line, score = line.strip().split("<SPLIT>")

                values, line = line.strip().split("<COMMENT>")

                parent_score, time, sv, edit = values.strip().split(" ")

                if LOG:
                    if int(score) <= 0:
                        score = int(score)
                    else:
                        score = math.log(int(score))

                sentences.append(line)
                scores.append(float(score))
                parent_scores.append(int(parent_score))
                times.append(float(time))
                saved.append(int(sv))
                edited.append(int(edit))
                # author_karma.append(int(ac))
                # author_age.append(float(a_age))

                running = ''

        data = { \
            'Parent_Score' : parent_scores, \
            'Time'      : times,             \
            'Comment' : sentences,          \
            'Saved'   : saved, \
            'Score' : scores, \
            'Edited' : edited \
            # 'Author_Karma' : author_karma, \
            # 'Author_Age' : author_age
        }
        data = pd.DataFrame(data, columns = ['Parent_Score', \
                                        'Comment', 'Time', 'Saved', 'Edited', \
                                         'Score'])

        return data

    def model_diagnostics(self, y_test, y_predicted):
        """
        Returns and prints the R-squared, RMSE and the MAE for a trained model
        Though simple, I should mention I did not write this
        https://towardsdatascience.com/predicting-reddit-comment-karma-a8f570b544fc
        """

        r2 = r2_score(y_test, y_predicted)
        mse = mean_squared_error(y_predicted, y_test)
        mae = mean_absolute_error(y_predicted, y_test)

        print(f"R-Sq: {r2:.4}")
        print(f"RMSE: {np.sqrt(mse)}")
        print(f"MAE: {mae}")

    def export_guesses(self):
        output = ''
        for model_name, guesses in GUESSES.items():
            output += model_name + '\n'
            for guess in guesses:
                output += str(guess) + " "
            output += '\n'
        f = open("results.txt", "w")
        f.write(output)
        f.close()

if __name__ == "__main__":
    _, train, test = sys.argv


    model = Models(train, test)

    print()
    print("Mode Baseline")
    model.baseline_mode()
    print()
    print("Average Baseline")
    model.baseline_avg()

    print()
    print("SVM regression")
    model.svm_regressor()

    print()
    print("RandomForesetClassifier")
    model.RandomForesetClassifier()

    print()
    print("Reinforced RandomForesetClassifier")
    cbm = model.combined_model_1()

    print()
    print("GradiantBoostingRegression")
    gbr = model.GradiantBoostingRegression()

    print()
    print("Reinforced GradiantBoostingRegression")
    gbr = model.combined_model_2()

    print()
    print("Neural Net")
    ml = model.neural()

    model.export_guesses()
