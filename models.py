import pandas as pd
import numpy as np
import sys

import sklearn
from sklearn import svm
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score

class Models:
    def __init__(self, train, test):
        self.train_data = self.create_df(train)
        self.test_data  = self.create_df(test)
        self.train_prediction = None
        self.test_predicition = None

    def combined_model(self):
        '''
        This takes the Forrest Classifier, and makes it learn with the svm values
        '''

        if not self.train_prediction.any() or not self.test_predicition.any():
            print("Run an svm regrerssor model first!")
            return

        print("Running Combined Model")

        self.train_data['Predicted'] = self.train_prediction
        self.test_data['Predicted'] = self.test_predicition

        x_train = self.train_data[['Parent_Score', 'Time', 'Saved', 'Predicted']]
        x_test = self.test_data[['Parent_Score', 'Time', 'Saved', 'Predicted']]
        y_train = self.train_data['Score']
        y_test = self.test_data['Score']

        rf = RandomForestRegressor(n_jobs=-1, n_estimators=70, min_samples_leaf=10, random_state = 10)
        rf.fit(x_train, y_train)

        y_pred = rf.predict(x_test)

        self.model_diagnostics(y_test, y_pred)

        return rf

    def LR_classifier(self):
        '''
        This takes attributes that are not words and learns the upvotes

        The goal is to improve this with the text
        '''


        x_train = self.train_data[['Parent_Score', 'Time']]
        x_test = self.test_data[['Parent_Score', 'Time']]
        y_train = self.train_data['Score']
        y_test = self.test_data['Score']

        linear = LinearRegression()

        LR = linear.fit(x_train, y_train)
        y_pred = LR.predict(x_test)

        print(y_pred)

        self.model_diagnostics(y_test, y_pred)

    def RandomForesetClassifier(self):
        '''
        This takes attributes that are not words and learns the upvotes

        The goal is to improve this with the text
        '''

        x_train = self.train_data[['Parent_Score', 'Time', 'Saved']]
        x_test = self.test_data[['Parent_Score', 'Time', 'Saved']]
        y_train = self.train_data['Score']
        y_test = self.test_data['Score']

        rf = RandomForestRegressor(n_jobs=-1, n_estimators=70, min_samples_leaf=10, random_state = 10)
        rf.fit(x_train, y_train)

        y_pred = rf.predict(x_test)
        self.model_diagnostics(y_test, y_pred)
        return rf

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

    def baseline_mode(self):
        '''
        this baseline uses mode
        '''
        pass

    def baseline_avg(self):
        '''
        this baseline uses average
        '''
        pass

    def create_df(self, file):
        sentences = []
        scores    = []
        parent_scores = []
        times = []
        saved = []

        running = ''
        with open(file) as tf:
            for line in tf:
                if "<SPLIT>" not in line:
                    running += line
                    continue
                line = running + line

                line, score = line.strip().split("<SPLIT>")

                values, line = line.strip().split("<COMMENT>")

                parent_score, time, sv = values.strip().split(" ")

                sentences.append(line)
                scores.append(int(score))
                parent_scores.append(int(parent_score))
                times.append(float(time))
                saved.append(int(sv))

                running = ''

        data = { \
            'Parent_Score' : parent_scores, \
            'Time'      : times,             \
            'Comment' : sentences,          \
            'Saved'   : saved, \
            'Score' : scores
        }
        data = pd.DataFrame(data, columns = ['Parent_Score', \
                                        'Comment', 'Time', 'Saved', 'Score'])

        return data

    def model_diagnostics(self, y_test, y_predicted):
        """
        Returns and prints the R-squared, RMSE and the MAE for a trained model
        """

        r2 = r2_score(y_test, y_predicted)
        mse = mean_squared_error(y_test, y_predicted)
        mae = mean_absolute_error(y_test, y_predicted)

        print(f"R-Sq: {r2:.4}")
        print(f"RMSE: {np.sqrt(mse)}")
        print(f"MAE: {mae}")

if __name__ == "__main__":
    _, train, test = sys.argv

    model = Models(train, test)
    model.svm_regressor()
    cbm = model.combined_model()
