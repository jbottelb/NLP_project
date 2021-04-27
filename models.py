import pandas as pd
import numpy as np
import sys

import sklearn
from sklearn import svm
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

def model_diagnostics(model, pr=True):
    """
    Returns and prints the R-squared, RMSE and the MAE for a trained model
    """
    y_predicted = model.predict(X_test)
    r2 = r2_score(y_test, y_predicted)
    mse = mean_squared_error(y_test, y_predicted)
    mae = mean_absolute_error(y_test, y_predicted)
    if pr:
        print(f"R-Sq: {r2:.4}")
        print(f"RMSE: {np.sqrt(mse)}")
        print(f"MAE: {mae}")

    return [r2,np.sqrt(mse),mae]

class ML_Models:
    def __init__(self, train, test):
        self.train_data = self.create_df(train)
        self.test_data  = self.create_df(test)



    def svm(self):
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


    def linear(self):
        X_train = self.train_data[['Parent_Score', "Comment"]]
        Y_train = self.train_data[['Parent_Score', "Comment"]]
        X_test = self.train_data[['Score']]
        Y_test = self.train_data[['Score']]

        print(X_train)

        linear = linear_model.LinearRegression()
        linear.fit(X_train, Y_train)

    def create_df(self, file):
        test_sentences = []
        test_scores    = []
        test_parent_scores = []

        running = ''
        with open(file) as tf:
            for line in tf:
                if "<SPLIT>" not in line:
                    running += line
                    continue
                line = running + line

                line = line.strip().split("<SPLIT>")

                test_parent_scores.append(int(line[0].split(" ")[0]))
                comment = " ".join(line[0].split(" ")[1:])

                test_sentences.append(comment)
                test_scores.append(int(line[1]))
                running = ''

        data = { \
            'Parent_Score' : test_parent_scores, \
            'Comment' : test_sentences,          \
            'Score' : test_scores
        }
        data = pd.DataFrame(data, columns = ['Parent_Score', \
                                        'Comment', 'Score'])

        return data

if __name__ == "__main__":
    _, train, dev, test = sys.argv


    model = ML_Models(train, test)
    model.svm()
