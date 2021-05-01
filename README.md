# NLP_project
Project for Natural Language Processing
This project seeks to be able to guess how well a reddit comment will do based off the text. 

### How to run the code
To run the models on a train and test set, run the command:
`python3 model.py data/memes.train data/memes.test`

And preformance for each model can be viewed. This command will also store the guesses at score in results.txt

To get more data to train on, data_getter.py will need to be altered. Change the subreddit name and number of posts, as well as the name of the output file. 
For a test set, change "top" to "hot". This will give exclusive data sets. 

`python3 data_getter`

## Results
Unfortuneatly, using the text would not get a better RSME or RME score than finding the score that would best minimize the loss to the function each time. 
No other implimentations of loss were found that would use the text to get a better insight into comment quailty. 

I was hoping that one of the models may be able to predict using certain words would reslut in extreme scores either way, but the network unformtunatley just learned how to guess the value that would minmize the loss function best when guessed each time. 
Some of the models preformed poorly overall, such as Random Forrest Regressor, but this did not learn that based on the comment text, rather attributes about the comment such as how long it was posted after the post and how popular the post was. 

These models were reinforced with the svm (support vector machine) and preformed better, but unfortunatley the svm also just learned a constant value to return that would maximise preformace. 

Overall, the results were not too satisfying, but it was an interesting project and I learned quite a bit about fitting models and building a neural network. 


##### Consulted Web resources
https://towardsdatascience.com/predicting-reddit-comment-karma-a8f570b544fc
https://github.com/importdata/Twitter-Sentiment-Analysis/blob/master/Twitter_Sentiment_Analysis_Support_Vector_Classifier.ipynb

