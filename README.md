# CSCI3202-Final-Project

##To Run
python  dataClassifier.py  -c  perceptron  -f

## Problem
Today, image classification is becoming increasingly more relied upon by the public for tasks
such as examining drone footage collected for national defense purposes, for analyzing footage
for self-driving cars, face recognition in Facebook posts, and to give us ever more relevant
Google searches. For the wannabe artist, a tool that can analyze our drawings and correctly
guess what it is we were trying to draw is thrilling, since people can’t tell what it is we were
trying to draw (it gets us). Therefore, this project aims to apply some simple image
classification by deciphering drawings into 10 types of categories (for example, the Eiffle Tower,
Great wall of China, baseball bat, etc.)
## Method
The method used will be one discussed in class for a similar task, classification of numbers.
This project will therefore use a perceptron classifier to implement the classification. To fine
tune, I will be finding and tuning features of the images that might help differentiate one from
the others. An alternate way of implementing this could be using Naive Bayes, which will be
explored as an alternate method. The training data comes from the QuickDraw Sketches
dataset, and test data will be able to be called as a JPG drawing created by the user of the tool.
A smaller portion of this dataset will be saved for use as validation data. Because the training
data includes information such as time stamp, and country code, it could be interesting to
explore how information like this can be leveraged to make more accurate predictions if
included in the test data. The output will be the tool’s guess for what has been drawn.
## Evaluation of Success
The first standard of success will be doing better than a completely randomly generated
solution, in which case it would choose one of the 10 categories at random, with 10% chance of
success. The second standard of success is guessing correctly about 50% of the time, and a
third standard of success is 70%. Further improvements will try to get up to 90% accuracy, and
add more category types that can be deciphered, which will be the final standard of success.
