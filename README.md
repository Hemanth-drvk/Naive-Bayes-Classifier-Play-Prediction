# Naïve Bayes Classifier to Determine if a Person is Likely to Play Tennis
This is a Python implementation of a Naïve Bayes classifier to predict whether a person is likely to play tennis based on certain weather conditions. The classifier uses two data sets: one with integer labels and another with labels represented as strings surrounded by quotes.

### Dependencies
- Python 3.x
- NumPy
- Collections

### How to use 
To use this classifier, you need to specify the training data, outcome (labels), and the new sample (data point for prediction) in the main function. In both implementations, you can modify the training, outcome, and new_sample variables to fit your use case.

The first implementation uses integer labels to represent the data, while the second implementation uses strings surrounded by quotes. Ensure that the new sample is of the same format as the training data in use.

### Methodology
The classifier works by calculating the probabilities of the outcome variable given each possible value of the predictor variables. It then returns the class with the highest probability.

### Function Description 
The implementation has a function naive_bayes() that takes three arguments:

- training - A NumPy array that contains the training data.
- outcome - A NumPy array that contains the labels or outcomes of the training data.
- new_sample - A NumPy array that contains the data point to predict.
The occurrences() function calculates the probabilities of each value in a given list, while the defaultdict() function creates a dictionary of lists.

### Output 
The output of the naive_bayes() function is a dictionary containing the probabilities for each outcome class. In the first implementation, the output is rounded to 4 decimal places.
