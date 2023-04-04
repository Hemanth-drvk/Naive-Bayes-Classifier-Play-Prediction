#!/usr/bin/env python
# coding: utf-8

# # Naïve Bayes classifier to determine if a person is likely to Play Tennis

# ### Classifier with labels 

#  - Yes = 1, No = 0 <br>
#  - Sunny = 0, Overcast = 1, Rain = 2<br>
#  - Hot = 0, Mild = 1, Cool = 2 <br>
#  - Normal = 0, High = 1 <br>
#  - Weak = 0, Strong = 1 <br>

# In[1]:


import numpy as np
from collections import Counter, defaultdict
 
def occurrences(list1):
    no_of_examples = len(list1)
    prob = dict(Counter(list1))
    for key in prob.keys():
        prob[key] = prob[key] / float(no_of_examples)
    return prob
 
def naive_bayes(training, outcome, new_sample):
    classes     = np.unique(outcome)
    rows, cols  = np.shape(training)
    likelihoods = {}
    for cls in classes:
        likelihoods[cls] = defaultdict(list)
 
    class_probabilities = occurrences(outcome)
 
    for cls in classes:
        row_indices = np.where(outcome == cls)[0]
        subset      = training[row_indices, :]
        r, c        = np.shape(subset)
        for j in range(0,c):
            likelihoods[cls][j] += list(subset[:,j])
 
    for cls in classes:
        for j in range(0,cols):
             likelihoods[cls][j] = occurrences(likelihoods[cls][j])
 
    results = {}
    for cls in classes:
         class_probability = class_probabilities[cls]
         for i in range(0,len(new_sample)):
             relative_values = likelihoods[cls][i]
             if new_sample[i] in relative_values.keys():
                 class_probability *= relative_values[new_sample[i]]
             else:
                 class_probability *= 0
             results[cls] = class_probability
    print(results)
 
if __name__ == "__main__":
    training   = np.asarray(((0,0,1,0),(0,0,1,1),(1,0,1,0),(2,1,1,0),(2,2,0,0),(2,2,0,1),(1,2,0,1),(0,1,1,0),(0,2,0,0),(2,1,0,0),(0,1,0,1),(1,1,1,1),(1,0,0,0),(2,1,1,1)));
    outcome    = np.asarray((0,0,1,1,1,0,1,0,1,1,1,1,1,0))
    new_sample = np.asarray((0,2,1,1))
    naive_bayes(training, outcome, new_sample)


# ##### Output: 
# #### 0 rounded to 4 decimals is 0.0206
# #### 1 rounded to 4 decimals is 0.0053

# ### Classifier with words surrounded by quotes

# In[2]:


import numpy as np
from collections import Counter, defaultdict
 
def occurrences(list1):
    no_of_examples = len(list1)
    prob = dict(Counter(list1))
    for key in prob.keys():
        prob[key] = prob[key] / float(no_of_examples)
    return prob
 
def naive_bayes(training, outcome, new_sample):
    classes     = np.unique(outcome)
    rows, cols  = np.shape(training)
    likelihoods = {}
    for cls in classes:
        likelihoods[cls] = defaultdict(list)
 
    class_probabilities = occurrences(outcome)
 
    for cls in classes:
        row_indices = np.where(outcome == cls)[0]
        subset      = training[row_indices, :]
        r, c        = np.shape(subset)
        for j in range(0,c):
            likelihoods[cls][j] += list(subset[:,j])
 
    for cls in classes:
        for j in range(0,cols):
             likelihoods[cls][j] = occurrences(likelihoods[cls][j])
 
    results = {}
    for cls in classes:
         class_probability = class_probabilities[cls]
         for i in range(0,len(new_sample)):
             relative_values = likelihoods[cls][i]
             if new_sample[i] in relative_values.keys():
                 class_probability *= relative_values[new_sample[i]]
             else:
                 class_probability *= 0
             results[cls] = class_probability
    print(results)
 
if __name__ == "__main__":
    training   = np.asarray((('Sunny','Hot','High','Weak'),('Sunny','Hot','High','Strong'),('Overcast','Hot','High','Weak'),('Rain','Mild','High','Weak'),('Rain','Cool','Normal','Weak'),('Rain','Cool','Normal','Strong'),('Overcast','Cool','Normal','Strong'),('Sunny','Mild','High','Weak'),('Sunny','Cool','Normal','Weak'),('Rain','Mild','Normal','Weak'),('Sunny','Mild','Normal','Strong'),('Overcast','Mild','High','Strong'),('Overcast','Hot','Normal','Weak'),('Rain','Mild','High','Strong')));
    outcome    = np.asarray(('No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No'))
    new_sample = np.asarray(('Sunny','Cool','High','Strong'))
    naive_bayes(training, outcome, new_sample)


# ##### Output: 
# #### 'No' rounded to 4 decimals is 0.0206
# #### 'Yes' rounded to 4 decimals is 0.0053

# Rinda Digamarthi(157742d)

# In[ ]:




