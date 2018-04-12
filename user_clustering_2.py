from movielens import *
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import sys
import time
import random
import math
import operator
import pickle
user = []
item = []
rating = []
rating_test = []
import skfuzzy as fuzz
import matplotlib.pyplot as plt
# Load the movie lens dataset into arrays
d = Dataset()
d.load_users("data/u10.user", user)
d.load_items("data/u.item", item)
d.load_ratings("data/u10.base", rating)
d.load_ratings("data/u10.test", rating_test)
n_users = len(user)

n_items = len(item)


utility = np.zeros((n_users, n_items))
for r in rating:
    utility[r.user_id-1][r.item_id-1] = r.rating

# Finds the average rating for each user and stores it in the user's object
for i in range(n_users):
    rated = np.nonzero(utility[i]) #np.nonzero returns indices of the elements that are non-zero.
    n = len(rated[0])
    if n != 0:
        user[i].avg_r = np.mean(utility[i][rated])
    else:
        user[i].avg_r = 0.


test = np.zeros((n_users, n_items))
for r in rating_test:
    test[r.user_id - 1][r.item_id - 1] = r.rating

utility_transposed = np.transpose(utility)
cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(utility_transposed, 6, 2, error=0.005, maxiter=300)
labels = list(np.argmax(u_orig, axis=0) + 1)

utility_copy = np.copy(utility)
def predict(user_id,item_id):
    cluster_number = labels[user_id]

    indices = [i for i, x in enumerate(labels) if x == cluster_number]

    y = []
    for user in indices:
        x = utility[user][item_id]
        y.append(x)

    y = list(filter((0.0).__ne__, y))

    if len(y) == 0:
        return 0.0
    else:
        max_r = max(y,key=y.count)

    return max_r

distance = {}
ratings = [1.0,2.0,3.0,4.0,5.0]
def predict_weighted(user_id,item_id):
    scores = [0,0,0,0,0]

    for i in indices:
        if not distance[i]:
            sum_d = sum((utility[user_id][j] - utility[i][j]) for j in range(0,n_items))
        #     sum_f = sum_d ** 0.5
            lamb = 1/ (sum_d ** 2)
            distance[i] = lamb
    for j in ratings:
            for i in indices:
                if utility[i][item_id] == j:
                    scores[int(j-1)]+=distance[i]
    max_s = max(scores)

    return float(scores.index(max_s) + 1)

for i in range(0,n_users):
  for j in range(0,n_items):
      if utility_copy[i][j] == 0:
        sys.stdout.write("\rGuessing [User:Rating] = [%d:%d]" % (i, j))
        sys.stdout.flush()
        time.sleep(0.00005)
        utility_copy[i][j] = predict(i,j)
# print ("\rGuessing [User:Rating] = [%d:%d]" % (i, j))
pickle.dump( utility_copy, open("utility_matrix.pkl", "wb"))

# print (utility_copy)
y_true = []
y_pred = []
f = open('test.txt', 'w')
for i in range(0, n_users):
    for j in range(0, n_items):
        if test[i][j] > 0:
            y_true.append(test[i][j])
            y_pred.append(utility_copy[i][j])
f.close()
# print (y_true)
# print (y_pred)

print ("Mean Squared Error: %f" % mean_squared_error(y_true, y_pred))