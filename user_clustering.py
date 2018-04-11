from movielens import *
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import sys
import time
import pickle
user = []
item = []
rating = []
rating_test = []

# Load the movie lens dataset into arrays
d = Dataset()
d.load_users("data/u.user", user)
d.load_items("data/u.item", item)
d.load_ratings("data/u.base", rating)
d.load_ratings("data/u.test", rating_test)
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

# print (utility)
test = np.zeros((n_users, n_items))
for r in rating_test:
    test[r.user_id - 1][r.item_id - 1] = r.rating

cluster = KMeans(n_clusters=19)
cluster.fit_predict(utility)

# print (cluster.labels_)
utility_copy = np.copy(utility)
def predict(user_id,item_id):
	cluster_number = cluster.labels_[user_id]
	c = []
	for i in range(0,n_users):
		if(cluster.labels_[i] == cluster_number):
		  c.append(i)

	y = []
	for user in c:
	    x = utility[user][item_id]
	    y.append(x)

	max_r = max(y,key=y.count)

	if max_r == 0.0:
		y = list(filter((0.0).__ne__, y))
		if len(y) == 0:
			return 0.0
		else:
			max_r = max(y,key=y.count)

	# print (y)

	return max_r


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
#  6