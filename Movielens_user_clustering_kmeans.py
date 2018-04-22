import numpy as np
import time
import pickle
from movielens import *
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

user = []
item = []
rating = []
rating_test = []

n_users = 0
n_items = 0

# Load the movie lens dataset into arrays
def load_data():
    global user
    global item
    global rating
    global rating_test
    
    d = Dataset()
    d.load_users("data/u10.user", user)
    d.load_items("data/u.item", item)
    d.load_ratings("data/u10.base", rating)
    d.load_ratings("data/u10.test", rating_test)

def create_rating_matrix():
    global n_users
    global n_items

    n_users = len(user)
    n_items = len(item)
    user_rating = np.zeros((n_users, n_items))
    for r in rating:
        user_rating[r.user_id-1][r.item_id-1] = r.rating

    return user_rating

# Finds the average rating for each user and stores it in the user's object
def find_avg_rating_per_user(user_rating):
    for i in range(n_users):
        rated = np.nonzero(user_rating[i]) #np.nonzero returns indices of the elements that are non-zero.
        n = len(rated[0])
        if n != 0:
            user[i].avg_r = np.mean(user_rating[i][rated])
        else:
            user[i].avg_r = 0

def cluster_users(user_rating):

    cluster = KMeans(n_clusters=6)
    
    cluster.fit_predict(user_rating)

    return cluster

def guess(user_id, item_id, cluster, user_rating):
    
    cluster_number = cluster.labels_[user_id]

    indices = [i for i, x in enumerate(cluster.labels_) if x == cluster_number]

    y = []
   
    for user in indices:
        x = user_rating[user][item_id]
        y.append(x)

    y = list(filter((0.0).__ne__, y))

    if len(y) == 0:
        return 0.0
    else:
        max_r = max(y,key=y.count)

    return max_r

def guess_weighted(user_id, item_id, cluster, user_rating):
    
    distance = {}
    
    ratings = [1.0,2.0,3.0,4.0,5.0]

    cluster_number = cluster.labels_[user_id]

    indices = [i for i, x in enumerate(cluster.labels_) if x == cluster_number]
    scores = [0,0,0,0,0]

    for i in indices:
        if not 'i' in distance :
            sum_d = sum((user_rating[user_id][j] - user_rating[i][j]) for j in range(0,n_items))
        #     sum_f = sum_d ** 0.5
            lamb = 1/ (sum_d ** 2)
            distance[i] = lamb
  
    for j in ratings:
            for i in indices:
                if user_rating[i][item_id] == j:
                    scores[int(j-1)] += distance[i]

    max_s = max(scores)

    return float(scores.index(max_s) + 1)

def predict_user_rating(user_rating, cluster):

    user_rating_copy = np.copy(user_rating)

    for i in range(0, n_users):
        for j in range(0, n_items):
            if user_rating_copy[i][j] == 0:
                time.sleep(0.00005)
                user_rating_copy[i][j] = guess(i, j, cluster, user_rating)

    pickle.dump(user_rating_copy, open("user_rating_movie_user_kmeans.pkl", "wb"))

    return user_rating_copy

def create_test_matrix():
   
    test = np.zeros((n_users, n_items))

    for r in rating_test:
        test[r.user_id - 1][r.item_id - 1] = r.rating

    return test

def calculate_error(test, predicted_rating, cluster):
    # Predict ratings for u.test and find the mean squared error
    y_true = []
    y_pred = []
    
    f = open('test_movie_user_kmeans.txt', 'w')
    
    for i in range(0, n_users):
        for j in range(0, n_items):
            if test[i][j] > 0:
                y_true.append(test[i][j])
                y_pred.append(predicted_rating[i][j])
    f.close()

    print ("Mean Squared Error: %f" % mean_squared_error(y_true, y_pred))

def test_model(predicted_rating, cluster):
    test_matrix = create_test_matrix()
    calculate_error(test_matrix, predicted_rating, cluster)

def main():
    load_data()
    user_rating = []
    user_rating = create_rating_matrix()
    find_avg_rating_per_user(user_rating)
    cluster = cluster_users(user_rating)
    predicted_rating = predict_user_rating(user_rating, cluster)
    test_model(predicted_rating, cluster)

if __name__ == '__main__':
    main()
