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
avg_rating_per_cluster = []
similarity_matrix = []

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
            user[i].avg_r = 0.

def cluster_movies():

    movie_genre = []

    for movie in item:
        movie_genre.append([movie.unknown, movie.action, movie.adventure, movie.animation, movie.childrens, movie.comedy,
                            movie.crime, movie.documentary, movie.drama, movie.fantasy, movie.film_noir, movie.horror,
                            movie.musical, movie.mystery, movie.romance, movie.sci_fi, movie.thriller, movie.war, movie.western])

    movie_genre = np.array(movie_genre)
    cluster = KMeans(n_clusters=19)

    #Compute cluster centers and predict cluster index for each sample movie.
    cluster.fit_predict(movie_genre)

    return cluster

def update_avg_rating(user_rating, cluster):

    global avg_rating_per_cluster

    # find average rating of each movie cluster based on ratings of ith user for each movie in movie cluster
    for i in range(0, n_users):
        average = np.zeros(19)
        tmp = []
       
        for m in range(0, 19):
            tmp.append([])
        
        for j in range(0, n_items):
            if user_rating[i][j] != 0:
                tmp[cluster.labels_[j] - 1].append(user_rating[i][j])#find the cluster of each movie the user has rated and append the rating
        
        for m in range(0, 19):
            if len(tmp[m]) != 0:
                average[m] = np.mean(tmp[m])
            else:
                average[m] = 0
        
        avg_rating_per_cluster.append(average)

    avg_rating_per_cluster = np.array(avg_rating_per_cluster) #stores average rating of each cluster of all users
    # print (avg_rating_per_cluster)

    for i in range(0, n_users):
        x = avg_rating_per_cluster[i]
        user[i].avg_r = sum(a for a in x if a > 0) / sum(a > 0 for a in x)#final average rating of each user

def pcs(x, y):
    
    global avg_rating_per_cluster

    num = 0
    den1 = 0
    den2 = 0
    
    A = avg_rating_per_cluster[x - 1]
    B = avg_rating_per_cluster[y - 1]
    
    num = sum((a - user[x - 1].avg_r) * (b - user[y - 1].avg_r) for a, b in zip(A, B) if a > 0 and b > 0)
    
    den1 = sum((a - user[x - 1].avg_r) ** 2 for a in A if a > 0)
    den2 = sum((b - user[y - 1].avg_r) ** 2 for b in B if b > 0)
    
    den = (den1 ** 0.5) * (den2 ** 0.5)
    
    if den == 0:
        return 0
    else:
        return num/den

def find_similarity_matrix():
    pcs_matrix = np.zeros((n_users, n_users))#Pearson Correlation Similarity

    for i in range(0, n_users):
        for j in range(0, n_users):
            if i!=j:
                pcs_matrix[i][j] = pcs(i + 1, j + 1)
                time.sleep(0.00005)

    global similarity_matrix

    similarity_matrix = pcs_matrix

# Guesses ratings user might give to item
# We will consider the top_n similar users to do this.
def norm():

    global avg_rating_per_cluster

    normalize = np.zeros((n_users, 19))
    for i in range(0, n_users):
        for j in range(0, 19):
            if avg_rating_per_cluster[i][j] != 0:
                normalize[i][j] = avg_rating_per_cluster[i][j] - user[i].avg_r
            else:
                normalize[i][j] = float('Inf')
    return normalize

def guess(user_id, i_id, top_n):
    
    global similarity_matrix

    similarity = []
    for i in range(0, n_users):
        if i+1 != user_id:
            similarity.append(similarity_matrix[user_id-1][i]) #append the row of user from similarity matrix
    temp = norm()
    temp = np.delete(temp, user_id-1, 0)
    top = [x for (y,x) in sorted(zip(similarity,temp), key=lambda pair: pair[0], reverse=True)] 
    s = 0
    c = 0
    for i in range(0, top_n):
        if top[i][i_id-1] != float('Inf'):
            s += top[i][i_id-1]
            c += 1
    g = user[user_id-1].avg_r if c == 0 else s/float(c) + user[user_id-1].avg_r
    if g < 1.0:
        return 1.0
    elif g > 5.0:
        return 5.0
    else:
        return g

def predict_user_rating():

    global avg_rating_per_cluster

    user_rating_copy = np.copy(avg_rating_per_cluster)
    
    for i in range(0, n_users):
        for j in range(0, 19):
            if user_rating_copy[i][j] == 0:
                time.sleep(0.00005)
                user_rating_copy[i][j] = guess(i+1, j+1, 5)

    pickle.dump( user_rating_copy, open("user_rating_matrix.pkl", "wb"))
    
    return user_rating_copy

def create_test_matrix():
    test = np.zeros((n_users, n_items))

    for r in rating_test:
        test[r.user_id - 1][r.item_id - 1] = r.rating

    return test

def calculate_error(test, predicted_rating,cluster):
    # Predict ratings for u.test and find the mean squared error
    y_true = []
    y_pred = []
    
    f = open('test.txt', 'w')
    
    for i in range(0, n_users):
        for j in range(0, n_items):
            if test[i][j] > 0:
                f.write("%d, %d, %.4f\n" % (i+1, j+1, predicted_rating[i][cluster.labels_[j]-1]))
                y_true.append(test[i][j])
                y_pred.append(predicted_rating[i][cluster.labels_[j]-1])
    f.close()

    print ("Mean Squared Error: %f" % mean_squared_error(y_true, y_pred))

def test_model(predicted_rating,cluster):
    test_matrix = create_test_matrix()
    calculate_error(test_matrix, predicted_rating, cluster)

def main():
    load_data()
    user_rating = []
    user_rating = create_rating_matrix()
    find_avg_rating_per_user(user_rating)
    cluster = cluster_movies()
    update_avg_rating(user_rating, cluster)
    find_similarity_matrix()
    predicted_rating = predict_user_rating()
    test_model(predicted_rating,cluster)

# # %matplotlib inline%
# import matplotlib.pyplot as plt
# import numpy as np
# #x, y = np.meshgrid(x, y)
# plt.pcolormesh(pcs_matrix)
# plt.colorbar() #need a colorbar to show the intensity scale
# plt.show() #boom
# plt.pcolormesh(user_rating_copy)
# plt.colorbar() #need a colorbar to show the intensity scale
# plt.show() #boom

if __name__ == '__main__':
    main()
