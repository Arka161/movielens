#Movie Recommender System 
#Algorithms used: Content & Collaborative Filtering
import pandas as pd
import numpy as np
from collections import OrderedDict
#DataSet used for programming: ml-20m-GroupLens
#20 million ratings
#27,000 movies
#138,000 users
#URL: http://grouplens.org/datasets/movielens/20m/


variable=True
while variable:
    #Menu based to give user algorithm options
    print ("""WELCOME TO MY MOVIE RECOMMENDER SYSTEM. YOU HAVE TWO ALGORITHMS TO CHOOSE FROM
    1. COLLABORATIVE FILTERING
    2. CONTENT FILTERING
    (ALGORITHMS ARE EXPLAINED IN THE README FILE)
    """)
variable=raw_input("What would you like to do? ")
if variable=="1":
#collaborative filtering section
ratings_df = pd.read_table('ml-20m/ratings.csv', header=None, sep=',', names=['user_id', 'movie_id', 'rating', 'timestamp'],engine='python')
movies_df = pd.read_table('ml-20m/movies.csv', header=None, sep=',', names=['movie_id', 'movie_title', 'movie_genre'],engine='python')


#Seperating the given generes in the movies file
movies_df = pd.concat([movies_df, movies_df.movie_genre.str.get_dummies(sep='|')], axis=1)


#Individual Consideration
movie_categories = movies_df.columns[3:]
movies_df.loc[0]

#Timestamp is not needed in the code
del ratings_df['timestamp']

#Rename movie_id to movie_title
ratings_df = pd.merge(ratings_df, movies_df, on='movie_id')[['user_id', 'movie_title', 'movie_id','rating']]


ratings_mtx_df = ratings_df.pivot_table(values='rating', index='user_id', columns='movie_title')
ratings_mtx_df.fillna(0, inplace=True)

movie_index = ratings_mtx_df.columns
#We have userid in rows and movie title in the coloumns

#Calculate the Pearson Product Moment Correlation Coefficient(PMCC)
#Reference: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
matrix = np.corrcoef(ratings_mtx_df.T)
matrix.shape
#We invert(transpose) the matrix to avoid getting similarity between users.we need movies

favoured_movie_title = 'Beauty and the Beast (1991)'

favoured_movie_index = list(movie_index).index(favoured_movie_title)

P = matrix[favoured_movie_index]

#Return the movie which are similar to Beauty and the Beast
list(movie_index[(P>0.5) & (P<1.0)])

def movie_similarity(movie_title):
    movie_idx = list(movie_index).index(movie_title)
    return matrix[movie_idx]

def movie_recommendations(user_movies):

  movie_similarities = np.zeros(matrix.shape[0])
    for movie_id in user_movies:
        movie_similarities = movie_similarities + movie_similarity(movie_id)
    similarities_df = pd.DataFrame({
        'movie_title': movie_index,
        'sum_similarity': movie_similarities
        })
    similarities_df = similarities_df[~(similarities_df.movie_title.isin(user_movies))]
    similarities_df = similarities_df.sort_values(by=['sum_similarity'], ascending=False)
    return similarities_df



#Sample user selection
sample_user = 28 #User_Id
ratings_df[ratings_df.user_id==sample_user].sort_values(by=['rating'], ascending=False)


sample_user_movies = ratings_df[ratings_df.user_id==sample_user].movie_title.tolist()
recommendations = movie_recommendations(sample_user_movies)

#List movie Recommendations- 20 items
recommendations.movie_title.head(20)
print("Thank you for trying my recommender!")


elif variable=="2":
#Contenet Filtering
movies_df = pd.read_table('ml-20m/movies.csv', header=None, sep=',', names=['movie_id', 'movie_title', 'movie_genre'],engine='python')


#Seperating the given generes in the movies file
movies_df = pd.concat([movies_df, movies_df.movie_genre.str.get_dummies(sep='|')], axis=1)
movies_df.head()


#Individual Consideration
movie_categories = movies_df.columns[3:]
movies_df.loc[0]

#User inputs his/her movie preference

user_preferences = OrderedDict(zip(movie_categories, []))

user_preferences['Action'] =input('Enter Action Preference(1-5)')
user_preferences['Adventure'] = input('Enter Adventure Preference(1-5)')
user_preferences['Animation'] = input('Enter Animation Preference(1-5)')
user_preferences["Children's"] =input('Enter Children Movie Preference(1-5)')
user_preferences["Comedy"] = input('Enter Comedy Preference(1-5)')
user_preferences['Crime'] = input('Enter Crime movie Preference(1-5)')
user_preferences['Documentary'] = input('Enter Documentary Preference(1-5)')
user_preferences['Drama'] = input('Enter Drama Preference(1-5)')
user_preferences['Fantasy'] = input('Enter Fantasy Preference(1-5)')
user_preferences['Film-Noir'] = input('Enter Film-Noir Preference(1-5)')
user_preferences['Horror'] = input('Enter Horror Preference(1-5)')
user_preferences['Musical'] = input('Enter Musical Preference(1-5)')
user_preferences['Mystery'] = input('Enter Mystery Preference(1-5)')
user_preferences['Romance'] = input('Enter Romance Preference(1-5)')
user_preferences['Sci-Fi'] = input('Enter Sci-Fi Preference(1-5)')
user_preferences['War'] = input('Enter War Preference(1-5)')
user_preferences['Thriller'] = input('Enter Thriller Preference(1-5)')
user_preferences['Western'] =input('Enter Western Preference(1-5)')

#Dot product manual computation
#we can also use numpy dot product function
def dot_product(vector_1, vector_2):
    return sum([ i*j for i,j in zip(vector_1, vector_2)])

def get_movie_score(movie_features, user_preferences):
    return dot_product(movie_features, user_preferences)

#You can consider any movie in the given dataset. I consider Jumaji with id as 1
jumanji_features=movies_df.loc[1][movie_categories]
jumanji_features
jumanji_predicted_score=dot_product(jumanji_features, user_preferences.values())
jumanji_predicted_score

def get_movie_recommendations(user_preferences, n_recommendations):
    movies_df['score'] = movies_df[movie_categories].apply(get_movie_score,
                                                           args=([user_preferences.values()]), axis=1)
    #New coloumn is added for given calculations
    return movies_df.sort_values(by=['score'], ascending=False)['movie_title'][:n_recommendations]


print(get_movie_recommendations(user_preferences, 20))
#Recomending the user 20 films
print("Thank you for trying my recommender!")


elif variable!="1" | "2":
print("You selected an invalid processing algorithm try again ")
