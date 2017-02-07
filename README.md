# Movie Recommendation Engine based on the MovieLens Dataset
Used the MovieLens 20M DataSet having 27,000 movies. Implemented Content Filtering and Collaborative Filtering in the program. 

# 1.Content Filtering
The user enters his preference in every genre; The program checks the characteristics of movies based on his preference and suggests movies. 

# 2.Collaborative Filtering
The program checks correlation between a movie the user likes and other movies already rated in the data set. Used the ratings file in the process.
Also calculated Pearson Product Moment Correlation Coefficient(PMCC) for improving suggestions. However, Collaborative Filtering has the 
Cold Start issue in computation. 

# How to use the program? 
```
python index.py
```
You'll be asked to select the type of filtering you need. Make sure you have atleast ~2.5GB of free RAM to avoid system crashes. 
For Collaborative Filtering, you will be asked to enter a movie you like. 

```
favoured_movie_title = 'Beauty and the Beast (1991)'
```
You can edit the variable with a movie of your choice to make the system get to recommend movies you enjoy. 



