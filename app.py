import time

import streamlit as st
import pandas as pd
import  numpy as np
import difflib
import pickle
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('final_data.csv')
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(data['tag']).toarray()
simillarity = cosine_similarity(vectors)
moviesList = data['original_title'].values.tolist()

data2 = pd.read_csv("collaborative.csv")
ratings = pd.DataFrame(data2.groupby('title')['rating'].mean(), columns=['rating'])
ratings['total no of ratings'] = data2.groupby('title')['rating'].count()
movieMatrix = pd.pivot_table(index='user_id', columns='title', values='rating', data=data2, fill_value =0)
def content_based_recommend(movie_name):
    flag = 0
    try:
        movie_name = difflib.get_close_matches(movie_name, moviesList)[0]
    except Exception as e:
        flag = 1
    if flag==0:
        movie_index = data[data['original_title'] == movie_name].index[0]
        cosine_distance = simillarity[movie_index]
        required_movies = sorted(enumerate(cosine_distance), reverse=True, key=lambda x: x[1])[:9]
        movies = []
        for i in required_movies:
            item = []
            item.append(data[data['imdb_id'] == data.iloc[i[0]]['imdb_id']]['original_title'].values[0])
            item.append(data[data['imdb_id'] == data.iloc[i[0]]['imdb_id']]['wiki_link'].values[0])
            item.append(data[data['imdb_id'] == data.iloc[i[0]]['imdb_id']]['poster_path'].values[0])
            movies.append(item)
        return (movies)
    else:
        return("No match")


def collaborative_recommendation(movieName):
    movieName = difflib.get_close_matches(movieName, movieMatrix.columns.values.tolist())
    if len(movieName) != 0:
        movieName = movieName[0]
        related_movies_user_ratings = movieMatrix[movieName]

        related_movies = movieMatrix.corrwith(related_movies_user_ratings)
        related_movies = pd.DataFrame(related_movies, columns=['corelation'])
        related_movies['total no of ratings'] = ratings['total no of ratings']

        return(related_movies[related_movies['total no of ratings'] > 100].sort_values(by='corelation',
                                                                                       ascending=False)[:8].index.tolist())
    else:
        return([ratings.index.tolist()[i] for i in np.random.randint(0, 663,8)])

st. set_page_config(layout="wide")
st.title("HYBRID MOVIE RECOMMENDATION SYSTEM :popcorn::popcorn::popcorn:")
flag1 = 0
with st.form('recommendMovies'):
    movieName = st.text_input("ENTER MOVIE NAME")
    if st.form_submit_button("SUBMIT"):
        flag1 = 1
        movies = content_based_recommend(movieName)
        if movies!="No match":
            st.title("Movies recommended for you")
            index = 0
            while index < len(movies):
                col1, col2, col3 = st.columns(3)

                # Display information for the first movie
                col1.markdown(f"<h5>{movies[index][0]}</h5",True)
                try:
                    col1.image(movies[index][2], width=200)
                except Exception as e:
                    col1.image("https://images.unsplash.com/photo-1509281373149-e957c6296406?q=80&w=1456&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D", width=200)
                col1.markdown(f"<a href = '{movies[index][1]}'>Click</a>", True)

                # Increment index
                index += 1

                # Display information for the second movie
                col2.markdown(f"<h5>{movies[index][0]}</h5", True)
                try:
                    col2.image(movies[index][2], width=200)
                except Exception as e:
                    col2.image("https://images.unsplash.com/photo-1509281373149-e957c6296406?q=80&w=1456&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D", width=200)
                col2.markdown(f"<a href = '{movies[index][1]}'>Click</a>", True)

                # Increment index
                index += 1

                # Display information for the third movie
                col3.markdown(f"<h5>{movies[index][0]}</h5", True)
                try:
                    col3.image(movies[index][2], width=200)
                except Exception as e:
                    col3.image(
                        "https://images.unsplash.com/photo-1509281373149-e957c6296406?q=80&w=1456&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D", width=200)
                col3.markdown(f"<a href = '{movies[index][1]}'>Click</a>", True)

                # Increment index
                index += 1
        else:
            st.error("SEARCHED MOVIE NOT FOUND !!!")

        search = collaborative_recommendation(movieName)
        st.title("Peoples also search for")
        i = 0
        while i<len(search):
            col1, col2, col3, col4 = st.columns(4)
            col1.markdown(f"<h5>{search[i]}</h5", True)
            col1.image("https://plus.unsplash.com/premium_photo-1676049461573-09279845663e?q=80&w=1374&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D", width=100)
            i+=1

            col2.markdown(f"<h5>{search[i]}</h5", True)
            col2.image(
                "https://plus.unsplash.com/premium_photo-1676049461573-09279845663e?q=80&w=1374&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
                width=100)
            i += 1

            col3.markdown(f"<h5>{search[i]}</h5", True)
            col3.image(
                "https://plus.unsplash.com/premium_photo-1676049461573-09279845663e?q=80&w=1374&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
                width=100)
            i += 1

            col4.markdown(f"<h5>{search[i]}</h5", True)
            col4.image(
                "https://plus.unsplash.com/premium_photo-1676049461573-09279845663e?q=80&w=1374&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
                width=100)
            i += 1

st.markdown("<br><br>",True)
st.title("New to you...")
cur = pd.read_csv("curresntUserRate.csv")
rate = cur['rate'][len(cur)-1]
homePage = ratings[ratings['rating']>rate].sort_values(by=["total no of ratings",'rating'], ascending=True)[:4].index.tolist()
i=0
while i < len(homePage):
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"<h5>{homePage[i]}</h5", True)
    col1.image(
        "https://images.unsplash.com/photo-1512070679279-8988d32161be?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTJ8fG1vdmllfGVufDB8fDB8fHww",width=200)
    i += 1

    col2.markdown(f"<h5>{homePage[i]}</h5", True)
    col2.image(
        "https://images.unsplash.com/photo-1512070679279-8988d32161be?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTJ8fG1vdmllfGVufDB8fDB8fHww",width=200)
    i += 1

    col3.markdown(f"<h5>{homePage[i]}</h5", True)
    col3.image(
        "https://images.unsplash.com/photo-1512070679279-8988d32161be?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTJ8fG1vdmllfGVufDB8fDB8fHww",width=200)
    i += 1

    col4.markdown(f"<h5>{homePage[i]}</h5", True)
    col4.image(
        "https://images.unsplash.com/photo-1512070679279-8988d32161be?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTJ8fG1vdmllfGVufDB8fDB8fHww",width=200)
    i += 1

st.markdown("<br><br>",True)
with st.form("rate"):
    rate = st.slider("GIVE RATINGS FOR RECOMMENDATION",0.0,5.0)
    if st.form_submit_button("SUBMIT REVIEW"):
        cur = pd.read_csv("curresntUserRate.csv")
        new_rating = pd.DataFrame({'rate': [rate]})
        cur = pd.concat([cur, new_rating])
        cur.to_csv("curresntUserRate.csv", index=False)
