"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import time

# Custom Libraries
from app_functions import *
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    # Create the sidebar and add the logo to it
    logo = Image.open("resources/imgs/logo.png")
    st.sidebar.image(logo, width=200)
    page_options = ["Recommender System","Solution Overview","Our Team","Analysis and Insights","Model Performance"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('First Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    #define a function for loading css files
    def local_css(file_name):
            with open(file_name) as f:
                st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)
    
    

    if page_selection == "Solution Overview":
        st.title("Solution Overview")

        st.image("resources/imgs/collage.png", use_column_width=True)
        st.write("---")
        st.title(
            """
            Discover new films and TV shows you will love with ease.
            """
        )
        st.write(
            """
            We use state-of-the-art machine learning algorithms to create 
            personalized recommendations based on your viewing history, favorite genres, 
            and other preferences. 
            
            Our recommendation algorithm combines collaborative filtering and 
            content-based filtering methods to recommend movies and TV shows that are most 
            likely to interest you. 
            """
            )
        
        st.write("---")
        l_col,r_col = st.columns(2)
        with l_col:
            st.write(
                """
            **The collaborative filtering** method is based on the idea that users who have similar 
            tastes will enjoy the same movies and TV shows. We use this method to identify users with 
            similar preferences and recommend movies and TV shows that they have liked.
            """
            )
        with r_col:
            st.image("resources/imgs/collab.PNG")

        st.write("---")

        first_col,second_col = st.columns(2)
        with first_col:
            st.image("resources/imgs/content.PNG")
        with second_col:
            st.write(
                """
                The **content-based filtering** method, on the other hand, is based on the idea 
                that users are more likely to enjoy movies and TV shows that are similar to ones 
                they've already watched. We use this method to recommend movies and TV shows that 
                are similar to the ones you've liked based on factors such as plot, cast, and genre.
                """
            )
        
        st.write("---")

        st.write(
            """
            By combining these two methods, we can provide accurate and diverse recommendations to our users. 
            Plus, we are constantly updating and improving our recommendation algorithm to ensure that you 
            have the best possible experience.
            We hope you enjoy using our movie recommendation system and discover new films and TV shows that 
            you'll love!
            """
            )
    if page_selection == "Our Team":    
        # Open the image file
        st.title("Our Team")
        image1 = Image.open("resources/imgs/building.jpg")
        st.image(image1, caption = "DataCOP Headquarters")
        #getting the css file with the typewriter effect
        local_css("text.css")
        #applying the typewriter effect on the text
        st.markdown('<h1 class="typewriter">Hi👋, We are DataCOP</h1>', unsafe_allow_html=True)
        st.write('---')
        st.header('A Market Research team focused on creating real-world solutions')
        st.write(""" \n We are passionate about the use of data to help
			companies to make informed decisions about  marketing strategies""")
        
        st.write('---')
        left_column, right_column = st.columns(2)
        with left_column:
            st.subheader("What do we do?")
            st.write(
					"""
					We create viable market solutions to clients to increase their reach while reducing 
					marketing costs by:
					 - Leveraging available data to analyse the market trends
					 - Creating machine learning models to analyse the data
					 - Using classification to accurately predict a user's opinion on a product
					 - Building ready to use web applications that clients can use to get a user's sentiment
					 - Deploying our web applications to make them available to a wide array of users
					    """
				)
        with right_column:
            st.image("resources/imgs/people.jpg")
        st.write('---')

        st.subheader("Meet the Team👨‍💼🏢👩‍💼")
        st.info("""
        Our research team is led by a highly experienced and skilled professionals, 
        who has a deep understanding of the industry and the latest trends. Together with the team, 
        they work on creating cutting-edge research methods and tools to deliver accurate and actionable 
        insights for our clients.
        """
        )
        check1,check2,check3=st.columns(3)
        with check1:
            rumbie=st.checkbox("Rumbie Chitongo")
        with check2:
            chisom=st.checkbox("Chisom Nwankwo")
        with check3:
            eliza=st.checkbox("Elizabeth Olorunleke")

        check4,check5,check6=st.columns(3)
        with check4:
            ben=st.checkbox("Benjamin Michael")
        with check5:
            thando=st.checkbox("Thandolwethu Madondo")
        with check6:
            koke=st.checkbox("Koketso Maleka")

        if chisom:
            c_image,c_text=st.columns(2)
            with c_image:
                st.image("resources/imgs/team1.jpg")
            with c_text:
                st.write("""
                Chisom is a highly skilled and experienced data scientist. 
                With a background in statistics and computer science, she has a deep understanding of 
                data analysis, machine learning, and statistical modeling. She has extensive experience 
                in the industry and has worked on a wide range of projects, from consumer research to 
                market analysis. Her ability to turn data into actionable insights has made her 
                a valuable asset to the team. Chisom is passionate about using data to drive business 
                decisions and is dedicated to 
                providing accurate and reliable results for our clients.
                """
                )
        if rumbie:
            r_text,r_image=st.columns(2)
            with r_text:
                st.write("""
            Rumbie is a highly skilled and experienced data scientist with a passion for uncovering insights 
            from complex data sets. With a background in statistics and machine learning, Rumbie has a 
            strong understanding of the latest techniques and tools in the field. Rumbie has a proven 
            track record of delivering actionable insights that drive business decisions and has 
            experience working with a variety of industries. Rumbie is a team player, with a strong 
            ability to communicate complex technical concepts to non-technical stakeholders. 
            She is always looking for new challenges and opportunities to expand her knowledge and skills.
            """
            )
            with r_image:
                st.image("resources/imgs/team2.jpg")
        if koke:
            k_text,k_image=st.columns(2)
            with k_image:
                st.image("resources/imgs/team3.jpg")
            with k_text:
                st.write(
                    """
             Koketso is a junior data analyst with a passion for using data to drive business decisions
              and solve complex problems. With a strong background in statistics and programming, Koketso 
              is able to turn large sets of data into actionable insights that can be used to improve 
              operations and increase revenue.
            """
            )

        if ben:
            b_text,b_image=st.columns(2)
            with b_image:
                st.image("resources/imgs/team4.jpeg")
            with b_text:
                st.write(
                    """
                Benjamin is a Junior Data Analyst with a passion for uncovering insights and 
                solving problems through data analysis. With a strong background in statistics 
                and programming, he is able to turn raw data into actionable information for 
                businesses and organizations. His attention to detail and ability to communicate 
                findings clearly make him a valuable asset to any team
                """
                )
        if eliza:
            e_image,e_text=st.columns(2)
            with e_image:
                st.image("resources/imgs/team5.png")
            with e_text:
                st.write(
                    """
                    Elizabeth is a highly skilled data scientist and machine learning engineer with 
                    a passion for uncovering insights and driving business growth through data analysis. 
                    With a strong background in both statistical analysis and programming, she excels at 
                    developing predictive models and identifying patterns in large data sets. Elizabeth's 
                    expertise in machine learning and artificial intelligence allows her to create 
                    innovative solutions for a wide range of industries, from healthcare to finance. 
                    Her commitment to staying up-to-date with the latest industry developments and technologies 
                    ensures that her clients receive the best possible service.
                    """
                )

        if thando:
            t_image,t_text=st.columns(2)
            with t_image:
                st.image("resources/imgs/team6.png")
            with t_text:
                st.write(
                    """
             With a background in statistics, machine learning and data mining,
             Thandolwethu has a deep understanding of the latest techniques and technologies in the field. 
             She is an expert in statistical modeling, data visualization and predictive analytics. 
             Whether working on a large-scale project or a small-scale research, Thandolwethu brings 
             her analytical skills and attention to detail to deliver actionable insights for clients. 
             She is always eager to take on new challenges and push the boundaries of what's possible 
             with data.
             """
                )
        st.write('---')

        st.subheader("📬 Get in Touch with us!")
        contact_form = """
        <form action="https://formsubmit.co/cynthiarapheals@gmail.com" method="POST">
     <input type="hidden" name="_captcha" value="false">
     <input type="text" name="name" placeholder="Your name" required>
     <input type="email" name="email" placeholder="Your email" required>
     <textarea name="message" placeholder="Your message"></textarea>
     <button type="submit">Send</button>
</form>
					"""
        st.markdown(contact_form, unsafe_allow_html = True)
    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.

        local_css("style.css")

    if page_selection == "Analysis and Insights":
        st.title("Data Insights")
        st.info("Data Source")
        st.write(
            """
            This [dataset](https://www.kaggle.com/competitions/edsa-movie-recommendation-predict/data) 
            is derived from an online movie recommendation service called [MovieLens](https://grouplens.org/). 
            It consists of millions of 5-star ratings that are provided by the MovieLens' users. 
            Additional data has been scraped from IMDB and added with the purpose to enhance the dataset. 
            """
        )
       # ratings = st.checkbox("Ratings insights")
        #movies = st.checkbox("Movie insights")
        #if ratings:
         #   st.subheader("These plots give insights about the ratings given for the movies")
        
        st.header("1. Movie Insights")
        st.info(
            """
            This is the exploration of the movie dataset and explanation of the insights that we derived 
            from it
        """
        )
        genre = st.checkbox("Genres Distribution")
        director = st.checkbox("Director Distribution")
        plot = st.checkbox("Plot Keywords Distribution")

        if genre:
            st.subheader('1.1 Genres distribution')
            st.image("resources/imgs/movie_genre.png",use_column_width=True)
            st.markdown(
                """
                Dramatic and comic movies seem to be most popular with the Film-noir and Imax being
                the least popular as observed in this movies sample data.
                """
                )
        if director:
            st.subheader('1.2 Director distribution')
            first_half,second_half = st.columns(2)
            with first_half:
                st.image("resources/imgs/movie_director.png",use_column_width=True)
            with second_half:
                st.image("resources/imgs/wordcloud1.png",use_column_width=True)
            st.markdown(
                """
                Analysing the popularity of directors that have made atleast 1o movies, we can see that
                Luc Besson and Alex Gibney are the most popular, with Woody Allen and Robert Rodriguez being 
                the least popular
                """
                )
        if plot:
            st.subheader("1.3 Plot Keywords Distribution")
            st.image('resources/imgs/plot.png',use_column_width=True)
            st.markdown(
                """
                Plot keywords are terms or phrases that describe the main themes or events in a movie. 
                They can affect a user's movie choice and rating by providing information about the 
                movie's content and genre. On analysing the plot keywords of movie in this dataset, we can
                see that most users love movies with keywords like action, comedy and f-rated
                """
                )
        st.write("---")
        st.header("2. Rating Insights")
        st.info(
            """
            This is the exploration of the user ratings and explanation of the insights that we derived 
            from it
        """
        )
        rating=st.checkbox("Rating Distribution")
        if rating:
            st.subheader('2.1 Rating distribution')
            st.image("resources/imgs/movie_rating1.png",use_column_width=True)
            st.image("resources/imgs/movie_rating2.png",use_column_width=True)
            st.image("resources/imgs/movie_rating3.png",use_column_width=True)
            st.markdown(
                """
                On exploring the ratings users give to movies they've watched, we noticed that user 
                reviews are relatively positive. For better insights, we grouped the user ratings 
                according to the year the movies were made
                """
                )
    if page_selection == "Model Performance":
        st.title("Model Evaluation")
        st.info(
            """To reduce computation time, we train and evaluate the following models 
        on a 100k subset of the data. We will look at each model and  and compare their 
        performance using a statistical measure known as the root mean squared error (RMSE)
        """)
        svd= st.checkbox("SVD")
        nom= st.checkbox("NormalPredictor")
        bas = st.checkbox("BaselineOnly")
        nmf = st.checkbox("NMF")
        slop = st.checkbox("SlopeOne")
        clus = st.checkbox("CoClustering")

        if svd:
            st.write(
                """
                The Singular Value Decomposition algorithm is a matrix factorization technique 
                which reduces the number of features of a dataset and was popularized by 
                Simon Funk during the [Neflix Prize](https://en.wikipedia.org/wiki/Netflix_Prize)
                 contest [[6]](#ref6). In the matrix structure, each row represents a user and 
                 each column represents a movie. The matrix elements are ratings that are given
                  to movies by users.
                  """
            )
        if nom:
            st.write(
                """
                The Normal Predictor algorithm predicts a random rating for each movie based 
                on the distribution of the training set, which is assumed to be normal
                """
            )
        if bas:
            st.write(
                """
                The Baseline Only algorithm predicts the baseline estimate for a given user and movie. 
            A baseline is calculated using either Stochastic Gradient Descent (SGD) or 
            Alternating Least Squares (ALS).
            """
            )
        if nmf:
            st.write("""
            NMF is a collaborative filtering algorithm based on Non-negative Matrix Factorization. 
            The optimization procedure is a (regularized) stochastic gradient descent with a specific 
            choice of step size that ensures non-negativity of factors, provided that their initial 
            values are also positive""")
        if slop:
            st.write("""
            The SlopeOne algorithm is a simple yet accurate collaborative 
            filtering algorithm that uses a simple linear regression model to 
            solve the data sparisity problem. 
            """)
        if clus:
            st.write(
                """
                The Co-clustering algorithm assigns clusters using a straightforward optimization method, 
                much like k-means.
                """
            )

        st.subheader("Model Performance")
        st.info("""We built and tested six different collaborative filtering models 
        and compared their performance using a statistical measure known as the root 
        mean squared error (**RMSE**), which determines the average squared 
        difference between the estimated values and the actual value. A low 
        RMSE value indicates high model accuracy.""")
        st.image("resources/imgs/model_compare.png",use_column_width=True)




# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
    main()
 