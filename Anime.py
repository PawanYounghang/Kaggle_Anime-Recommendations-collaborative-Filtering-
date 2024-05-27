import streamlit as st

# importing the required modules and libraries for the core funcrtion /// 
import numpy as np
import csv
import pandas as pd
from math import sqrt
import regex as re
# import matplotlib.pyplot as plt

#classes for the program defined ////
class Anime:

    def __init__(self,anime_id,name,episodes,public_rating,members):
        self.anime_id = anime_id
        self.name = name
        self.public_rating = public_rating
        self.members = members
        self.episodes = episodes
    
class Rating:
    
    def __init__(self,user_id,anime_id,user_rating):
        self.user_id = user_id
        self.anime_id = anime_id
        self.user_rating = user_rating
        
class Input:
    def __init__(self,input_name,input_rating):
        self.name = input_name 
        self.input_rating = input_rating

# main definitations that are used for the program ///
def get_anime_info():
    # Reads anime data from a CSV file, cleans it using pandas, and creates a list of Anime objects.
    # Args: csv_file: The path to the CSV file containing anime data.
    # Returns: A list of Anime class objects.
    anime = pd.read_csv('anime.csv')
    anime = anime.dropna()
    anime = anime.drop(columns = ['genre','type'])
    # anime['name'] = [re.sub(r'[^a-zA-Z0-9\s]', '', i)for i in anime['name']]
    # anime['name'] = anime['name'].str.strip().str.lower()

    # Create Anime objects from the cleaned DataFrame
    animes = anime.apply(lambda row: Anime(*row.tolist()), axis=1).tolist()

    return animes

# animes = get_anime_info()
# print(animes)

def get_rating_info():
    # Reads ratuing data from a CSV file, cleans it using pandas, and creates a list of rating objects.
    # Args: csv_file: The path to the CSV file containing anime rating data.
    # Returns: A list of Rating class objects.
    rating = pd.read_csv('rating.csv')
    rating = rating.dropna()
    
    # Create Anime objects from the cleaned DataFrame
    ratings = rating.apply(lambda row: Rating(*row.tolist()), axis=1).tolist()

    return ratings
# ratings = get_rating_info()
# print(ratings)

# Steve = [
#             {'name':'Gintama', 'rating':9},
#             {'name':'Violence Gekiga David no Hoshi', 'rating':7},
#             {'name':'Violence Gekiga Shin David no Hoshi Inma Densetsu', 'rating':9},
#             {'name':"Yasuji no Pornorama Yacchimae", 'rating':5},
#             {'name':'Fullmetal Alchemist Brotherhood', 'rating':9},
#             {'name':'Kimi no Na wa', 'rating':6.5},
#          ]
# steve_data = pd.DataFrame(Steve)
# steve_data['name'] = steve_data['name'].str.strip().str.lower()
# steve_data

#getting the User subset data 
def get_user_subset_group(steve_data):
    #getting the anime_data 
    anime = pd.read_csv('anime.csv', usecols=(['anime_id','name']))
    anime = anime.dropna()
    anime['name'] = [re.sub(r'[^a-zA-Z0-9\s]', '', i)for i in anime['name']]
    anime['name'] = anime['name'].str.strip().str.lower()
    
    #getting the rating_data 
    rating = pd.read_csv('rating.csv')

    #formatting the user input data
    steve_data['name'] = [re.sub(r'[^a-zA-Z0-9\s]', '', i)for i in steve_data['name']]
    steve_data['name'] = steve_data['name'].str.strip().str.lower()

    # Filtering out animes by title
    Id = anime[anime['name'].isin(steve_data['name'].tolist())]
    
    # Merging data to get anime ID
    input_data = pd.merge(Id, steve_data, on='name')
    
    # Filtering out users that have watched animes the input user has watched
    users = rating[rating['anime_id'].isin(input_data['anime_id'].tolist())]
    
    # Grouping users by user_id
    userSubsetGroup = users.groupby('user_id')
    
    # Sorting user groups based on the number of common animes watched
    userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]>=5), reverse=True)
    
    # Filter out users who have watched at least 5 similar anime
    userSubsetGroup = [(user_id, group) for user_id, group in userSubsetGroup if len(group) >= 4]
    
    return userSubsetGroup,input_data,anime,rating

# calling the function:
# userSubsetGroup,input_data,anime,rating = get_user_subset_group(steve_data)
# print(len(userSubsetGroup))
# userSubsetGroup

#getting simillarity index using the person's coefficient
def get_prearson_coefficient(userSubsetGroup,input_data):
    #Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
    pearsonCorDict = {}

    #For every user group in our subset
    for name, group in userSubsetGroup:
        #Let's start by sorting the input and current user group so the values aren't mixed up later on
        group = group.sort_values(by='anime_id')
        inputMovie = input_data.sort_values(by='anime_id')
        #Get the N for the formula
        n = len(group)
        #Get the review scores for the movies that they both have in common
        temp = input_data[input_data['anime_id'].isin(group['anime_id'].tolist())]
        #And then store them in a temporary buffer variable in a list format to facilitate future calculations
        tempRatingList = temp['rating'].tolist()
        #put the current user group reviews in a list format
        tempGroupList = group['rating'].tolist()
        #Now let's calculate the pearson correlation between two users, so called, x and y
        Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(n)
        Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(n)
        Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(n)

        #If the denominator is different than zero, then divide, else, 0 correlation.
        if Sxx != 0 and Syy != 0:
            pearsonCorDict[name] = Sxy/sqrt(Sxx*Syy)
        else:
            pearsonCorDict[name] = 0

    pearsonDF = pd.DataFrame.from_dict(pearsonCorDict, orient='index')
    pearsonDF.columns = ['similarityIndex']
    pearsonDF['user_Id'] = pearsonDF.index
    pearsonDF.index = range(len(pearsonDF))
    return pearsonDF

# pearsonDF = get_prearson_coefficient()


def get_recommended_animes(pearsonDF,rating,anime):
    # Sort users by similarity index
    topUsers = pearsonDF.sort_values(by='similarityIndex', ascending=False)
    
    # Filter top users based on similarity threshold
    topUsers = topUsers[topUsers['similarityIndex'] >= 0.90]
    
    # Initialize list to store related anime IDs
    anime_list = []
    
    # Iterate over top users
    for user_id in topUsers.index:
        # Filter ratings of the current user with high ratings
        related_anime_id = rating[(rating['user_id'] == user_id) & (rating['rating'] >= 9)]['anime_id']
        # Append the related anime IDs to the list
        anime_list.extend(related_anime_id)
    
    # Concatenate the list of related anime IDs into a DataFrame
    new_anime_data = pd.DataFrame({'anime_id': anime_list})
    
    # Drop duplicate anime IDs
    unique_anime_df = pd.DataFrame({'recommended_animes': new_anime_data['anime_id'].drop_duplicates()})
    
    #getting names from the dataframe of actucal animes
    Id = anime[anime['anime_id'].isin(unique_anime_df['recommended_animes'].tolist())]
 
    return Id

# anime = get_recommended_animes(pearsonDF,rating,anime)

def final_list(anime, steve_data):
    # Filter anime titles that are not present in the user input data
    final_anime_list = anime[~anime['name'].isin(steve_data['name'])]
    return final_anime_list



# /// main program logic /// 
def main():
    st.title("Anime Reccomendation system")
    # Initialize session state for storing inputs
    if 'anime_list' not in st.session_state:
        st.session_state.anime_list = []

    #getting the anime names into a list for dropdowm    
    X = get_anime_info()
    name = []
    for i in X:
        k =i.name
        name.append(k)

    # Filter out already selected animes
    available_anime_names = [name for name in name if name not in [anime['name'] for anime in st.session_state.anime_list]]

    # Dropdown for user selection
    if available_anime_names:
        Title = st.selectbox('Anime Title:', available_anime_names)

        # Number input for rating
        rating = st.number_input('Rating:', min_value=0.0, max_value=10.0, step=0.1)

        # Check if rating is within range
        if rating is not None and 0 <= rating <= 10:
            # Add button to store the input
            if st.button('Add'):
                st.session_state.anime_list.append({'name': Title, 'rating': rating})
        else:
            st.warning("Please enter a valid rating between 0 and 10.")
    else:
        st.write("All animes have been selected.")

    # Create DataFrame from the list
    if st.session_state.anime_list:
        steve_data = pd.DataFrame(st.session_state.anime_list)
        st.write("Anime Title/Ratings:")

        # Style the DataFrame
        styled_df = steve_data.style \
            .format({'rating': '{:.1f}'}) \
            .set_properties(**{'text-align': 'center'}) \
            .set_table_styles([{
                'selector': 'th',
                'props': [('font-size', '16px'),
                          ('text-align', 'center')]
            }]) \
            .set_caption('User input of Animes and Ratings') \
            .hide(axis='index')
    
        # st.write(styled_df)
        st.write(styled_df.to_html(), unsafe_allow_html=True)

    if st.button("Submit"):
        st.write("Anime Recommendations:")
        userSubsetGroup,input_data,anime,rate = get_user_subset_group(steve_data)
        pearsonDF = get_prearson_coefficient(userSubsetGroup,input_data)
        anime = get_recommended_animes(pearsonDF,rate,anime)
        final = final_list(anime, steve_data)

        final_names = []
        for obj in X:
            k = obj.name
            k = re.sub(r'[^a-zA-Z0-9\s]', '', k)
            k = k.strip().lower()
            final_names.append({'name':k,'object': obj})
        output = []
        for name in final['name']:
            for item in final_names:
                if name == item['name']:
                    output.append({'Anime Title':item['object'].name, 'Public rating': item['object'].public_rating, 'Episodes': item['object'].episodes})

        Output = pd.DataFrame(output)
        # Style the DataFrame
        style = Output.style \
            .format({'rating': '{:.1f}'}) \
            .set_properties(**{'text-align': 'center'}) \
            .set_table_styles([{
            'selector': 'th',
            'props': [('font-size', '16px'),
            ('text-align', 'center')]
            }]) \
            .set_caption('Recommended Animes using ML') \
            .hide(axis='index')
                
        # st.write(styled_df)
        st.write(style.to_html(), unsafe_allow_html=True)

if __name__ == "__main__":
    main()




    