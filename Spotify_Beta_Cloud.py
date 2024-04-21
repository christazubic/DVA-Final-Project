import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import pandas as pd
from pyvis.network import Network

# Function to fetch data (assuming you already have data loaded in audio_features_df)
def fetch_data():
    # Your data fetching code goes here
    return audio_features_df


# Streamlit app title
st.title('Spotify Song Network Analysis')

# Fetch data
audio_features_df = pd.read_csv('df_christa_edit.csv')

# Select Artist dropdown
selected_artist = st.selectbox('Select Artist', audio_features_df['ArtistName'].unique())

if selected_artist:
    # Filter songs by selected artist
    artist_songs = audio_features_df.loc[audio_features_df['ArtistName'] == selected_artist, 'TrackName']

    # Select Song dropdown
    selected_song = st.selectbox('Select Song', artist_songs)

    if selected_song:
        # Add a multi-select to select one or more features
        default_features = ['danceability', 'energy', 'key', 'loudness', 'mode',
                            'speechiness', 'acousticness', 'instrumentalness',
                            'liveness', 'valence', 'tempo']
        selected_features = st.multiselect('Select One or More Features', default_features, default=default_features)

        if selected_features:
            # Filter data for selected song and features
            selected_track_id = audio_features_df.loc[(audio_features_df['ArtistName'] == selected_artist) &
                                                       (audio_features_df['TrackName'] == selected_song), 'TrackID']
            
            if not selected_track_id.empty:
                selected_track_id = selected_track_id.iloc[0]
                selected_song_values = audio_features_df.loc[selected_track_id, selected_features]

                selected_song_values = np.array(selected_song_values).reshape(1, -1)

                # Fit StandardScaler on selected feature columns
                ss = StandardScaler()
                song_values_scaled = ss.fit_transform(audio_features_df[selected_features])

                # Calculate cosine similarity with other songs using selected features
                cosine_similarities = cosine_similarity(selected_song_values, song_values_scaled)

                # Sort and get top similar songs
                similar_indices = np.argsort(cosine_similarities[0])[::-1][:20]  # Top 20 similar songs
                
                # Create a table to display similar songs
                similar_songs_data = {
                    'Song Name': [],
                    'Artist Name': [],
                    'Cosine Score': []
                }
                for idx in similar_indices:
                    similar_songs_data['Song Name'].append(audio_features_df.iloc[idx]['TrackName'])
                    similar_songs_data['Artist Name'].append(audio_features_df.iloc[idx]['ArtistName'])
                    similar_songs_data['Cosine Score'].append(cosine_similarities[0][idx]*100)

                similar_songs_df = pd.DataFrame(similar_songs_data)
                similar_songs_df = similar_songs_df.rename(columns = {"Cosine Score": "Percent Similarity", "Artist Name": "Artist Name", "Song Name": "Song Name" })

                # Display the table
                st.write(similar_songs_df)
                #add padding
                st.markdown("---") 

                # Visualize song network using NetworkX and pyvis
                G = nx.Graph()
                G.add_node(str(selected_track_id), label=selected_song)  # Convert ID to string explicitly
                # Add selected song as a node with its label

                for i in similar_indices:
                    track_id = audio_features_df.iloc[i]['TrackID']
                    track_name = audio_features_df.iloc[i]['TrackName']
                    cosine_score = cosine_similarities[0][i]
                    G.add_node(str(track_id), label=track_name)  # Convert ID to string explicitly
                    # Add similar song as a node with its label
                    G.add_edge(str(selected_track_id), str(track_id), weight=cosine_score)  
                    # Add edge between selected song and similar song

                graph_title = f"Most Similar Songs to {selected_song}"

                # Visualize the graph using pyvis
                g4 = Network(height="500px", width="750px", notebook=True)
                g4.from_nx(G)
                # g4.show_buttons(filter_=['physics'])
                g4.show("graph.html")  # Save the graph as an HTML file
                html_str = open("graph.html").read()
                html_str = f"<h2>{graph_title}</h2>\n{html_str}"
                try:
                    path = '/tmp'
                    g4.save_graph(f'{path}/graph.html')
                    HtmlFile = open(f'{path}/graph.html', 'r', encoding='utf-8')
                
                # Save and read graph as HTML file (locally)
                except:
                    path = '/html_files'
                    g4.save_graph(f'{path}/graph.html')
                    HtmlFile = open(f'{path}/graph.html', 'r', encoding='utf-8')
                
                
                st.components.v1.html(html_str, height=600)
                
            else:
                st.error("Selected song not found.")