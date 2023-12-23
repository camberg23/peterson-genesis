import streamlit as st
import os
import pickle
import numpy as np
import pandas as pd
import json
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
# from get_ideas import *
from process_ideas import *

def get_cluster_explanation(cluster_texts):
    SYSTEM_MESSAGE = "Provide a concise summary of the following ideas: "
    transcript = " ".join(cluster_texts)
    client = OpenAI(api_key=openai_api)
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": transcript}
        ]
    )
    return completion.choices[0].message

# Function to load ideas data
def load_ideas(ideas_folder):
    ideas_data = {}
    for file in os.listdir(ideas_folder):
        if file.endswith('.pickle'):
            with open(os.path.join(ideas_folder, file), 'rb') as f:
                ideas_data[file] = pickle.load(f)
    return ideas_data

# Function to aggregate embeddings and corresponding texts
def aggregate_embeddings(ideas_data):
    embeddings = []
    idea_texts = []  # To keep track of idea text for each embedding
    references = []  # To keep track of video ID and timestamp for each idea
    for file, ideas in ideas_data.items():
        video_id = file.split('aggregated_ideas_')[1].replace('.pickle', '')
        for timestamp, idea in ideas.items():
            embeddings.append(idea['embedding'])
            idea_texts.append(idea['text'])
            references.append((video_id, timestamp))
    return np.array(embeddings), idea_texts, references

# Function to perform PCA and K-means clustering
def pca_and_cluster(embeddings, n_components=3, n_clusters=6, random_state=42):
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=n_clusters,  n_init='auto', random_state=random_state)
    labels = kmeans.fit_predict(reduced_embeddings)

    return reduced_embeddings, labels

def split_text(text, max_line_length=75):
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 <= max_line_length:
            current_line += (" " + word) if current_line else word
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return '<br>'.join(lines)

def plot_3d_scatter(embeddings, labels, texts, cluster_explanations):
    df = pd.DataFrame(embeddings, columns=['x', 'y', 'z'])
    df['Idea'] = ['Idea ' + str(i) for i in range(len(texts))]
    df['Text'] = [split_text(text) for text in texts]
    df['Clusters'] = [cluster_explanations[str(label)] for label in labels]

    fig = px.scatter_3d(df, x='x', y='y', z='z', color='Clusters', hover_name='Idea', hover_data={'Text': True, 'Clusters': False, 'x': False, 'y': False, 'z': False})
    fig.update_traces(marker=dict(size=4),  selector=dict(type='scatter3d'))
    fig.update_layout(
        width=1000, 
        height=600, 
        legend=dict(itemsizing='constant', x=1.05, y=0.5, xanchor='left', yanchor='middle', font=dict(size=14)),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    return fig


# Function to convert timestamp to YouTube URL format
def timestamp_to_youtube_url(video_id, timestamp):
    return f"https://www.youtube.com/watch?v={video_id}&t={timestamp}s"

def on_cluster_change():
    # Reset the selected idea index when the cluster changes
    st.session_state.selected_idea_index = 0

def on_idea_change():
    # Update the selected idea index
    st.session_state.selected_idea_index = st.session_state.idea_selectbox

def find_similar_ideas(query_embedding, idea_embeddings, top_n):
    # Calculate cosine similarity between the query and all idea embeddings
    similarities = cosine_similarity([query_embedding], idea_embeddings)[0]
    
    # Get the indices of the top N similar ideas
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    return top_indices

# Streamlit App
def main():
    st.set_page_config(page_title="JBP Genesis Ideas", page_icon="https://pbs.twimg.com/media/DHDP6tWW0AE8lL_.jpg")
    
    st.title('Explore Every Idea from Dr. Peterson\'s Genesis Lectures')
    st.markdown("**Explore the mathematical relationship between thousands of self-contained ideas from Dr. Peterson's lecture series, [The Psychological Significance of the Biblical Stories: Genesis](https://youtube.com/playlist?list=PL22J3VaeABQD_IZs7y60I3lUrrFTzkpat&si=oiNNM-jcnIzSi57c).**")

    with st.expander("**How to use this**"):
        st.markdown("""
        There are three tools to explore:
        1. *Search for any idea:* see the most relevant ideas to your query from Dr. Peterson's lectures, complete with links to the moment where the idea is brought up, as well as their conceptual category.\n\n
        2. *Explore the interactive visualization of every idea from this lecture series:* each point in the plot below represents a self-contained idea. Drag and mouseover the plot to explore ideas and the six core idea-clusters from these lectures.\n\n
        3. *Explore ideas by cluster:* explore any of the ideas within a given conceptual cluster, complete with links to the ideas within the original lectures.
        """)
    st.markdown('---')

    ideas_folder = 'ideas'
    ideas_data = load_ideas(ideas_folder)

    if not ideas_data:
        st.error("No ideas data found in the 'ideas' folder.")
        return

    embeddings, idea_texts, references = aggregate_embeddings(ideas_data)
    if embeddings.size == 0:
        st.error("No embeddings found in the data.")
        return

    # Load cluster explanations
    with open('cluster_explanations.json', 'r') as file:
        cluster_explanations = json.load(file)

    reduced_embeddings, labels = pca_and_cluster(embeddings, n_clusters=len(cluster_explanations))

    st.markdown("### Tool one: search for any idea")
    col1, col2, col3 = st.columns([2.5, 1, 1])
    with col1:
        user_query = st.text_input("Search any idea from the lectures", placeholder="hemispheric differences, psychological significance of the serpent, etc.")
    with col2:
        top_n = st.number_input("# related ideas", min_value=1, max_value=20, value=5)
    with col3:
        submit_search = st.button("Search")
    
    if submit_search and user_query:
        query_embedding = get_embedding(user_query)
        top_indices = find_similar_ideas(query_embedding, embeddings, top_n)
        with st.expander(f"**{top_n} most relevant ideas (click on the idea to see the moment it is introduced):**", expanded=True):
            for i, index in enumerate(top_indices, 1):
                idea = idea_texts[index]
                cluster_label = labels[index]
                cluster_name = cluster_explanations[str(cluster_label)]
                video_id, timestamp = references[index]
                video_url = f"https://www.youtube.com/watch?v={video_id}&t={round(float(timestamp))}s"
                st.markdown(f"[**Idea {index}**]({video_url}): {idea} *This idea was assigned to conceptual cluster: {cluster_name}.*", unsafe_allow_html=True)
                
    st.markdown('---')

    st.markdown("### Tool two: interactive 3D visualization and clustering of all ideas")
    fig = plot_3d_scatter(reduced_embeddings, labels, idea_texts, cluster_explanations)
    st.plotly_chart(fig)

    st.markdown('---')
    st.markdown("### Tool three: explore ideas by conceptual cluster")
    # Initialize session state
    if 'selected_cluster_name' not in st.session_state:
        st.session_state.selected_cluster_name = list(cluster_explanations.values())[0]
    if 'selected_idea_index' not in st.session_state:
        st.session_state.selected_idea_index = 0

    # Radio button for selecting a cluster
    cluster_selection = st.radio("Please select one of the six conceptual clusters:", list(cluster_explanations.values()), index=list(cluster_explanations.values()).index(st.session_state.selected_cluster_name), on_change=on_cluster_change)
    st.session_state.selected_cluster_name = cluster_selection
    selected_cluster_index = list(cluster_explanations.values()).index(st.session_state.selected_cluster_name)
    
    # Filter ideas based on selected cluster
    filtered_ideas = [(index, idea) for index, (label, idea) in enumerate(zip(labels, idea_texts)) if label == selected_cluster_index]
    idea_indices = [index for index, _ in filtered_ideas]

    # Check if there are ideas in the selected cluster and update the session state accordingly
    if idea_indices:
        # If the current selected idea index is not in the filtered list, reset it to the first item
        if st.session_state.selected_idea_index not in idea_indices:
            st.session_state.selected_idea_index = idea_indices[0]

        # Dropdown for selecting ideas within the selected cluster
        idea_selection = st.selectbox(f"Select an idea within the cluster *{st.session_state.selected_cluster_name}*", idea_indices, index=idea_indices.index(st.session_state.selected_idea_index), format_func=lambda x: f'Idea {x}', key="idea_selectbox", on_change=on_idea_change)
        st.session_state.selected_idea_index = idea_selection
    else:
        # Reset the selected idea index if no ideas are available
        st.session_state.selected_idea_index = None

    # Display selected idea and video
    if filtered_ideas and st.session_state.selected_idea_index in idea_indices:
        selected_idea_text = idea_texts[st.session_state.selected_idea_index]
        st.markdown(f"<p style='font-size: 20px;'><b>{selected_idea_text}</b></p>", unsafe_allow_html=True)

        selected_video_id, selected_timestamp = references[st.session_state.selected_idea_index]
        st.video(f'https://www.youtube.com/watch?v={selected_video_id}', start_time=round(float(selected_timestamp)))
    st.markdown("---")
    st.markdown("Built by [Cameron Berg](https://www.linkedin.com/in/cameron-berg-080b8b1b7/)")
    st.markdown("*To make this tool, I (1) extracted every idea from the lecture transcripts with the help of GPT-4, (2) translated these ideas to their 1536-dimensional embedding representation, (3) reduced these embeddings to 3 dimensions, and (4) clustered these dimensionality-reduced embeddings into the plot you see above. I was then able to make the search and ideas-by-clusters tools on top of this.*")

if __name__ == "__main__":
    main()
