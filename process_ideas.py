import os
import pickle
from openai import OpenAI
from tqdm import tqdm
import streamlit as st

openai_api = st.secrets['OPENAI_API']

def get_embedding(text):
    client = OpenAI(api_key=openai_api)
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
        encoding_format="float"
    )
    # Extracting the embedding vector correctly
    embedding_vector = response.data[0].embedding
    return embedding_vector

def process_ideas(ideas_folder):
    idea_files = [f for f in os.listdir(ideas_folder) if f.startswith('aggregated_ideas_') and f.endswith('.pickle')]

    for file in tqdm(idea_files, desc="Processing Ideas", unit="file"):
        file_path = os.path.join(ideas_folder, file)
        with open(file_path, 'rb') as infile:
            ideas = pickle.load(infile)

        # Updated processing
        updated_ideas = {}
        for timestamp, text in ideas.items():
            embedding = get_embedding(text)
            updated_ideas[timestamp] = {'text': text, 'embedding': embedding}

        # Save the updated ideas
        with open(file_path, 'wb') as outfile:
            pickle.dump(updated_ideas, outfile)

if __name__ == "__main__":
    ideas_folder = 'ideas'
    process_ideas(ideas_folder)