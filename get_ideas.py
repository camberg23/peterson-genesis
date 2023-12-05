import json
import pickle
import tiktoken
import os
import ast
from openai import OpenAI
from tqdm import tqdm
from process_ideas import *
from get_ideas_message import SYSTEM_MESSAGE

# Function to count tokens in a string
def num_tokens_from_string(string: str, encoding_name: str = 'cl100k_base') -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Function to process transcript with GPT-4
def process_transcript_with_gpt(transcript):
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

# Function to chunk transcript
def chunk_transcript(transcript_entries, chunk_size=5000):
    chunks = []
    current_chunk = ""
    for entry in transcript_entries:
        entry_with_timestamp = f"[{entry['start']}s] {entry['text']}"
        if num_tokens_from_string(current_chunk + entry_with_timestamp) > chunk_size:
            chunks.append(current_chunk)
            current_chunk = entry_with_timestamp
        else:
            current_chunk += " " + entry_with_timestamp
    chunks.append(current_chunk)  # Append the last chunk
    return chunks

# Main function to process all transcripts
def main():
    # Load transcripts from JSON file
    with open('transcripts.json', 'r') as infile:
        transcripts = json.load(infile)

    # Create the 'ideas' folder if it doesn't exist
    ideas_folder = 'ideas'
    if not os.path.exists(ideas_folder):
        os.makedirs(ideas_folder)

    # Process each video with a tqdm progress bar
    for video_id in tqdm(transcripts.keys(), desc="Processing Lectures", unit="lecture"):
        chunks = chunk_transcript(transcripts[video_id])  # Adjust chunk_size as needed
        aggregated_ideas = {}

        # Process each chunk with another tqdm progress bar
        for chunk in tqdm(chunks, desc=f"Processing Chunks of Video {video_id}", leave=False, unit="chunk"):
            gpt_response = process_transcript_with_gpt(chunk)

            # Check if '```plaintext' is in the response
            if '```plaintext' in gpt_response.content:
                dict_str = gpt_response.content.split('```plaintext')[1].strip().strip('```')
            elif '```json' in gpt_response.content:
                dict_str = gpt_response.content.split('```json')[1].strip().strip('```')
            else:
                dict_str = gpt_response.content

            try:
                chunk_ideas = ast.literal_eval(dict_str)  # Convert string to dictionary
                aggregated_ideas.update(chunk_ideas)
            except ValueError as e:
                print(f"Error processing chunk for video {video_id}: {e}")

        # Save the aggregated ideas
        with open(os.path.join(ideas_folder, f'aggregated_ideas_{video_id}.pickle'), 'wb') as outfile:
            pickle.dump(aggregated_ideas, outfile)

if __name__ == "__main__":
    main()