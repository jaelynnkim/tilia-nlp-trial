import streamlit as st
import pandas as pd
import re
import os
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Set NLTK data path to the local directory
nltk.data.path.append('./nltk_data')

# Helper functions
def extract_details_from_filename(filename):
    parts = filename.split(',')
    company = parts[0].strip()
    quarter_match = re.search(r"Q\d+", filename)
    quarter = quarter_match.group() if quarter_match else 'Unknown'
    year_match = re.search(r"\d{4}\.txt$", filename)
    year = year_match.group()[:-4] if year_match else '0000'
    return company, quarter, int(year)

def load_texts_to_dataframe(files):
    data = []
    for file in files:
        filename = file.name
        if filename.endswith('.txt'):
            company, quarter, year = extract_details_from_filename(filename)
            content = file.read().decode('utf-8')
            data.append({'Company': company, 'Quarter': quarter, 'Year': year, 'File Name': filename, 'Text': content})
    return pd.DataFrame(data)

def clean_text(text, patterns):
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    return text

def remove_text_before_presentation(text):
    presentation_index = text.find("Presentation")
    if presentation_index != -1:
        text = text[presentation_index:]
    return text

def split_text_into_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)

def chunk_text10(text, chunk_size=10):
    sentences = sent_tokenize(text)
    chunks = [' '.join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]
    return chunks

def compress_sentence(sentence, pipeline):
    try:
        output = pipeline(sentence, max_length=50, truncation=True)
        return output[0]['generated_text']
    except Exception as e:
        print(f"Error compressing sentence: {e}")
        return sentence  # Return the original sentence if compression fails

def extract_nouns(sentence):
    words = word_tokenize(sentence)
    pos_tags = pos_tag(words)
    nouns = [word for word, tag in pos_tags if tag in {'NN', 'NNS', 'NNP', 'NNPS'}]
    return ' '.join(nouns)

# Initialize Streamlit app
st.title('Document Upload and Processing App')

# Upload files section
uploaded_files = st.file_uploader("Upload your documents", type=['txt', 'pdf', 'docx'], accept_multiple_files=True)

# Clear all files button
if st.button('Clear All Files'):
    uploaded_files = []

if uploaded_files:
    df = load_texts_to_dataframe(uploaded_files)

    # Clean and process the text
    patterns = [
        r"These materials have been prepared solely for information purposes based upon information generally available to the public\s+and from sources believed to be reliable.*",
        r"COPYRIGHT\s+©\s+\d{4}\s+(by\s+)?S&P\s+Global\s+Market\s+Intelligence,\s+a\s+division\s+of\s+S&P\s+Global\s+Inc\.\s+All\s+rights\s+reserved",
        r"spglobal\.com/marketintelligence(\s*\d+)?"
    ]
    df['Text'] = df['Text'].apply(lambda x: clean_text(x, patterns))
    df['Text'] = df['Text'].apply(remove_text_before_presentation)

    # Split into sentences
    rows = []
    for _, row in df.iterrows():
        sentences = split_text_into_sentences(row['Text'])
        for sentence in sentences:
            rows.append({'Company': row['Company'], 'Year': row['Year'], 'Quarter': row['Quarter'], 'Sentence': sentence})

    sentence_df = pd.DataFrame(rows)

    # Summarize text
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    rows = []
    for _, row in sentence_df.iterrows():
        text_chunks = chunk_text10(row['Sentence'])
        for chunk in text_chunks:
            summary = summarizer(chunk, max_length=160, min_length=40, do_sample=False)[0]['summary_text']
            rows.append({
                'Company': row['Company'],
                'Year': row['Year'],
                'Quarter': row['Quarter'],
                'Text': chunk,
                'pipeline_summary': summary
            })

    summary_df = pd.DataFrame(rows)

    # Compress sentences
    compression_pipeline = pipeline("text2text-generation", model="jaelynnkk/sentence_compression")
    summary_df['Compressed_Sentence'] = summary_df['pipeline_summary'].apply(lambda x: compress_sentence(x, compression_pipeline))

    # Extract nouns
    summary_df['Nouns_Only'] = summary_df['Compressed_Sentence'].apply(extract_nouns)

    st.write("Processed Data:")
    st.dataframe(summary_df)

    # Display keywords as clickable buttons
    unique_keywords = summary_df['Nouns_Only'].unique()
    keyword_state = {keyword: False for keyword in unique_keywords}

    def toggle_keyword(keyword):
        keyword_state[keyword] = not keyword_state[keyword]

    for keyword in unique_keywords:
        if st.button(keyword):
            toggle_keyword(keyword)

    for keyword, active in keyword_state.items():
        if active:
            filtered_df = summary_df[summary_df['Nouns_Only'] == keyword]
            st.write(f"Details for keyword: {keyword}")
            st.dataframe(filtered_df)
else:
    st.write("No files uploaded.")
