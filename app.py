import streamlit as st
import pandas as pd
import re
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Helper functions
def extract_details_from_filename(filename):
    parts = filename.split(',')
    company = parts[0].strip()
    quarter_match = re.search(r"Q\d+", filename)
    quarter = quarter_match.group() if quarter_match else 'Unknown'
    year_match = re.search(r"\d{4}\.txt$", filename)
    year = year_match.group()[:-4] if year_match else '0000'
    return company, quarter, int(year)

@st.cache_data
def load_texts_to_dataframe(file_list):
    data = []
    for uploaded_file in file_list:
        if uploaded_file.name.endswith('.txt'):
            company, quarter, year = extract_details_from_filename(uploaded_file.name)
            content = uploaded_file.read().decode('utf-8')
            data.append({'Company': company, 'Quarter': quarter, 'Year': year, 'File Name': uploaded_file.name, 'Text': content})
    return pd.DataFrame(data)

def clean_text(text, patterns):
    for pattern in patterns:
        combined_pattern = re.compile(pattern, re.VERBOSE | re.IGNORECASE)
        text = combined_pattern.sub('', text)
    return text

def remove_text_before_second_call_participants(text):
    first_occurrence = text.find("Call Participants")
    second_occurrence = text.find("Call Participants", first_occurrence + 1)
    if first_occurrence != -1 and second_occurrence != -1:
        text = text[second_occurrence:]
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

def summarize_text(text, summarizer):
    try:
        summary = summarizer(text, max_length=160, min_length=40, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return "Summarization failed: " + str(e)

def split_into_sentences(df, text_column):
    rows = []
    for index, row in df.iterrows():
        sentences = sent_tokenize(row[text_column])
        for sentence in sentences:
            rows.append({
                'Company': row['Company'],
                'Year': row['Year'],
                'Quarter': row['Quarter'],
                'Sentence': sentence
            })
    return pd.DataFrame(rows)

def compress_sentence(sentence, compression_pipeline):
    try:
        output = compression_pipeline(sentence, max_length=50, truncation=True)
        return output[0]['generated_text']
    except Exception as e:
        print(f"Error compressing sentence: {e}")
        return sentence  # Return the original sentence if compression fails

def extract_nouns(sentence):
    words = word_tokenize(sentence)
    pos_tags = pos_tag(words)
    nouns = [word for word, tag in pos_tags if tag in {'NN', 'NNS', 'NNP', 'NNPS'}]
    return ' '.join(nouns)

# Streamlit App
st.title("Document Upload and Processing App")

uploaded_files = st.file_uploader("Upload your documents", accept_multiple_files=True, type=['txt'])

if uploaded_files:
    st.write("Processing uploaded files... This may take some time.")
    progress_bar = st.progress(0)
    
    originaltxt = load_texts_to_dataframe(uploaded_files)

    if 'Text' not in originaltxt.columns:
        st.error("Error: 'Text' column not found in the loaded DataFrame.")
        st.stop()

    st.write("Original DataFrame")
    st.dataframe(originaltxt)

    eda_df = originaltxt.copy()
    cutoff_regex = r"These materials have been prepared solely for information purposes based upon information generally available to the public\s+and from sources believed to be reliable.*"
    patterns = [
        cutoff_regex,
        r"COPYRIGHT\s+©\s+\d{4}\s+(by\s+)?S&P\s+Global\s+Market\s+Intelligence,\s+a\s+division\s+of\s+S&P\s+Global\s+Inc\.\s+All\s+rights\s+reserved",
        r"spglobal\.com/marketintelligence(\s*\d+)?"
    ]
    eda_df['Text'] = eda_df['Text'].apply(lambda x: clean_text(x, patterns))
    fix_df = eda_df.copy()
    fix_df['Text'] = fix_df['Text'].apply(remove_text_before_second_call_participants)
    fix_df['Text'] = fix_df['Text'].apply(remove_text_before_presentation)
    cleantext = fix_df.copy()
    sentence_df = split_into_sentences(cleantext, 'Text')

    progress_bar.progress(30)

    # Load summarization pipeline
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    rows = []
    for index, row in cleantext.iterrows():
        text_chunks = chunk_text10(row['Text'])
        for i, chunk in enumerate(text_chunks):
            summary = summarize_text(chunk, summarizer)
            rows.append({
                'Company': row['Company'],
                'Year': row['Year'],
                'Quarter': row['Quarter'],
                'Text': chunk,
                'pipeline_summary': summary
            })

    summary_df10 = pd.DataFrame(rows)
    sentence_df = split_into_sentences(summary_df10, 'pipeline_summary')

    progress_bar.progress(60)

    # Load compression pipeline
    model_id = "jaelynnkk/sentence_compression"
    tokenizer = T5Tokenizer.from_pretrained(model_id)
    model = T5ForConditionalGeneration.from_pretrained(model_id)
    compression_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    def remove_first_last_three(df):
        # If the group has fewer than seven rows, return an empty DataFrame
        if len(df) <= 6:
            return pd.DataFrame()
        # Otherwise, return the DataFrame excluding the first three and last three rows
        return df.iloc[3:-3]

    filtered_df = sentence_df.groupby(['Company', 'Year', 'Quarter']).apply(remove_first_last_three).reset_index(drop=True)
    filtered_df['Compressed_Sentence'] = filtered_df['Sentence'].apply(lambda x: compress_sentence(x, compression_pipeline))
    filtered_keywords_seq = filtered_df.copy()

    filtered_keywords_seq['Nouns_Only'] = filtered_keywords_seq['Compressed_Sentence'].apply(extract_nouns)

    unique_keywords = filtered_keywords_seq['Nouns_Only'].unique()
    selected_keyword = st.selectbox('Select a Keyword', unique_keywords)

    progress_bar.progress(90)

    if selected_keyword:
        st.write(f"Selected Keyword: {selected_keyword}")
        keyword_df = filtered_keywords_seq[filtered_keywords_seq['Nouns_Only'] == selected_keyword]
        st.write(keyword_df[['Company', 'Year', 'Quarter', 'Compressed_Sentence']])

    progress_bar.progress(100)

    if st.button('Clear All Files'):
        uploaded_files = None
        st.experimental_rerun()
