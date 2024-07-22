import streamlit as st
import fitz  # PyMuPDF
from docx import Document
import os
import tempfile

# Function to read PDFs
def read_pdf(file):
    doc = fitz.open(file)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to read Word files
def read_word(file):
    doc = Document(file)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return "\n".join(text)

# Main Streamlit app
def main():
    st.title("File Upload and Processing App")
    st.write("Upload a folder of PDF or Word files for processing.")

    uploaded_files = st.file_uploader("Choose PDF or Word files", type=['pdf', 'docx'], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_filename = temp_file.name
            
            if uploaded_file.name.endswith('.pdf'):
                st.write(f"**Processing PDF file:** {uploaded_file.name}")
                text = read_pdf(temp_filename)
            elif uploaded_file.name.endswith('.docx'):
                st.write(f"**Processing Word file:** {uploaded_file.name}")
                text = read_word(temp_filename)
            
            st.text_area(f"Content of {uploaded_file.name}:", text, height=200)
            os.remove(temp_filename)

if __name__ == "__main__":
    main()
