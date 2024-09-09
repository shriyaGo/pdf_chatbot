import streamlit as st
from io import StringIO
import os
from pathlib import Path
from create_kg import get_text_from_files
from read_kg import get_hybrid_index, get_answer


# Set up the Streamlit app layout
st.set_page_config(layout="wide")
vector_index = None;

# Side menu for file upload
st.sidebar.header("Upload File")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["pdf"])


# Store chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat interface
st.title("Pdf chatbot")
st.write("Ask questions based on the uploaded file content")

# File reading and displaying
if uploaded_file is not None:
    FILE_PATH = f"{os.getcwd()}"
    path = os.path.join(FILE_PATH,'docs', uploaded_file.name)
    get_text_from_files(path)
    vector_index = get_hybrid_index()
    st.sidebar.write("File uploaded")

# Input for user query
query = st.text_input("Ask a question:")

# # Process the query and respond based on file content
if st.button("Submit"):
    answer = get_answer(query)
    st.session_state.messages.append((query, answer))
for query, response in st.session_state.messages:
        st.write(f"**You:** {query}")
        st.write(f"**Bot:** {response}")
       
if "messages" not in st.session_state:
        st.session_state.messages = []
    

