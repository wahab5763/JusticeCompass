import os
import streamlit as st
import pdfplumber
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline

# Set up the page configuration
st.set_page_config(page_title="RAG-based PDF Chat", layout="centered", page_icon="📄")

# Load the summarization pipeline model
@st.cache_resource
def load_summarization_pipeline():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer

summarizer = load_summarization_pipeline()

# Split text into manageable chunks
@st.cache_data
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Initialize embedding function
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a FAISS vector store with embeddings, checking for empty chunks
@st.cache_resource
def load_or_create_vector_store(text_chunks):
    if not text_chunks:
        st.error("No valid text chunks found to create a vector store. Please check your PDF files.")
        return None
    vector_store = FAISS.from_texts(text_chunks, embedding=embedding_function)
    return vector_store

# Helper function to process a single PDF
def process_single_pdf(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except Exception as e:
        st.error(f"Failed to read PDF: {file_path} - {e}")
    return text

# Function to load PDFs with progress display
def load_pdfs_with_progress(folder_path):
    all_text = ""
    pdf_files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith('.pdf')]
    num_files = len(pdf_files)

    if num_files == 0:
        st.error("No PDF files found in the specified folder.")
        st.session_state['vector_store'] = None
        st.session_state['loading'] = False
        return

    # Title for the progress bar
    st.markdown("### Loading data...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    processed_count = 0

    for file_path in pdf_files:
        result = process_single_pdf(file_path)
        all_text += result
        processed_count += 1
        progress_percentage = int((processed_count / num_files) * 100)
        progress_bar.progress(processed_count / num_files)
        status_text.text(f"Loading documents: {progress_percentage}% completed")

    progress_bar.empty()  # Remove the progress bar when done
    status_text.text("Document loading completed!")  # Show completion message

    if all_text:
        text_chunks = get_text_chunks(all_text)
        vector_store = load_or_create_vector_store(text_chunks)
        st.session_state['vector_store'] = vector_store
    else:
        st.session_state['vector_store'] = None

    st.session_state['loading'] = False  # Mark loading as complete

# Generate summary based on the retrieved text
def generate_summary_with_huggingface(query, retrieved_text):
    summarization_input = f"{query} Related information:{retrieved_text}"
    max_input_length = 1024
    summarization_input = summarization_input[:max_input_length]
    summary = summarizer(summarization_input, max_length=500, min_length=50, do_sample=False)
    return summary[0]["summary_text"]

# Generate response for user query
def user_input(user_question):
    vector_store = st.session_state.get('vector_store')
    if vector_store is None:
        return "The app is still loading documents or no documents were successfully loaded."
    docs = vector_store.similarity_search(user_question)
    context_text = " ".join([doc.page_content for doc in docs])
    return generate_summary_with_huggingface(user_question, context_text)

# Main function to run the Streamlit app
def main():
    # Use HTML to style the title with a larger font size
    st.markdown(
        """
        <h1 style="font-size:30px; text-align: center;">
        📄 JusticeCompass: Your AI-Powered Legal Navigator for Swift, Accurate Guidance.
        </h1>
        """,
        unsafe_allow_html=True
    )

    # Start loading documents if not already loaded
    if 'loading' not in st.session_state or st.session_state['loading']:
        st.session_state['loading'] = True
        load_pdfs_with_progress('documents1')

    user_question = st.text_input("Ask a Question:", placeholder="Type your question here...")

    if st.session_state.get('loading', True):
        st.info("The app is loading documents in the background. You can type your question now and submit once loading is complete.")

    if st.button("Get Response"):
        if not user_question:
            st.warning("Please enter a question before submitting.")
        else:
            with st.spinner("Generating response..."):
                answer = user_input(user_question)
                st.markdown(f"**🤖 AI:** {answer}")

if __name__ == "__main__":
    main()
