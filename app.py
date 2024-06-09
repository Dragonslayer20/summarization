import streamlit as st
import asyncio
from concurrent.futures import ThreadPoolExecutor
from transc import get_transcript
from summarizer_and_paraphraser import summarize_text, paraphrase_text
from qadrant_1 import setup_qdrant_collection, recommend_items

#st.title("NLP Toolkit")

# Load models once
summarizer_model = None
paraphraser_model = None
def load_models():
    global summarizer_model, paraphraser_model
    if summarizer_model is None:
        summarizer_model = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    if paraphraser_model is None:
        paraphraser_model = AutoModelForSeq2SeqLM.from_pretrained("tuner007/pegasus_paraphrase")
    return summarizer_model, paraphraser_model

def process_text_async(func, text):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    with ThreadPoolExecutor() as pool:
        result = loop.run_in_executor(pool, func, text)
        return loop.run_until_complete(result)

toolbar = st.sidebar.radio(
    "Choose a tool",
    ("Video Processing Tool", "Text Processing Tool",)
)

if toolbar == "Video Processing Tool":
    st.header("Video Processing Tool 🎬")

    full_yt = st.text_input("Enter video link", "")

    if st.button("Get Transcript"):
        try:
            video_id = full_yt.split("=")[1]
            get_transcript(video_id)
            st.success("Transcript fetched successfully. Choose an action below.")
        except Exception as e:
            st.error(f"Error fetching transcript: {e}")

    col1, col2 = st.columns([2, 1])
    with col1:
        summarize_button = st.button("Summarize Transcript")
    with col2:
        paraphrase_button = st.button("Paraphrase Transcript")

    if summarize_button:
        try:
            with open("ms_kitco.txt", "r") as f:
                tx = f.read()
            
            summarized_text = process_text_async(summarize_text, tx)
            st.write(summarized_text)
        except Exception as e:
            st.error(f"Error summarizing transcript: {e}")

    if paraphrase_button:
        try:
            with open("ms_kitco.txt", "r") as f:
                tx = f.read()
            
            paraphrased_text = process_text_async(paraphrase_text, tx)
            st.write(paraphrased_text)
        except Exception as e:
            st.error(f"Error paraphrasing transcript: {e}")

elif toolbar == "Text Processing Tool":
    st.header("Text Processing Tool 📝🔄")

    text_to_process = st.text_area("Enter text", height=300)

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("Summarize Text"):
            try:
                summarized_text = process_text_async(summarize_text, text_to_process)
                st.write(summarized_text)
            except Exception as e:
                st.error(f"Error summarizing text: {e}")

    with col2:
        if st.button("Paraphrase Text"):
            try:
                paraphrased_text = process_text_async(paraphrase_text, text_to_process)
                st.write(paraphrased_text)
            except Exception as e:
                st.error(f"Error paraphrasing text: {e}")

elif toolbar == "Qdrant Setup & Recommendations":
    st.header("Qdrant Setup & Recommendations")

    collection_name = st.text_input("Collection Name", "first_collection")

    if st.button("Setup Qdrant Collection"):
        try:
            client, collection_name = setup_qdrant_collection(collection_name)
            st.success(f"Collection '{collection_name}' set up successfully.")
        except Exception as e:
            st.error(f"Error setting up collection: {e}")

    query_vector = st.text_input("Query Vector (comma-separated)", "")
    positive_ids = st.text_input("Positive IDs (comma-separated)", "")
    negative_ids = st.text_input("Negative IDs (comma-separated)", "")
    limit = st.number_input("Limit", min_value=1, value=5)

    if st.button("Get Recommendations"):
        try:
            query_vector = [float(x) for x in query_vector.split(",")]
            positive_ids = [int(x) for x in positive_ids.split(",")]
            negative_ids = [int(x) for x in negative_ids.split(",")]

            client, collection_name = setup_qdrant_collection(collection_name)  # Ensure collection is set up
            recommendations = recommend_items(client, collection_name, query_vector, positive_ids, negative_ids, limit)
            st.write(recommendations)
        except Exception as e:
            st.error(f"Error getting recommendations: {e}")
