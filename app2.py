import os
import tempfile
import streamlit as st
from voice import (
    get_voice_input,
    speak_answer
)

from dotenv import load_dotenv
from gtts import gTTS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from rag_utils import process_pdfs
from recommendation import get_recommendations

# -----------------------------
# Load Environment Variables
# -----------------------------

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# -----------------------------
# Streamlit Config
# -----------------------------

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide"
)

# -----------------------------
# Session State
# -----------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "recommendations" not in st.session_state:
    st.session_state.recommendations = []

# -----------------------------
# UI
# -----------------------------

st.title("📚 AI-Powered Study Assistant")

with st.sidebar:

    st.header("📄 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    process_btn = st.button(
        "Process Documents"
    )

    st.subheader("📌 Recommended Topics")

    for topic in st.session_state.recommendations:
        st.write(f"•{topic}")

# -----------------------------
# Process PDFs
# -----------------------------

if process_btn and uploaded_files:

    pdf_paths = []

    for uploaded_file in uploaded_files:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(uploaded_file.read())
            pdf_paths.append(tmp.name)

    with st.spinner("Processing PDFs..."):

        vector_store, split_docs = process_pdfs(
            pdf_paths
        )

        st.session_state.vector_store = (
            vector_store
        )

        st.session_state.recommendations = (
            get_recommendations(split_docs)
        )

    st.success(
        "Documents processed successfully!"
    )

# -----------------------------
# Ask Question
# -----------------------------
# st.subheader("🎤 Voice Question")

# voice_question = get_voice_input()

# question = st.text_input(
#     "Enter your question",
#     value=voice_question,
#     key="voice_question"
# )
st.subheader("💬 Ask a Question")


question = st.text_input(
    "Enter your question"
)

if st.button("Get Answer"):

    if st.session_state.vector_store is None:

        st.warning(
            "Please upload and process PDFs first."
        )

    elif not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        docs = (
            st.session_state.vector_store
            .similarity_search(
                question,
                k=4
            )
        )

        context = "\n\n".join(
            [
                doc.page_content
                for doc in docs
            ]
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3
        )

        prompt = ChatPromptTemplate.from_template(
            """
            You are an AI Study Assistant.

            Use only the provided context
            to answer the question.

            Context:
            {context}

            Question:
            {question}

            Answer:
            """
        )

        chain = prompt | llm

        response = chain.invoke(
            {
                "context": context,
                "question": question
            }
        )

        answer = response.content

        st.write("### 🤖 Answer")
        st.write(answer)

        try:

           speak_answer(answer)

        except Exception:
            pass

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": answer
            }
        )

# -----------------------------
# Chat History
# -----------------------------

st.subheader("📜 Conversation History")

for chat in reversed(
    st.session_state.chat_history
):

    st.markdown(
        f"### 🧑 Question\n{chat['question']}"
    )

    st.markdown(
        f"### 🤖 Answer\n{chat['answer']}"
    )

    st.divider()