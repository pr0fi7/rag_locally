import streamlit as st
from PyPDF2 import PdfReader
from langchain.vectorstores.chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema.document import Document
import io
from langchain.prompts import ChatPromptTemplate
import requests
import json
from langchain_community.embeddings.ollama import OllamaEmbeddings

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings


def get_pdf_text(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def split_documents(documents: list[Document]):
    text_splitter = CharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks


with st.sidebar:
    st.write("This is a sidebar")
    docs = st.file_uploader("Upload a file", accept_multiple_files=True, type=["pdf"])

    if st.button("Process"):
        if docs:
            for doc in docs:
                file_bytes = doc.read()
                file_io = io.BytesIO(file_bytes)
                text = get_pdf_text(file_io)
                documents = [Document(page_content=text, metadata={"source": doc.name})]
                chunks = split_documents(documents)
                st.write('Text is chunked')
                add_to_chroma(chunks)
                st.write("Processing complete!")
        else:
            st.write("Please upload at least one PDF file.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if message := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": message})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(message)

    with st.chat_message("assistant"):
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        results = db.similarity_search_with_score(message, k=5)

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=message)

        url = "http://localhost:11434/api/generate"
        data = {
            "model": "llama3",
            "prompt": prompt,
        }

        response = requests.post(url, json=data)

        lines = response.text.strip().split('\n')

        full_response = ""
        for line in lines:
            json_line = json.loads(line)
            full_response += json_line['response']
        
        sources = [doc.metadata.get("id", None) for doc, _score in results]
        formatted_response = f"Response: {full_response}\nSources: {sources}"
        st.markdown(formatted_response)

    st.session_state.messages.append({"role": "assistant", "content": formatted_response})
