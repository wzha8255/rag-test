import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI 
import tempfile
from dotenv import load_dotenv

load_dotenv()


st.set_page_config(layout="wide")
st.title("test simple rag application")

uploaded_file = st.file_uploader("Upload your pdf", type='pdf')
if uploaded_file:

    # save file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_pdf_path = tmp_file.name

    # extract text from PDF
    pdf_reader = PdfReader(temp_pdf_path)
    raw_text = ""
    for page in pdf_reader.pages:
        raw_text += page.extract_text()

    # split text
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=500, chunk_overlap= 50)
    chunks = text_splitter.split_text(raw_text)
    st.info(f"{len(chunks)} chunks splitted.")

    # embed splitted chunks
    st.info("embedding pdf ....")
    embedding_model = OpenAIEmbeddings()
    embeddings = FAISS.from_texts(chunks, embedding_model)

    # set up LLM and QA chain
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type='stuff')

    # ask questions
    st.success("PDF processed! Ask your question!")
    query = st.text_input("ask somthing about content in the uploaded pdf")
    if query:
        docs = embeddings.similarity_search(query, k=3)
        for idx,doc in enumerate(docs):
            # st.markdown(f"*** Retrieval {idx+1}: {doc}")
            st.text_area(f"Retrieved Content {idx+1}", value=doc, height=300)
        response = chain.run(input_documents=docs, question=query)
        st.markdown(f"*** Answer: {response}")