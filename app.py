import os, tempfile
import streamlit as st
from langchain.llms.openai import OpenAI
from langchain.vectorstores.pinecone import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from PyPDF2 import PdfReader
import pinecone 

def set_reuse_pinecone_index():
    # Set to true because the chunks are already saved to the pinecone. 
    # So we call Pinecone.from_existing_index that will not embed the chunks again which will cost credit.
    st.session_state.box = True

def main():
    # Streamlit app
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
                
    # Get OpenAI API key, Pinecone API key and environment, and source document input
    with st.sidebar:
        openai_api_key       = st.text_input("OpenAI API key", type="password")
        pinecone_api_key     = st.text_input("Pinecone API key", type="password")
        pinecone_env         = st.text_input("Pinecone environment")
        pinecone_index       = st.text_input("Pinecone index name")
        reuse_pinecone_index = st.checkbox('reuse pinecone index', key='box')


    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None or reuse_pinecone_index:

        if pdf is not None:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

        user_question = st.text_input("Ask a question about your PDF:")
        if st.button("Submit", on_click=set_reuse_pinecone_index):
            # Validate inputs
            if not openai_api_key or not pinecone_api_key or not pinecone_env or not pinecone_index:
                st.warning(f"Please provide the missing fields.")
            else:
                try:
                    # split into chunks
                    with get_openai_callback() as cb:
                        

                        # Generate embeddings for the pages, insert into Pinecone vector database, and expose the index in a retriever interface
                        pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
                        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

                        if reuse_pinecone_index:
                            knowledge_base = Pinecone.from_existing_index(pinecone_index, embeddings)
                        else:
                            knowledge_base = Pinecone.from_texts(chunks, embeddings, index_name=pinecone_index)

                        # show user input
                        if user_question:
                            docs = knowledge_base.similarity_search(user_question)
                            st.write("Docs with Best Similarity:")
                            st.write(docs[0])
                            llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
                            chain = load_qa_chain(llm, chain_type="stuff")
                            response = chain.run(input_documents=docs[:1], question=user_question)
                            st.write(response)
                            st.write(cb)
                           
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
