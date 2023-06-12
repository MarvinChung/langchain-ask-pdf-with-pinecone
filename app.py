import os, tempfile
import streamlit as st
from langchain.llms.openai import OpenAI
from langchain.vectorstores.pinecone import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
# from langchain.callbacks import get_openai_callback
from PyPDF2 import PdfReader
import pinecone 

os.environ["OPENAI_API_BASE"]= "http://35.189.163.143:8080/v1"
os.environ["OPENAI_API_KEY"]="EMPTY"

def main():
    # Streamlit app
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
                
    # Get OpenAI API key, Pinecone API key and environment, and source document input
    with st.sidebar:
        # openai_api_key       = st.text_input("OpenAI API key", type="password")
        pinecone_api_key     = st.text_input("Pinecone API key", type="password")
        pinecone_env         = st.text_input("Pinecone environment")
        pinecone_index       = st.text_input("Pinecone index name")
        # reuse_pinecone_index = st.checkbox('reuse pinecone index', key='box')


    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:

        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=512,
            chunk_overlap=0,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # if pdf is not None:
        #     from langchain.document_loaders import PyPDFLoader
        #     loader = PyPDFLoader(pdf)
        #     documents = loader.load()

        #     text_splitter = CharacterTextSplitter(
        #         separator="\n",
        #         chunk_size=1000,
        #         chunk_overlap=0,
        #         length_function=len
        #     )
        #     docs = text_splitter.split_documents(documents)

        # from langchain.document_loaders import TextLoader
        # loader = TextLoader('state_of_the_union.txt')
        # documents = loader.load()
        # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        # docs = text_splitter.split_documents(documents)
        # end

        user_question = st.text_input("Ask a question about your PDF:")
        if st.button("Submit"):
            # Validate inputs
            if not pinecone_api_key or not pinecone_env or not pinecone_index:
                st.warning(f"Please provide the missing fields.")
            else:
                try:
                    # split into chunks
                    # Generate embeddings for the pages, insert into Pinecone vector database, and expose the index in a retriever interface
                    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
                    embeddings = OpenAIEmbeddings(model="multilingual-e5-base")

                    # knowledge_base = Pinecone.from_documents(docs, embeddings, index_name=pinecone_index)
                    knowledge_base = Pinecone.from_texts(chunks, embeddings, index_name=pinecone_index, batch_size=1)

                    # show user input
                    if user_question:
                        docs = knowledge_base.similarity_search(user_question)
                        st.write("Docs with Best Similarity:")
                        st.write(docs[0])
                        llm = OpenAI(model="bloom-zh-1b1")
                        chain = load_qa_chain(llm, chain_type="stuff")
                        response = chain.run(input_documents=docs[:1], question=user_question)
                        st.write(response)
                           
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
