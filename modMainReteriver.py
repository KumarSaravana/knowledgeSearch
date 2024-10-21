import os
import openai
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain_community.llms import Ollama
import ollama as ol
from dotenv import load_dotenv
import os
load_dotenv()

from modResponse import search_in_pinecone, refine_answer_with_llm

openai.api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Streamlit app logic
def main():
    st.title("Knowledge Search")
    st.write('This utility uses free accounts of HuggingFace and PineCone and so is subjective to limits. Pls. reissue the query after few minutes in case you notice errors or issues..')
    with st.sidebar:
        fn = st.radio("Choose the operation", [ "Search existing Index"], horizontal=True)
    if fn == "Search existing Index":
        with st.sidebar:
            index_name = st.selectbox("Select from list of Indexes", pc.list_indexes().names())
        if index_name:
            search_query = st.text_input("Enter your search query", key="search_query")
            button = st.button ("Query")
            with st.sidebar:
                top_k = int(st.text_input("Enter the top K results to retrieve", "1",disabled=True))
                max_tokens = st.slider("Max Tokens for Answer", 50, 500, 200)
                temperature = st.slider("Temperature (Answer Creativity)", 0.0, 1.0, 0.2)
            if button and search_query :
                with st.spinner('We are fetching your results and depending on complexity it might take anywhere between 1-5 mins, appreciate your patience...'):
                    search_results = search_in_pinecone(search_query, index_name, top_k,"OpenAI")
                    #st.write(f"search results: {search_results}")
                    if search_results:
                        sorted_results = sorted(search_results, key=lambda x: x['score'], reverse=True)
                        #st.write(f"sorted results: {sorted_results}")
                        top_text_result = None
                        for result in sorted_results:
                            if 'text_chunk' in result['metadata'] and not top_text_result:
                                top_text_result = result['metadata']['text_chunk']
                                #st.write(f"top results: {top_text_result}")
                                
                        if top_text_result:
                            #refined_text = refine_answer_with_llm(top_text_result, search_query, temperature,max_tokens,'llama3.2:latest', 'Local')
                            refined_text = refine_answer_with_llm(top_text_result, search_query, temperature,max_tokens,'gpt-3.5-turbo', 'Hosted')
                            #refined_text = refine_answer_with_llm(top_text_result, search_query, temperature,max_tokens,'meta-llama/Llama-3.2-1B', 'HuggingFace')
                            st.write(f"**Refined Answer: Generated using Llama-3.2-3B-Instruct**")
                            st.write(refined_text)
                            st.write('\n\n Disclaimer: Above generated text is AI generated and is prone to errors and this utility is in beta phase. Use with caution')                            
                        else:
                            st.write(f"No text found")

if __name__ == "__main__":
    main()
