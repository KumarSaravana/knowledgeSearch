import streamlit as st
import openai
from langchain_community.llms import Ollama
from pinecone import Pinecone, ServerlessSpec
#import torch
#from transformers import pipeline
#from huggingface_hub import login
#from llama_index.llms.groq import Groq

from modProcess import clean_text, compute_image_hash, extract_text_from_image, get_embedding

OPENAI_API_KEY = "sk-5JcbXuYcGKobVHUVQ4yFGFmwmoHyZ2EBmvaCwTcXVNT3BlbkFJef1rtXtB-1kv6V5wIBJkhF_pAnfjQ_tjgNQXCswEsA"
PINECONE_API_KEY = "fa89aa36-c208-4901-9ae1-2db0d4601f0b"
#HUGGINGFACE_API_KEY = "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"

# Initialize OpenAI and Pinecone
openai.api_key = OPENAI_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)


# Search function for querying Pinecone and handling the response
def search_in_pinecone(query, index_name, top_k,embedType):
    query_embedding = get_embedding(query, embedType)
    #print(query_embedding)
    index = pc.Index(index_name)
    try:
        result = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        if 'matches' in result:
            return result['matches']
        else:
            return []
    except Exception as e:
        st.error(f"An error occurred while querying Pinecone: {str(e)}")
        return []

def refine_answer_with_llm(top_text_result, query, temperature,max_tokens, model, model_type):
    context = top_text_result if isinstance(top_text_result, str) else top_text_result ['metadata'].get('text', top_text_result['metadata'].get('ocr_text', ''))
    if not context:
        return "No text available for refinement."
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    msg = [{"role": "system", "content": "You are an assistant that helps summarize and refine text. Prioritize text in the input from index more than generated data"},
            {"role": "user", "content": prompt}]
    
    if model_type == "Local":
        ollama = Ollama(base_url = 'http://localhost:11434',
                    model = model)  
        response = ollama(prompt)
        resp = response
    elif model_type == 'Hosted' and model == 'gpt-3.5-turbo':
        response = openai.ChatCompletion.create(
            model=model,
            messages=msg,
            max_tokens=max_tokens,
            temperature=temperature
        )
        resp = response['choices'][0]['message']['content'].strip()
    elif  model_type == 'HuggingFace' and model == 'meta-llama/Llama-3.2-1B':
        #login(HUGGINGFACE_API_KEY)
        #model_id = model
        #llm = Groq(model="llama3-70b-8192", api_key="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")
        #response = llm.complete(msg)
        #resp = response[0]["generated_text"][-1].strip()
        resp = None
    return resp