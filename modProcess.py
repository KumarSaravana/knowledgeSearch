import re  # For text cleaning
import io
import hashlib
import pytesseract  # For OCR
from PIL import Image
import openai
import ollama as ol

def clean_text(text):
    cleaned_text = re.sub(r'\s+', ' ', text)  # Remove extra spaces, tabs, newlines
    cleaned_text = re.sub(r'[^\w\s.,!?]', '', cleaned_text)  # Remove special characters
    return cleaned_text

def compute_image_hash(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()

def extract_text_from_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'  # your path may be different
    text = pytesseract.image_to_string(image)
    return text.strip()

# Function to get embeddings from OpenAI's text-embedding-ada-002
def get_embedding(text, embedType):
    if embedType == "OpenAI":
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        resp = response['data'][0]['embedding']
    elif embedType == "nomic-embed-text":
        response = ol.embeddings(prompt=text, model="nomic-embed-text")
        resp = response['embedding']
    return resp