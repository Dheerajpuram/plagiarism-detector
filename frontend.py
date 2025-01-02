import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
import openai

# Load the model and vectorizer
MODEL_PATH = 'model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'

try:
    model = pickle.load(open(MODEL_PATH, 'rb'))
    tfidf_vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))
except FileNotFoundError as e:
    st.error(f"Required files not found: {e}")
    st.stop()


# Set your OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Function to preprocess the text
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Remove stop words
    stop_words = set(stopwords.words("english"))
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

# Function to detect plagiarism and originality percentage
def detect(input_text):
    # Preprocess the input text
    input_text = preprocess_text(input_text)
    
    # Vectorize the input text using the saved TF-IDF vectorizer
    vectorized_text = tfidf_vectorizer.transform([input_text])
    
    # Predict plagiarism (0: No Plagiarism, 1: Plagiarism)
    result = model.predict(vectorized_text)
    
    # Calculate cosine similarity with the training data (to get originality percentage)
    similarities = cosine_similarity(vectorized_text, tfidf_vectorizer.transform([input_text]))
    max_similarity = similarities.max()
    
    # Originality percentage as (1 - max_similarity) * 100
    originality_percentage = (1 - max_similarity) * 100
    
    # Return results
    plagiarism_result = "Plagiarism Detected" if result[0] == 1 else "No Plagiarism Detected"
    return plagiarism_result, originality_percentage

# Function to read text from a .txt or .docx file
def read_file(uploaded_file):
    if uploaded_file.type == "text/plain":
        return uploaded_file.getvalue().decode("utf-8")
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(uploaded_file)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return "\n".join(full_text)
    else:
        st.error("Unsupported file type")
        return None

# Function to rephrase text using OpenAI's GPT-3 (or GPT-4)


# Function to rephrase text using OpenAI's GPT-3 (or GPT-4)
import openai
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Set your OpenAI API key from the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to rephrase text using OpenAI's GPT-3 (or GPT-4)
def rephrase_text(input_text):
    try:
        if len(input_text.strip()) < 10:
            return "Please enter a longer text for rephrasing."

        # Request the rephrasing from GPT
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Or "gpt-4" if you have access to GPT-4
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Please rephrase the following text to make it unique and more original, while maintaining its meaning:\n\n{input_text}"}
            ],
            max_tokens=500,
            temperature=0.7,
        )

        # Debugging: Print the raw response
        print("API Response:", response)

        # Check if the response has any valid content
        rephrased_text = response['choices'][0]['message']['content'].strip()

        if not rephrased_text:
            print("Warning: No rephrased content generated!")
            return "Sorry, I couldn't generate a more original version."
        
        return rephrased_text
    except Exception as e:
        print(f"Error rephrasing text: {e}")
        return "Sorry, there was an error processing the request."

# Streamlit app layout
st.title("Plagiarism Detector and Chat Assistant")
st.write("Choose how you want to provide the text.")

# Radio button for selecting input method
input_method = st.radio("Select input method", ("Upload a file", "Input text"))

# If "Input text" is selected
if input_method == "Input text":
    input_text = st.text_area("Enter your text here:")
    
    if st.button("Detect Plagiarism"):
        if input_text.strip():
            detection_result, originality = detect(input_text)
            st.success(f"Result: {detection_result}")
            st.write(f"Originality Percentage: {originality:.2f}%")
        else:
            st.warning("Please enter some text for detection.")
    
    # Chat assistant to rephrase text
    if st.button("Generate Original Text"):
        if input_text.strip():
            rephrased_text = rephrase_text(input_text)
            if rephrased_text:
                st.success("Rephrased Text:")
                st.write(rephrased_text)
        else:
            st.warning("Please enter text to generate an original version.")

# If "Upload a file" is selected
elif input_method == "Upload a file":
    uploaded_file = st.file_uploader("Choose a .txt or .docx file", type=["txt", "docx"])
    
    if uploaded_file is not None:
        file_text = read_file(uploaded_file)
        
        if file_text:
            detection_result, originality = detect(file_text)
            st.success(f"Result: {detection_result}")
            st.write(f"Originality Percentage: {originality:.2f}%")
            
            # Chat assistant to rephrase text from uploaded file
            if st.button("Generate Original Text from File"):
                rephrased_text = rephrase_text(file_text)
                if rephrased_text:
                    st.success("Rephrased Text:")
                    st.write(rephrased_text)