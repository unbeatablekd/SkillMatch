from flask import Flask, render_template, request, send_file
import shutil
import zipfile
import os
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances
import fitz  # PyMuPDF
import docx2txt
from flask import send_file


app = Flask('SKILLMATCH2')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/job')
def job():
    return render_template('job.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form['jd']

    # Load the English language model in spaCy
    nlp = spacy.load("en_core_web_sm")

    # Initialize CountVectorizer with desired parameters
    vectorizer = CountVectorizer(max_features=1000, stop_words='english', vocabulary=None)

    # Function to extract text from docx file
    def extract_text_from_docx(docx_file_path):
        try:
            return docx2txt.process(docx_file_path)
        except ImportError:
            print("Error: 'docx2txt' module not found. Make sure to install it.")
            return None

    # Function to extract text from PDF file using PyMuPDF (fitz)
    def extract_text_from_pdf(pdf_file_path):
        try:
            text = ""
            with fitz.open(pdf_file_path) as pdf:
                for page in pdf:
                    text += page.get_text()
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return None

    # Function to extract text from resume file (either docx or pdf)
    def extract_text_from_resume(resume_file_path):
        _, extension = os.path.splitext(resume_file_path)
        if extension.lower() == '.pdf':
            return extract_text_from_pdf(resume_file_path)
        elif extension.lower() == '.docx':
            return extract_text_from_docx(resume_file_path)
        else:
            print(f"Unsupported file format: {extension}")
            return None

    # Function to extract features from text
    def extract_features(text):
        doc = nlp(text)
        return " ".join([token.text.lower() for token in doc if not token.is_stop and not token.is_punct])

    # Function to calculate similarity score between job description and resume
    def calculate_similarity(job_features, resume_features):
        # Convert features to vectors using CountVectorizer
        X_bow = vectorizer.fit_transform([job_features, resume_features])
        # Calculate cosine distance
        cosine_distance = cosine_distances(X_bow[0], X_bow[1])
        # Convert cosine distance to similarity score (1 - cosine distance)
        similarity_score = 1 - cosine_distance[0, 0]
        return similarity_score

    # Get user input for job description
    job_description = input_data

    # Specify the path to the zip file containing resumes
    zip_file_path = "resumes.zip"
    extraction_path = "resumes_unzipped"

    # Clear the extraction path if it exists
    if os.path.exists(extraction_path):
        shutil.rmtree(extraction_path)

    # Extract the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_path)

    # Initialize a list to store similarity scores
    similarity_scores = []

    # Iterate over the unzipped files
    for root, dirs, files in os.walk(extraction_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_name.lower().endswith(('.pdf', '.docx')):
                resume_text = extract_text_from_resume(file_path)
                if resume_text:
                    resume_features = extract_features(resume_text)
                    job_features = extract_features(job_description)
                    similarity_score = calculate_similarity(job_features, resume_features)
                    similarity_scores.append((file_name, similarity_score))

    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    return render_template('results.html', similarity_scores=similarity_scores[:10])

@app.route('/view_resume/<file_name>')
def view_resume(file_name):
    # Serve the resume file to be viewed
    resume_dir = "resumes_unzipped"
    return send_file(os.path.join(resume_dir, file_name))

if __name__ == '__main__':
    app.run("localhost", 9999, debug=True)
