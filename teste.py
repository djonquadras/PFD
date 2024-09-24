"""
import os
import PyPDF2  # or pdfplumber, or fitz (from PyMuPDF) for text extraction
from transformers import pipeline  # Hugging Face Transformers for summarization

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

# Function to summarize text using Hugging Face Summarization pipeline
def split_text(text, max_length=1000):  # Ensure max_length is within the model's token limit
    words = text.split()
    for i in range(0, len(words), max_length):
        yield ' '.join(words[i:i + max_length])

MAX_INPUT_LENGTH = 1024  # or adjust based on model

def summarize_text(text, model_name="facebook/bart-large-cnn", max_length=300):
    summarizer = pipeline("summarization", model=model_name, device=0)  # Use GPU

    # Truncate input if too long
    truncated_text = text[:MAX_INPUT_LENGTH]
    
    summaries = []
    for chunk in split_text(truncated_text):
        try:
            summary = summarizer(chunk, max_length=max_length, min_length=100, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
            summaries.append("Error in summarization")
    
    return ' '.join(summaries)

# Analyze and summarize all papers
def analyze_papers(paper_folder):
    all_summaries = []
    for filename in os.listdir(paper_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(paper_folder, filename)
            print(f"Processing {filename}...")

            # Step 1: Extract text
            paper_text = extract_text_from_pdf(pdf_path)

            # Step 2: Summarize text
            paper_summary = summarize_text(paper_text)

            # Store summary
            all_summaries.append(f"Summary for {filename}:\n{paper_summary}\n")

    # Combine all summaries into one overview
    overview = "\n".join(all_summaries)
    return overview

# Main function to run the analysis
if __name__ == "__main__":
    paper_folder = "Papers/"
    overview = analyze_papers(paper_folder)
    print("Consolidated Overview of All Papers:\n")
    print(overview)
    
"""

import os
import pdfplumber  # Biblioteca recomendada para extração de texto
from transformers import pipeline  # Hugging Face Transformers para sumarização
import tensorflow as tf  # Atualize TensorFlow para evitar funções desatualizadas

# Configura ambiente para CUDA (opcional, pode ser removido se não usar GPU)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Função para extrair texto do PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def split_text(text, max_length=512):  # Reduza o tamanho para 512 ou menor
    words = text.split()
    for i in range(0, len(words), max_length):
        yield ' '.join(words[i:i + max_length])

MAX_INPUT_LENGTH = 1024  # Ajuste conforme o modelo

# Função para sumarizar o texto usando Hugging Face Summarization pipeline
def summarize_text(text, model_name="facebook/bart-large-cnn", max_length=300):
    summarizer = pipeline("summarization", model=model_name, device=-1)  # Usando CPU

    summaries = []
    for chunk in split_text(text, max_length=512):  # Use o mesmo tamanho de chunk que acima
        try:
            summary = summarizer(chunk, max_length=max_length, min_length=50, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
            summaries.append("Error in summarization")

    return ' '.join(summaries)



# Analisar e sumarizar todos os documentos juntos (em vez de separadamente)
def analyze_papers_comparatively(paper_folder):
    combined_text = ""  # Coleta todo o texto dos PDFs

    for filename in os.listdir(paper_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(paper_folder, filename)
            print(f"Processing {filename}...")

            # Passo 1: Extrair texto de cada PDF e adicionar ao texto combinado
            paper_text = extract_text_from_pdf(pdf_path)
            combined_text += f"\n\n{paper_text}"

    # Passo 2: Gerar um resumo abrangente do texto combinado
    consolidated_summary = summarize_text(combined_text)

    return consolidated_summary

# Função principal para rodar a análise
if __name__ == "__main__":
    paper_folder = "Papers/"  # Substitua pelo seu diretório contendo os PDFs
    overview = analyze_papers_comparatively(paper_folder)
    print("Consolidated Comparative Overview of All Papers:\n")
    print(overview)
