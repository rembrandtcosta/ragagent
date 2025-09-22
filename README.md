# Assistente de Legislação Condominial Brasileira Baseado em RAG

**Sistema de Agentes com RAG para Consulta Dúvidas Sobre Condomínios**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-green.svg)](https://langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)](https://streamlit.io)


## Installing

### 1. Clone the Repository
```bash
git clone https://github.com/rembrandtcosta/ragagent
cd ragagent
```
### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Set Up Environment Variables
```bash
export GOOGLE_API_KEY='your_google_api_key'
```
### 5. Index Documents 
```bash
python ingest/ingest.py
```

## Usage
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.


