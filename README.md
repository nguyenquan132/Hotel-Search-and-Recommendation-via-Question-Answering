<div align="center">
    <h1>Hotel Search and Recommendation via Question Answering</h1>
</div>

## Introduction
In the era of travel and technological advancement, searching for and recommending hotels that meet users' needs has become a critical challenge. This problem requires not only the ability to retrieve accurate information but also the capability to understand the context and specific requirements of users through natural language questions.

**Objective**: To build a system capable of answering user questions about hotels while providing recommendations based on criteria such as location, desciprtion, ratings.

**Key Features**: 
* Preprocess and analyze data before chunking to improve retrieval efficiency.
* Build a vector database for efficient storage and retrieval of hotel information.
* Implement retrieval-augmented generation (RAG) techniques to provide accurate answers and relevant recommendations.
* Leverage a large language model (LLM) to generate accurate answers to user queries.

## Technologies
- Framework: Langchain, FastAPI, Streamlit
- Vector Database: PineCone
- Large Language Model: Gemini

## Installation 

### **1. Clone the repository:**
```bash
https://github.com/nguyenquan132/Hotel-Search-and-Recommendation-via-Question-Answering.git
```

### **2. Create virtual environment and a file '.env' to store API_KEY of PINECONE, GEMINI, LANGGRAPH**

```bash
python -m venv myvenv
```

Install the requirements:

```bash
pip install -r requirements.txt
```

### **3. Run Application**
To start the application, follow these steps:
1️⃣ **Run the backend** using the following command:
```bash
uvicorn main:app --reload
```
2️⃣ Start the frontend with:
```bash
streamlit run app.py
```

## Demo
<video>
    <source src="demo/Hotel Search and Recommendation via Question Answering - Brave 2025-03-15 01-57-37.mp4" type="video/mp4">
</video>