# 📚 Book Recommendation Chatbot (RAG)

This project implements an **interactive book recommendation chatbot** using a **Retrieval-Augmented Generation (RAG)** architecture.  
Users can describe what they want to read in natural language, and the system retrieves relevant books from a catalog and generates personalized recommendations using a language model.

The project was developed as a **Natural Language Processing course project**, with a strong focus on:
- retrieval quality,
- explainable recommendations,
- modular system design,
- and a clean, interactive user interface.

---

## 🚀 Features

- 🔍 **Semantic book retrieval** using sentence embeddings + FAISS
- 🤖 **LLM-powered recommendation generation** (RAG pipeline)
- 🎭 **Answer style control** (friendly, formal, concise, detailed)
- 🧠 **Optional mood detection** to adapt recommendation tone
- 📖 **Explain-why mode** for transparent recommendations
- 🔁 **Alternative recommendations** (“second opinion”)
- 📊 **Analytics page** (most recommended books, retrieval statistics)
- 🕸️ **Book similarity graph** based on embedding distances
- 🧪 **Evaluation notebook** using RAGAS and manual IR metrics
- 🖥️ **Streamlit web interface**

---

## 🧠 System Architecture

```

User Query
↓
Embedding Model (Sentence-Transformers)
↓
FAISS Vector Store (Book embeddings)
↓
Top-K Retrieved Books
↓
RAG Pipeline
├── Context construction
├── Prompting
└── LLM Generation
↓
Final Answer + Book Recommendations

```

---

## 📁 Project Structure

```

rag-book-recommender/
│
├── data/
│   ├── clean_books.csv
│   ├── books_with_genres.csv
│   └── eval/
│       ├── eval_queries.json
│       ├── rag_outputs.jsonl
│       ├── manual_retrieval_metrics.csv
│       └── results.jsonl
│
├── models/
│   ├── faiss_index.bin
│   └── metadata.pkl
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_generator_and_prompts.ipynb
│   ├── 03_rag_pipeline_test.ipynb
│   ├── 04_evaluation_ragas.ipynb
│   ├── 05_advanced_features.ipynb
│   ├── dev_faiss_test.ipynb
│   ├── dev_test_embeddings.ipynb
│   └── dev_test_generator.ipynb
│
├── src/
│   ├── generator/
│   ├── logging/
│   ├── pipeline/
│   ├── retriever/
│   ├── service/
│   └── utils/
│
├── ui/
│   ├── app.py
│   └── pages/
│       ├── 1_Book_Graph.py
│       └── 2_Analytics.py
│
├── requirements.txt
├── README.md
└── .gitignore

```

---

## ⚙️ Installation & Setup

### 1. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate 
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Ensure model files exist

The following files must be present:

```
models/faiss_index.bin
models/metadata.pkl
```

These are generated during preprocessing and embedding creation.

---

## ▶️ Running the Application

```bash
streamlit run ui/app.py
```

Then open the local URL shown in the terminal.

---

## 🧪 Evaluation

The project includes both **quantitative** and **qualitative** evaluation:

* **RAGAS metrics**:

  * Faithfulness
  * Answer relevance
  * Context precision / recall
* **Manual IR metrics**:

  * Recall@K
  * Precision@K
  * Mean Reciprocal Rank (MRR)

Evaluation code and results are available in:

```
notebooks/04_evaluation_ragas.ipynb
data/eval/
```
Classical IR metrics (Recall@K, Precision@K, MRR) were often zero due to strict title matching between retrieved results and manually defined gold labels. Since the system performs semantic retrieval and focuses on explainable recommendations rather than exact title matching, these metrics underestimate practical performance. We therefore rely primarily on RAGAS metrics and qualitative analysis.

---

## 📊 Analytics & Visualizations

* **Analytics page**:

  * Most frequently recommended books
  * Retrieval distribution insights

* **Book similarity graph**:

  * Built from FAISS nearest neighbors
  * Interactive exploration in the UI

---

## 👥 Team & Contributions

This project was developed as a **two-person team project**.

* **Bianca-Gabriela Leoveanu - Data & Infrastructure**:

  * Dataset cleaning & preprocessing
  * Embeddings & FAISS vector store
  * Retrieval logic
  * Backend & Streamlit UI
  * Analytics & visualization

* **Berin Venedik - LLM & RAG Orchestration**:

  * Prompt design
  * RAG pipeline logic
  * Evaluation with RAGAS
  * Explanation & refinement modes

---

## 🎯 Key Learning Outcomes

* Practical implementation of **Retrieval-Augmented Generation**
* Embedding-based semantic search with FAISS
* Prompt engineering for controlled generation
* Evaluating RAG systems beyond accuracy
* Building modular, explainable GenAI applications

---

## 📝 License

This project is for academic purposes.