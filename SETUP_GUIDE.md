<img width="2872" height="1614" alt="Screenshot 2026-06-06 232213" src="https://github.com/user-attachments/assets/6355413b-82ff-4ecf-8c96-bfeaadb955f7" />
# Bot — Setup & Run Guide (Anaconda Prompt)

# Project Structure
```
Bot_Project:
├── main.py                  ← Streamlit app (the full RAG pipeline)
├── requirements.txt         ← All Python dependencies
├── .env                     ← Your OpenAI API key (never commit this!)
└── faiss_store_openai.pkl   ← Auto-generated after first URL processing


# Step-by-Step Setup in Anaconda Prompt

### Step 1 — Create a Conda Virtual Environment
```bash
conda create -n rockybot python=3.10 -y
conda activate rockybot
```
> Always use a dedicated environment to avoid package conflicts.

---

### Step 2 — Navigate to Your Project Folder
```bash
cd path\to\Bot_Project
# Example: cd C:\Users\YourName\Desktop\RockyBot_Project
```

---

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

>  **Windows Users**: If you get an error about `python-magic`, also run:
> ```bash
> pip install python-magic-bin==0.4.14
> ```

---

### Step 4 — Add Your OpenAI API Key
Open the `.env` file in any text editor and replace the placeholder:
```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```
> Get your key from: https://platform.openai.com/api-keys

---

### Step 5 — Run the App
```bash
streamlit run main.py
```
The app opens automatically at **http://localhost:8501** in your browser.

---

##  How to Use the App

1. **Paste up to 3 news article URLs** in the left sideba

- Indian Stock Market (Moneycontrol)
- `https://www.moneycontrol.com/news/business/stocks/buy-tata-motors-target-of-rs-743-kr-choksey-11080811.html`
- `https://www.moneycontrol.com/news/business/tata-motors-mahindra-gain-certificates-for-production-linked-payouts-11281691.html`
- `https://www.moneycontrol.com/news/business/tata-motors-launches-punch-icng-price-starts-at-rs-7-1-lakh-11098751.html`

- NVIDIA / US Tech Stocks (GuruFocus / Reuters)
- `https://www.reuters.com/technology/nvidia-results-beat-expectations-2023-08-23/`
- `https://finance.yahoo.com/news/nvidia-stock-analysis-2024/`

- General Finance (Good for testing)
- `https://www.bbc.com/news/business`
- `https://economictimes.indiatimes.com/markets/stocks/news`


2. **Click "Process URLs"** — this will:

   - Load article content using LangChain's UnstructuredURLLoader
   - Split text into 1000-character chunks
   - Generate OpenAI embeddings for each chunk
   - Store them in a FAISS vector index (saved as `faiss_store_openai.pkl`)


3. **Type your question** in the main area

   - Example: *"What is the price of Tiago iCNG?"*
   - The RAG chain retrieves relevant chunks and generates an answer with source links

---

##  RAG Pipeline Architecture

```
User Question
     │
     ▼
[OpenAI Embeddings]  ←── Question → vector
     │
     ▼
[FAISS Vector Store] ←── Similarity search → top-k chunks
     │
     ▼
[RetrievalQAWithSourcesChain]
     │  (chunks + question → LLM prompt)
     ▼
[OpenAI LLM (gpt-3.5-turbo)]
     │
     ▼
Answer + Source URLs
```

---

##  Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: langchain` | Run `pip install -r requirements.txt` again |
| `AuthenticationError: OpenAI` | Check your `.env` file has a valid API key |
| `python-magic` error on Windows | Run `pip install python-magic-bin==0.4.14` |
| Streamlit not found | Run `pip install streamlit==1.22.0` |
| App runs but no answer | Make sure you clicked "Process URLs" before asking |

---

##  Tips
- The `faiss_store_openai.pkl` file persists between runs — you don't need to re-process URLs every time.
- Delete `faiss_store_openai.pkl` to reset and load fresh articles.
- You can swap OpenAI for a free model (HuggingFace) by replacing `OpenAIEmbeddings` with `HuggingFaceEmbeddings`.
