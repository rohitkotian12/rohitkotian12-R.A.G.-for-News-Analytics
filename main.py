import os
import streamlit as st
import pickle
import time
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document                        # FIX: was langchain.schema
from langchain_text_splitters import RecursiveCharacterTextSplitter  # FIX: was langchain.text_splitter
from dotenv import load_dotenv

load_dotenv()

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Bot: News Research Tool", page_icon="🤖", layout="wide")
st.title("🤖 Bot: News Research Tool 📈")
st.sidebar.title("📰 News Article URLs")

# ── Sidebar: URL Input ────────────────────────────────────────────────────────
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", placeholder=f"Paste article URL {i+1} here")
    urls.append(url)

process_url_clicked = st.sidebar.button("⚙️ Process URLs")
file_path = "faiss_store_gemini.pkl"
main_placeholder = st.empty()

# ── LLM Setup (Groq - Llama 3.3 70B) ──────────────────────────────────────────
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("❌ GROQ_API_KEY not found. Please add it to your .env file.")
    st.stop()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=api_key,
    temperature=0.9,
    max_retries=2
)

# ── News Loader ───────────────────────────────────────────────────────────────
def load_articles(url_list):
    from newspaper import Article
    docs = []
    for url in url_list:
        if not url.strip():
            continue
        try:
            article = Article(url)
            article.download()
            article.parse()
            if article.text.strip():
                docs.append(Document(
                    page_content=article.text,
                    metadata={"source": url}
                ))
                st.sidebar.success(f"✅ Loaded: {url[:50]}...")
            else:
                st.sidebar.warning(f"⚠️ Empty content: {url[:50]}...")
        except Exception as e:
            st.sidebar.error(f"❌ Failed: {url[:50]}...\n{e}")
    return docs

# ── Step 1: Process URLs ──────────────────────────────────────────────────────
if process_url_clicked:
    valid_urls = [u for u in urls if u.strip()]
    if not valid_urls:
        st.sidebar.error("Please enter at least one URL.")
    else:
        main_placeholder.text("📥 Loading articles...")
        data = load_articles(valid_urls)

        if not data:
            main_placeholder.error("❌ Could not extract text. Please use direct article URLs.")
            st.stop()

        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("✂️ Splitting text into chunks...")
        docs = text_splitter.split_documents(data)

        if not docs:
            main_placeholder.error("❌ No chunks produced. Try different URLs.")
            st.stop()

        main_placeholder.text("🔢 Building embeddings (this may take a moment)...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        time.sleep(1)

        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)

        main_placeholder.success(f"✅ Done! {len(data)} article(s) → {len(docs)} chunks indexed. Ask your question below.")

# ── Step 2: Q&A ───────────────────────────────────────────────────────────────
st.divider()
query = st.text_input("💬 Ask a question:", placeholder="e.g. What is the target price of Tata Motors?")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        relevant_docs = retriever.invoke(query)   # FIX: was get_relevant_documents() — deprecated

        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        prompt = f"""You are a helpful financial news analyst. Use the following news article excerpts to answer the question accurately and concisely.
If the answer is not in the provided context, say "I couldn't find this information in the loaded articles."

Context:
{context}

Question: {query}

Answer:"""

        with st.spinner("🤔 Thinking..."):
            try:
                response = llm.invoke(prompt)
                st.header("📋 Answer")
                st.write(response.content)

                st.subheader("🔗 Sources:")
                sources = set([doc.metadata["source"] for doc in relevant_docs])
                for source in sources:
                    st.markdown(f"- [{source}]({source})")
            except Exception as e:
                st.error(f"❌ LLM Error: {e}\n\nCheck your API key at https://aistudio.google.com/app/apikey")
    else:
        st.warning("⚠️ No articles loaded yet. Enter URLs in the sidebar and click 'Process URLs' first.")
