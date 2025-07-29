# AI-Powered-Reasearch-Tool

AI-POWERED REASEARCH TOOL 📈:

It is a user-friendly news research tool designed for effortless information retrieval. Users can input article URLs and ask questions to receive relevant insights from the stock market and financial domain.

Features:

1.Load URLs or upload text files containing URLs to fetch article content. 2.Process article content through LangChain's UnstructuredURL Loader 3.Construct an embedding vector using OpenAI's embeddings and leverage FAISS, a powerful similarity search library, to enable swift and effective retrieval of relevant information 4.Interact with the LLM's (Chatgpt) by inputting queries and receiving answers along with source URLs.

Project Structure:

1 main.py: The main Streamlit application script. 2 requirements.txt: A list of required Python packages for the project. 3 faiss_store_openai.pkl: A pickle file to store the FAISS index. 4 .env: Configuration file for storing your Gemini API key.
