# Facbot
FacBot is a faculty research helper that can find relevant information and summarize research papers for exploring the faculty of UIUC.
Although the app is specialized for this purpose and has UIUC-specific data in DB, it can also perform general searching and scraping as it has access to the internet. 

This LangChain-based agent has access to the following tools : 

- SERP API — Google Search results for a given query and get URLs and context from different websites.
- Browserless — Scrape the provided URL from the SERP API to get more information from any webpage.
- Arxiv — Access recent publications data via Arxiv API and even get a summary of research papers.
- GPT3 summarizer — Summarize data from any of the above using GPT3.


# How to set up? 

1. Clone the repo to your local 
```git clone https://github.com/Trevahok/facboy
cd facboy/
touch .env
```

2. Add the following keys to your newly created `.env` : 
```
BROWSERLESS_API_KEY=
SERP_API_KEY=
OPENAI_API_KEY=
PINECONE_API_KEY=
PINECONE_ENV=< SOMETHING LIKE "asia-northeast1-gcp" >
PINECONE_INDEX=
```
```
python3 -m venv venv
source venv/bin/activate # if you're on linux/mac
pip3 install -r requirements.txt
```

3. Populate the PineCone DB with scraped data for later retrieval: 
`python3 ingestion.py` 

4. Run the streamlit UI:
   
`streamlit run app.py`


## Find how it works here: 

https://medium.com/@trevahok/save-hundreds-of-hours-using-gpt4-langchain-agent-for-exploration-4e97dfc5f94d
