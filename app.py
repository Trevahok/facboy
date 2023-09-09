from dotenv import load_dotenv
import streamlit as st
from src.agent import agent

load_dotenv()


st.set_page_config(page_title="ActualAppleGenius", page_icon=":apple:")

st.header("FacBoy - Your faculty research helper :teacher:")
query = st.text_input("What do you want me to find out?")

if query:
    st.write("Doing research for: ", query)

    result = agent({"input": query})

    st.info(result['output'])

