import os
import streamlit as st
from langchain_community.llms import OpenAI
from query import query
import base64

PATH_TO_DOCS = "/Users/jakesimmons/repos/Langchain-RAG/docs/osp_docs"


def displayPDF(file : str, page=1):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}#page={page}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def generate_response(input_text):
    collection_name = "osp_initial_docs_V1"
    response, context, sources = query(input_text, collection_name=collection_name)
    #st.info(response)
    # with st.expander("Context"):
    #     st.write(context)
    return response, context



st.title("OSP Chatbot V1")
#displayPDF("/Users/jakesimmons/repos/Langchain-RAG/docs/osp_docs/AU Principal Investigator Handbook (1).pdf")

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    st.info("This is a prototype of the OSP Chatbot V1")
    
    view_context = st.checkbox("View context")


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    reply, context = generate_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)
    with st.expander("Context"):
        st.write(context)

#
# with st.form("my_form"):
#     text = st.text_area("Enter text:", "What does the OSP do?")
#     submitted = st.form_submit_button("Submit")
#     if submitted:
#         generate_response(text)
