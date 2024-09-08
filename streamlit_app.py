import os
import streamlit as st
from query import query
import base64
import time

PATH_TO_DOCS = "docs/osp_docs"


def displayPDF(file : str, page=1):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}#page={page+2}" width="650" height="400" type="application/pdf"></iframe>'

    st.markdown(pdf_display, unsafe_allow_html=True)

def generate_response(input_text):
    collection_name = "osp_initial_docs_V1"
    response = query(input_text, collection_name=collection_name)
    return response



st.title("OSP Chatbot V1")

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    if st.secrets["openai_api_key"]:
        openai_api_key = st.secrets["openai_api_key"]
    os.environ["OPENAI_API_KEY"] = openai_api_key
    st.info("This is a prototype of the OSP Chatbot V1")
    


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = generate_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response["response"]})
    st.chat_message("assistant").write(response["response"])
    with st.expander("Context"):

        for i, (page, source) in enumerate(zip(response["page_nums"], response["sources"])):
            #st.write(response["display_texts"][i] + f" (Page) {page}")
            st.write(f"#### Source {i + 1}: \n{source} p. {page + 1}")
            displayPDF(os.path.join(PATH_TO_DOCS, source), page=page + 1)
#
# with st.form("my_form"):
#     text = st.text_area("Enter text:", "What does the OSP do?")
#     submitted = st.form_submit_button("Submit")
#     if submitted:
#         generate_response(text)
