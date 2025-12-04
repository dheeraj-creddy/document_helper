from typing import Set

from backend.core import run_llm

import streamlit as st

st.header("LG Helper Bot")

prompt = st.text_input("Prompt", placeholder="Enter your prompt here..")

if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []


#takes list of URLs and prints it very nicely with numbers and nice format
def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"- {source}\n"
    return sources_string


if prompt:
    with st.spinner("Generating response.."):
        generated_response = run_llm(query=prompt, chat_history=st.session_state["chat_history"])
        sources = set(
            [doc.metadata["source"] for doc in generated_response["source_documents"]]
        )

        formatted_response = (
            f"{generated_response['result']} \n\n {create_sources_string(sources)}"
        )

        #store previous chats
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        #memory for LLM
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["result"]))

if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(st.session_state["chat_answers_history"],st.session_state["user_prompt_history"],):
        #display previous chats
        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(generated_response)

#ok so now with session states we are storing the entire history of chats and printing on the screen
#chat message user and assistant give the logos lol with name of the author as 1st param "user" "assistant" "ai" "human"
#write method reveives the text and displays it!
# but how does the LLM know this chat history => memory