import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()

st.set_page_config(page_title="Agente IA - Phi-4 Mini", page_icon="🤖")
st.title("Agente Conversacional (Phi-4-mini-instruct)")

azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "phi-4-mini-instruct")

history = StreamlitChatMessageHistory(key="chat_messages")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente virtual prestativo, claro e direto nas respostas."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

if azure_endpoint and azure_key:
    llm = ChatOpenAI(
        base_url=azure_endpoint,
        api_key=azure_key,
        model=deployment_name,
        temperature=0.7
    )

    chain = prompt | llm

    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: history,
        input_messages_key="input",
        history_messages_key="history"
    )

    for msg in history.messages:
        st.chat_message(msg.type).write(msg.content)

    if user_input := st.chat_input("Digite sua mensagem..."):
        st.chat_message("human").write(user_input)
        
        response = chain_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "default_session"}}
        )
        
        st.chat_message("ai").write(response.content)
else:
    st.error("Configuração ausente! Defina as variáveis AZURE_OPENAI_ENDPOINT e AZURE_OPENAI_API_KEY.")