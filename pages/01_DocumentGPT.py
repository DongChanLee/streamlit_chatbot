from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationSummaryBufferMemory
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


def callback_llm(chat_callback: bool):
    if chat_callback:
        callbacks = [ChatCallbackHandler()]
    else:
        callbacks = []

    llm = ChatOpenAI(
        model="gpt-3.5-turbo-1106",
        temperature=0.1,
        streaming=True,
        callbacks=callbacks,
    )

    return llm


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


memory = ConversationSummaryBufferMemory(
    llm=callback_llm(chat_callback=False),
    max_token_limit=120,
    return_messages=True,
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


# input을 받아줘야하기 때문에 무시하기 위해서 _라는 argument 입력
def load_memory(_):
    return memory.load_memory_variables({})["history"]


st.title("DocumentGPT")

st.markdown(
    """
안녕하세요.

GPT에게 당신이 업로드한 파일에 대한 질문을 해주세요.

파일 업로드는 사이드바를 통해 가능합니다.
"""
)

with st.sidebar:
    file = st.file_uploader(
        "문서를 업로드해 주세요. (.pdf, .txt, .docx)",
        type=["pdf", "txt", "docx"],
    )

if file:
    retriever = embed_file(file)
    send_message("준비됐습니다. 질문을 해주세요!", "ai", save=False)
    paint_history()
    message = st.chat_input("업로드한 파일과 관련하여 질문을 해주세요.")
    # 메시지 전용 llm 정의
    llm = callback_llm(chat_callback=True)
    
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "history": load_memory,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)
        
        memory.save_context(
            {"input": message},
            {"output": response.content},
        )


else:
    st.session_state["messages"] = []