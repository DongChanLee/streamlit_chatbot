import json
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser, output_parser


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")


function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}


llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4-turbo",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
).bind(
    function_call={
        "name": "create_quiz",
    },
    functions=[
        function,
    ],
)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
        """
        You are a helpful assistant that is role playing as a teacher.
        Make 10 (TEN) questions to test the user's knowledge about the text and make a quiz that corresponds to the following context in Korean.
        Please keep all questions short and unique.
         
        Context: {context}
        """
        ),
    ]
)


@st.cache_resource(show_spinner="파일 업로드 중...")
def split_file(file):
    file_content = file.read()
    file_path = f"./cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_resource(show_spinner="문제 생성 중...")
def run_quiz_chain(_docs, topic):
    chain = {"context": format_docs} | prompt | llm
    response = chain.invoke(_docs)
    response = response.additional_kwargs["function_call"]["arguments"]
    print(response)
    response = json.loads(response)
    return response


@st.cache_resource(show_spinner="위키피디아 검색 중...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5, lang="ko")
    docs = retriever.get_relevant_documents(term)
    return docs


with st.sidebar:
    docs = None
    topic = None
    choice = st.selectbox(
        "어떤 정보를 사용 하실지 선택해 주세요.",
        (
            "파일",
            "위키피디아",
        ),
    )
    if choice == "파일":
        file = st.file_uploader(
            "문서를 업로드해 주세요. (.pdf, .txt, .docx)",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("위키피디아에서 검색...", placeholder="검색할 내용을 입력해 주세요.")
        if topic:
            docs = wiki_search(topic)


if not docs:
    st.markdown(
        """
    QuizGPT에 오신 것을 환영합니다.

    이 GPT는 위키피디아의 자료나 당신이 업로드한 파일을 이용해서 당신의 학습을 도울 것입니다.
                
    사이드바에서 위키피디아에 특정 주제를 검색하거나 파일을 직접 업로드하여 퀴즈 학습을 시작해보세요.
    """
    )
else:
    response = run_quiz_chain(docs, topic if topic else file.name)
    with st.form("questions_form"):
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio(
                "아래에서 한 개를 선택하세요.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("✅ 정답입니다!")
            elif value is not None:
                st.error("❌ 오답입니다!")
        button = st.form_submit_button()