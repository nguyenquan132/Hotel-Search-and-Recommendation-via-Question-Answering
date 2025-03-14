from fastapi import FastAPI, status
from pydantic import BaseModel
from pinecone import Pinecone as PineconeClient
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.embeddings import GPT4AllEmbeddings
from langgraph.graph import START, StateGraph
from langchain_core.prompts import PromptTemplate
import re, json, os
from typing_extensions import List, TypedDict
from langchain_core.documents import Document

# Load env gồm các API_KEY
load_dotenv()

app = FastAPI()

# Initilize llm and vector database 
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = PineconeClient(
    api_key=PINECONE_API_KEY,
)
index_name = "langchainretrieval"
index = pc.Index(index_name)

embedding = GPT4AllEmbeddings()
vector_store = PineconeVectorStore(embedding=embedding, index_name=index_name)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-pro-exp-02-05",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1
)
meta = ''

class QuestionRequest(BaseModel):
    question: str

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    meta: dict

def retrieve(state: State):
    prompt_template = PromptTemplate(
            input_variable=["question"],
            template="""Tìm các thông tin location, rating, hotel_name (nếu có) trong câu: {question}. 
            Nếu không có thông tin thì loại bỏ biến đó khỏi kết quả.
            Trả về JSON chỉ chứa các key có giá trị, theo format sau:
            {{"location": "string", "rating": int, "hotel_name": "string"}}"""
    )
    filter = llm.invoke(prompt_template.invoke({"question": state['question']}))
    meta = re.sub(r'```json\n|```', '', filter.content).strip()
    meta = json.loads(meta)
    retrieved_docs = vector_store.similarity_search(state['question'], 
                                                    k=30,
                                                    filter=meta)
    return {"context": retrieved_docs, "meta": meta}

def generate(state: State):
    meta = state["meta"] if "meta" in state else {}
    if not state['context']:
        return {"answer": f"Xin lỗi, hệ thống không tìm thấy khách sạn nào ở {meta.get('location', '')} với rating {meta.get('rating', '')}. Vui lòng chọn lại rating khác."}
    docs_content = "\n\n".join(f"{doc.page_content}\nmetadata: {doc.metadata}" for doc in state['context'])
    prompt_generation = PromptTemplate(
        input_variables=['context'],
        template="""- Tóm tắt thông tin của tất cả khách sạn trong {context}, sử dụng dữ liệu từ metadata để bổ sung thông tin nhưng không in trực tiếp metadata.
        Không cần in ra ID của khách sạn. Nếu có các khách sạn trùng ID trong context thì có thể nhóm vào 1 khách sạn (vì cùng 1 khách sạn) in ra với các thông tin
        'Địa chỉ, Mô tả, Đánh giá (rating), URL_khách sạn.'
        Ngoài ra, nếu nhiều hơn 1 khách sạn thì câu đầu tiên nên ghi là 'Dưới đây là thông tin một số khách sạn ở location với rating' (map location và rating tương ứng ở metadata).
        Còn hỏi khách sạn cụ thể thì câu đầu nên ghi là 'Dưới đây là thông tin khách sạn hotel_name' (map hotel_name ở metadata)."""
    )
    messages = prompt_generation.invoke({"context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Create Control Flow
def control_flow():
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    return graph

graph = control_flow()
@app.get("/")
async def root():
    return {"message": "Hello"}

@app.post('/predict', status_code=status.HTTP_200_OK)
def get_response(request: QuestionRequest):
    results = graph.invoke({"question": request.question})
    
    return {"answer": results['answer'], "status_code": 200}