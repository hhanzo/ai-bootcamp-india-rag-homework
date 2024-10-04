import os
import getpass
import json
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads

# Environment keys
os.environ["OPENAI_API_KEY"] = getpass.getpass()
os.environ["GROQ_API_KEY"] = getpass.getpass()


# Function for unique document extraction
def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]



# Setup LLMs
llm_sub_questions = ChatGroq(model="llama3-8b-8192", temperature=0)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Load and split documents
loader = TextLoader('./docs/intro-to-llms-karpathy.txt')
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# Store embeddings and create retrievers
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})


# Multi Query: Different Perspectives
template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_perspectives 
    | llm_sub_questions 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

# Define a function to handle retrieval and chain setup
def create_final_rag_chain(question):
    retrieved_docs=[]
    queries = generate_queries.invoke({"question": question})
    for query in queries:
        retrieved_doc = retriever.invoke(query)
        retrieved_docs.append(retrieved_doc)
    unique_docs = get_unique_union(retrieved_docs)
    
    # Generate answers using the context from retrieved documents
    context_text = "\n\n".join([doc.page_content for doc in unique_docs])

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{context_text}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", question),
        ]
    )
    rag_chain = (prompt|llm|StrOutputParser())
    response = rag_chain.invoke({"input":question})
    #print(response)
    return {
        "question": question,
        "answer": response,
        "contexts": [doc.page_content for doc in unique_docs]
    }

# Load questions from JSON
file_path = './questions.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Extract questions and process them
test_questions = [item['question'] for item in data]
json_results = []

for question in test_questions:
    print(question)
    response = create_final_rag_chain(question)
    json_results.append(response)


# Write results to file
with open('./results/my_rag_output_mq5.json', mode='w', encoding='utf-8') as f:
    f.write(json.dumps(json_results, indent=4))

