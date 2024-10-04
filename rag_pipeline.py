import os
import getpass
import openai
import json
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever



os.environ["OPENAI_API_KEY"] = getpass.getpass()
llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0)


#indexing: Load
loader = TextLoader('./docs/intro-to-llms-karpathy.txt')
docs = loader.load()


#indexing: Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)


#indexing: Store
#Setting the context k 
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 9})


# Generation
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

#create the rag chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


json_results = []
#load json
file_path = './questions.json'
with open(file_path, 'r') as file:
        data = json.load(file)
#extract questions from the file
test_questions = [item['question'] for item in data]
#for each generate the answer using the context
for question in test_questions:
  response = rag_chain.invoke({"input" : question})
  json_results.append( {
    "question" : question,
    "answer" : response["answer"],
    "contexts" : [context.page_content for context in response["context"]]
} )

with open('./results/my_rag_output.json', mode='w', encoding='utf-8') as f:
  f.write( json.dumps(json_results, indent=4) )

