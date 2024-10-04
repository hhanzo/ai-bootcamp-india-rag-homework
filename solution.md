# RAG Pipeline Experiments
### Simple RAG Pipeline

Built a simple RAG pipeline. Experimented with the "k" (different documents retrieved) for the context. 
Implementation in the file **rag_pipeline.py**
The RAGAS score for different "k" documents retrieved are as follows:
| k   | faithfulness | answer_relevancy | context_recall | context_precision | answer_correctness | ragas_score |
| ---: |:------------:| :----------------: |:--------------:|:-----------------:| :------------------:| :-----------:|
|  3   |      0.88       |     0.888         |      0.825           |        0.855           |        0.676          |      0.825       |
|  4  |     0.904         |      0.867            |       0.871         |      0.849             |       0.718             |      0.842       |
|   5  |      0.915        |        0.834          |        0.882        |         0.804          |       0.663             |     0.82        |
|   6  |      0.94        |         0.864         |      0.91          |       0.831            |       0.69             |     0.847        |
|  7   |     0.878         |       0.853           |      0.828          |      0.787             |        0.681            |     0.805        |
|  8  |     0.955         |         0.847        |      0.937         |      0.858          |      0.701              |      0.86     |
|  9  |     0.93        |         0.89        |      0.925          |       0.812            |      0.69              |      0.849       |


### Experimenting with Ensemble Retrievers

Experimented with ensemble retrievers in the pipeline. Included the BM25 retriever and the vectorstore retriever based on similarity with equal weights and varying "k" (the number of retrieved documents). There are two "k"'s in the follwing case:
k1 - No of retrieved documents by the vectorstore retriever
k2 - No of retrieved documents by the BM25 retriever
Implementation can be found in **rag_pipeline_ensemble_retriever.py**
The RAGAS Score for different k1,k2 as follows:

| k1,k2   | faithfulness | answer_relevancy | context_recall | context_precision | answer_correctness | ragas_score |
| ---: |:------------:| :----------------: |:--------------:|:-----------------:| :------------------:| :-----------:|
|  2,2   |     0.899         |      0.84            |     0.831           |          0.873         |        0.69            |     0.827        |
| 3,3    |    0.919          |       0.868           |      0.876          |        0.712               |         0.747           |   0.824      |

### RAG Pipeline with Multi-Query

Created a RAG pipeline by breaking down a query into multi-queries. 
For each question, 5 different variants of the questions are generated. 
Contexts for each of the questions are retrieved. 
Unique contexts are retrieved from all the contexts for all the questions. 
Used Groq-llama3-8b-8192 model for generating the variants of the question. 
The implementation can be found in the **rag_multi_query.py**. 
It also takes relatively longer for this method to generate answers.
I'm still experimenting with results. 

### Conclusion
After all the above experiments, the best **ragas_score = 0.86** is achieved in simple rag pipeline and using the number of retrieved documents for the context **k=8**.
