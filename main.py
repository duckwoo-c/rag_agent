import os

## FASTAPI

from fastapi import FastAPI
from pydantic import BaseModel

## LANGCHAIN

from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings

from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict

from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from langgraph.prebuilt import tools_condition

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.managed.is_last_step import RemainingSteps

## 
from utils import embedding_model, vectorstore_to_retriever_tool, create_llm

IS_LOCAL = False

if IS_LOCAL==False:
    ## Chroma
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    ## Environment parameter
    AZURE_API_INFO = os.getenv("AZURE_API_INFO")
    print(f"Api info : {AZURE_API_INFO}")
    MODEL_PARAMETER = os.getenv("MODEL_PARAMETER")
    print(f"Model Para : {MODEL_PARAMETER}")

if IS_LOCAL==True:
    from dev.config import AZURE_API_INFO, MODEL_PARAMETER

azure_embeddings = embedding_model(AZURE_API_INFO)

kt_glossary_vector = Chroma(collection_name="kt_glossary", persist_directory="./vectorstore/kt_glossary", embedding_function=azure_embeddings)
kt_glossary_tool = vectorstore_to_retriever_tool(kt_glossary_vector, "kt_glossary_rag_tool", "search meaning of specific word, from a glossary containing words that are used in telecommunication industries.")

mobile_manual_vector = Chroma(collection_name="mobile_manual", persist_directory="./vectorstore/mobile_manual", embedding_function=azure_embeddings)
mobile_manual_tool = vectorstore_to_retriever_tool(mobile_manual_vector, "mobile_manual_rag_tool", "search data from user manual pdf files of mobile devices")

retriever_tools = [kt_glossary_tool, mobile_manual_tool]
azure_gpt4o_mini = create_llm(AZURE_API_INFO, MODEL_PARAMETER)

class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]
    remaining_steps: RemainingSteps


def grade_documents(state) -> Literal["generate", "nodocument_end"]:
    """
    Determines whether the retrieved documents are relevant to the question.
    Args:
        state (messages): The current state
    Returns:
        str: A decision for whether the documents are relevant or not
    """
    
    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = azure_gpt4o_mini

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"
    
    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "nodocument_end"

### Nodes

def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]
 
    model = azure_gpt4o_mini
    model_with_tools = model.bind_tools(retriever_tools)
    response = model_with_tools.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n 
    Formulate an improved question: """,
        )
    ]

    # Grader
    model = azure_gpt4o_mini
    response = model.invoke(msg)
    return {"messages": [response]}


def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    ### Default Prompt
    #prompt = hub.pull("rlm/rag-prompt")
    
    prompt = PromptTemplate(
        template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Give answer in Korean language.\nQuestion: {question}\nContext: {context}\nAnswer:",
        input_variables=["question", "context"]
    )
    # LLM

    model = azure_gpt4o_mini
    
    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | model | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}

def recursive_end(state):
    return {"messages":[BaseMessage(type="ErrorMessage",content="Recursive Limit Exceeded")]}
def nodocument_end(state):
    return {"messages":[BaseMessage(type="ErrorMessage",content="답변을 찾을 수 없습니다.")]}


# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
retrieve = ToolNode(retriever_tools)
workflow.add_node("retrieve", retrieve)  # retrieval
workflow.add_node("nodocument_end", nodocument_end)
#workflow.add_node("rewrite", rewrite)  # Re-writing the question
workflow.add_node(
    "generate", generate
)  # Generating a response after we know the documents are relevant
# Call agent node to decide to retrieve or not
workflow.add_node("recursive_end", recursive_end)
workflow.add_edge(START, "agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate", END)
#workflow.add_edge("rewrite", "agent")
workflow.add_edge("nodocument_end", END)
workflow.add_edge("recursive_end", END)

# Compile
graph = workflow.compile()


import pprint

app = FastAPI()

# Define a Pydantic model for the request body
class InputData(BaseModel):
    input: str

@app.get("/test")
def test_output():
    return "hello world"

# Define a POST route to accept JSON data
@app.post("/process-data")
def process_data(data: InputData):
    processed_input ={
        "messages": [
            ("user", data.input)
        ]
    }
    output_list = []
    output_str = ""
    for output in graph.stream(processed_input):
        for key, value in output.items():
            pprint.pprint(f"Output from node '{key}':")
            pprint.pprint("---")
            pprint.pprint(value, indent=2, width=80, depth=None)
            output_list.append([key, value])
            output_str += str(value)
        pprint.pprint("\n---\n")
    final_node = output_list[-1][0]
    if final_node=="generate":
        final_response=output_list[-1][1]["messages"][0]
    else:
        final_response=output_list[-1][1]["messages"][0].content
    return {
        "received_input": data.input,
        "final_node": final_node,
        "final_response": final_response,
        "total_output": output_str,
        "message": "Data processed successfully"
    }
