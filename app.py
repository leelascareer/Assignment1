from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


load_dotenv()

# Initialize the OpenAI client
llm=ChatOpenAI(model="gpt-4.1-nano", seed=6)

#String input to the model

resp1 = llm.invoke("We are building an AI system for processing medical insurance claims.")
resp2 = llm.invoke("What are the main risks in this system?")
print("Response from OpenAI:")
print("--------Response to String Input--------")
print("Response1:", resp1.content)
print("Response2:", resp2.content)


# Message input to the model
messages = [
    SystemMessage(content="You are a senior AI architect reviewing production systems.."),
    HumanMessage(content="We are building an AI system for processing medical insurance claims.") ,
    HumanMessage(content="What are the main risks in this system?")
]
response = llm.invoke(messages)
print("--------Response to MESSAGE Input--------")
print("Response1:",response.content)

"""
Reflection:

1. Why did string-based invocation fail?
The string-based invocation failed because the model does not retain the context of previous interactions. 
So the model does not know what to refer to when it recieves the second input . 
It does not know what "this system" is.

2. Why does message-based invocation work?
LangChain Messages retain state and converstaion history. This allows the model to understand the context of the current user query. It now knows that "this system" referes to the medical insurance claims processing system mentioned in the first message.
Also there it supports clear role based distinction of what is the system message, user message and AI message which is the model's response.

3. What would break in a production AI system if we ignore message history?
Ignoring message history could lead to confusion, inconsistent responses, and a poor user experience.
The model may end up hallucinating, giving incorrect or irrelevant responses.
This is a bad user expereince leading to lack of trust in the model.
"""
