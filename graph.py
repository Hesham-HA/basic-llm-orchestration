from dotenv import load_dotenv
load_dotenv()

import asyncio
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage, HumanMessage
from crewai import Crew, Agent, Task
from textwrap import dedent
import json
import time

# create llm chain
openai_llm = ChatOpenAI(temperature=0.6, streaming=True)

# define the running langgraph state
class State(TypedDict):
    messages: Annotated[list, add_messages]
    email: str
    query: str
    category: str

# create short-term memory
checkpointer = InMemorySaver()

# create config to assign thread ids
config = {'configurable': {'thread_id': "session_1"}}

##### In this application we have 4 langgraph nodes #####
# 1. create an entry node
async def entry_node(state: State):
    t0 = time.perf_counter()
    input = state["query"]
    agent = await openai_llm.ainvoke(f"""
        User input
        ---
        {input}
        ---
        Your last response
        ---
        {state['messages'][-2] if len(state['messages']) > 1 else None}
        ---
        You have given one user input and you have to perform actions on it based on given instructions

        Categorize the user input in below categories
        email_query: If user wrote you an email and wants to get a reply
        weather_query: If user want any weather info (or alarms) about given location
        other: If it is any other query

        After categorizing your final RESPONSE must be in json format with these properties:
        category: category of user input
        email: If category is 'email_query' then extract the email body from user input with proper line breaks and add it here else keep it blank
        query: If category is 'weather_query' or 'other' then add the user's query here else keep it blank
    """, config
    )
    response = json.loads(agent.content)
    t1 = time.perf_counter()
    print(f"Time to generate response at entry node: {t1-t0} sec")
    return {'email': response["email"], 'query': response['query'] if 'query' in response else None, 'category': response['category']}
# 2. create a node for finding weather information
# # create an mcp client instance to retrieve available weather tools
client = MultiServerMCPClient(
    {
        'weather': {
            "url": 'http://localhost:9090/mcp',
            "transport": "streamable_http"
        }
    }
)
tools = asyncio.get_event_loop().run_until_complete(client.get_tools())
# # create an instance of the llm with tools binded
openai_with_tools = openai_llm.bind_tools(tools)
# # create a node to use the weather tools
async def weather_node(state: State):
    t0 = time.perf_counter()
    res = await openai_with_tools.ainvoke(
        state['messages'], config
    )
    t1 = time.perf_counter()
    print(f"Time to generate response at weather node: {t1-t0} sec")
    return {'messages': [res]}
# # create a tool node to use these tools
weather_as_node = ToolNode(tools, name="weather_tools")
# 3. create a node for Email reading/writing
# # create Email agents
# # # Email classifier agent
eclassifier = Agent(
    role="Email Classifier",
    goal="You will be given an email and you have to classify the given email in one of these 2 categories: 1) Important 2) Casual ",
    backstory="An email classifier who is expert in classifying every type of email and have classified so many emails so far",
    verbose=True,
    allow_delegation=False
)
# # # Email writer agent
ewriter = Agent(
    role="Email writing expert",
    goal="You are email writing assistant for Hesham Hassan. You will be given an email and a category of that email and your job is to write a reply for that email. If email category is 'Important' then write the reply in professional way and If email category is 'Casual' then write in a casual way",
    backstory="An email writer with an expertise in email writing for more than 10 years",
    verbose=True,
    allow_delegation=False
)
# # create tasks for Email agents
# # # Email classification task
def eclassification(agent, email):
    return Task(
        description=dedent(f"""
        You have given an email and you have to classify this email 
        {email}
        """),
        agent = agent,
        expected_output = "Email category as a string"
    )
# # # Email writing task
def ewriting(agent, email):
    return Task(
        description=dedent(f"""
        Create an email response to the given email based on the category provided by 'Email Classifier' Agent
        {email}
        """),
        agent = agent,
        expected_output = "A very concise response to the email based on the category provided by 'Email Classifier' Agent"
	)
# # create the Email crew
class EmailCrew:
    def __init__(self, email):
        self.email = email
    async def run(self):
        classificationTask = eclassification(eclassifier, self.email)
        writingTask = ewriting(ewriter, self.email)
        crew = Crew(
            agents=[eclassifier, ewriter],
            tasks=[classificationTask, writingTask],
            verbose=True
        )
        result = await crew.kickoff_async()
        return result
# # create a node for Email stuff
async def email_node(state: State) -> State:
    t0 = time.perf_counter()
    email = state["email"]
    emailcrew = EmailCrew(email)
    crewresults = await emailcrew.run()
    t1 = time.perf_counter()
    print(f"Time to generate response at email node: {t1-t0} sec")
    return {'messages': [AIMessage(content=crewresults.raw)]}
# 4. Create a node for typical llm replies
async def reply_node(state: State) -> State:
    t0 = time.perf_counter()
    query = state["query"]
    res = await openai_llm.ainvoke(
        f"""{query}""",config
    )
    t1 = time.perf_counter()
    print(f"Time to generate response at reply node: {t1-t0} sec")
    return {'messages': [res]}

# Create the langgraph's state graph
graph_builder = StateGraph(State)
# # Add all nodes
graph_builder.add_node('entry', entry_node)
graph_builder.add_node('weather', weather_node)
graph_builder.add_node('weather_tools', weather_as_node)
graph_builder.add_node('emails', email_node)
graph_builder.add_node('reply', reply_node)
# # Add functions to define conditional routing
def category_condition(state: State):
    cat = state['category']
    if cat == "email_query":
        return "to_emails"
    elif cat == "weather_query":
        return "to_weather"
    else:
        return "to_reply"
# # Add conditional edges
graph_builder.add_conditional_edges(
    'entry',
    category_condition,
    {
        'to_weather': 'weather',
        'to_emails': 'emails',
        'to_reply': 'reply'
    }
)
graph_builder.add_conditional_edges(
    'weather',
    tools_condition,
    {
        'tools': 'weather_tools',
        '__end__': END
    }
)
# # Add normal edges
graph_builder.add_edge('weather_tools', 'weather')
graph_builder.add_edge('emails', END)
graph_builder.add_edge('reply', END)
# # set entry point
graph_builder.set_entry_point('entry')
# # compile the graph against the checkpointer
graph = graph_builder.compile(checkpointer)

# define a function to stream graph updates (responses)
async def stream_graph_updates(input_state: State):
    started = False
    async for chunk, metadata in graph.astream(input_state, config, stream_mode="messages"):
        if metadata["langgraph_node"] in ['reply', 'emails', 'weather']:
            if not started:
                print("Assistant: ", end='', flush=True)
                started = True
            if hasattr(chunk, 'content') and chunk.content:
                # Stream the content character by character
                for char in chunk.content:
                    print(char, end='', flush=True)
                    await asyncio.sleep(0.05)
    print('\n')

# draw the graph
try:
    with open('graph.png', '+wb') as file:
        file.write(graph.get_graph().draw_mermaid_png())
except Exception as e:
    print(f"Faild to draw langgraph graph due to exception: {e}")

# run the agent
if __name__ == '__main__':
    while True:
        t0 = time.perf_counter()
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        asyncio.get_event_loop().run_until_complete(stream_graph_updates({"query": user_input, "messages": [HumanMessage(content=user_input)]}))
        t1 = time.perf_counter()
        print(f"Assistant took {t1-t0} seconds to respond!")