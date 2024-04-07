import os
import streamlit as st
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

import operator
from typing import Annotated, Sequence, TypedDict, List
from langchain_community.document_loaders import WebBaseLoader
# Import StateGraph from langgraph.graph
from langgraph.graph import StateGraph, END
import streamlit_shadcn_ui as ui
from streamlit.components.v1 import html
import  clipboard



def main():
    footer="""<style>
        a:link , a:visited{
        color: blue;
        background-color: transparent;
        text-decoration: underline;
        }

        a:hover,  a:active {
        color: red;
        background-color: transparent;
        text-decoration: underline;
        }

        .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: black;
        text-align: center;
        }
        </style>
        <div class="footer">
        <p>Developed with ❤ by <a style='display: block; text-align: center;' href="https://twitter.com/tuantruong/" target="_blank">Tuan Truong</a></p>
        </div>
        """
    with open( "style.css" ) as css:
        st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
   
    st.title("Tweet Generator  🐦")
    st.markdown("***")
    st.markdown("#Workflow: Keyword Extractor --> Trend Analyzer --> Engagement Optimizer")
    st.markdown("***")
    # Add a sidebar for model selection
    OPENAI_MODEL = st.sidebar.selectbox(
        "Select Model",
        ["gpt-4-turbo-preview", "gpt-3.5-turbo"]
    )
  
    api_key = ""
    if 'api_key' not in st.session_state:
        api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
        st.session_state['api_key'] = api_key
    else:
        api_key = st.session_state['api_key']
        st.session_state['api_key'] = st.sidebar.text_input("Enter your OpenAI API Key", type="password", value=api_key)
    
    if st.session_state['api_key']:
        os.environ["OPENAI_API_KEY"] = st.session_state['api_key']

    tweet_topic = st.text_input("Enter the topic for the tweet:")
    with st.sidebar:
        st.markdown("---")
        st.markdown(
            '<h5>Made with ❤ in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://twitter.com/tuantruong">@tuantruong</a></h5>',
            unsafe_allow_html=True,
        )
  
    st.markdown("")
    trigger_btn = ui.button(text="Generate Tweet", key="trigger_btn")
    # Run the workflow
    #if st.button("Generate Tweet"):
    if trigger_btn:
        with st.spinner("Generating Tweet..."):
            #for chundk tweet
            def chunk_content_for_tweets(content):
                max_length = 280
                words = content.split()
                tweets = []
                current_tweet = ""
                
                for word in words:
                    # Check if adding the next word exceeds the max length
                    if len(current_tweet + " " + word) <= max_length:
                        current_tweet += " " + word
                    else:
                        # If the current tweet plus the new word exceeds the limit, save the current tweet and start a new one
                        tweets.append(current_tweet.strip())
                        current_tweet = word
                # Make sure to add the last tweet if it's not empty
                if current_tweet:
                    tweets.append(current_tweet.strip())
                    
                return tweets
            
            #for return tweet
            def display_tweets(tweets):
                if len(tweets) == 1:
                    st.markdown("### Tweet:")
                else:
                    st.markdown("### Tweet Thread:")

                for i, tweet in enumerate(tweets, start=1):
                    st.text_area(f"Tweet {i}", tweet, height=75)

                st.markdown("Copy and paste these into your Twitter feed!")
            
            llm = ChatOpenAI(model=OPENAI_MODEL,openai_api_key=api_key )

            def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="messages"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ])
                agent = create_openai_tools_agent(llm, tools, prompt)
                return AgentExecutor(agent=agent, tools=tools)
            
            @tool("Keyword_Extractor")
            def extract_keywords(content: str) -> str:
                """Extract relevant keywords from the given content."""
                chat = ChatOpenAI(openai_api_key=api_key)
                messages = [
                    SystemMessage(
                        content="You are a keyword extraction expert. Your task is to identify the most relevant keywords from the given content.Make sure that you come back with the list of quality keywords, present the result with bullet points."
                    ),
                    HumanMessage(
                        content=content
                    ),
                ]
                response = chat(messages)
                return response.content

            @tool("Trend_Analyzer")
            def trend_analyzer(topic: str) -> str:
                """Analyze current trends related to the topic to inspire the tweet content. Making sure you have the list of the ponts and present with bullet points."""
                chat = ChatOpenAI(openai_api_key=api_key)               
                messages = [
                    SystemMessage(content="Analyze current trends related to the topic."),
                    HumanMessage(content=topic),
                ]
                response = chat(messages)
                return response.content

            @tool("Engagement_Optimizer")
            def engagement_optimizer(content: str) -> str:
                """Optimize the tweet for maximum engagement based on content trends and best practices. It should be human-friendly and engaging. Make sure you present the tweet in a way that it is engaging and human-friendly, casual  intriguing and engaging."""
                chat = ChatOpenAI(openai_api_key=api_key)
                messages = [
                    SystemMessage(content="Optimize the tweet for engagement."),
                    HumanMessage(content=content),
                ]
                response = chat(messages)
                return response.content

            def keyword_extractor_agent() -> Runnable:
                prompt = (
                    "You are a keyword extraction agent."
                )
                return create_agent(llm, [extract_keywords], prompt)
            
            def trend_analyzer_agent() -> Runnable:
                prompt = "You are responsible for analyzing current trends related to the topic."
                return create_agent(llm, [trend_analyzer], prompt)

            def engagement_optimizer_agent() -> Runnable:
                prompt = "You are responsible for optimizing the tweet for engagement."
                return create_agent(llm, [engagement_optimizer], prompt)

            KEYWORD_EXTRACTOR = "Keyword_Extractor"
            TREND_ANALYZER = "Trend_Analyzer"
            ENGAGEMENT_OPTIMIZER = "Engagement_Optimizer"
            SUPERVISOR = "Supervisor"

            agents = [KEYWORD_EXTRACTOR,TREND_ANALYZER, ENGAGEMENT_OPTIMIZER]

            # Modify the AgentState definition to include a list or indicator of completed steps
            class AgentState(TypedDict):
                messages: Annotated[Sequence[HumanMessage], operator.add]
                next: str
                completed_steps: List[str]  # Add this line to keep track of completed steps
            
    
                
            def keyword_extractor_node(state: AgentState) -> dict:
                result = keyword_extractor_agent().invoke(state)
                # Ensure completed_steps is a list
                completed_steps = state.get('completed_steps') or []
                completed_steps.append(KEYWORD_EXTRACTOR)
                return {"messages": [HumanMessage(content=result["output"], name=KEYWORD_EXTRACTOR)], "completed_steps": completed_steps}

            def trend_analyzer_node(state: AgentState) -> dict:
                result = trend_analyzer_agent().invoke(state)
                # Ensure completed_steps is a list
                completed_steps = state.get('completed_steps') or []
                completed_steps.append(TREND_ANALYZER)
                return {"messages": [HumanMessage(content=result["output"], name=TREND_ANALYZER)], "completed_steps": completed_steps}

            def engagement_optimizer_node(state: AgentState) -> dict:
                result = engagement_optimizer_agent().invoke(state)
                # Ensure completed_steps is a list
                completed_steps = state.get('completed_steps') or []
                completed_steps.append(ENGAGEMENT_OPTIMIZER)
                return {"messages": [HumanMessage(content=result["output"], name=ENGAGEMENT_OPTIMIZER)], "completed_steps": completed_steps}
            # Define the sequence of steps in the order they should be executed
            STEPS_ORDER = [KEYWORD_EXTRACTOR, TREND_ANALYZER, ENGAGEMENT_OPTIMIZER]
            def supervisor_node(state: AgentState) -> Runnable:
                completed_steps = state.get('completed_steps') or []
    
                # Determine the next step based on what has already been completed
                next_step = None
                for step in STEPS_ORDER:
                    if step not in completed_steps:
                        next_step = step
                        break
                
                # If all steps are completed, the next action is to finish
                if next_step is None:
                    next_step = "FINISH"

                system_prompt = "You are the supervisor. Please assign tasks to the agents or choose 'FINISH' to complete the tweet generation."
                options = ["FINISH"] if next_step == "FINISH" else [next_step]
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="messages"),
                    ("system", "Please select the next action: {options}."),
                ]).partial(options=str(options), agents=", ".join(agents))
                
                function_def = {
                    "name": "supervisor",
                    "description": "Select the next agent.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "next": {"anyOf": [{"enum": options}]},
                        },
                        "required": ["next"],
                    },
                }
                
                # Bind the next step based on the current state
                return (prompt | llm.bind_functions(functions=[function_def], function_call="supervisor") | JsonOutputFunctionsParser())
                # system_prompt = (
                #     "You are the supervisor. Please assign tasks to the agents or choose 'FINISH' to complete the tweet generation."
                # )
                # options = ["FINISH"] + agents
                # prompt = ChatPromptTemplate.from_messages([
                #     ("system", system_prompt),
                #     MessagesPlaceholder(variable_name="messages"),
                #     ("system", "Please select the next action: {options}."),
                # ]).partial(options=str(options), agents=", ".join(agents))
                # function_def = {
                #     "name": "supervisor",
                #     "description": "Select the next agent.",
                #     "parameters": {
                #         "type": "object",
                #         "properties": {
                #             "next": {"anyOf": [{"enum": options}]},
                #         },
                #         "required": ["next"],
                #     },
                # }
                # return (prompt | llm.bind_functions(functions=[function_def], function_call="supervisor") | JsonOutputFunctionsParser())
            # Define an explicit end node function (it could simply pass if no action is needed)
           
            workflow = StateGraph(AgentState)
            workflow.add_node(KEYWORD_EXTRACTOR, keyword_extractor_node)
            workflow.add_node(TREND_ANALYZER, trend_analyzer_node)
            workflow.add_node(ENGAGEMENT_OPTIMIZER, engagement_optimizer_node)
            workflow.add_node(SUPERVISOR, supervisor_node)
            # Add the end node to your workflow
            workflow.add_edge(KEYWORD_EXTRACTOR, SUPERVISOR)
            workflow.add_edge(TREND_ANALYZER, SUPERVISOR)
            workflow.add_edge(ENGAGEMENT_OPTIMIZER, SUPERVISOR)
            
            workflow.add_conditional_edges(
                SUPERVISOR,
                lambda x: x["next"],
                {
                    KEYWORD_EXTRACTOR: KEYWORD_EXTRACTOR,
                    TREND_ANALYZER: TREND_ANALYZER,
                    ENGAGEMENT_OPTIMIZER: ENGAGEMENT_OPTIMIZER,
                   "FINISH": END
                }
            )
            workflow.set_entry_point(SUPERVISOR)
            graph = workflow.compile()
            

            # Assuming the loop is already in place as shown in your code
            #for s in graph.stream({"messages": [HumanMessage(content=tweet_topic)]}):
                # if "__end__" not in s:
                #     print(s)
                #     # Assuming `s` is a dictionary that contains an output or message you want to display
                #     # Check if there's a specific key in `s` you want to check for message content
                #     if 'output' in s:
                #         # Format and display the message in markdown
                #         st.markdown(s['output'])
                # st.write("-----")
            final_tweet = ""  # Initialize an empty string to hold the final tweet content

            for s in graph.stream({"messages": [HumanMessage(content=tweet_topic)]}):
                for key, value in s.items():  # Iterate through each key and its corresponding value in the state object
                    if 'messages' in value:  # Check if the value contains a 'messages' key
                        for message in value['messages']:  # Iterate through the list of messages
                            if isinstance(message, HumanMessage):  # Ensure the object is an instance of HumanMessage
                                # Display the content of the message in markdown format
                                st.markdown(f"**{key}:** {message.content}")
                                st.markdown("___")
                                if key == 'Engagement_Optimizer':  # Check if this is the final tweet
                                    final_tweet = message.content  # Save the final tweet content for later display

                                # Display the final tweet in a card format after the loop
                                if final_tweet:
                                    # HTML template for the tweet card
                                    tweet_card_html = f"""
                                    <style>
                                    .tweet-card {{
                                        border: 2px solid #1DA1F2;
                                        border-radius: 15px;
                                        padding: 20px;
                                        margin-top: 10px;
                                        background-color: #F5F8FA;
                                        height: auto;
                                    }}
                                    .tweet-text {{
                                        font-size: 16px;
                                    }}
                                    </style>
                                    <div class="tweet-card rounded-xl px-2 ">
                                        <h2>Final Tweet</h2>
                                        <p class="tweet-text">{final_tweet}</p>
                                    </div>
                                    """
                                    # Use html to render the HTML tweet card
                                    
                                    # Add a copy-to-clipboard button for the final tweet
                                # if 'copy_clicked' not in st.session_state:
                                #     st.session_state.copy_clicked = False

                                # if st.button('📋Copy Tweet'):
                                #     st.session_state.copy_clicked = True

                                # if st.session_state.copy_clicked:
                                #     clipboard.copy(final_tweet)
                                #     st.markdown('Tweet copied to clipboard!')
    
if __name__ == "__main__":
    main()
