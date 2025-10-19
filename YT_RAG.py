import streamlit as st
import re
import asyncio
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from operator import add
from dotenv import load_dotenv

# MCP imports
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

load_dotenv()

st.title("🎥 YouTube Video Chat Q&A + MCP Tools")
st.caption("Ask questions about your video, do math, check weather, and more!")

# Define the state
class ChatState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    context: str
    question: str

# Initialize session state
if 'graph' not in st.session_state:
    st.session_state.graph = None
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = "default_thread"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'mcp_client' not in st.session_state:
    st.session_state.mcp_client = None
if 'mcp_tools' not in st.session_state:
    st.session_state.mcp_tools = []
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'video_transcript' not in st.session_state:
    st.session_state.video_transcript = ""

def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'^([0-9A-Za-z_-]{11})$'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Invalid YouTube URL")

def get_transcript(video_id: str) -> str:
    """Fetch transcript from YouTube video"""
    try:
        api=YouTubeTranscriptApi()
        transcript_list = api.fetch(video_id)
        transcript = " ".join(chunk.text for chunk in transcript_list)
        return transcript
    except Exception as e:
        raise Exception(f"Error fetching transcript: {str(e)}")

def create_vector_store(transcript: str):
    """Create vector store from transcript"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    
    return retriever

async def initialize_mcp_client():
    """Initialize MCP client and get tools"""
    try:
        client = MultiServerMCPClient({
            "math": {
                "command": "python",
                "args": ["mathserver.py"],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://127.0.0.1:8000/mcp",
                "transport": "streamable_http",
            }
        })
        
        tools = await client.get_tools()
        return client, tools
    except Exception as e:
        st.error(f"Error initializing MCP: {str(e)}")
        return None, []

def retrieve_context(state: ChatState) -> ChatState:
    """Retrieve relevant context from video transcript"""
    question = state["question"]
    
    if st.session_state.retriever:
        retrieved_docs = st.session_state.retriever.get_relevant_documents(question)
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    else:
        context = ""
    
    return {"context": context}

def generate_answer(state: ChatState) -> ChatState:
    """Generate answer based on context, chat history, and MCP tools"""
    
    # If we have MCP tools, use the agent
    if st.session_state.agent and st.session_state.mcp_tools:
        # Check if question is video-related
        video_keywords = ["video", "transcript", "what did", "explain", "summary", "about this"]
        is_video_question = any(keyword in state["question"].lower() for keyword in video_keywords)
        
        if is_video_question and state["context"]:
            # Use context-aware prompt for video questions
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant with access to various tools (math, weather, etc.) 
                AND you can answer questions about a YouTube video transcript.
                
                For video-related questions, use this transcript context:
                {context}
                
                For other questions (math, weather, etc.), use the appropriate tools.
                You will only answer questions related to the video topic for video questions. 
                For anything else apart from the video, say: "Sorry I can't answer that question about the video."
                Be conversational and refer back to previous messages when relevant."""),
                MessagesPlaceholder(variable_name="messages"),
            ])
            
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
            chain = prompt | llm | StrOutputParser()
            
            response = chain.invoke({
                "context": state["context"],
                "messages": state["messages"]
            })
        else:
            # Use agent for non-video questions or when no context
            agent_response = asyncio.run(st.session_state.agent.ainvoke({
                "messages": [{"role": "user", "content": state["question"]}]
            }))
            response = agent_response['messages'][-1].content
    else:
        # Fallback to basic RAG without MCP tools
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant answering questions about a YouTube video transcript. 
            Use the provided context from the transcript and the conversation history to answer questions.
            Note: You will only answer the questions related to the Video topic, anything else apart from the video,
            you will say: "Sorry I can't answer that question."
            Be conversational and refer back to previous messages when relevant.
            
            Transcript Context:
            {context}"""),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        chain = prompt | llm | StrOutputParser()
        
        response = chain.invoke({
            "context": state["context"],
            "messages": state["messages"]
        })
    
    return {"messages": [AIMessage(content=response)]}

def create_chat_graph():
    """Create LangGraph workflow"""
    workflow = StateGraph(ChatState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_context)
    workflow.add_node("generate", generate_answer)
    
    # Add edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    # Compile with memory
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    
    return graph

# Sidebar for video loading and MCP configuration
with st.sidebar:
    st.header("🔧 MCP Tools Setup")
    
    if st.button("🔌 Connect MCP Servers", use_container_width=True):
        with st.spinner("Connecting to MCP servers..."):
            try:
                client, tools = asyncio.run(initialize_mcp_client())
                if client and tools:
                    st.session_state.mcp_client = client
                    st.session_state.mcp_tools = tools
                    
                    # Create agent with MCP tools
                    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
                    st.session_state.agent = create_react_agent(model, tools)
                    
                    st.success(f"✅ Connected! {len(tools)} tools available")
                    for tool in tools:
                        st.write(f"  • {tool.name}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    if st.session_state.mcp_tools:
        st.info(f"🛠️ {len(st.session_state.mcp_tools)} MCP tools active")
    
    st.divider()
    
    st.header("📹 Load Video")
    video_input = st.text_input("YouTube URL or Video ID:")
    
    if st.button("Load Video", use_container_width=True):
        try:
            with st.spinner("Loading video transcript..."):
                video_id = extract_video_id(video_input)
                transcript = get_transcript(video_id)
                
                # Store transcript
                st.session_state.video_transcript = transcript
                
                # Create retriever
                st.session_state.retriever = create_vector_store(transcript)
                
                # Create graph
                st.session_state.graph = create_chat_graph()
                
                # Reset chat history
                st.session_state.chat_history = []
                
                st.success("✅ Video loaded successfully!")
                st.info(f"Video ID: {video_id}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    if st.session_state.graph:
        if st.button("🔄 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# Main chat interface
if st.session_state.graph is None:
    st.info("👈 Please load a YouTube video from the sidebar to start chatting!")
    
    with st.expander("💡 Available Capabilities"):
        st.markdown("""
        **After loading a video, you can:**
        - 📺 Ask questions about the video content
        - ➕ Do math calculations (if MCP tools connected)
        - 🌤️ Check weather (if MCP tools connected)
        - 💬 Have contextual conversations
        
        **Example Questions:**
        - "What is this video about?"
        - "Calculate 25 + 37"
        - "What's the weather in Kestopur?"
        - "Summarize the main points of the video"
        """)
else:
    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)
    
    # Chat input
    if question := st.chat_input("Ask about the video, do math, check weather..."):
        # Add user message to history
        user_message = HumanMessage(content=question)
        st.session_state.chat_history.append(user_message)
        
        # Display user message
        with st.chat_message("user"):
            st.write(question)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Invoke graph with state
                    result = st.session_state.graph.invoke(
                        {
                            "messages": st.session_state.chat_history,
                            "question": question,
                            "context": ""
                        },
                        config={"configurable": {"thread_id": st.session_state.thread_id}}
                    )
                    
                    # Get the AI response
                    ai_message = result["messages"][-1]
                    st.session_state.chat_history.append(ai_message)
                    
                    # Display response
                    st.write(ai_message.content)
                except Exception as e:
                    st.error(f"Error: {str(e)}")