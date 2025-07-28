# import libraries
import os 
import time
import streamlit as st
import speech_recognition as sr
from langchain.agents import Tool
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
#from langchain.vectorstores import FAISS
from langchain.agents import initialize_agent
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import OpenWeatherMapAPIWrapper, WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper,SerpAPIWrapper


from dotenv import load_dotenv
_ = load_dotenv()

OPENAI_API_KEY  = os.getenv('OPENAI_API_KEY')
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]="Travel Assistant"

# Setup recognizer and APIs
# assign APIs libraries into variable
weather_api = OpenWeatherMapAPIWrapper(OPENWEATHERMAP_API_KEY)  #https://home.openweathermap.org/api_keys
duck_api =DuckDuckGoSearchAPIWrapper()
wiki_api = WikipediaAPIWrapper()
trav_api = TavilySearchResults()
serp_api = SerpAPIWrapper()
recognizer = sr.Recognizer()
embed = OpenAIEmbeddings(model='text-embedding-ada-002') 
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Wikipidia retriever system
def retriever_tool(split_docs,query):
    
    # creates a FAISS vector store for similarity search
    vectorstore = FAISS.from_documents(split_docs, embed)
    retriever = vectorstore.as_retriever()
    results = retriever.get_relevant_documents(query)
    return "\n".join([doc.page_content for doc in results])

# Wikipidia search system 
def wiki_search(query):
    try:
        data = wiki_api.load(query)
        if not data:
            return "i didn't find any information at wikipedia"
        split_docs = splitter.split_documents(data)
        return retriever_tool(split_docs, query)
    except Exception as e:
        return f"error! {str(e)}"

# document identification function
def docs(article):
  return Document(
      page_content=article['snippet'],
      metadata={"title": article['title']}
  )


# DuckDuckGo search system 
def serp_search(query):
    try:
        time.sleep(2) 
        data = serp_api.results()
        documents = []
        for raw in data:
            documents.append(docs(raw))
        split_docs = splitter.split_documents(documents)
        return retriever_tool(split_docs, query)
    except Exception as e:
        return f"⚠️ Web search failed (rate limited or error): {str(e)}"


# Define the listen_to_user function
def listen_to_user(timeout=3, phrase_time_limit=5):
    """Convert speech to text with error handling"""
    try:
        with sr.Microphone() as source:
            print("\n🎤 Listening... (Speak now)")
            audio = recognizer.listen(source, timeout=timeout)
        return recognizer.recognize_google(audio)
    except sr.WaitTimeoutError:
        print("⌛ No speech detected, please try again!")
        return ""
    except Exception as e:
        print(f"🔇 : {str(e)}. Try speaking again.")
        return ""

# Define the prompt for the agent
tour_guide_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
    You are a SmartTourGuideAgent. Your task is to help users with their travel-related questions.
    If the user asks about a city or country, provide them with relevant details about it.
    If the user asks or ordered or request for a trip plan, create a detailed itinerary day-by-day time-by-time for them.
    be smart, add your touches and be more friendly.
    Answer questions with relevant emojis to make the answers more engaging and fun. 
    Query: {query}
    Answer:
    """
)

st.title("My Smart TourGuide 💬")
st.subheader("🚀 Adventures Around the World")



# Define the tools (API wrappers)
tools = [
    Tool(
        name="TravelInfoRetriever",
        func=wiki_search,
        description=(
            "You are a SmartTourGuideAgent. Use this tool when the user wants general information or suggestions about a place. "
            "Provide cultural facts, famous landmarks, popular activities, or create basic travel itineraries. "
            "- For cities: Mention must-see sights, activities, local foods, or typical weather with emojis. "
            "- For countries: Mention popular destinations, customs, or iconic elements with emojis. "
            "- For trip plans: Generate a daily schedule or highlights with emoji-enhanced bullet points. "
            "Avoid using this for real-time weather or current events."
        )
    ),
    Tool(
        name="Weather",
        func=weather_api.run,
        description=(
            "Use this to fetch current or forecasted weather conditions. "
            "Perfect for questions like: 'Is it snowing in Tokyo?', 'What's the temperature in Dubai in July?', or "
            "'Will it rain in Paris next weekend?'. Only use if the user specifically asks about weather."
        )
    ),
    Tool(
        name="SerpSearch",
        func=serp_api.run,
        description=(
            "Use this tool to retrieve Google-style search results. Ideal for rich, broad, or deeper queries like: "
            "'Compare Tokyo and Seoul for nightlife', 'Recent travel advisories for Thailand', or "
            "'Top-rated ski resorts in Switzerland'. Suitable when DuckDuckGo returns limited info."
            "Use when you need current news, regulations, or live information not covered by Wikipedia or guidebooks."
        )
    ),
    Tool(
        name="TavilySearch",
        func=trav_api.run,
        description=(
            "Use this powerful real-time search tool to find fast and reliable web results for travel, news, or events. "
            "Great for: 'New Year's events in New York 2025', 'COVID travel updates for Canada', or "
            "'Unique cultural festivals in Africa'. Useful when you need well-organized search results quickly."
            "Search the web for up-to-date or obscure travel-related information. "
            "Great for things like: 'Travel rules for 2025', 'Best hiking events in Switzerland this summer', or "
            "'Is the Venice Carnival happening this year?'. "
        )
    )
]


# Initialize agent with memory
if "memory" not in st.session_state:
    st.session_state.memory =  ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


conversational_agent = initialize_agent(
    agent="conversational-react-description",
    tools=tools,
    llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),  # Using a compatible model for chat completions
    verbose=True,
    agent_kwargs={"prompt": tour_guide_prompt},
    memory=st.session_state.memory,
    handle_parsing_errors=True,
)

# Display existing chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])




query = None
input_user = ""
agent = ""

# Toggle for Audio or Text
on = st.toggle("🎙️ Use Voice Input")
if on:
    if st.button("Start Recording"):
        st.write("🎤 Listening...") 
        query = listen_to_user()  # make sure this function returns a valid string
        st.success("✅ Voice input received!")
        time.sleep(2)
        st.empty()          
else:
    query = st.chat_input("Ask Anything ✍️ ")

if query:
    input_user = query   
        # Get agent response with spinner to show processing
    with st.spinner("Thinking..."):
        agent = conversational_agent.run(query)

# Show the interaction in chat format
if query:
    st.chat_message("user").markdown(input_user)
    st.chat_message("assistant").markdown(agent)
    # Optionally: Save to session state for chat history
    st.session_state.messages.append({"role": "user", "content": input_user})
    st.session_state.messages.append({"role": "assistant", "content": agent})

with st.sidebar:
    st.title("Travel Assistant Agent 🌍")
    st.subheader("Your Smart Travel Companion")
    st.markdown(
        """
        Meet your **AI-powered Travel Assistant Agent** – your all-in-one solution for discovering new cities and planning smarter trips.

         **What it does**✨
        - Gives you a short overview of any city you ask about
        - Recommends top tourist attractions
        - Provides live weather updates
        - Checks for nearby airports and informs you if travel access is limited
        - Supports both text and voice interaction

        🔧 **Powered by**:
        - [Wikipedia API](https://www.mediawiki.org/wiki/API:Main_page) for city facts and history
        - [DuckDuckGo API](https://duckduckgo.com/api) for additional insights
        - [OpenWeatherMap API](https://openweathermap.org/api) for real-time weather

        Perfect for travelers, trip planners, and anyone curious about the world 🌐✈️
        """
        
    )
    st.markdown("**Travel Tip**: When you visit a new place, don’t forget to try the local food! 🌮🍣")
        # Add a button to clear conversation history
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.experimental_rerun()

