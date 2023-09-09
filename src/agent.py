from src.tools import SearchSERPTool, ScrapeWebsiteTool, SummarizerTool, UIUCDatabaseTool

from langchain.agents import initialize_agent, load_tools
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import SystemMessage


# 3. Create langchain agent with the tools above
tools = load_tools( [
            "arxiv"
        ])
tools.extend([
    SearchSERPTool(),
    ScrapeWebsiteTool(),
    SummarizerTool(),
    UIUCDatabaseTool(),
])

system_message = SystemMessage(
    content = """
        You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
        you do not make things up, you will try as hard as possible to gather facts & data to back up the research
        
        Please make sure you complete the objective above with the following rules:
        1/ You should do enough search to gather as much information as possible about the objective
        2/ If there are url of relevant links & articles, you will scrape it to gather more information
        3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iterations
        4/ You should not make things up, you should only write facts & data that you have gathered
        5/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
    """
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0, model="gpt-4")

memory = ConversationBufferWindowMemory(
    k=10, memory_key="memory", return_messages=True, max_token_limit=1000
)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory
)

