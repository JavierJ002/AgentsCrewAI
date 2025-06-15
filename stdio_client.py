from crewai import Agent, Task, Crew, LLM, Process
import os
from dotenv import load_dotenv
from mcp import StdioServerParameters
from crewai_tools.adapters.mcp_adapter import MCPServerAdapter
import json
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Literal, Optional
import logging


load_dotenv(dotenv_path='../.env')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

topic = "Which team scored the most goals on the premier league on the 22-23 season?"

server_env = {**os.environ}
server_env["PYTHONUTF8"] = "1"

#Setup servidor MCP
server_params = StdioServerParameters(
    command="python",
    args = ["./MCP_PSQL/server/server.py"],
    env = server_env
)


#Setup LLM del agente
llm = LLM(
    model="gemini/gemini-2.0-flash", 
    verbose=True,
    temperature=0.5,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

#Setup Tarea Agente
database_query_agent_config = {
    'role': f'{topic} Database Query Agent', 
    'goal': f'Retrieve and analyze {topic} data in response to user queries', 
    'backstory': f"You are a skilled database query specialist with a talent for swiftly retrieving and analyzing {topic} data. Your expertise lies in understanding user inquiries and efficiently providing accurate responses from the database, ensuring clarity and relevance in every answer.",
    # 'verbose': True,
    # 'allow_delegation': False,
}

database_query_task_config = {
    'description': (
        f"Your primary goal is to count the total number of tables in the database. "
        f"To do this, you must use the `psql_list_tables` tool and then accurately count the tables returned by the tool. "
        f"The original user question is: '{topic}'"
    ),
    'expected_output': (
        "A single numerical value representing the total count of tables in the database, e.g., 'There are 15 tables in the database.'"
    ),
}



with MCPServerAdapter(serverparams=server_params) as mcp_tools:
    logger.info(f"Herramientas MCP disponibles y adaptadas para CrewAI: {[tool.name for tool in mcp_tools]}")

    if not mcp_tools:
            logger.error("No se cargaron herramientas desde el servidor MCP. Revisa los logs del servidor (stdout/stderr).")
            exit(1)

    agent_psql = Agent(
        **database_query_agent_config,
        tools=mcp_tools,
        llm = llm
    )
    logger.info("Agente CrewAI creado con herramientas MCP.")

    database_query_task_instance = Task(
            description=database_query_task_config['description'],
            expected_output=database_query_task_config['expected_output'],
            agent=agent_psql,
            #human_input=True 
        )
    logger.info("Tarea CrewAI creada.")
    crew = Crew(
            agents=[agent_psql],
            tasks=[database_query_task_instance],
            process=Process.sequential,
            verbose=True 
        )

    logger.info(f"Ejecutando la Crew con la tarea: {database_query_task_instance.description}")
    result = crew.kickoff()

    print("\n--------------------------------------------------")
    print("Resultado final de la Crew:")
    print(result)
    print("--------------------------------------------------")




