import json
import os
import logging
import re
from typing import Any, Dict, List, Optional
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool 

from langchain_experimental.utilities.python import PythonREPL 

from mcp import StdioServerParameters
from crewai_tools.adapters.mcp_adapter import MCPServerAdapter
from dotenv import load_dotenv

# --- Logging and Environment Setup ---
load_dotenv(dotenv_path='./.env') 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
TARGET_TEAM_NAME = "Barcelona" 
NUM_GAMES_TO_ANALYZE = 3 

# Setup LLM for all agents
llm = LLM(
    model="gemini/gemini-2.0-flash", 
    verbose=True, 
    temperature=0.55,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

server_env = {**os.environ}
server_env["PYTHONUTF8"] = "1"

MCP_SERVER_SCRIPT_PATH = "./MCP_PSQL/server/server.py" 

server_params = StdioServerParameters(
    command="python",
    args=[MCP_SERVER_SCRIPT_PATH],
    env=server_env
)


def parse_mcp_table_string(table_string: str) -> List[Dict[str, Any]]:
    """
    Parses the formatted table string output from the MCP psql_query tool
    into a list of dictionaries.
    """
    lines = table_string.strip().split('\n')
    if len(lines) < 3 or "No rows returned." in lines[0] or "Error:" in lines[0]:
        logger.warning(f"Parsing returned empty or error data: {table_string[:100]}...")
        return [] 

    header_line_index = -1
    for i, line in enumerate(lines):
        if ' | ' in line and all(col.strip() for col in line.split(' | ')):
            header_line_index = i
            break
    
    if header_line_index == -1 or header_line_index + 1 >= len(lines):
        logger.error(f"Could not find valid header in string: {table_string[:200]}...")
        return []

    headers = [h.strip() for h in lines[header_line_index].split(' | ')]
    
    data_start_index = header_line_index + 2
    data_lines = []
    for line in lines[data_start_index:]:
        if re.match(r'^\(\d+ rows\)$', line.strip()): 
            break
        data_lines.append(line)

    parsed_data = []
    for line in data_lines:
        values = [v.strip() for v in line.split(' | ')]
        row_dict = {}
        for i, header in enumerate(headers):
            if i < len(values): 
                val_str = values[i]
                try:
                    if '.' in val_str and val_str.replace('.', '').replace('-', '').isdigit():
                        row_dict[header] = float(val_str)
                    elif val_str.replace('-', '').isdigit():
                        row_dict[header] = int(val_str)
                    elif val_str.lower() in ['true', 'false']:
                        row_dict[header] = val_str.lower() == 'true'
                    else:
                        row_dict[header] = val_str
                except (ValueError, IndexError):
                    row_dict[header] = val_str
            else:
                row_dict[header] = None
        parsed_data.append(row_dict)
    logger.info(f"Successfully parsed {len(parsed_data)} rows.")
    return parsed_data

class DatabaseTool(BaseTool):
    name: str = "psql_query"
    description: str = (
        "Executes a SQL SELECT query against the 'sofaproject_schema' PostgreSQL database "
        "via the MCP server. Input: `query: str` (the SQL SELECT statement as a string), "
        "`params: list` (optional list of parameters for the query). "
        "Returns a formatted string representing a table of results. "
        "IMPORTANT: This string output MUST be parsed by Python code to extract structured data."
    )
    

# --- Custom Python REPL Tool (Wrapping langchain_experimental.utilities.python.PythonREPL) ---
class CustomPythonREPLTool(BaseTool):
    name: str = "Python_REPL"
    description: str = (
        "Executes arbitrary Python code. Input should be a string containing the Python code. "
        "Useful for data manipulation, file operations (read/write JSON), and complex calculations. "
        "The `parse_mcp_table_string` function is available in the global scope for parsing database query results. "
        "Ensure all necessary imports (like `json`) are included in the code string."
    )
    

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._python_repl = PythonREPL(globals=globals()) 

    def _run(self, code: str) -> str:
        """Synchronously executes the Python code."""
        logger.info(f"Executing Python REPL code:\n{code}")
        result = self._python_repl.run(code) # Use the internal attribute
        logger.info(f"Python REPL output:\n{result}")
        return result

    async def _arun(self, code: str) -> str:
        """Asynchronously executes the Python code (wraps sync for async context)."""
        return self._run(code)


# --- 4. Agent Definitions (using Classes for Organization) ---

class SportsAnalyticsAgents:
    def __init__(self, mcp_tools: List[BaseTool]):
        self.psql_query_tool = next((tool for tool in mcp_tools if tool.name == 'psql_query'), None)
        if not self.psql_query_tool:
            raise ValueError("psql_query tool not found in MCP tools list. Ensure it's exposed by your MCP server.")
        
        self.python_repl_tool = CustomPythonREPLTool() # Instantiate our custom Python REPL tool

    def create_extractor_agent(self) -> Agent:
        return Agent(
            role='Soccer Data Extractor & Initial Processor',
            goal=f'To reliably extract, parse, clean, and standardize soccer team and player performance data for the last {NUM_GAMES_TO_ANALYZE} games of {TARGET_TEAM_NAME} from the database.',
            backstory=(
                "A meticulous data engineer specialized in soccer analytics, deeply proficient in "
                "SQL querying against PostgreSQL databases (specifically 'sofaproject_schema') "
                "via the MCP server. Excels at extracting raw game and player statistics and "
                "critically transforming the formatted string outputs from the database tool "
                "into clean, structured JSON files for downstream analysis. Highly skilled in "
                "handling team ID lookups and date-based queries to fetch recent match data, "
                "ensuring data integrity."
            ),
            # Order matters for explicit indexing in tasks (e.g., tools[0], tools[1])
            tools=[self.psql_query_tool, self.python_repl_tool], 
            llm=llm,
            verbose=True, 
            allow_delegation=False
        )

    def create_team_statistician_agent(self) -> Agent:
        return Agent(
            role='Soccer Team Statistician',
            goal='To calculate comprehensive, accurate standard and advanced team-level soccer statistics from provided raw game data.',
            backstory=(
                "A seasoned quantitative analyst specializing in soccer team performance metrics. "
                "Expert in calculating metrics like possession, pass accuracy, shots on target ratio, "
                "and defensive effectiveness from raw game data. Ensures all statistics are contextually "
                "relevant for soccer and numerically sound, providing valuable quantitative foundations."
            ),
            tools=[self.python_repl_tool], 
            llm=llm,
            verbose=True, 
            allow_delegation=False
        )

    

    def create_team_performance_analyst_agent(self) -> Agent:
        return Agent(
            role='Soccer Team Performance Analyst',
            goal='To analyze calculated team statistics and provide actionable insights into team strengths, weaknesses, and tactical patterns in soccer.',
            backstory=(
                "A former soccer coach and tactical expert, skilled at interpreting statistical data "
                "to understand overall team performance. Focuses on identifying trends in offensive "
                "and defensive efficiency, set-piece effectiveness, and possession play, providing "
                "insights valuable for coaching adjustments and strategic planning."
            ),
            tools=[], # Primarily uses reasoning and synthesis
            llm=llm,
            verbose=True,
            allow_delegation=False
        )


    def create_momentum_analyst_agent(self) -> Agent:
        return Agent(
            role='Soccer Team Momentum Analyst',
            goal=f'To assess the current momentum of {TARGET_TEAM_NAME} in the last {NUM_GAMES_TO_ANALYZE} based on recent team and individual player performances, providing a momentum score and detailed justification.',
            backstory=(
                "A seasoned soccer strategist and performance evaluator with an intuitive understanding "
                "of team dynamics and player form. Excels at synthesizing complex analytical reports "
                "into a concise, actionable assessment of a team's current trajectory. Highly skilled "
                "in identifying whether a team is building positive momentum, struggling, or maintaining "
                "a steady state, crucial for coaching and management decisions."
            ),
            tools=[], 
            llm=llm,
            verbose=True,
            allow_delegation=False 
        )

# --- 5. Task Definitions (using Classes for Organization) ---

class SportsAnalyticsTasks:
    def __init__(self):
        pass # Tasks are defined as methods, agents are passed in at creation

    def define_get_team_id_task(self, team_name: str, agent: Agent) -> Task:
        return Task(
            description=(
                f"**Objective**: Find the `team_id` for '{team_name}' in the `sofaproject_schema.teams` table.\n"
                "**Instructions**:\n"
                "1.  Use the `psql_query` tool (accessible via `self.agent.tools[0]`) to execute a SQL SELECT query: "
                "`SELECT team_id FROM sofaproject_schema.teams WHERE team_name = '{team_name}'`.\n"
                "2.  **Parse Result**: Use the `Python_REPL` tool (accessible via `self.agent.tools[1]`) to execute Python code that calls `parse_mcp_table_string` on the raw string output from `psql_query`. For example, your code should look like: \n"
                "```python\n"
                "import json\n"
                "output_str = tools.psql_query.run(query='SELECT team_id FROM sofaproject_schema.teams WHERE team_name = \\'{team_name}\\'')\n"
                "parsed_data = parse_mcp_table_string(output_str)\n"
                "```\n"
                "3.  **Validate & Extract**: If `parsed_data` is not empty and contains a `team_id` (e.g., `parsed_data[0]['team_id']`), extract the first `team_id`. If no `team_id` is found, return an error status.\n"
                "**Output Format**: A JSON object with `status: 'success'` and `team_id: '...'`, or `status: 'error'` and `message: 'Team not found or query failed.'`."
            ),
            expected_output=(
                "A JSON object with `status: 'success'` and `team_id: '...'`, or `status: 'error'` and `message: 'Team not found or query failed.'`.\n"
                "Example success:\n"
                "```json\n"
                "{\"status\": \"success\", \"team_id\": \"team_barcelona_id\"}\n"
                "```\n"
                "Example error:\n"
                "```json\n"
                "{\"status\": \"error\", \"message\": \"Team 'Barcelona' not found in database.\"}\n"
                "```"
            ),
            agent=agent,
        )

    def define_extract_game_data_task(self, team_id_lookup_output_json_str: str, team_name: str, agent: Agent) -> Task:
        return Task(
            description=(
                f"**Objective**: Extract raw soccer data for '{team_name}' (ID: obtained from previous step)s last {NUM_GAMES_TO_ANALYZE} games from the `sofaproject_schema` database.\n"
                f"**Instructions**:\n"
                f"1.  First, parse the `team_id_lookup_output_json_str` (JSON string) to get the `team_id`. If `status` is 'error' in the parsed JSON, immediately report an error for this task.\n"
                f"2.  **Formulate SQL Queries**: Construct the SQL SELECT query for team statistics: "
                f"`SELECT * FROM sofaproject_schema.team_match_stats WHERE team_id = '{{team_id}}' AND period = 'ALL' ORDER BY match_datetime DESC LIMIT {NUM_GAMES_TO_ANALYZE}`.\n"
                f"3.  **Execute Query**: Execute the SQL query using the `psql_query` tool. Store the *string output*.\n"
                f"4.  **Parse Data**: Use the `Python_REPL` tool to execute Python code that calls `parse_mcp_table_string` on the raw string output from `psql_query` into a list of Python dictionaries. Ensure correct type conversion (e.g., numbers to `int`/`float`).\n"
                f"5.  **Validate Data**: Check if the parsed list is not empty. If crucial data is missing for a game, note it.\n"
                f"6.  **Save as JSON**: Use the `Python_REPL` tool to execute Python code that saves the parsed team stats data to `last{NUM_GAMES_TO_ANALYZE}games_team_stats_{team_name.replace(' ', '_').lower()}.json`. Ensure the file content is valid JSON.\n"
                f"**Error Handling**: If no data is found for the team or extraction/parsing fails, generate an error message with `status: 'error'`."
            ),
            expected_output=(
                f"A JSON object indicating success or failure, including the file path of the generated JSON file and a brief summary.\n"
                f"Example successful output:\n"
                f"```json\n"
                f"{{\n"
                f"  \"status\": \"success\",\n"
                f"  \"team_stats_file\": \"last{NUM_GAMES_TO_ANALYZE}games_team_stats_barcelona.json\",\n"
                f"  \"summary\": \"Extracted and saved data for Barcelona's last {NUM_GAMES_TO_ANALYZE} games.\"\n"
                f"}}\n"
                f"```\n"
                f"Example error output:\n"
                f"```json\n"
                f"{{\n"
                f"  \"status\": \"error\",\n"
                f"  \"message\": \"Could not extract data for [Team Name]. Reason: [Specific Error, e.g., 'No data returned from DB', 'Parsing failed'].\"\n"
                f"}}\n"
                f"```"
            ),
            agent=agent,
            context=[team_id_lookup_output_json_str] # Expects JSON string output from GetTeamIDTask
        )

    def define_team_statistician_task(self, extractor_output_json_str: str, agent: Agent) -> Task:
        return Task(
            description=(
                f"**Objective**: Calculate standard and advanced soccer team statistics from the data extracted by the 'Data Extractor' agent.\n"
                f"**Instructions**:\n"
                f"1.  **Load Data**: Parse the `extractor_output_json_str` (which is a JSON string) to get the `team_stats_file` path. Use `Python_REPL` to read the JSON data from this file.\n"
                f"2.  **Calculate Metrics**: For each game and overall for the {NUM_GAMES_TO_ANALYZE} games, calculate the following soccer-specific metrics using `Python_REPL` for computations:\n"
                f"    -   **Standard Metrics**: Goals Scored, Goals Conceded, Shots, Shots on Target, Possession Percentage, Fouls, Yellow Cards, Red Cards, Pass Accuracy.\n"
                f"    -   **Advanced Metrics**: Goal Difference (Goals Scored - Goals Conceded), Shot Conversion Rate (Shots on Target / Shots), Defensive Effectiveness (e.g., Goals Conceded per 90 minutes or per opponent shot).\n"
                f"3.  **Aggregate**: Compute averages for the 'Last {NUM_GAMES_TO_ANALYZE} Games' period for all metrics. Also provide per-game breakdown.\n"
                f"4.  **Validate Calculations**: Ensure percentages are between 0-100, counts are non-negative, etc.\n"
                f"**Output Format**: A JSON object containing team name, games analyzed count, 'last_{NUM_GAMES_TO_ANALYZE}_games_average' (key-value pairs of metrics), and 'per_game_stats' (list of dicts for each game)."
            ),
            expected_output=(
                f"A JSON object with team name, and detailed statistics for the last {NUM_GAMES_TO_ANALYZE} games.\n"
                f"Example:\n"
                f"```json\n"
                f"{{\n"
                f"  \"team_name\": \"Barcelona\",\n"
                f"  \"games_analyzed_count\": {NUM_GAMES_TO_ANALYZE},\n"
                f"  \"last_{NUM_GAMES_TO_ANALYZE}_games_average\": {{\n"
                f"    \"goals_scored_avg\": 2.4,\n"
                f"    \"goals_conceded_avg\": 1.2,\n"
                f"    \"possession_percentage_avg\": 65.0,\n"
                f"    \"shot_conversion_rate_avg\": 0.45,\n"
                f"    \"pass_accuracy_avg\": 0.90\n"
                f"  }},\n"
                f"  \"per_game_stats\": [\n"
                f"    {{\"game_id\": \"G001\", \"goals_scored\": 2, \"possession_percentage\": 60.0, ...}},\n"
                f"    // ... data for {NUM_GAMES_TO_ANALYZE} games\n"
                f"  ]\n"
                f"}}\n"
                f"```"
            ),
            agent=agent,
            context=[extractor_output_json_str] # Expects the JSON string output from ExtractGameDataTask
        )

    def define_team_performance_analysis_task(self, team_stats_json_str: str, agent: Agent) -> Task:
        return Task(
            description=(
                f"**Objective**: Analyze the calculated soccer team statistics for {TARGET_TEAM_NAME} to identify key strengths, weaknesses, and tactical patterns for the last {NUM_GAMES_TO_ANALYZE} games.\n"
                f"**Instructions**:\n"
                f"1.  **Load Data**: Parse the `team_stats_json_str` (JSON string) into a Python dictionary.\n"
                f"2.  **Comparative Analysis**: Compare the team's 'last_{NUM_GAMES_TO_ANALYZE}_games_average' metrics against general league averages or ideal soccer performance benchmarks (using inherent knowledge). Identify significant over/under-performances and trends over the {NUM_GAMES_TO_ANALYZE} games.\n"
                f"3.  **Infer Reasons**: Infer underlying reasons or tactical implications for statistical trends (e.g., 'high pass accuracy with low shots on target might indicate sterile possession').\n"
                f"4.  **Formulate Insights**: Formulate 2-3 clear strengths and 2-3 clear weaknesses for the team's performance, each supported by specific statistics and observations.\n"
                f"5.  **Identify Patterns**: Identify 1-2 overarching performance patterns or tactical observations that characterize the team's play in recent games.\n"
                f"**Output Format**: A Markdown report structured to directly feed into a FODA analysis. Ensure all points are supported by data references (e.g., 'Goals Scored Avg: 2.4')."
            ),
            expected_output=(
                f"A Markdown report detailing team strengths, weaknesses, and key observations.\n"
                f"```markdown\n"
                f"### Team Performance Analysis for [Team Name] - Last {NUM_GAMES_TO_ANALYZE} Games\n\n"
                f"**Key Strengths (Potential Advantages/Opportunities):**\n"
                f"- **Clinical Finishing:** Averaging X goals per game (Y conversion rate), indicating efficiency in front of goal. (Supported by: `goals_scored_avg`, `shot_conversion_rate_avg`)\n"
                f"- **High Possession & Control:** Consistently maintaining high possession percentages (Avg. Y%), allowing them to dictate game tempo. (Supported by: `possession_percentage_avg`)\n\n"
                f"**Key Weaknesses (Potential Dangers/Factors):**\n"
                f"- **Defensive Lapses Post-Possession Loss:** Despite high possession, concedes a high number of goals from relatively few opponent shots, suggesting vulnerability during transitions. (Supported by: `goals_conceded_avg`, analysis of opponent shots vs. goals)\n"
                f"- **Foul Prone Midfield:** High number of fouls (Avg. X per game), leading to dangerous set-piece opportunities for opponents. (Supported by: `fouls_avg`)\n\n"
                f"**Key Performance Patterns/Observations (Potential Factors):**\n"
                f"- **Slow Starts:** Team often starts games slowly, with most offensive action concentrated in the second half. (Observation based on `per_game_stats` trends).\n"
                f"```"
            ),
            agent=agent,
            context=[team_stats_json_str] # Expects the JSON string output from Team Statistician task
        )

    def define_momentum_analysis_task(self, team_analysis_md: str, agent: Agent) -> Task:
        return Task(
            description=(
                f"**Objective**: Based on the provided 'Team Performance Analysis' and 'Player Performance Analysis' reports for {TARGET_TEAM_NAME} (last {NUM_GAMES_TO_ANALYZE} games), assess the team's current momentum.\n"
                "**Instructions**:\n"
                "1.  Review both Markdown reports thoroughly, focusing on recent trends, consistency, and key statistical indicators.\n"
                "2.  **Evaluate Team Momentum**: Consider:\n"
                "    -   Recent match results (wins, losses, draws, scorelines, comebacks, collapses).\n"
                "    -   Overall offensive and defensive efficiency trends.\n"
                "    -   Tactical effectiveness and adaptability.\n"
                "    -   Team cohesion and apparent confidence.\n"
                "3.  **Evaluate Player Momentum**: Consider:\n"
                "    -   Form of key players (goal scorers, playmakers, defenders, goalkeeper).\n"
                "    -   Consistency of individual performances.\n"
                "    -   Impact of player strengths/weaknesses on overall team performance.\n"
                "    -   Any notable individual slumps or surges.\n"
                "4.  **Synthesize & Score**: Based on the combined analysis, assign a momentum score from 1 to 10 (10 being amazing momentum, 1 being a severe loss streak with poor performances).\n"
                "5.  **Justify Score**: Provide a clear, concise justification for the assigned score, referencing specific points from both the team and player analyses to support your assessment.\n"
                "**Output Format Constraint**: Strict Markdown as specified below."
            ),
            expected_output=(
                f"A Markdown report detailing the team's current momentum and score.\n"
                f"```markdown\n"
                f"### Momentum Analysis for {TARGET_TEAM_NAME} - Last {NUM_GAMES_TO_ANALYZE} Games\n\n"
                f"**Current Momentum Score: [X]/10**\n"
                f"*(Where X is an integer from 1 to 10)*\n\n"
                f"**Momentum Assessment:**\n"
                f"The team is currently experiencing [positive/negative/mixed/stable] momentum due to the following factors:\n"
                f"-   **Team Performance Indicators:**\n"
                f"    -   [Summary of team's recent form, e.g., 'Consistent high possession (Avg. Y%) but struggling with defensive transitions (X goals conceded from Z opponent shots).']\n"
                f"    -   [Reference to specific trends, e.g., 'Offensive output has been sporadic, with X goals in Y games.']\n"
                f"-   **Key Player Performance Indicators:**\n"
                f"    -   [Summary of key player form, e.g., 'Player A's goal-scoring form (X goals in Y games) is a major positive.']\n"
                f"    -   [Reference to player struggles/consistency, e.g., 'However, Player B's defensive contributions (Avg. X tackles won) have been inconsistent.']\n"
                f"-   **Recent Results & Trends:**\n"
                f"    -   [Brief overview of recent match outcomes, e.g., 'A recent win streak of X matches suggests growing confidence.']\n"
                f"    -   [Overall trajectory, e.g., 'The team appears to be on an upward trajectory, showing resilience in tight games.']\n"
                f"```"
            ),
            agent=agent,
            context=[team_analysis_md] 
        )

# --- 6. Crew Definition and Execution ---

class SoccerAnalyticsCrew:
    def __init__(self, target_team: str, mcp_tools: List[BaseTool]):
        self.target_team = target_team
        self.mcp_tools = mcp_tools

        self.agent_factory = SportsAnalyticsAgents(mcp_tools=self.mcp_tools)
        self.task_factory = SportsAnalyticsTasks()

        # Define Agents (excluding player-related ones)
        self.extractor = self.agent_factory.create_extractor_agent()
        self.team_statistician = self.agent_factory.create_team_statistician_agent()
        self.team_performance_analyst = self.agent_factory.create_team_performance_analyst_agent()
        self.momentum_analyst = self.agent_factory.create_momentum_analyst_agent()

    def run_crew(self):
        # Define Tasks
        get_team_id_task = self.task_factory.define_get_team_id_task(
            team_name=self.target_team,
            agent=self.extractor 
        )

        extract_game_data_task = self.task_factory.define_extract_game_data_task(
            team_id_lookup_output_json_str=get_team_id_task, # Context from previous task
            team_name=self.target_team,
            agent=self.extractor
        )

        # Team stats/analysis tasks
        team_statistician_task = self.task_factory.define_team_statistician_task(
            extractor_output_json_str=extract_game_data_task,
            agent=self.team_statistician
        )

        team_performance_analysis_task = self.task_factory.define_team_performance_analysis_task(
            team_stats_json_str=team_statistician_task,
            agent=self.team_performance_analyst
        )

        momentum_analysis_task = self.task_factory.define_momentum_analysis_task(
            team_analysis_md=team_performance_analysis_task,
            agent=self.momentum_analyst 
        )

        # Instantiate Crew with hierarchical process
        soccer_crew = Crew(
            agents=[
                self.extractor,
                self.team_statistician,
                self.team_performance_analyst,
                self.momentum_analyst
            ],
            tasks=[
                get_team_id_task,
                extract_game_data_task,
                team_statistician_task,
                team_performance_analysis_task,
                momentum_analysis_task
            ],
            process=Process.hierarchical, 
            manager_llm=llm, 
            verbose=True, 
            max_iterations=3, 
            
        )

        logger.info(f"\n--- Starting Soccer Analytics Crew for {self.target_team} ---")
        result = soccer_crew.kickoff()
        logger.info("\n--- Final Soccer Analytics Crew Result ---")
        return result

# --- Main Execution Block ---
if __name__ == "__main__":
    logger.info("Initializing MCP Server Adapter...")
    with MCPServerAdapter(serverparams=server_params) as mcp_tools_list:
        if not mcp_tools_list:
            logger.error("No tools loaded from MCP server. Check server logs (stdout/stderr) and MCP_SERVER_SCRIPT_PATH.")
            exit(1)
        
        logger.info(f"MCP tools available and adapted for CrewAI: {[tool.name for tool in mcp_tools_list]}")

        soccer_analytics = SoccerAnalyticsCrew(
            target_team=TARGET_TEAM_NAME,
            mcp_tools=mcp_tools_list
        )
        
        final_report = soccer_analytics.run_crew()
        print(final_report)

    # Optional: Clean up generated JSON files after the run
    team_file = f"last{NUM_GAMES_TO_ANALYZE}games_team_stats_{TARGET_TEAM_NAME.replace(' ', '_').lower()}.json"
    
    if os.path.exists(team_file):
        os.remove(team_file)
        logger.info(f"Cleaned up {team_file}")