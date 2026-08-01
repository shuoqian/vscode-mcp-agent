# VS Code Intelligent Signal Tracking Agent

An intelligent agent built with **LangGraph** and **Groq** to track, filter, and summarize GitHub issues from the `microsoft/vscode` repository based on specific user symptoms and operating systems.

## 🚀 Features
- **Intelligent Planning**: Automatically breaks down user symptoms into targeted search and filtering tasks.
- **GitHub Integration**: Fetches live data from the GitHub REST API.
- **ReAct Workflow**: Iteratively processes issues to filter for relevance and deduplicate findings.
- **Self-Reflection**: Uses a dynamic loop to verify the quality of the findings before final synthesis.
- **Executive Summaries**: Generates concise, bulleted reports highlighting common patterns and root causes.

## 🛠️ Prerequisites
- Python 3.9 or higher.
- A **Groq API Key** (Get one at [console.groq.com](https://console.groq.com/)).

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd vscode-mcp-agent
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install langgraph langchain-groq httpx rich python-dotenv
   ```

## ⚙️ Configuration

Create a `.env` file in the root directory and add your Groq API key:

```env
GROQ_API_KEY=your_lp_api_key_here
```

## 🖥️ Usage

Run the agent using the following command:

```bash
python main.py
```

Follow the interactive prompts:
1. **Target OS Environment**: (e.g., Windows, macOS, Linux)
2. **Symptoms/Interests**: (e.g., "memory leak when opening large files")
3. **Time Range**: Number of days to look back (e.g., 7, 30)

## 🏗️ Architecture

The agent follows a modular **LangGraph** workflow:
1. **Planner**: Analyzes input and generates a 3-step execution plan.
2. **Fetch Issues**: Retrieves recent open issues from the VS Code repo.
3. **ReAct Processor**: Filters each issue against the plan and checks for duplicates.
4. **Reflection**: Evaluates the candidate pool; if insufficient, it triggers a refinement loop.
5. **Final Summary**: Synthesizes the gathered intelligence into a final report.

## 📄 License
MIT
