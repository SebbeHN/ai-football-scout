# AI Football Scout

An intelligent football scouting application powered by LangChain, LangGraph, and Google Gemini. The system uses RAG (Retrieval-Augmented Generation) and multi-agent AI to analyze player data and provide professional scouting recommendations.

## Features

| Feature | Description |
|---------|-------------|
| **Scout Chat** | Conversational AI with memory for natural scouting discussions |
| **Scout Search** | Hybrid search combining semantic AI + SQL filtering |
| **Multi-Agent System** | LangGraph orchestration with specialized Scout, Stats, and Transfer agents |
| **Player Comparison** | Interactive radar charts comparing player profiles |
| **Similar Players** | Statistical similarity analysis across different teams |
| **Transfer Prediction** | Estimate player market values based on age, position, league, and performance |

## Tech Stack

- **LLM**: Google Gemini 2.0 Flash
- **Embeddings**: Google Generative AI Embeddings
- **Vector Database**: ChromaDB
- **Orchestration**: LangGraph (multi-agent)
- **Framework**: LangChain + LCEL
- **Frontend**: Streamlit
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy, Scikit-learn

## Project Structure

```
ai-football-scout/
├── app.py                    # Main Streamlit application
├── scout_notebook.ipynb      # Data preprocessing & vector DB creation
├── requirements.txt          # Python dependencies
├── data/
│   ├── players24_25.xlsx     # Raw player data
│   └── players24_25.csv      # Processed player data
├── chroma_db/                # Vector database (generated)
└── transfers/                # Transfer prediction ML project
    ├── app/app_ml.py
    └── notebook/transfers.ipynb
```

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/SebbeHN/ai-football-scout.git
cd ai-football-scout
```

### 2. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or: .venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API key
Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your-google-api-key-here
```

Get a free API key at: https://aistudio.google.com/apikey

### 5. Generate the vector database
Run all cells in `scout_notebook.ipynb` to:
- Load and preprocess player data
- Generate embeddings
- Create the ChromaDB vector database

### 6. Run the application
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Usage Examples

### Scout Chat
Ask natural language questions:
- *"Who are the best young forwards under 23?"*
- *"Find me a creative midfielder with high progressive passes"*
- *"Which defenders have the best goal contribution?"*

### Multi-Agent System
The supervisor automatically routes queries to specialized agents:
- **Scout Agent**: Player recommendations and scouting analysis
- **Stats Agent**: Statistical comparisons and data analysis
- **Transfer Agent**: Market value assessments

### Transfer Prediction
Factors considered:
- Age profile (peak value 24-27)
- Position value
- League strength
- Performance stats (goals, assists, xG, progression)
- Market conditions

## Data Source

Player statistics from FBref (2024/25 season), including:
- Goals, Assists, xG, xAG
- Progressive Carries & Passes
- Minutes played, Age, Position, Team

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                    │
├─────────────────────────────────────────────────────────┤
│                  LangGraph Supervisor                    │
│         ┌──────────┬──────────┬──────────┐              │
│         │  Scout   │  Stats   │ Transfer │              │
│         │  Agent   │  Agent   │  Agent   │              │
│         └────┬─────┴────┬─────┴────┬─────┘              │
├──────────────┼──────────┼──────────┼────────────────────┤
│              │   ChromaDB + Hybrid Search               │
│              │   (Semantic + SQL Filtering)             │
├──────────────┴──────────────────────────────────────────┤
│                 Google Gemini 2.0 Flash                  │
└─────────────────────────────────────────────────────────┘
```

## License

MIT License

## Author

Sebastian Holmberg - NBI Handelsakademin
