import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import os
import operator
from typing import TypedDict, Annotated, Sequence, Literal
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Football Scout",
    layout="wide"
)

# --- CONSTANTS ---
STAT_COLUMNS = ['Gls', 'Ast', 'npxG', 'xAG', 'PrgC', 'PrgP']
STAT_LABELS = ['Goals', 'Assists', 'npxG', 'xAG', 'Prog Carries', 'Prog Passes']

def predict_transfer_value(age, position, to_club, from_club_tier, league, 
                           goals=0, assists=0, xg=0, xag=0, minutes=0,
                           progressive_carries=0, progressive_passes=0,
                           transfer_period='Summer', transfer_year=2025):
    """
    Predict transfer value using multiple factors including player performance.
    
    Key factors:
    - Age profile (peak value 24-27)
    - Position value
    - League strength  
    - Club tiers (buying/selling)
    - PLAYER PERFORMANCE (goals, assists, xG, progression)
    - Market inflation
    """
    # Position mapping
    position_mapping = {
        'GK': 'GK', 'DF': 'DEF', 'MF': 'MID', 'FW': 'FWD',
        'FW,MF': 'WING', 'MF,FW': 'WING', 'DF,MF': 'MID', 'MF,DF': 'DEF'
    }
    position_simple = position_mapping.get(position, 'MID')
    
    # === AGE FACTOR ===
    # Peak value at 25-28
    if age <= 20:
        age_factor = 0.90
    elif age <= 23:
        age_factor = 1.10
    elif age <= 27:
        age_factor = 1.20  # Peak years
    elif age <= 29:
        age_factor = 1.05
    elif age <= 31:
        age_factor = 0.80
    elif age <= 33:
        age_factor = 0.50
    else:
        age_factor = 0.25
    
    # === LEAGUE STRENGTH ===
    league_strength_map = {
        'Premier League': 25.0, 'La Liga': 20.0, 'Serie A': 18.0,
        '1. Bundesliga': 17.0, 'Ligue 1': 14.0, 'Eredivisie': 8.0,
        'Liga Nos': 6.0, 'Championship': 4.0, 'Other': 3.0
    }
    league_avg_fee = league_strength_map.get(league, 10.0)
    league_factor = league_avg_fee / 25.0
    
    # === POSITION BASE VALUE (calibrated to 2024/25 market) ===
    position_value_map = {
        'FWD': 28.0, 'WING': 25.0, 'MID': 18.0, 'DEF': 14.0, 'GK': 8.0
    }
    base_value = position_value_map.get(position_simple, 15.0)
    
      # Goals/Assists per 90 (if enough minutes)
    if minutes >= 450:  # At least 5 full games
        games_90 = minutes / 90
        goals_p90 = goals / games_90
        assists_p90 = assists / games_90
        xg_p90 = xg / games_90
        prog_p90 = (progressive_carries + progressive_passes) / games_90
        
        # Performance multipliers based on position
        if position_simple in ['FWD', 'WING']:
            # Attackers valued on goals
            if goals_p90 >= 0.8:
                perf_factor = 1.8  # Elite scorer
            elif goals_p90 >= 0.5:
                perf_factor = 1.5  # Strong scorer
            elif goals_p90 >= 0.3:
                perf_factor = 1.25
            else:
                perf_factor = 0.90
            
            # Bonus for overperforming xG (clinical finisher)
            if goals > xg * 1.2:
                perf_factor *= 1.05
            
            # Bonus for progressive carries (important for wingers)
            if prog_p90 >= 8:
                perf_factor *= 1.15
            elif prog_p90 >= 5:
                perf_factor *= 1.08
                
        elif position_simple == 'MID':
            # Midfielders valued on contribution + progression
            contribution_p90 = goals_p90 + assists_p90
            if contribution_p90 >= 0.6:
                perf_factor = 1.7
            elif contribution_p90 >= 0.4:
                perf_factor = 1.4
            elif contribution_p90 >= 0.2:
                perf_factor = 1.15
            else:
                perf_factor = 0.9
            
            # Bonus for progressive actions
            if prog_p90 >= 10:
                perf_factor *= 1.15
                
        elif position_simple == 'DEF':
            # Defenders valued on progressive actions + clean sheets proxy
            if prog_p90 >= 8:
                perf_factor = 1.5  # Ball-playing defender
            elif prog_p90 >= 5:
                perf_factor = 1.25
            else:
                perf_factor = 1.0
            
            # Bonus for goal contributions (rare for defenders)
            if goals + assists >= 5:
                perf_factor *= 1.2
        else:
            perf_factor = 1.0
    else:
        perf_factor = 0.85  # Not enough minutes - discount
    
    # === CLUB TIER FACTORS ===
    tier_mapping = {
        'Elite': 1.4, 'Top': 1.15, 'Mid': 0.9, 'Lower': 0.6, 'Unknown': 0.4
    }
    tier_factor = tier_mapping.get(to_club, 1.0)
    
    from_tier_mapping = {
        'Elite (Top 10 clubs)': 0.9,   # Hard to buy from elite
        'Top (Top leagues, good clubs)': 1.0,
        'Mid (Mid-table clubs)': 1.05,
        'Lower (Lower leagues)': 1.15,
        'Academy/Unknown': 1.35
    }
    from_factor = from_tier_mapping.get(from_club_tier, 1.0)
    
    # === MARKET FACTORS ===
    period_factor = 0.90 if transfer_period == 'Winter' else 1.0
    year_inflation = 1.0 + (transfer_year - 2020) * 0.03  # 3% yearly inflation
    
    # Young talent premium for elite clubs (only if performing well)
    young_talent_bonus = 1.15 if (age <= 23 and to_club == 'Elite' and perf_factor >= 1.3) else 1.0
    
    # === FINAL CALCULATION ===
    estimated_value = (base_value * age_factor * league_factor * tier_factor * 
                       from_factor * perf_factor * young_talent_bonus * 
                       period_factor * year_inflation)
    
    # === FREE TRANSFER PROBABILITY ===
    free_prob = 0.03
    if age >= 35:
        free_prob = 0.70
    elif age >= 33:
        free_prob = 0.45
    elif age >= 31:
        free_prob = 0.25
    elif age >= 29:
        free_prob = 0.12
    
    # Players leaving elite clubs on decline more likely free
    if from_club_tier == 'Elite (Top 10 clubs)' and to_club in ['Lower', 'Mid'] and age >= 29:
        free_prob += 0.25
    
    free_prob = min(0.95, max(0.02, free_prob))
    
    return {
        'estimated_value': round(estimated_value, 1),
        'free_probability': round(free_prob * 100, 1),
        'age_factor': round(age_factor, 2),
        'league_factor': round(league_factor, 2),
        'position_value': round(base_value, 1),
        'tier_factor': round(tier_factor, 2),
        'performance_factor': round(perf_factor, 2)
    }

# --- LOAD DATA ---
@st.cache_data
def load_data():
    """Load and prepare player data from CSV"""
    df = pd.read_csv("data/players24_25.csv")
    df = df[df['Min'] >= 180].copy()
    df = df.fillna(0)
    
    # Add per 90 stats for fair comparison
    df['Gls_p90'] = (df['Gls'] / df['Min']) * 90
    df['Ast_p90'] = (df['Ast'] / df['Min']) * 90
    df['npxG_p90'] = (df['npxG'] / df['Min']) * 90
    df['xAG_p90'] = (df['xAG'] / df['Min']) * 90
    df['PrgC_p90'] = (df['PrgC'] / df['Min']) * 90
    df['PrgP_p90'] = (df['PrgP'] / df['Min']) * 90
    
    return df

@st.cache_resource
def load_vector_db():
    """Load the ChromaDB vector database"""
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_function
    )
    return db

@st.cache_resource
def get_llm():
    """Initialize the Google Gemini LLM"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

# --- SIMILAR PLAYERS ---
def find_similar_players_statistical(df, player_name, top_k=5):
    """
    Find players with similar playing style using statistical similarity.
    Uses normalized stats to find players with similar profiles regardless of team.
    """
    player_row = df[df['Player'] == player_name]
    if len(player_row) == 0:
        return []
    
    player_data = player_row.iloc[0]
    player_team = player_data.get('Squad', '')
    player_pos = player_data.get('Pos', '')
    
    # Stats to compare for similarity
    compare_stats = ['Gls', 'Ast', 'npxG', 'xAG', 'PrgC', 'PrgP', 'G+A']
    
    # Filter to available stats
    available_stats = [s for s in compare_stats if s in df.columns]
    
    # Get players with same general position (exclude same team)
    # Extract primary position
    primary_pos = player_pos.split(',')[0] if ',' in player_pos else player_pos
    
    # Filter candidates: different team, similar position type
    candidates = df[
        (df['Player'] != player_name) & 
        (df['Squad'] != player_team)
    ].copy()
    
    if len(candidates) == 0:
        return []
    
    # Normalize stats for comparison
    scaler = MinMaxScaler()
    
    # Get player's normalized stats
    player_stats = player_data[available_stats].values.reshape(1, -1)
    all_stats = df[available_stats].values
    
    # Fit scaler on all data
    scaler.fit(all_stats)
    
    player_normalized = scaler.transform(player_stats)[0]
    
    # Calculate similarity for each candidate
    similarities = []
    for idx, row in candidates.iterrows():
        candidate_stats = row[available_stats].values.reshape(1, -1)
        candidate_normalized = scaler.transform(candidate_stats)[0]
        
        # Euclidean distance (lower = more similar)
        distance = np.sqrt(np.sum((player_normalized - candidate_normalized) ** 2))
        
        # Position bonus: prefer same position
        candidate_pos = row.get('Pos', '')
        candidate_primary = candidate_pos.split(',')[0] if ',' in candidate_pos else candidate_pos
        pos_bonus = 0 if candidate_primary == primary_pos else 0.5
        
        similarities.append({
            'player_name': row['Player'],
            'team': row['Squad'],
            'position': row['Pos'],
            'age': int(row['Age']) if pd.notna(row['Age']) else 0,
            'distance': distance + pos_bonus
        })
    
    # Sort by similarity (lowest distance first)
    similarities.sort(key=lambda x: x['distance'])
    
    return similarities[:top_k]

# --- RADAR CHART ---
def create_radar_chart(df, player_names, use_per90=True):
    """
    Create a radar chart comparing players across key metrics.
    """
    if use_per90:
        stats = ['Gls_p90', 'Ast_p90', 'npxG_p90', 'xAG_p90', 'PrgC_p90', 'PrgP_p90']
        labels = ['Goals/90', 'Assists/90', 'npxG/90', 'xAG/90', 'PrgC/90', 'PrgP/90']
    else:
        stats = STAT_COLUMNS
        labels = STAT_LABELS
    
    fig = go.Figure()
    
    # Normalize stats to 0-100 percentile for visualization
    for stat in stats:
        df[f'{stat}_pct'] = df[stat].rank(pct=True) * 100
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, player_name in enumerate(player_names):
        player = df[df['Player'] == player_name]
        if len(player) == 0:
            continue
        player = player.iloc[0]
        
        values = [player[f'{stat}_pct'] for stat in stats]
        values.append(values[0])  # Close the radar
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels + [labels[0]],
            fill='toself',
            name=player_name,
            line_color=colors[i % len(colors)],
            opacity=0.7
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=True,
        title="Player Comparison (Percentile Ranks)"
    )
    
    return fig

def create_full_comparison_chart(df, player1_name, player2_name):
    """
    Create a comprehensive spider chart comparing two players across ALL categories.
    Shows the difference clearly between two players.
    """
    # Define all stat categories for complete comparison
    categories = {
        'Scoring': ['Gls', 'G-PK', 'PK', 'xG', 'npxG'],
        'Creating': ['Ast', 'xAG', 'G+A'],
        'Progression': ['PrgC', 'PrgP', 'PrgR'],
        'Involvement': ['MP', 'Starts', 'Min', '90s']
    }
    
    # Flatten to get all stats we need
    all_stats = ['Gls', 'Ast', 'xG', 'npxG', 'xAG', 'G+A', 'PrgC', 'PrgP', 'PrgR', 'G-PK']
    all_labels = ['Goals', 'Assists', 'xG', 'npxG', 'xAG', 'G+A', 'Prog Carries', 'Prog Passes', 'Prog Received', 'Non-PK Goals']
    
    # Get players
    p1 = df[df['Player'] == player1_name]
    p2 = df[df['Player'] == player2_name]
    
    if len(p1) == 0 or len(p2) == 0:
        return None
    
    p1 = p1.iloc[0]
    p2 = p2.iloc[0]
    
    # Calculate percentile ranks for each stat
    percentiles = {}
    for stat in all_stats:
        if stat in df.columns:
            percentiles[stat] = df[stat].rank(pct=True) * 100
    
    # Get values for both players
    p1_values = []
    p2_values = []
    valid_labels = []
    
    for stat, label in zip(all_stats, all_labels):
        if stat in df.columns:
            p1_idx = df[df['Player'] == player1_name].index[0]
            p2_idx = df[df['Player'] == player2_name].index[0]
            p1_values.append(percentiles[stat].loc[p1_idx])
            p2_values.append(percentiles[stat].loc[p2_idx])
            valid_labels.append(label)
    
    # Close the radar
    p1_values.append(p1_values[0])
    p2_values.append(p2_values[0])
    valid_labels_closed = valid_labels + [valid_labels[0]]
    
    fig = go.Figure()
    
    # Player 1
    fig.add_trace(go.Scatterpolar(
        r=p1_values,
        theta=valid_labels_closed,
        fill='toself',
        name=player1_name,
        line_color='#1f77b4',
        fillcolor='rgba(31, 119, 180, 0.3)',
        opacity=0.8
    ))
    
    # Player 2
    fig.add_trace(go.Scatterpolar(
        r=p2_values,
        theta=valid_labels_closed,
        fill='toself',
        name=player2_name,
        line_color='#ff7f0e',
        fillcolor='rgba(255, 127, 14, 0.3)',
        opacity=0.8
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickmode='array',
                tickvals=[0, 25, 50, 75, 100],
                ticktext=['0%', '25%', '50%', '75%', '100%']
            ),
            angularaxis=dict(
                tickfont=dict(size=11)
            )
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.05,
            xanchor="center",
            x=0.5,
            font=dict(size=13)
        ),
        title=dict(
            text=f"{player1_name} vs {player2_name}",
            x=0.5,
            font=dict(size=16)
        ),
        margin=dict(t=80, b=80),
        height=550
    )
    
    return fig

def create_category_bar_chart(df, player1_name, player2_name):
    """
    Create a grouped bar chart showing direct stat comparison.
    """
    p1 = df[df['Player'] == player1_name].iloc[0]
    p2 = df[df['Player'] == player2_name].iloc[0]
    
    stats = ['Gls', 'Ast', 'npxG', 'xAG', 'PrgC', 'PrgP']
    labels = ['Goals', 'Assists', 'npxG', 'xAG', 'Prog Carries', 'Prog Passes']
    
    p1_vals = [float(p1[s]) for s in stats]
    p2_vals = [float(p2[s]) for s in stats]
    
    fig = go.Figure(data=[
        go.Bar(name=player1_name, x=labels, y=p1_vals, marker_color='#1f77b4'),
        go.Bar(name=player2_name, x=labels, y=p2_vals, marker_color='#ff7f0e')
    ])
    
    fig.update_layout(
        barmode='group',
        title="Direct Statistical Comparison",
        xaxis_title="Stat Category",
        yaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=400
    )
    
    return fig

# --- QUERY PARSER ---
def parse_query_filters(query):
    """
    Parse natural language query to extract filters.
    Returns dict with position, max_age, min_goals extracted from query.
    """
    query_lower = query.lower()
    
    extracted = {
        'position': None,
        'max_age': None,
        'min_goals': None
    }
    
    # Position detection
    position_keywords = {
        'striker': 'FW', 'forward': 'FW', 'attacker': 'FW', 'anfallare': 'FW', 'forwards': 'FW',
        'midfielder': 'MF', 'midfield': 'MF', 'mittfältare': 'MF', 'playmaker': 'MF',
        'defender': 'DF', 'centre-back': 'DF', 'center-back': 'DF', 'centreback': 'DF',
        'mittback': 'DF', 'försvarare': 'DF', 'back': 'DF', 'cb': 'DF', 'fullback': 'DF',
        'goalkeeper': 'GK', 'keeper': 'GK', 'målvakt': 'GK', 'gk': 'GK',
        'winger': 'FW', 'wing': 'FW', 'ytter': 'FW'
    }
    
    for keyword, pos in position_keywords.items():
        if keyword in query_lower:
            extracted['position'] = pos
            break
    
    # Age detection - look for patterns like "under 23", "younger than 25", "under 25 år"
    import re
    age_patterns = [
        r'under\s+(\d+)',
        r'younger than\s+(\d+)',
        r'below\s+(\d+)',
        r'max\s+(\d+)\s*(?:år|years|year)',
        r'(\d+)\s*(?:år|years)?\s*(?:or younger|eller yngre)',
        r'u(\d+)',  # U23, U21 etc.
    ]
    
    for pattern in age_patterns:
        match = re.search(pattern, query_lower)
        if match:
            extracted['max_age'] = int(match.group(1))
            break
    
    # Goals detection
    goal_patterns = [
        r'at least\s+(\d+)\s*goals',
        r'minimum\s+(\d+)\s*goals',
        r'(\d+)\+\s*goals',
        r'minst\s+(\d+)\s*mål',
    ]
    
    for pattern in goal_patterns:
        match = re.search(pattern, query_lower)
        if match:
            extracted['min_goals'] = int(match.group(1))
            break
    
    return extracted

# --- HYBRID SEARCH ---
def hybrid_search(df, db, query, position=None, max_age=None, min_goals=None, min_minutes=None, top_k=10):
    """
    Hybrid search: Parse query for filters, then filter and sort by stats.
    Query-based filters OVERRIDE sidebar filters for more intuitive behavior.
    """
    # Parse query to extract filters from natural language
    query_filters = parse_query_filters(query)
    
    # Query filters override sidebar filters
    effective_position = query_filters['position'] or position
    effective_max_age = query_filters['max_age'] or max_age
    effective_min_goals = query_filters['min_goals'] or min_goals
    
    filtered_df = df.copy()
    
    # Apply position filter
    if effective_position and effective_position != "All":
        filtered_df = filtered_df[filtered_df['Pos'].str.contains(effective_position, na=False)]
    
    # Apply age filter
    if effective_max_age and effective_max_age < 40:
        filtered_df = filtered_df[filtered_df['Age'] <= effective_max_age]
    
    # Apply goals filter
    if effective_min_goals and effective_min_goals > 0:
        filtered_df = filtered_df[filtered_df['Gls'] >= effective_min_goals]
    
    # Apply minutes filter
    if min_minutes and min_minutes > 180:
        filtered_df = filtered_df[filtered_df['Min'] >= min_minutes]
    
    # If no players match, return empty
    if len(filtered_df) == 0:
        return [], filtered_df
    
    # Sort by composite score to get TOP performers
    # Adjust scoring based on position
    if effective_position == 'DF':
        # For defenders, prioritize progressive actions
        filtered_df['score'] = (
            filtered_df['PrgC'] * 0.4 + 
            filtered_df['PrgP'] * 0.4 + 
            filtered_df['Gls'] * 0.5 + 
            filtered_df['Ast'] * 0.3
        )
    elif effective_position == 'MF':
        # For midfielders, balance goals/assists and progression
        filtered_df['score'] = (
            filtered_df['Gls'] * 0.5 + 
            filtered_df['Ast'] * 0.5 + 
            filtered_df['PrgC'] * 0.3 + 
            filtered_df['PrgP'] * 0.3
        )
    elif effective_position == 'GK':
        # For goalkeepers, just use minutes (limited stats)
        filtered_df['score'] = filtered_df['Min']
    else:
        # For forwards, prioritize goals
        filtered_df['score'] = (
            filtered_df['Gls'] + 
            filtered_df['Ast'] * 0.7 + 
            filtered_df['npxG'] * 0.5 + 
            filtered_df['PrgC'] * 0.1
        )
    
    top_players = filtered_df.nlargest(top_k * 2, 'score')['Player'].tolist()
    
    # Get documents for top players from vector DB
    all_docs = db.similarity_search(query, k=100)
    
    filtered_docs = [
        doc for doc in all_docs 
        if doc.metadata.get('player_name') in top_players
    ][:top_k]
    
    # If not enough found, manually add top scorers
    if len(filtered_docs) < top_k:
        found_names = {doc.metadata.get('player_name') for doc in filtered_docs}
        for player_name in top_players:
            if player_name not in found_names and len(filtered_docs) < top_k:
                player_docs = db.similarity_search(player_name, k=1)
                if player_docs:
                    filtered_docs.append(player_docs[0])
                    found_names.add(player_name)
    
    return filtered_docs, filtered_df

def format_docs(docs):
    """Format retrieved documents into context string"""
    return "\n\n".join(doc.page_content for doc in docs)

# =============================================================================
# LANGGRAPH MULTI-AGENT SYSTEM
# =============================================================================

class AgentState(TypedDict):
    """State schema for the multi-agent system"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    context: str
    agent_type: str
    final_response: str

def create_scout_agent(llm):
    """Agent specialized in finding players matching criteria"""
    template = """You are a SCOUT AGENT - an expert at finding football players.
    
Your specialty: Identifying players that match specific criteria (position, age, playing style).

Use this player data:
{context}

User query: {query}

Provide a structured scouting report:
1. List the top matching players with their key stats
2. Explain why each player fits the criteria
3. Recommend the single best match

Be specific with numbers. Answer in the user's language."""
    
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | StrOutputParser()

def create_stats_agent(llm):
    """Agent specialized in statistical analysis"""
    template = """You are a STATS ANALYST AGENT - a football data scientist.
    
Your specialty: Deep statistical analysis, per-90 metrics, xG interpretation, progression stats.

Use this player data:
{context}

User query: {query}

Provide analytical insights:
1. Statistical breakdown of relevant metrics
2. Compare players using advanced metrics (per-90, xG over/underperformance)
3. Identify statistical outliers and trends
4. Give data-driven conclusions

Use precise numbers and percentages. Answer in the user's language."""
    
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | StrOutputParser()

def create_transfer_agent(llm):
    """Agent specialized in transfer market analysis"""
    template = """You are a TRANSFER MARKET AGENT - an expert in player valuations.
    
Your specialty: Assessing player value, contract situations, market trends, transfer feasibility.

Use this player data:
{context}

User query: {query}

Provide transfer market analysis:
1. Estimated market value based on age, stats, and position
2. Compare to similar recent transfers
3. Assess if the player is over/undervalued
4. Recommend whether to pursue and at what price

Consider age curves and market inflation. Answer in the user's language."""
    
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | StrOutputParser()

def create_supervisor(llm):
    """Supervisor that routes queries to the right agent"""
    template = """You are a SUPERVISOR that routes football queries to specialist agents.

Analyze this query: "{query}"

Available agents:
- SCOUT: Finding players matching criteria (position, age, style, recommendations)
- STATS: Statistical analysis (xG, per-90, comparisons, trends)
- TRANSFER: Market value, transfer fees, contract analysis

Respond with ONLY one word - the agent type: SCOUT, STATS, or TRANSFER"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain

def route_query(state: AgentState, llm) -> str:
    """Route the query to the appropriate agent"""
    supervisor = create_supervisor(llm)
    result = supervisor.invoke({"query": state["query"]})
    
    # Parse supervisor decision
    result_upper = result.strip().upper()
    if "STATS" in result_upper:
        return "STATS"
    elif "TRANSFER" in result_upper:
        return "TRANSFER"
    else:
        return "SCOUT"  # Default to scout

def run_scout_agent(state: AgentState, llm) -> AgentState:
    """Execute the scout agent"""
    agent = create_scout_agent(llm)
    response = agent.invoke({"context": state["context"], "query": state["query"]})
    return {
        "messages": [AIMessage(content=response)],
        "final_response": response,
        "agent_type": "Scout Agent"
    }

def run_stats_agent(state: AgentState, llm) -> AgentState:
    """Execute the stats agent"""
    agent = create_stats_agent(llm)
    response = agent.invoke({"context": state["context"], "query": state["query"]})
    return {
        "messages": [AIMessage(content=response)],
        "final_response": response,
        "agent_type": "Stats Analyst Agent"
    }

def run_transfer_agent(state: AgentState, llm) -> AgentState:
    """Execute the transfer agent"""
    agent = create_transfer_agent(llm)
    response = agent.invoke({"context": state["context"], "query": state["query"]})
    return {
        "messages": [AIMessage(content=response)],
        "final_response": response,
        "agent_type": "Transfer Market Agent"
    }

def build_agent_graph(llm):
    """Build the LangGraph multi-agent workflow"""
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes for each agent
    workflow.add_node("scout", lambda state: run_scout_agent(state, llm))
    workflow.add_node("stats", lambda state: run_stats_agent(state, llm))
    workflow.add_node("transfer", lambda state: run_transfer_agent(state, llm))
    
    # Router function
    def router(state: AgentState) -> Literal["scout", "stats", "transfer"]:
        agent_type = route_query(state, llm)
        if agent_type == "STATS":
            return "stats"
        elif agent_type == "TRANSFER":
            return "transfer"
        return "scout"
    
    # Set entry point with conditional routing
    workflow.set_conditional_entry_point(
        router,
        {
            "scout": "scout",
            "stats": "stats",
            "transfer": "transfer"
        }
    )
    
    # All agents lead to END
    workflow.add_edge("scout", END)
    workflow.add_edge("stats", END)
    workflow.add_edge("transfer", END)
    
    return workflow.compile()

def run_multi_agent(query: str, docs: list, llm) -> tuple[str, str]:
    """Run the multi-agent system and return (response, agent_type)"""
    context = format_docs(docs)
    
    # Build and run the graph
    graph = build_agent_graph(llm)
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "context": context,
        "agent_type": "",
        "final_response": ""
    }
    
    result = graph.invoke(initial_state)
    
    return result["final_response"], result["agent_type"]

# =============================================================================
# LEGACY ASK_SCOUT (fallback)
# =============================================================================

def ask_scout(query, docs, llm):
    """Generate AI response using RAG"""
    template = """You are an expert football scout for a top European club.
Use the following context (player statistics) to answer the question.

Rules:
1. Only recommend players found in the context.
2. PRIORITIZE players with the BEST stats (highest goals, xG, assists).
3. Always justify your recommendations with SPECIFIC numbers from the data.
4. Compare players objectively using their actual statistics.
5. If asking for "best" players, recommend those with the highest output.
6. Keep the tone professional and analytical.
7. Answer in the same language as the question.

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    context = format_docs(docs)
    response = chain.invoke({"context": context, "question": query})
    return response

# --- MAIN APP ---
def main():
    st.title("AI Football Scout")
    st.markdown("*Advanced AI-powered player analysis and scouting*")
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY is missing! Add it to your .env file.")
        return
    
    # Load resources
    try:
        df = load_data()
        db = load_vector_db()
        llm = get_llm()
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.info("Run all cells in scout_notebook.ipynb first to create the database.")
        return
    
    # Default settings (query parser extracts filters from questions)
    use_per90 = True
    min_minutes = 500
    
    # Initialize chat history in session state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # --- TABS ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Scout Chat",
        "Scout Search", 
        "Agent System",
        "Compare Players", 
        "Similar Players",
        "Transfer Prediction"
    ])
    
    # --- TAB 1: SCOUT CHAT ---
    with tab1:
        st.subheader("Chat with Scout AI")
        st.markdown("*Have a conversation about players, tactics, and scouting*")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_messages:
                if message["role"] == "user":
                    st.chat_message("user").write(message["content"])
                else:
                    st.chat_message("assistant").write(message["content"])
        
        # Chat input
        user_input = st.chat_input("Ask me anything about football scouting...")
        
        if user_input:
            # Add user message to history
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            
            # Create context from dataset
            query_filters = parse_query_filters(user_input)
            
            # Get relevant players if the question seems to be about specific players
            relevant_context = ""
            if any(word in user_input.lower() for word in ['player', 'spelare', 'striker', 'forward', 'midfielder', 'defender', 'find', 'hitta', 'recommend', 'best', 'top', 'bäst']):
                docs, _ = hybrid_search(df, db, user_input, min_minutes=min_minutes)
                if docs:
                    relevant_context = f"\n\nRelevant player data:\n{format_docs(docs[:5])}"
            
            # Build conversation history for context
            conversation_history = "\n".join([
                f"{'User' if m['role'] == 'user' else 'Scout'}: {m['content']}" 
                for m in st.session_state.chat_messages[-6:]  # Last 6 messages
            ])
            
            # Create the chat prompt
            chat_template = """You are an expert football scout assistant. You help users with:
- Finding and analyzing players from the database
- Explaining football statistics (xG, xAG, progressive carries, etc.)
- Discussing tactics and player roles
- Providing scouting insights and recommendations

Database info: You have access to player stats from top European leagues (2024/25 season).
Available stats: Goals, Assists, xG, npxG, xAG, Progressive Carries, Progressive Passes, Minutes played.

Previous conversation:
{conversation}

{context}

Current question: {question}

Provide a helpful, conversational response. Be specific when discussing players.
Answer in the same language as the question."""

            prompt = ChatPromptTemplate.from_template(chat_template)
            chain = prompt | llm | StrOutputParser()
            
            with st.spinner("Thinking..."):
                response = chain.invoke({
                    "conversation": conversation_history,
                    "context": relevant_context,
                    "question": user_input
                })
            
            # Add assistant response to history
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            
            # Rerun to update chat display
            st.rerun()
        
        # Clear chat button
        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button("Clear Chat"):
                st.session_state.chat_messages = []
                st.rerun()
    
    # --- TAB 2: SCOUT SEARCH ---
    with tab2:
        st.subheader("Ask Scout AI")
        
        query = st.text_input(
            "What are you looking for?",
            placeholder="E.g., Find a young forward with high xG and good progressive carries"
        )
        
        if st.button("Search", type="primary", key="search"):
            if query:
                with st.spinner("Searching..."):
                    docs, filtered_df = hybrid_search(
                        df, db, query,
                        min_minutes=min_minutes
                    )
                    
                    if not docs:
                        st.warning("No players found. Try adjusting filters.")
                    else:
                        st.markdown("### Retrieved Players:")
                        player_data = []
                        player_names_found = []
                        for doc in docs:
                            meta = doc.metadata
                            player_row = df[df['Player'] == meta['player_name']]
                            if len(player_row) > 0:
                                p = player_row.iloc[0]
                                player_names_found.append(meta['player_name'])
                                if use_per90:
                                    player_data.append({
                                        "Player": meta['player_name'],
                                        "Team": meta['team'],
                                        "Pos": meta['position'],
                                        "Age": meta['age'],
                                        "Gls/90": round(p['Gls_p90'], 2),
                                        "Ast/90": round(p['Ast_p90'], 2),
                                        "xG/90": round(p['npxG_p90'], 2),
                                        "PrgC/90": round(p['PrgC_p90'], 2)
                                    })
                                else:
                                    player_data.append({
                                        "Player": meta['player_name'],
                                        "Team": meta['team'],
                                        "Pos": meta['position'],
                                        "Age": meta['age'],
                                        "Goals": int(p['Gls']),
                                        "Assists": int(p['Ast']),
                                        "xG": round(p['npxG'], 1),
                                        "PrgC": int(p['PrgC'])
                                    })
                        
                        st.dataframe(pd.DataFrame(player_data), use_container_width=True)
                        
                        # Radar chart for top players
                        if len(player_names_found) >= 2:
                            st.markdown("### Visual Comparison:")
                            fig = create_radar_chart(df, player_names_found[:4], use_per90)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # AI Analysis with Multi-Agent System
                        st.markdown("### Multi-Agent Analysis:")
                        
                        with st.spinner("Supervisor routing to specialist agent..."):
                            response, agent_type = run_multi_agent(query, docs, llm)
                        
                        # Show which agent responded
                        st.info(f"**Agent Selected:** {agent_type}")
                        st.markdown(response)
    
    # --- TAB 3: AGENT SYSTEM ---
    with tab3:
        st.subheader("Multi-Agent System Architecture")
        
        st.markdown("""
        This application uses a **LangGraph-based Multi-Agent System** with intelligent routing.
        
        ### Architecture Overview
        """)
        
        # Visual architecture diagram using columns
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            ```
            ┌─────────────────┐
            │   USER QUERY    │
            └────────┬────────┘
                    │
                    ▼
            ┌─────────────────┐
            │   SUPERVISOR    │ ← Routes to specialist
            └────────┬────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   ┌───────┐  ┌───────┐  ┌─────────┐
   │ SCOUT │  │ STATS │  │TRANSFER│
   │ AGENT │  │ AGENT │  │  AGENT │
   └───┬───┘  └───┬───┘  └────┬────┘
        │           │           │
        └───────────┼───────────┘
                    ▼
            ┌─────────────────┐
            │    RESPONSE     │
            └─────────────────┘
            ```
            """)
        
        # Agent descriptions
        st.markdown("### Specialist Agents")
        
        agent_col1, agent_col2, agent_col3 = st.columns(3)
        
        with agent_col1:
            st.markdown("""
            #### Scout Agent
            **Specialty:** Player discovery
            
            - Finding players by criteria
            - Position-based search
            - Age/style matching
            - Recommendations
            
            *"Find me a young striker with pace"*
            """)
            
        with agent_col2:
            st.markdown("""
            #### Stats Agent
            **Specialty:** Data analysis
            
            - Per-90 metrics analysis
            - xG interpretation
            - Statistical comparisons
            - Performance trends
            
            *"Analyze Haaland's xG performance"*
            """)
            
        with agent_col3:
            st.markdown("""
            #### Transfer Agent
            **Specialty:** Market analysis
            
            - Player valuations
            - Transfer feasibility
            - Market comparisons
            - ROI assessment
            
            *"What is Bellingham worth?"*
            """)
        
        st.markdown("---")
        
        # Direct agent query
        st.markdown("### Direct Agent Query")
        st.markdown("*Bypass the supervisor and query an agent directly:*")
        
        agent_choice = st.radio(
            "Select Agent:",
            ["Scout Agent", "Stats Agent", "Transfer Agent"],
            horizontal=True
        )
        
        direct_query = st.text_area(
            "Your question:",
            placeholder="Ask the selected agent directly...",
            height=100
        )
        
        if st.button("Query Agent", type="primary"):
            if direct_query:
                with st.spinner(f"Querying {agent_choice}..."):
                    # Get relevant docs
                    docs, _ = hybrid_search(
                        df, db, direct_query,
                        min_minutes=min_minutes
                    )
                    
                    if docs:
                        context = format_docs(docs)
                        
                        # Call the selected agent directly
                        if "Scout" in agent_choice:
                            agent = create_scout_agent(llm)
                        elif "Stats" in agent_choice:
                            agent = create_stats_agent(llm)
                        else:
                            agent = create_transfer_agent(llm)
                        
                        response = agent.invoke({"context": context, "query": direct_query})
                        
                        st.success(f"Response from {agent_choice}:")
                        st.markdown(response)
                    else:
                        st.warning("No players found matching the criteria.")
            else:
                st.warning("Please enter a question.")
        
        # Technical details expander
        with st.expander("Technical Implementation Details"):
            st.markdown("""
            ### LangGraph Implementation
            
            This system uses **LangGraph** to orchestrate multiple AI agents:
            
            ```python
            # State definition
            class AgentState(TypedDict):
                messages: Annotated[Sequence[BaseMessage], operator.add]
                query: str
                context: str
                agent_type: str
                final_response: str
            
            # Graph construction
            workflow = StateGraph(AgentState)
            workflow.add_node("scout", run_scout_agent)
            workflow.add_node("stats", run_stats_agent)
            workflow.add_node("transfer", run_transfer_agent)
            
            # Conditional routing
            workflow.set_conditional_entry_point(router, {...})
            ```
            
            ### Key Features:
            - **State Management:** Tracks conversation and context
            - **Conditional Routing:** Supervisor LLM routes to specialists
            - **Modular Design:** Easy to add new agents
            - **Tool Integration:** Each agent can use different tools
            """)
    
    # --- TAB 4: COMPARE PLAYERS ---
    with tab4:
        st.subheader("Compare Players")
        
        player_names = df['Player'].tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            player1 = st.selectbox("Player 1", player_names, index=0)
        with col2:
            player2 = st.selectbox("Player 2", player_names, index=1)
        
        if st.button("Compare", type="primary", key="compare"):
            p1 = df[df['Player'] == player1].iloc[0]
            p2 = df[df['Player'] == player2].iloc[0]
            
            # Full Spider Chart with ALL categories
            st.markdown("### Complete Profile Comparison (Spider Chart)")
            st.markdown("*Shows percentile rank across all key metrics - higher = better than more players*")
            
            full_spider = create_full_comparison_chart(df, player1, player2)
            if full_spider:
                st.plotly_chart(full_spider, use_container_width=True)
            
            # Bar chart for direct comparison
            st.markdown("### Direct Statistical Comparison")
            bar_chart = create_category_bar_chart(df, player1, player2)
            st.plotly_chart(bar_chart, use_container_width=True)
            
            # Detailed stats table
            st.markdown("### Detailed Statistics")
            if use_per90:
                compare_df = pd.DataFrame({
                    "Stat": ["Team", "Position", "Age", "Minutes", "Gls/90", "Ast/90", "xG/90", "xAG/90", "PrgC/90", "PrgP/90"],
                    player1: [p1['Squad'], p1['Pos'], int(p1['Age']), int(p1['Min']), 
                              round(p1['Gls_p90'], 2), round(p1['Ast_p90'], 2), round(p1['npxG_p90'], 2),
                              round(p1['xAG_p90'], 2), round(p1['PrgC_p90'], 2), round(p1['PrgP_p90'], 2)],
                    player2: [p2['Squad'], p2['Pos'], int(p2['Age']), int(p2['Min']),
                              round(p2['Gls_p90'], 2), round(p2['Ast_p90'], 2), round(p2['npxG_p90'], 2),
                              round(p2['xAG_p90'], 2), round(p2['PrgC_p90'], 2), round(p2['PrgP_p90'], 2)]
                })
            else:
                compare_df = pd.DataFrame({
                    "Stat": ["Team", "Position", "Age", "Minutes", "Goals", "Assists", "npxG", "xAG", "PrgC", "PrgP"],
                    player1: [p1['Squad'], p1['Pos'], int(p1['Age']), int(p1['Min']),
                              int(p1['Gls']), int(p1['Ast']), round(p1['npxG'], 1),
                              round(p1['xAG'], 1), int(p1['PrgC']), int(p1['PrgP'])],
                    player2: [p2['Squad'], p2['Pos'], int(p2['Age']), int(p2['Min']),
                              int(p2['Gls']), int(p2['Ast']), round(p2['npxG'], 1),
                              round(p2['xAG'], 1), int(p2['PrgC']), int(p2['PrgP'])]
                })
            
            st.dataframe(compare_df, use_container_width=True, hide_index=True)
            
            # AI comparison
            with st.spinner("Analyzing..."):
                comparison_query = f"Compare {player1} and {player2}. Who is better and why? Analyze strengths and weaknesses."
                p1_doc = db.similarity_search(player1, k=1)
                p2_doc = db.similarity_search(player2, k=1)
                docs = p1_doc + p2_doc
                response = ask_scout(comparison_query, docs, llm)
                st.markdown("### AI Analysis:")
                st.markdown(response)
    
    # --- TAB 5: SIMILAR PLAYERS ---
    with tab5:
        st.subheader("Find Similar Players")
        st.markdown("*Find players with similar statistical profile from OTHER teams*")
        
        player_names = df['Player'].tolist()
        selected_player = st.selectbox("Select a player", player_names, key="similar")
        
        if st.button("Find Similar", type="primary", key="find_similar"):
            with st.spinner("Finding similar players..."):
                similar = find_similar_players_statistical(df, selected_player, top_k=5)
                
                if similar:
                    st.markdown(f"### Players similar to {selected_player}:")
                    
                    similar_names = [s['player_name'] for s in similar]
                    similar_data = []
                    for s in similar:
                        player_row = df[df['Player'] == s['player_name']]
                        if len(player_row) > 0:
                            p = player_row.iloc[0]
                            similar_data.append({
                                "Player": s['player_name'],
                                "Team": s['team'],
                                "Position": s['position'],
                                "Age": s['age'],
                                "Goals": int(p['Gls']),
                                "Assists": int(p['Ast']),
                                "xG": round(p['npxG'], 1)
                            })
                    
                    st.dataframe(pd.DataFrame(similar_data), use_container_width=True)
                    
                    # Radar comparison
                    st.markdown("### Style Comparison:")
                    fig = create_radar_chart(df, [selected_player] + similar_names[:3], use_per90)
                    st.plotly_chart(fig, use_container_width=True)
    
    # --- TAB 6: TRANSFER PREDICTION ---
    with tab6:
        st.subheader("Transfer Value Prediction")
        st.markdown("*Estimate transfer fees based on age, position, performance & market factors*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Select player from database
            player_names = df['Player'].tolist()
            selected_player_transfer = st.selectbox(
                "Select player from database", 
                player_names, 
                key="transfer_player"
            )
            
            # Get player data
            player_data = df[df['Player'] == selected_player_transfer].iloc[0]
            player_age = int(player_data['Age'])
            player_pos = player_data['Pos']
            player_team = player_data['Squad']
            player_goals = int(player_data['Gls'])
            player_assists = int(player_data['Ast'])
            player_xg = float(player_data['npxG'])
            player_xag = float(player_data['xAG'])
            player_minutes = int(player_data['Min'])
            player_prgc = int(player_data['PrgC'])
            player_prgp = int(player_data['PrgP'])
            
            st.markdown(f"**Current Team:** {player_team}")
            st.markdown(f"**Position:** {player_pos} | **Age:** {player_age}")
            st.markdown(f"**Stats:** {player_goals} goals, {player_assists} assists ({player_minutes} min)")
            st.markdown(f"**xG:** {round(player_xg, 1)} | **xAG:** {round(player_xag, 1)}")
            st.markdown(f"**Progression:** {player_prgc} carries, {player_prgp} passes")
        
        with col2:
            # Transfer scenario inputs
            to_club_tier = st.selectbox(
                "Destination Club Tier",
                ["Elite", "Top", "Mid", "Lower"],
                help="Elite: Real Madrid, Man City | Top: Arsenal, Juventus | Mid: Everton, Valencia"
            )
            
            from_club_tier = st.selectbox(
                "Current Club Tier",
                ["Elite (Top 10 clubs)", "Top (Top leagues, good clubs)", "Mid (Mid-table clubs)", "Lower (Lower leagues)", "Academy/Unknown"]
            )
            
            league = st.selectbox(
                "Destination League",
                ["Premier League", "La Liga", "1. Bundesliga", "Serie A", "Ligue 1", "Eredivisie", "Liga Nos", "Championship", "Other"]
            )
            
            transfer_period = st.radio("Transfer Window", ["Summer", "Winter"], horizontal=True)
        
        if st.button("Predict Transfer Value", type="primary", key="predict_transfer"):
            with st.spinner("Calculating transfer value..."):
                prediction = predict_transfer_value(
                    age=player_age,
                    position=player_pos,
                    to_club=to_club_tier,
                    from_club_tier=from_club_tier,
                    league=league,
                    goals=player_goals,
                    assists=player_assists,
                    xg=player_xg,
                    xag=player_xag,
                    minutes=player_minutes,
                    progressive_carries=player_prgc,
                    progressive_passes=player_prgp,
                    transfer_period=transfer_period
                )
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Estimated Transfer Value",
                        f"€{prediction['estimated_value']}M",
                        help="Predicted transfer fee in millions"
                    )
                
                with col2:
                    st.metric(
                        "Free Transfer Probability",
                        f"{prediction['free_probability']}%",
                        help="Likelihood of leaving on a free transfer"
                    )
                
                with col3:
                    st.metric(
                        "Performance Factor",
                        f"{prediction['performance_factor']}x",
                        help="Multiplier based on goals, assists, and progression"
                    )
                
                # Factor breakdown
                st.markdown("### Value Factors Breakdown:")
                
                factors_df = pd.DataFrame({
                    'Factor': ['Age Profile', 'League Strength', 'Club Tier', 'Performance', 'Position Base'],
                    'Multiplier': [
                        prediction['age_factor'], 
                        prediction['league_factor'], 
                        prediction['tier_factor'],
                        prediction['performance_factor'],
                        prediction['position_value'] / 20
                    ],
                    'Description': [
                        f"Age {player_age} impact",
                        f"{league} market",
                        f"Moving to {to_club_tier} club",
                        f"Based on {player_goals}G, {player_assists}A",
                        f"{player_pos} base value"
                    ]
                })
                
                fig = px.bar(
                    factors_df,
                    x='Factor',
                    y='Multiplier',
                    color='Multiplier',
                    color_continuous_scale='RdYlGn',
                    title="Transfer Value Factors (>1.0 = adds value, <1.0 = reduces value)",
                    hover_data=['Description']
                )
                fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Baseline")
                st.plotly_chart(fig, use_container_width=True)
                
                # Explanation
                with st.expander("How is this calculated?"):
                    st.markdown(f"""
                    **Base Value:** €{prediction['position_value']}M (average for {player_pos})
                    
                    **Multipliers applied:**
                    - Age ({player_age}): **{prediction['age_factor']}x** - {'Peak years!' if prediction['age_factor'] >= 1.4 else 'Some age discount' if prediction['age_factor'] < 1.0 else 'Good age'}
                    - League ({league}): **{prediction['league_factor']}x** - Market strength
                    - Club tier ({to_club_tier}): **{prediction['tier_factor']}x** - Elite clubs pay premium
                    - Performance: **{prediction['performance_factor']}x** - Based on actual output
                    
                    **Key performance stats used:**
                    - Goals: {player_goals} | Assists: {player_assists}
                    - xG: {round(player_xg, 1)} | xAG: {round(player_xag, 1)}
                    - Progressive actions: {player_prgc + player_prgp}
                    - Minutes: {player_minutes}
                    """)
                
                # AI Analysis
                st.markdown("### AI Transfer Analysis:")
                transfer_query = f"""Analyze the transfer potential for {selected_player_transfer}:
                - Age: {player_age}
                - Position: {player_pos}
                - Current team: {player_team}
                - Goals: {player_goals}, Assists: {player_assists}
                - xG: {round(player_xg, 1)}, xAG: {round(player_xag, 1)}
                - Estimated value: €{prediction['estimated_value']}M
                
                Estimated value: {prediction['estimated_value']}M
                Is this player worth the investment? What are the risks?"""
                
                player_docs = db.similarity_search(selected_player_transfer, k=1)
                if player_docs:
                    response = ask_scout(transfer_query, player_docs, llm)
                    st.markdown(response)

if __name__ == "__main__":
    main()
