# AI Football Scout
## Utveckling av ett AI-drivet scoutingsystem för fotboll

**Kurs:** Kunskapskontroll AI  
**Datum:** Januari 2026  
**Författare:** Sebastian Holmberg

---

## Innehållsförteckning

1. [Inledning](#1-inledning)
2. [Syfte och mål](#2-syfte-och-mål)
3. [Teknisk arkitektur](#3-teknisk-arkitektur)
4. [Implementation](#4-implementation)
5. [AI- och ML-komponenter](#5-ai--och-ml-komponenter)
6. [Resultat och demonstration](#6-resultat-och-demonstration)
7. [Diskussion och reflektion](#7-diskussion-och-reflektion)
8. [Slutsats](#8-slutsats)
9. [Referenser](#9-referenser)

---

## 1. Inledning

Fotbollsindustrin har genomgått en digital transformation där dataanalys spelar en allt viktigare roll i spelarrekrytering och scouting. Traditionell scouting, som bygger på subjektiva observationer, kompletteras nu med avancerad statistik och AI-drivna analysverktyg.

Detta projekt presenterar **AI Football Scout** – en fullständig webbapplikation som kombinerar modern AI-teknologi med fotbollsstatistik för att assistera i scoutingprocessen. Applikationen använder tekniker som Retrieval-Augmented Generation (RAG), multi-agent system och maskininlärning för att analysera spelardata och ge rekommendationer.

### 1.1 Bakgrund

Moderna fotbollsklubbar investerar stora resurser i dataanalys. Mätvärden som *Expected Goals (xG)*, *Expected Assists (xAG)* och progressiva aktioner har blivit standard i spelarutvärdering. Utmaningen ligger i att:

- Hantera stora datamängder effektivt
- Extrahera meningsfulla insikter från statistik
- Kombinera kvantitativ data med kvalitativ analys

AI-teknologi, särskilt stora språkmodeller (LLM), erbjuder nya möjligheter att lösa dessa utmaningar.

---

## 2. Syfte och mål

### 2.1 Projektets syfte

Syftet med projektet är att utveckla en AI-driven applikation som demonstrerar praktisk tillämpning av:

- **Retrieval-Augmented Generation (RAG)** för kontextmedveten AI
- **Multi-agent system** med LangGraph för specialiserad analys
- **Vektordatabaser** för semantisk sökning
- **Maskininlärning** för klustring och värdeprediktion

### 2.2 Funktionella mål

Applikationen ska kunna:

1. Söka efter spelare baserat på naturligt språk
2. Jämföra spelare visuellt med radar-/spiderdiagram
3. Hitta liknande spelare baserat på statistisk profil
4. Prediktera transfervärden med multipla faktorer
5. Kategorisera spelare i typer med klustringsalgoritmer
6. Föra konversationer om scouting och spelareanalys

---

## 3. Teknisk arkitektur

### 3.1 Systemöversikt

Applikationen är uppbyggd i lager enligt följande arkitektur:

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND                        │
│              (Interaktivt webbgränssnitt)                   │
├─────────────────────────────────────────────────────────────┤
│              LANGGRAPH MULTI-AGENT SYSTEM                    │
│  ┌───────────┐    ┌───────────┐    ┌─────────────┐         │
│  │   SCOUT   │    │   STATS   │    │  TRANSFER   │         │
│  │   AGENT   │    │   AGENT   │    │    AGENT    │         │
│  └───────────┘    └───────────┘    └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│           HYBRID SEARCH (SQL + Semantic)                     │
├─────────────────────────────────────────────────────────────┤
│  ChromaDB      │  Google Gemini  │  Scikit-learn │  Pandas  │
│  (Vektorer)    │     (LLM)       │     (ML)      │  (Data)  │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Teknologistack

| Komponent | Teknologi | Syfte |
|-----------|-----------|-------|
| Frontend | Streamlit | Interaktivt webbgränssnitt |
| LLM | Google Gemini 2.0 Flash | Textgenerering och analys |
| Vektordatabas | ChromaDB | Semantisk sökning |
| Embeddings | Google Generative AI | Textrepresentation |
| Orkestrering | LangGraph | Multi-agent koordinering |
| ML | Scikit-learn | Klustring, normalisering |
| Visualisering | Plotly | Interaktiva diagram |
| Databehandling | Pandas, NumPy | Statistikberäkningar |

### 3.3 Dataflöde

1. **Indata:** Spelarstatistik från FBref (säsong 2024/25)
2. **Preprocessing:** Filtrering, beräkning av per-90 statistik
3. **Vektorisering:** Scouting-rapporter konverteras till embeddings
4. **Lagring:** Embeddings sparas i ChromaDB
5. **Sökning:** Hybrid av SQL-filter och semantisk sökning
6. **Analys:** LLM genererar insikter baserat på kontext
7. **Presentation:** Visualiseringar och text i Streamlit

---

## 4. Implementation

### 4.1 Dataförberedelse

Spelardata från FBref innehåller över 2300 spelare från Europas toppligor. Varje spelare har cirka 40 statistiska attribut. Förbehandling inkluderar:

```python
# Filtrera spelare med tillräcklig speltid
df = df[df['Min'] >= 180]

# Beräkna per-90 statistik för rättvis jämförelse
df['Gls_p90'] = (df['Gls'] / df['Min']) * 90
df['Ast_p90'] = (df['Ast'] / df['Min']) * 90
df['npxG_p90'] = (df['npxG'] / df['Min']) * 90
```

Per-90 statistik är viktigt för att kunna jämföra spelare med olika speltid på ett rättvist sätt.

### 4.2 Vektordatabas och RAG

För att möjliggöra semantisk sökning skapas textbaserade "scouting reports" för varje spelare:

```python
def create_scouting_report(row):
    return f"""
    Player: {row['Player']}
    Team: {row['Squad']}
    Position: {row['Pos']}
    Age: {row['Age']}
    Goals: {row['Gls']}, Assists: {row['Ast']}
    xG: {row['npxG']}, xAG: {row['xAG']}
    Progressive Carries: {row['PrgC']}
    Progressive Passes: {row['PrgP']}
    """
```

Dessa rapporter konverteras till vektorer med Google Generative AI Embeddings och lagras i ChromaDB. Vid sökning:

1. Användarens fråga konverteras till en vektor
2. ChromaDB hittar liknande spelarprofiler
3. Relevanta spelare skickas som kontext till LLM:en

### 4.3 Hybrid sökning

Ren semantisk sökning räcker inte för kvantitativa frågor som "hitta anfallare under 23 år". Därför implementerades en hybrid approach:

```python
def hybrid_search(df, db, query, ...):
    # 1. Parsera naturligt språk för filter
    query_filters = parse_query_filters(query)
    
    # 2. Applicera SQL-liknande filter
    if query_filters['position']:
        filtered_df = df[df['Pos'].str.contains(position)]
    if query_filters['max_age']:
        filtered_df = filtered_df[filtered_df['Age'] <= max_age]
    
    # 3. Sortera efter relevant statistik
    filtered_df['score'] = calculate_composite_score(...)
    
    # 4. Hämta semantiskt relevanta dokument
    docs = db.similarity_search(query)
    
    # 5. Kombinera resultaten
    return filtered_docs, filtered_df
```

Query-parsern extraherar automatiskt filter från naturligt språk:
- "mittback" → Position = DF
- "under 23" → Max ålder = 23
- "striker" → Position = FW

### 4.4 Multi-Agent System

LangGraph används för att orkestrera specialiserade AI-agenter:

**Scout Agent:** Specialiserad på att hitta spelare som matchar specifika kriterier.

**Stats Agent:** Fokuserar på djupgående statistisk analys och xG-tolkning.

**Transfer Agent:** Utvärderar marknadsvärde och transferpotential.

En **Supervisor** analyserar varje fråga och dirigerar den till lämplig agent:

```python
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    query: str
    context: str
    agent_type: str
    final_response: str

def build_agent_graph(llm):
    workflow = StateGraph(AgentState)
    
    workflow.add_node("scout", run_scout_agent)
    workflow.add_node("stats", run_stats_agent)
    workflow.add_node("transfer", run_transfer_agent)
    
    workflow.set_conditional_entry_point(router, {...})
    
    return workflow.compile()
```

### 4.5 Transfer Value Prediction

Transfervärdeprediktionen använder en regelbaserad modell kalibrerad mot verkliga transfersummor:

**Faktorer som påverkar värdet:**

| Faktor | Beskrivning | Påverkan |
|--------|-------------|----------|
| Ålder | Peak vid 25-28 år | ±20% |
| Position | Anfallare värderas högst | Bas 8-28M€ |
| Liga | Premier League = högst | ±25% |
| Prestation | Mål, assists, xG | ±50% |
| Progression | Carries, passes | ±15% |
| Klubbnivå | Elite/Top/Mid/Lower | ±40% |

Formeln kombinerar dessa faktorer multiplikativt:

```
Värde = Basvärde × Åldersfaktor × Ligafaktor × 
        Klubbfaktor × Prestationsfaktor × Inflation
```

### 4.6 Spelarklustring

K-Means klustring används för att automatiskt kategorisera spelare:

```python
def cluster_players(df, n_clusters=6):
    features = ['Gls_p90', 'Ast_p90', 'npxG_p90', 
                'xAG_p90', 'PrgC_p90', 'PrgP_p90']
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    kmeans = KMeans(n_clusters=n_clusters)
    df['cluster'] = kmeans.fit_predict(X_scaled)
```

Klustren mappas till beskrivande spelartypkategorier som "Elite Scorers", "Creative Playmakers" och "Progressive Carriers".

### 4.7 Similar Players

För att hitta liknande spelare används euklidiskt avstånd i normaliserat statistikrum:

```python
def find_similar_players(df, player_name, top_k=5):
    # Normalisera statistik
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(df[stats])
    
    # Beräkna avstånd
    player_vec = get_player_vector(player_name)
    distances = euclidean_distance(player_vec, all_players)
    
    # Exkludera samma lag för scouting-relevans
    similar = filter_different_team(distances)
    
    return similar[:top_k]
```

Viktigt: Funktionen exkluderar spelare från samma lag för att ge relevanta scoutingförslag.

---

## 5. AI- och ML-komponenter

### 5.1 Large Language Model (LLM)

Projektet använder **Google Gemini 2.0 Flash** för:

- Generering av naturliga textsvar
- Analys och jämförelse av spelare
- Konversationshantering i chatten
- Routing av frågor till rätt agent

Modellen är konfigurerad med låg temperatur (0.3) för mer deterministiska och faktabaserade svar.

### 5.2 Embeddings och vektorsökning

**Google Generative AI Embeddings** konverterar text till 768-dimensionella vektorer. Detta möjliggör:

- Semantisk likhet ("hitta snabba anfallare" matchar spelare med hög dribbling)
- Kontextförståelse (synonymer och relaterade koncept)
- Effektiv sökning i stor datamängd

### 5.3 LangGraph och agentorkestration

LangGraph implementerar ett **State Graph** där:

- **Noder** representerar agenter
- **Kanter** definierar arbetsflöde
- **State** bevaras genom processen

Detta möjliggör mer komplex resonering än enkel prompt-chaining.

### 5.4 Scikit-learn för klustring

K-Means algoritmen grupperar spelare baserat på likhet i statistik. Fördelar:

- Snabb att träna
- Tolkningsbara resultat
- Skalbar till stora datamängder

---

## 6. Resultat och demonstration

### 6.1 Applikationens funktioner

Den färdiga applikationen har **7 funktionella flikar**:

1. **Scout Chat** - Konversationsbaserad interaktion med minne
2. **Scout Search** - Avancerad sökning med multi-agent analys
3. **Agent System** - Visualisering av AI-arkitekturen
4. **Compare Players** - Spiderdiagram för spelarjämförelse
5. **Similar Players** - Hitta liknande spelare från andra lag
6. **Transfer Prediction** - Uppskatta transfervärde
7. **Player Clusters** - ML-baserad spelarkategorisering

### 6.2 Exempelfrågor och svar

**Fråga:** "Hitta en mittback under 25 år med bra progressiva aktioner"

**System:**
1. Query parser extraherar: Position=DF, MaxAge=25
2. Hybrid search filtrerar och sorterar på PrgC+PrgP
3. Scout Agent genererar analys med specifika rekommendationer

**Fråga:** "Vad är Noni Madueke värd?"

**System:**
1. Transfer Agent aktiveras
2. Spelardata hämtas: 22 år, FW, 7 mål, 154 prog carries
3. Modellen beräknar: ~65M€ (kalibrerat mot verklig transfer)

### 6.3 Prestanda

- Svarstid för sökning: ~2-3 sekunder
- Vektordatabas med 2300+ spelare
- Stabil drift med Streamlit

---

## 7. Diskussion och reflektion

### 7.1 Lärdomar

**RAG-arkitektur:** Att ge LLM:en kontext från en vektordatabas förbättrar avsevärt kvaliteten på svaren. Utan detta "hallucinerar" modellen ofta fakta.

**Hybrid sökning:** Ren semantisk sökning fungerar dåligt för kvantitativa frågor. Kombinationen med SQL-liknande filter är nödvändig.

**Kalibrering:** ML-modeller måste valideras mot verkliga data. Den initiala transfermodellen överskattade värden kraftigt.

### 7.2 Begränsningar

- **Dataaktualitet:** Statistiken är från 2024/25 och uppdateras inte i realtid
- **Kontraktsdata saknas:** Påverkar transfervärden men finns ej i datasetet
- **Begränsad position-data:** Endast övergripande positioner (FW/MF/DF/GK)

### 7.3 Förbättringsmöjligheter

- Integration med live-API för uppdaterad statistik
- Fler dataattribut (kontrakt, skador, spelminuter per match)
- Utökad klustring med fler dimensioner
- Jämförelse med faktiska transfersummor för modellvalidering

---

## 8. Slutsats

Projektet demonstrerar framgångsrik integration av flera AI- och ML-tekniker i en praktisk applikation. De viktigaste bidragen är:

1. **Multi-Agent arkitektur** med LangGraph som dirigerar specialiserade agenter
2. **Hybrid sökning** som kombinerar semantisk förståelse med kvantitativ filtrering
3. **RAG-implementation** som förankrar LLM-svar i faktisk spelardata
4. **Transfervärdemodell** kalibrerad mot verkliga marknadsdata

Applikationen visar hur moderna AI-verktyg kan tillämpas på ett verkligt domänproblem och ge mervärde genom att kombinera mänsklig intuition med datadriven analys.

---

## 9. Referenser

### Teknisk dokumentation

- LangChain Documentation. https://python.langchain.com/
- LangGraph Documentation. https://langchain-ai.github.io/langgraph/
- Streamlit Documentation. https://docs.streamlit.io/
- ChromaDB Documentation. https://docs.trychroma.com/
- Scikit-learn User Guide. https://scikit-learn.org/stable/user_guide.html

### Fotbollsstatistik

- FBref.com - Football Statistics and History
- StatsBomb - Expected Goals (xG) Methodology

### AI och Machine Learning

- Google AI - Gemini API Documentation
- Vaswani et al. (2017) - Attention Is All You Need
- Lewis et al. (2020) - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

---

## Bilagor

### A. Teknisk specifikation

**Python-version:** 3.13  
**Ramverk:** Streamlit  
**LLM:** Google Gemini 2.0 Flash  
**Vektordatabas:** ChromaDB  
**ML-bibliotek:** Scikit-learn  

### B. Filstruktur

```
ai-football-scout/
├── app.py                 # Huvudapplikation
├── scout_notebook.ipynb   # Dataförberedelse
├── requirements.txt       # Beroenden
├── data/
│   └── players24_25.csv   # Spelarstatistik
├── chroma_db/             # Vektordatabas
└── .env                   # API-nycklar
```

### C. Installationsinstruktioner

```bash
# Skapa virtuell miljö
python -m venv .venv
source .venv/bin/activate

# Installera beroenden
pip install -r requirements.txt

# Konfigurera API-nyckel
echo "GOOGLE_API_KEY=din_nyckel" > .env

# Kör notebook för att skapa vektordatabas
jupyter notebook scout_notebook.ipynb

# Starta applikationen
streamlit run app.py
```

---

*Rapport genererad januari 2026*
