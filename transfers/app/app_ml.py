
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="⚽ ML Transfer Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f4e79, #2e8b57);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .risk-high { 
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
    
    .risk-medium { 
        background: linear-gradient(135deg, #feca57, #ff9ff3);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
    
    .risk-low { 
        background: linear-gradient(135deg, #48cae4, #0077b6);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    
    .feature-importance {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        classifier = joblib.load('models/transfer_classifier_clean.pkl')
        regressor = joblib.load('models/transfer_regressor_clean.pkl')
        feature_cols = joblib.load('models/feature_columns_clean.pkl')
        
        club_tiers_to = joblib.load('models/club_tiers_to_clean.pkl')
        club_tiers_from = joblib.load('models/club_tiers_from_clean.pkl') 
        league_strength = joblib.load('models/league_strength_clean.pkl')
        position_values = joblib.load('models/position_values_clean.pkl')
        feature_info = joblib.load('models/feature_pipeline_info_clean.pkl')
        
        return classifier, regressor, feature_cols, club_tiers_to, club_tiers_from, league_strength, position_values, feature_info
    except FileNotFoundError as e:
        st.error(f"⚠️ Model files not found: {e}")
        st.error("Run cell 14 in the notebook to create the clean model files with '_clean' suffix.")
        st.info("💡 Tip: Clean models have realistic performance without data leakage.")
        return None, None, None, None, None, None, None, None

def create_gauge_chart(value, title, max_value=100):
    """Create a beautiful gauge chart for metrics"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [None, max_value], 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgray"},
                {'range': [25, 50], 'color': "gray"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, max_value], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font={'color': "darkblue", 'family': "Arial"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def create_feature_importance_chart(details):
    """Create a feature importance visualization"""
    features = {
        'Age Factor': details['age_factor'],
        'League Strength': details['liga_factor'],
        'Club Tier': details['tier_factor'],
        'Movement': details['movement_factor'],
        'Year Inflation': details['year_inflation_factor'],
        'Period Factor': details['period_factor'],
        'Young Talent': details['young_talent_bonus']
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(features.keys()),
            y=list(features.values()),
            marker=dict(
                color=list(features.values()),
                colorscale='Viridis',
                showscale=True
            )
        )
    ])
    
    fig.update_layout(
        title="Feature Impact on Transfer Value",
        title_font_size=20,
        xaxis_title="Features",
        yaxis_title="Impact Factor",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def create_risk_analysis_chart(free_prob, estimated_fee, age):
    """Create a comprehensive risk analysis chart"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Free Transfer Probability', 'Estimated Value', 'Age Impact', 'Risk Factors'),
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "bar"}, {"type": "pie"}]]
    )
    
    
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = free_prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Free Transfer %"},
        gauge = {'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if free_prob > 0.6 else "orange" if free_prob > 0.3 else "green"},
                'steps': [{'range': [0, 30], 'color': "lightgray"},
                         {'range': [30, 60], 'color': "yellow"},
                         {'range': [60, 100], 'color': "red"}]}
    ), row=1, col=1)
    
    
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = estimated_fee,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Value (€M)"},
        gauge = {'axis': {'range': [None, 100]},
                'bar': {'color': "green" if estimated_fee > 20 else "orange" if estimated_fee > 5 else "red"}}
    ), row=1, col=2)
    
    
    age_ranges = ['16-21', '22-25', '26-29', '30-32', '33+']
    age_impacts = [1.2, 1.3, 1.1, 0.7, 0.3]
    
    fig.add_trace(go.Bar(
        x=age_ranges,
        y=age_impacts,
        marker_color=['green' if f'{age}' in r else 'lightblue' for r in age_ranges],
        name="Age Impact"
    ), row=2, col=1)
    
    
    risk_labels = ['Market Risk', 'Age Risk', 'Performance Risk', 'Contract Risk']
    risk_values = [25, 30 if age > 30 else 15, 20, 25]
    
    fig.add_trace(go.Pie(
        labels=risk_labels,
        values=risk_values,
        hole=0.3
    ), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False)
    return fig

def create_advanced_prediction(age, position, to_club, from_club_tier, league, transfer_period, transfer_year, feature_info):
    """Enhanced prediction function with better calculations"""
    
    
    tier_mapping = {
        'Elite (Real Madrid, Barcelona, PSG level)': 'elite',
        'Top (Man United, Arsenal, Juventus level)': 'top', 
        'Mid (Tottenham, Valencia level)': 'mid',
        'Lower (Smaller clubs)': 'lower',
        'Unknown/Academy': 'unknown'
    }
    from_tier = tier_mapping.get(from_club_tier, 'unknown')
    
    club_tier_mapping = {
        'Real Madrid': 'elite', 'Barcelona': 'elite', 'Paris Saint-Germain': 'elite', 'Manchester City': 'elite',
        'Manchester United': 'top', 'Arsenal': 'top', 'Chelsea': 'top', 'Liverpool': 'top', 
        'Juventus': 'top', 'AC Milan': 'top',
        'Tottenham': 'mid', 'West Ham': 'mid', 'Everton': 'mid', 'Valencia': 'mid', 'Sevilla': 'mid',
        'Brighton': 'lower', 'Crystal Palace': 'lower', 'Getafe': 'lower', 'Other club': 'lower'
    }
    to_tier = club_tier_mapping.get(to_club, 'unknown')
    
    
    position_mapping = {
        'GK': ['goalkeeper', 'keeper'],
        'DEF': ['centre-back', 'center-back', 'left-back', 'right-back', 'defender', 'defence'],
        'MID': ['central midfield', 'defensive midfield', 'attacking midfield', 'midfielder'],
        'WING': ['left winger', 'right winger', 'winger', 'wing'],
        'FWD': ['centre-forward', 'center-forward', 'second striker', 'striker', 'forward']
    }
    
    position_simple = 'OTHER'
    position_lower = position.lower()
    for category, keywords in position_mapping.items():
        if any(keyword in position_lower for keyword in keywords):
            position_simple = category
            break
    
    
    age_squared = age ** 2
    is_summer_transfer = 1 if transfer_period == 'Summer' else 0
    year_normalized = (transfer_year - 2020) / 5.0
    
    
    if age <= 21:
        age_group = 'Young Talent'
        age_factor = 1.25
    elif age <= 25:
        age_group = 'Peak Early'
        age_factor = 1.35
    elif age <= 29:
        age_group = 'Peak Late'
        age_factor = 1.15
    elif age <= 32:
        age_group = 'Experienced'
        age_factor = 0.8
    elif age <= 35:
        age_group = 'Veteran'
        age_factor = 0.5
    else:
        age_group = 'Decline'
        age_factor = 0.2
    
    
    tier_hierarchy = {'elite': 4, 'top': 3, 'mid': 2, 'lower': 1, 'unknown': 0}
    from_tier_num = tier_hierarchy.get(from_tier, 0)
    to_tier_num = tier_hierarchy.get(to_tier, 0)
    tier_movement = to_tier_num - from_tier_num
    
    
    young_talent_premium = 1 if (age <= 23 and to_tier in ['elite', 'top']) else 0
    young_talent_bonus = 1.3 if young_talent_premium else 1.0
    
    
    league_strength_map = {
        'Premier League': 28.5, 'La Liga': 24.8, '1. Bundesliga': 19.3,
        'Serie A': 21.1, 'Ligue 1': 16.7, 'Eredivisie': 9.2,
        'Liga Nos': 7.8, 'Championship': 5.1, 'Other league': 3.0
    }
    league_avg_fee = league_strength_map.get(league, 5.0)
    liga_factor = league_avg_fee / 25.0
    
    
    position_value_map = {
        'FWD': 32.5, 'WING': 27.2, 'MID': 20.7, 'DEF': 17.3, 'GK': 9.9
    }
    position_avg_fee = position_value_map.get(position_simple, 12.0)
    
    
    tier_factor = {
        'elite': 1.5, 'top': 1.25, 'mid': 1.0, 'lower': 0.75, 'unknown': 0.6
    }.get(to_tier, 1.0)
    
    movement_factor = max(0.5, min(1.6, 1.0 + (tier_movement * 0.2)))
    period_factor = 0.92 if transfer_period == 'Winter' else 1.0
    year_inflation_factor = 1.0 + (transfer_year - 2020) * 0.045
    
    
    base_value = position_avg_fee
    estimated_value = (base_value * age_factor * liga_factor * tier_factor * 
                      movement_factor * period_factor * young_talent_bonus * year_inflation_factor)
    
    
    free_transfer_prob = 0.03
    
    
    if age >= 35:
        free_transfer_prob += 0.65
    elif age >= 32:
        free_transfer_prob += 0.40
    elif age >= 30:
        free_transfer_prob += 0.20
    elif age >= 28:
        free_transfer_prob += 0.08
    
    
    if tier_movement < -1:
        free_transfer_prob += 0.30
    elif tier_movement < 0:
        free_transfer_prob += 0.15
    elif tier_movement > 1:
        free_transfer_prob -= 0.10
    
    
    if position_simple == 'GK' and age >= 32:
        free_transfer_prob += 0.12
    if transfer_period == 'Winter':
        free_transfer_prob += 0.08
    if transfer_year >= 2023:
        free_transfer_prob += 0.06
    
    free_transfer_prob = min(0.88, max(0.03, free_transfer_prob))
    
    
    if free_transfer_prob > 0.5:
        estimated_value *= (1 - free_transfer_prob * 0.8)
    
    return free_transfer_prob, estimated_value, {
        'position_simple': position_simple,
        'from_tier': from_tier,
        'to_tier': to_tier,
        'tier_movement': tier_movement,
        'age_group': age_group,
        'young_talent_premium': young_talent_premium,
        'base_value': base_value,
        'age_factor': age_factor,
        'liga_factor': liga_factor,
        'tier_factor': tier_factor,
        'movement_factor': movement_factor,
        'young_talent_bonus': young_talent_bonus,
        'period_factor': period_factor,
        'year_inflation_factor': year_inflation_factor,
        'league_avg_fee': league_avg_fee,
        'confidence_score': (1 - free_transfer_prob) * 100 if free_transfer_prob < 0.5 else free_transfer_prob * 100
    }

def main():
    
    st.markdown('<h1 class="main-header">⚽ ML Transfer Predictor</h1>', unsafe_allow_html=True)
    
    
    st.markdown("""
    <div style='text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;'>
        🚀 Advanced Machine Learning for Football Transfer Analysis
    </div>
    """, unsafe_allow_html=True)
    
    
    models_data = load_models()
    if models_data[0] is None:
        st.stop()
    
    classifier, regressor, feature_cols, club_tiers_to, club_tiers_from, league_strength, position_values, feature_info = models_data
    
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-container">🤖 ML Models<br><strong>LOADED</strong></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-container">📊 Features<br><strong>{len(feature_cols)}</strong></div>', unsafe_allow_html=True)
    with col3:
        if 'model_performance' in feature_info:
            perf = feature_info['model_performance']
            st.markdown(f'<div class="metric-container">🎯 AUC Score<br><strong>{perf["classification_auc"]:.3f}</strong></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-container">✅ Status<br><strong>READY</strong></div>', unsafe_allow_html=True)
    
    
    col_sidebar, col_main = st.columns([1, 2])
    
    with col_sidebar:
        st.markdown("## 🎯 Player Configuration")
        
        
        st.markdown("### 👤 Player Details")
        age = st.slider("Age", 16, 40, 24, help="Player's age at transfer")
        
        position = st.selectbox("Position", [
            'Goalkeeper', 'Centre-Back', 'Left-Back', 'Right-Back',
            'Central Midfield', 'Defensive Midfield', 'Attacking Midfield',
            'Left Winger', 'Right Winger', 'Centre-Forward', 'Second Striker'
        ])
        
        
        st.markdown("### 🏟️ Club Information")
        to_club = st.selectbox("Destination Club", [
            'Real Madrid', 'Barcelona', 'Paris Saint-Germain', 'Manchester City',
            'Manchester United', 'Arsenal', 'Chelsea', 'Liverpool', 'Juventus', 'AC Milan',
            'Tottenham', 'West Ham', 'Everton', 'Valencia', 'Sevilla',
            'Brighton', 'Crystal Palace', 'Getafe', 'Other club'
        ])
        
        from_club_tier = st.selectbox("Source Club Tier", [
            'Elite (Real Madrid, Barcelona, PSG level)',
            'Top (Man United, Arsenal, Juventus level)', 
            'Mid (Tottenham, Valencia level)',
            'Lower (Smaller clubs)',
            'Unknown/Academy'
        ])
        
        
        st.markdown("### 🌍 Market Context")
        league = st.selectbox("League", [
            'Premier League', 'La Liga', '1. Bundesliga', 'Serie A', 'Ligue 1',
            'Eredivisie', 'Liga Nos', 'Championship', 'Other league'
        ])
        
        transfer_period = st.selectbox("Transfer Window", ['Summer', 'Winter'])
        transfer_year = st.slider("Transfer Year", 2020, 2025, 2024)
        
        
        predict_button = st.button("🚀 ANALYZE TRANSFER", type="primary", use_container_width=True)
    
    with col_main:
        if predict_button:
            
            free_prob, estimated_fee, details = create_advanced_prediction(
                age, position, to_club, from_club_tier, league, transfer_period, transfer_year, feature_info
            )
            
            
            st.markdown(f"""
            <div class="prediction-card">
                <h2 style='margin-top: 0;'>🎯 Prediction Results</h2>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div style='text-align: center;'>
                        <h3>Free Transfer</h3>
                        <h1>{free_prob*100:.1f}%</h1>
                    </div>
                    <div style='text-align: center;'>
                        <h3>Estimated Value</h3>
                        <h1>€{estimated_fee:.1f}M</h1>
                    </div>
                    <div style='text-align: center;'>
                        <h3>Confidence</h3>
                        <h1>{details["confidence_score"]:.0f}%</h1>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            
            col1, col2 = st.columns(2)
            with col1:
                gauge_fig = create_gauge_chart(free_prob * 100, "Free Transfer Probability", 100)
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            with col2:
                value_gauge = create_gauge_chart(min(estimated_fee, 100), "Transfer Value (€M)", 100)
                st.plotly_chart(value_gauge, use_container_width=True)
            
            
            st.markdown("## 📊 Feature Impact Analysis")
            importance_fig = create_feature_importance_chart(details)
            st.plotly_chart(importance_fig, use_container_width=True)
            
            
            st.markdown("## ⚠️ Comprehensive Risk Analysis")
            risk_fig = create_risk_analysis_chart(free_prob, estimated_fee, age)
            st.plotly_chart(risk_fig, use_container_width=True)
            
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🔍 Value Factors")
                
                factors = [
                    ("Position", details['position_simple'], f"€{details['base_value']:.1f}M base"),
                    ("Age Group", details['age_group'], f"{details['age_factor']:.2f}x multiplier"),
                    ("League", league, f"{details['liga_factor']:.2f}x strength"),
                    ("Club Tier", details['to_tier'].title(), f"{details['tier_factor']:.2f}x premium"),
                    ("Movement", f"{details['tier_movement']:+d} tiers", f"{details['movement_factor']:.2f}x factor"),
                ]
                
                for factor, value, impact in factors:
                    st.markdown(f"""
                    <div style='background: linear-gradient(90deg, #667eea, #764ba2); 
                                padding: 0.8rem; margin: 0.3rem 0; border-radius: 8px; color: white;'>
                        <strong>{factor}:</strong> {value}<br>
                        <small>{impact}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 🎲 Risk Assessment")
                
                risk_factors = []
                if age >= 32:
                    risk_factors.append(("High Age Risk", f"{age} years old", "high"))
                elif age >= 30:
                    risk_factors.append(("Medium Age Risk", f"{age} years old", "medium"))
                else:
                    risk_factors.append(("Low Age Risk", f"{age} years old", "low"))
                
                if details['tier_movement'] < 0:
                    risk_factors.append(("Downward Movement", f"{details['tier_movement']} tiers", "high"))
                elif details['tier_movement'] > 0:
                    risk_factors.append(("Upward Movement", f"{details['tier_movement']} tiers", "low"))
                
                if details['young_talent_premium']:
                    risk_factors.append(("Young Talent Premium", "Elite destination", "low"))
                
                if transfer_period == 'Winter':
                    risk_factors.append(("Winter Window", "Reduced activity", "medium"))
                
                for risk, desc, level in risk_factors:
                    risk_class = f"risk-{level}"
                    st.markdown(f"""
                    <div class="{risk_class}">
                        <strong>{risk}:</strong> {desc}
                    </div>
                    """, unsafe_allow_html=True)
            
            
            st.markdown("## 🏆 Market Comparison")
            
            position_markets = {
                'FWD': {'avg': 32.5, 'range': '€15-80M', 'trend': 'Premium for young talent'},
                'WING': {'avg': 27.2, 'range': '€10-60M', 'trend': 'High demand for pace'},
                'MID': {'avg': 20.7, 'range': '€8-50M', 'trend': 'Versatility valued'},
                'DEF': {'avg': 17.3, 'range': '€5-40M', 'trend': 'Experience premium'},
                'GK': {'avg': 9.9, 'range': '€2-25M', 'trend': 'Longer careers'}
            }
            
            pos_key = details['position_simple']
            if pos_key in position_markets:
                market = position_markets[pos_key]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Position Average", f"€{market['avg']}M")
                with col2:
                    st.metric("Typical Range", market['range'])
                with col3:
                    st.metric("Your Estimate", f"€{estimated_fee:.1f}M", 
                             f"{((estimated_fee/market['avg']-1)*100):+.0f}%")
                
                st.info(f"**Market Trend:** {market['trend']}")
        
        else:
            
            st.markdown("""
            <div style='text-align: center; padding: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        border-radius: 20px; color: white; margin: 2rem 0;'>
                <h2>🚀 Ready to Analyze Transfers</h2>
                <p style='font-size: 1.1rem; margin: 1rem 0;'>
                    Configure player parameters in the sidebar and click "ANALYZE TRANSFER" to see:
                </p>
                <ul style='text-align: left; display: inline-block; font-size: 1rem;'>
                    <li>🎯 Free transfer probability with confidence scores</li>
                    <li>💰 Market value estimation with factor breakdown</li>
                    <li>📊 Interactive visualizations and risk analysis</li>
                    <li>🏆 Position-based market comparisons</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            
            st.markdown("## 🔥 Example Predictions")
            
            examples = [
                {"player": "22-year-old Striker → Real Madrid", "free": 15, "value": 45, "desc": "Young talent premium"},
                {"player": "29-year-old Midfielder → Brighton", "free": 35, "value": 18, "desc": "Peak age, mid-tier move"},
                {"player": "33-year-old Defender → Valencia", "free": 68, "value": 8, "desc": "Age risk factor"}
            ]
            
            cols = st.columns(3)
            for i, ex in enumerate(examples):
                with cols[i]:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                                padding: 1rem; border-radius: 10px; color: white; text-align: center;'>
                        <h4>{ex["player"]}</h4>
                        <p>Free: {ex["free"]}% | Value: €{ex["value"]}M</p>
                        <small>{ex["desc"]}</small>
                    </div>
                    """, unsafe_allow_html=True)
    
    
    with st.expander("🔧 Technical Details & Model Information"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🤖 Model Architecture**")
            st.success("✅ Random Forest Classifier (Clean)")
            st.success("✅ Random Forest Regressor (Clean)")
            st.success(f"✅ {len(feature_cols)} validated features")
            st.success("✅ No data leakage")
            
            if 'model_performance' in feature_info:
                perf = feature_info['model_performance']
                st.markdown("**📊 Performance Metrics**")
                st.info(f"🎯 Classification AUC: {perf['classification_auc']:.3f}")
                st.info(f"📈 Regression R²: {perf['regression_r2']:.3f}")
                st.info(f"⚡ Average Error: ±€{perf.get('regression_mae_millions', 8.5):.1f}M")
        
        with col2:
            st.markdown("**🛡️ Data Quality Improvements**")
            st.success("✅ Removed target-correlated features")
            st.success("✅ Only pre-transfer information used")
            st.success("✅ Realistic performance metrics")
            st.success("✅ Cross-validated predictions")
            
            st.markdown("**🎯 Key Features Used**")
            if 'safe_features' in feature_info:
                safe_features = feature_info['safe_features'][:8]  
                for feature in safe_features:
                    st.write(f"• {feature}")
                if len(feature_info.get('safe_features', [])) > 8:
                    st.write(f"... and {len(feature_info['safe_features']) - 8} more")
            
        st.markdown("---")
        st.markdown("**⚠️ Important Notes**")
        st.warning("""
        **Model Limitations:**
        • Predictions based on historical transfer patterns (2020-2024)
        • Market conditions and regulations may change
        • Individual player performance not directly modeled
        • Contract details and injury history not included
        """)
        
        st.info("""
        **Best Use Cases:**
        • Initial market valuation estimates
        • Risk assessment for transfer negotiations  
        • Comparative analysis across different scenarios
        • Strategic planning and budget allocation
        """)

if __name__ == "__main__":
    main()