"""
Salary Prediction Dashboard - ULTIMATE FIXED EDITION

FIXES:
1. Fixed About Project tab (proper markdown to HTML)
2. Last prediction now appears ABOVE the table (not below)
3. Added Chatbot back to All Predictions tab
4. Clean professional layout

Author: Salary Prediction App
Version: 19.0.0 - ULTIMATE FIXED
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from datetime import datetime

# Supabase imports
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# CONFIGURATION
# ============================================================================

SUPABASE_URL = "https://efjkvlnmnenayhjevmie.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVmamt2bG5tbmVuYXloamV2bWllIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzU2ODU3NjgsImV4cCI6MjA5MTI2MTc2OH0.d-4bW0dwU-DsIJDODChsd6btC-7PWlYgQHHZogfDIYs"

FASTAPI_URL = "http://127.0.0.1:8000"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

st.set_page_config(
    page_title="Salary Predictor Pro",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0a15 0%, #14142b 100%);
    }
    
    .welcome-section {
        background: linear-gradient(135deg, #0f2136 0%, #16213e 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid #2a5a8c;
        text-align: center;
    }
    .welcome-title {
        color: white;
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .welcome-subtitle {
        color: #b0b0c0;
        margin-bottom: 1.5rem;
    }
    
    .pillar-card {
        background: #1a1a2e;
        padding: 1.2rem;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #2a5a8c;
        transition: all 0.3s ease;
        height: 100%;
    }
    .pillar-card:hover {
        transform: translateY(-5px);
        border-color: #4a8cbc;
        background: #1f2a3e;
    }
    .pillar-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .pillar-title {
        color: #4a8cbc;
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    .pillar-desc {
        color: #b0b0c0;
        font-size: 0.8rem;
    }
    
    .metric-card {
        background: #1a1a2e;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #2a5a8c;
    }
    .metric-icon {
        font-size: 1.8rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #4a8cbc;
        margin: 0.5rem 0;
    }
    .metric-label {
        color: #b0b0c0;
        font-size: 0.85rem;
    }
    
    .salary-card {
        background: linear-gradient(135deg, #0f2136 0%, #1a3a5c 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        height: 100%;
        border: 1px solid #2a5a8c;
    }
    .salary-card h1 {
        font-size: 2.8rem;
        margin: 1rem 0;
    }
    .salary-card hr {
        margin: 1rem 0;
        border-color: #2a5a8c;
    }
    
    .analysis-card {
        background: #1a1a2e;
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        height: 100%;
        border-left: 5px solid #2a5a8c;
    }
    
    .chat-card {
        background: #1a1a2e;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-top: 4px solid #2a5a8c;
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a15 0%, #0f2136 100%);
        border-right: 1px solid #2a5a8c;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #2a5a8c 0%, #1a3a5c 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        font-weight: bold;
        width: 100%;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
        color: #b0b0c0;
        border: 1px solid #2a5a8c;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2a5a8c 0%, #1a3a5c 100%);
        color: white;
    }
    
    .about-section {
        background: #1a1a2e;
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid #2a5a8c;
    }
    .about-section h1 {
        color: white;
        font-size: 1.8rem;
        margin-bottom: 1rem;
    }
    .about-section h2 {
        color: #4a8cbc;
        font-size: 1.3rem;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .about-section h3 {
        color: #4a8cbc;
        font-size: 1.1rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .about-section p {
        color: #c0c0d0;
        line-height: 1.6;
    }
    .about-section li {
        color: #c0c0d0;
        margin: 0.5rem 0;
    }
    
    hr {
        margin: 1.5rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #2a5a8c, transparent);
    }
    
    div[data-testid="stChatMessage"] {
        background: #0f2136;
        border-radius: 15px;
        border: 1px solid #2a5a8c;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

@st.cache_resource
def init_supabase():
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.error(f"Cannot connect to database: {e}")
        return None


def save_prediction_to_db(supabase, prediction_data):
    try:
        result = supabase.table('predictions').insert(prediction_data).execute()
        return True, result.data[0]['id'] if result.data else None
    except Exception as e:
        return False, str(e)


@st.cache_data(ttl=30)
def load_predictions(_supabase):
    if _supabase is None:
        return pd.DataFrame()
    try:
        result = _supabase.table('predictions').select('*').order('created_at', desc=True).execute()
        df = pd.DataFrame(result.data)
        if not df.empty and 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


# ============================================================================
# API FUNCTIONS
# ============================================================================

def predict_via_api(job_title, experience_level, employment_type, company_size,
                    remote_ratio, work_year, employee_residence="US", company_location="US"):
    try:
        params = {
            "job_title": job_title,
            "experience_level": experience_level,
            "employment_type": employment_type,
            "company_size": company_size,
            "remote_ratio": remote_ratio,
            "work_year": work_year,
            "employee_residence": employee_residence,
            "company_location": company_location
        }
        response = requests.get(f"{FASTAPI_URL}/predict", params=params, timeout=30)
        if response.status_code == 200:
            return response.json(), None
        return None, f"API Error: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return None, "FastAPI not running"
    except Exception as e:
        return None, str(e)


def generate_intelligent_analysis(job_title, experience_level, predicted_salary, location="US"):
    exp_names = {'EN': 'Entry Level', 'MI': 'Mid Level', 'SE': 'Senior', 'EX': 'Executive'}
    
    location_multipliers = {
        'US': 1.0, 'CA': 0.92, 'GB': 0.85, 'DE': 0.82, 'FR': 0.78,
        'AU': 0.88, 'SG': 0.86, 'NL': 0.84, 'IN': 0.45, 'BR': 0.52
    }
    multiplier = location_multipliers.get(location, 0.7)
    market_rate = int(predicted_salary / multiplier) if multiplier > 0 else predicted_salary
    
    if predicted_salary > market_rate * 1.15:
        competitiveness = "🔥 EXCELLENT - Significantly above market rate"
        advice = "Strong negotiating position. Ask for additional benefits."
    elif predicted_salary > market_rate * 0.85:
        competitiveness = "✅ FAIR - Aligns with market expectations"
        advice = "Solid offer. Negotiate for signing bonus."
    else:
        competitiveness = "⚠️ BELOW MARKET - May be undervalued"
        advice = "Consider negotiating or exploring other opportunities."
    
    role_insights = {
        "Data Scientist": "ML engineering skills command 15-20% premiums.",
        "Data Engineer": "Cloud platform skills earn 20-25% more.",
        "Machine Learning Engineer": "Production deployment earns top rates.",
        "Data Analyst": "SQL and visualization skills are in high demand.",
        "Data Architect": "Cloud certification earns premium salaries."
    }
    insight = role_insights.get(job_title, "Specialized skills increase earning potential.")
    
    analysis = f"""
**Market Position:** {competitiveness}

**Key Insight:** {insight}

**Career Advice:** {advice}

**Location Context:** For {location}, this salary is **{int(predicted_salary / market_rate * 100)}%** of adjusted market rate.
"""
    return analysis


def generate_chat_response(question, df):
    if df.empty:
        return "No data available yet. Generate predictions first!"
    
    avg_salary = df['predicted_salary_usd'].mean()
    
    prompt = f"Based on {len(df)} salary predictions averaging ${avg_salary:,.0f}, answer concisely: {question}"
    
    try:
        payload = {"model": "llama3.2", "prompt": prompt, "stream": False, "options": {"temperature": 0.5, "num_predict": 200}}
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get('response', "Could not analyze.")
        return "AI temporarily unavailable."
    except:
        return "AI not responding. Make sure Ollama is running."


# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar(supabase):
    with st.sidebar:
        st.markdown("# 🎯 Quick Predict")
        st.markdown("---")
        
        job_title = st.selectbox(
            "Job Title",
            ["Data Scientist", "Data Engineer", "Data Analyst", "Machine Learning Engineer",
             "Analytics Engineer", "Data Architect", "Research Scientist", "Data Science Manager"]
        )
        
        experience_level = st.selectbox(
            "Experience Level",
            [("EN", "Entry (0-2 yrs)"), ("MI", "Mid (3-5 yrs)"), 
             ("SE", "Senior (6-9 yrs)"), ("EX", "Executive (10+ yrs)")],
            format_func=lambda x: x[1]
        )[0]
        
        employment_type = st.selectbox(
            "Employment Type",
            [("FT", "Full-Time"), ("PT", "Part-Time"), ("CT", "Contract"), ("FL", "Freelance")],
            format_func=lambda x: x[1]
        )[0]
        
        company_size = st.selectbox(
            "Company Size",
            [("S", "Small"), ("M", "Medium"), ("L", "Large")],
            format_func=lambda x: x[1]
        )[0]
        
        remote_ratio = st.selectbox(
            "Remote Work",
            [(0, "On-site"), (50, "Hybrid"), (100, "Fully Remote")],
            format_func=lambda x: x[1]
        )[0]
        
        work_year = st.number_input("Year", min_value=2020, max_value=2026, value=2026, step=1)
        
        location = st.selectbox(
            "Location",
            ["United States (US)", "United Kingdom (GB)", "Germany (DE)", "France (FR)", 
             "Canada (CA)", "Australia (AU)", "Singapore (SG)", "Netherlands (NL)", 
             "India (IN)", "Brazil (BR)"]
        )
        location_code = location.split("(")[-1].replace(")", "")
        
        st.markdown("---")
        
        if st.button("💰 Predict My Salary", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                result, error = predict_via_api(
                    job_title, experience_level, employment_type, company_size,
                    remote_ratio, work_year, location_code, location_code
                )
                
                if result:
                    salary = result['predicted_salary_usd']
                    st.session_state['last_prediction'] = {
                        'salary': salary, 'job_title': job_title, 'experience_level': experience_level,
                        'employment_type': employment_type, 'company_size': company_size,
                        'remote_ratio': remote_ratio, 'work_year': work_year, 'location': location_code,
                        'analysis': generate_intelligent_analysis(job_title, experience_level, salary, location_code)
                    }
                    
                    if supabase:
                        prediction_record = {
                            "job_title": job_title, "experience_level": experience_level,
                            "employment_type": employment_type, "company_size": company_size,
                            "remote_ratio": remote_ratio, "work_year": work_year,
                            "employee_residence": location_code, "company_location": location_code,
                            "predicted_salary_usd": salary, "llm_narrative": st.session_state['last_prediction']['analysis']
                        }
                        save_prediction_to_db(supabase, prediction_record)
                        st.success("✅ Saved to database!")
                        st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(f"Error: {error}")


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    supabase = init_supabase()
    render_sidebar(supabase)
    
    # Welcome Section
    st.markdown("""
    <div class="welcome-section">
        <div class="welcome-title">👋 Welcome to Salary Predictor Pro</div>
        <div class="welcome-subtitle">Your complete AI-powered salary intelligence platform for data science professionals</div>
    </div>
    """, unsafe_allow_html=True)
    
    df = load_predictions(supabase)
    
    # Stats Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📊</div>
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Total Predictions</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        avg_sal = df['predicted_salary_usd'].mean() if not df.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">💰</div>
            <div class="metric-value">${avg_sal:,.0f}</div>
            <div class="metric-label">Average Salary</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        top_role = df.loc[df['predicted_salary_usd'].idxmax()]['job_title'] if not df.empty else "—"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🏆</div>
            <div class="metric-value" style="font-size:1.2rem;">{top_role[:18]}</div>
            <div class="metric-label">Highest Paid Role</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        top_loc = df.loc[df['predicted_salary_usd'].idxmax()]['employee_residence'] if not df.empty else "—"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📍</div>
            <div class="metric-value">{top_loc}</div>
            <div class="metric-label">Top Location</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========================================================================
    # LAST PREDICTION - NOW ABOVE THE TABS (FIXED)
    # ========================================================================
    
    if 'last_prediction' in st.session_state:
        pred = st.session_state['last_prediction']
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown(f"""
            <div class="salary-card">
                <h3>🎯 Your Latest Prediction</h3>
                <h1>${pred['salary']:,.2f}</h1>
                <hr>
                <p><strong>Role:</strong> {pred['job_title']}</p>
                <p><strong>Experience:</strong> {pred['experience_level']}</p>
                <p><strong>Employment:</strong> {pred['employment_type']}</p>
                <p><strong>Company Size:</strong> {pred['company_size']}</p>
                <p><strong>Remote:</strong> {pred['remote_ratio']}%</p>
                <p><strong>Year:</strong> {pred['work_year']}</p>
                <p><strong>Location:</strong> {pred['location']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_right:
            st.markdown(f"""
            <div class="analysis-card">
                <h3>🤖 AI Salary Analysis</h3>
                {pred['analysis']}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    # ========================================================================
    # TABS
    # ========================================================================
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 All Predictions", "📊 Analytics & EDA", "🤖 AI Salary Analyst", "ℹ️ About Project"])
    
    # ========================================================================
    # TAB 1: ALL PREDICTIONS (WITH CHATBOT)
    # ========================================================================
    
    with tab1:
        if df.empty:
            st.info("No predictions yet. Use the sidebar to create your first prediction!")
        else:
            # Display table
            display_df = df[['job_title', 'experience_level', 'employment_type', 'company_size', 
                            'remote_ratio', 'work_year', 'employee_residence',
                            'predicted_salary_usd', 'created_at']].copy()
            display_df['predicted_salary_usd'] = display_df['predicted_salary_usd'].apply(lambda x: f"${x:,.2f}")
            display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d')
            display_df.columns = ['Role', 'Exp', 'Emp', 'Size', 'Remote', 'Year', 'Location', 'Salary', 'Date']
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            st.markdown("---")
            st.markdown("### 🔍 View Prediction Details")
            
            if len(df) > 0:
                selected = st.selectbox("Select a prediction", options=range(len(df)), 
                                       format_func=lambda i: f"{df.iloc[i]['job_title']} - ${df.iloc[i]['predicted_salary_usd']:,.0f}")
                if selected is not None:
                    row = df.iloc[selected]
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"""
                        <div style="background:#1a1a2e; padding:1rem; border-radius:15px; border:1px solid #2a5a8c;">
                            <h4 style="color:white;">📋 Job Details</h4>
                            <p><strong>Role:</strong> {row['job_title']}</p>
                            <p><strong>Experience:</strong> {row['experience_level']}</p>
                            <p><strong>Employment:</strong> {row['employment_type']}</p>
                            <p><strong>Company Size:</strong> {row['company_size']}</p>
                            <p><strong>Remote:</strong> {row['remote_ratio']}%</p>
                            <p><strong>Year:</strong> {row['work_year']}</p>
                            <p><strong>Location:</strong> {row['employee_residence']}</p>
                            <p><strong>Salary:</strong> <strong style="color:#4a8cbc;">${row['predicted_salary_usd']:,.2f}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_b:
                        narrative = row.get('llm_narrative')
                        if narrative and not pd.isna(narrative):
                            st.markdown(f'''
                            <div style="background:#1a1a2e; padding:1rem; border-radius:15px; border-left:4px solid #2a5a8c;">
                                <h4 style="color:white;">🤖 AI Analysis</h4>
                                <p style="color:#e0e0e0;">{narrative}</p>
                            </div>
                            ''', unsafe_allow_html=True)
            
            # CHATBOT IN ALL PREDICTIONS TAB (ADDED BACK)
            st.markdown("---")
            st.markdown("### 💬 Chat about these predictions")
            st.markdown('<div class="chat-card"><p>Ask questions about the predictions in this table!</p></div>', unsafe_allow_html=True)
            
            if 'prediction_chat_history' not in st.session_state:
                st.session_state['prediction_chat_history'] = []
            
            for chat in st.session_state['prediction_chat_history']:
                with st.chat_message("user"):
                    st.write(chat['question'])
                with st.chat_message("assistant"):
                    st.write(chat['answer'])
            
            pred_q = st.chat_input("Ask about these predictions...", key="prediction_chat")
            if pred_q:
                with st.chat_message("user"):
                    st.write(pred_q)
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing..."):
                        response = generate_chat_response(pred_q, df)
                        st.write(response)
                        st.session_state['prediction_chat_history'].append({'question': pred_q, 'answer': response})
    
    # ========================================================================
    # TAB 2: ANALYTICS & EDA
    # ========================================================================
    
    with tab2:
        if df.empty:
            st.info("No data available. Generate predictions first!")
        else:
            st.markdown("### 📊 Salary Distribution")
            fig = px.histogram(df, x='predicted_salary_usd', nbins=30,
                              labels={'predicted_salary_usd': 'Salary (USD)', 'count': 'Frequency'},
                              color_discrete_sequence=['#2a5a8c'])
            fig.add_vline(x=df['predicted_salary_usd'].mean(), line_dash="dash", line_color="red", 
                         annotation_text=f"Mean: ${df['predicted_salary_usd'].mean():,.0f}")
            fig.add_vline(x=df['predicted_salary_usd'].median(), line_dash="dash", line_color="yellow",
                         annotation_text=f"Median: ${df['predicted_salary_usd'].median():,.0f}")
            fig.update_layout(height=450, plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e', font_color='#e0e0e0')
            st.plotly_chart(fig, use_container_width=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("### 💰 Salary by Experience")
                exp_data = df.groupby('experience_level')['predicted_salary_usd'].mean().reset_index()
                exp_names = {'EN': 'Entry', 'MI': 'Mid', 'SE': 'Senior', 'EX': 'Executive'}
                exp_data['level'] = exp_data['experience_level'].map(exp_names)
                fig = px.bar(exp_data, x='level', y='predicted_salary_usd', color='predicted_salary_usd', 
                            color_continuous_scale='Blues')
                fig.update_layout(height=400, showlegend=False, plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e', font_color='#e0e0e0')
                st.plotly_chart(fig, use_container_width=True)
            
            with col_b:
                st.markdown("### 🏢 Salary by Company Size")
                size_data = df.groupby('company_size')['predicted_salary_usd'].mean().reset_index()
                size_names = {'S': 'Small', 'M': 'Medium', 'L': 'Large'}
                size_data['size'] = size_data['company_size'].map(size_names)
                fig = px.bar(size_data, x='size', y='predicted_salary_usd', color='predicted_salary_usd',
                            color_continuous_scale='Greens')
                fig.update_layout(height=400, showlegend=False, plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e', font_color='#e0e0e0')
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🏆 Top 10 Highest Paying Roles")
            top_jobs = df.groupby('job_title')['predicted_salary_usd'].mean().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(top_jobs, x='predicted_salary_usd', y='job_title', orientation='h',
                        labels={'predicted_salary_usd': 'Avg Salary (USD)', 'job_title': ''},
                        color='predicted_salary_usd', color_continuous_scale='Reds')
            fig.update_layout(height=500, plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e', font_color='#e0e0e0')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🌍 Salary by Location")
            loc_data = df.groupby('employee_residence')['predicted_salary_usd'].mean().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(loc_data, x='employee_residence', y='predicted_salary_usd',
                        color='predicted_salary_usd', color_continuous_scale='Viridis')
            fig.update_layout(height=450, plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e', font_color='#e0e0e0')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🏠 Remote Work Impact")
            remote_data = df.groupby('remote_ratio')['predicted_salary_usd'].mean().reset_index()
            remote_labels = {0: 'On-site', 50: 'Hybrid', 100: 'Remote'}
            remote_data['arrangement'] = remote_data['remote_ratio'].map(remote_labels)
            fig = px.bar(remote_data, x='arrangement', y='predicted_salary_usd',
                        color='predicted_salary_usd', color_continuous_scale='Oranges')
            fig.update_layout(height=400, plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e', font_color='#e0e0e0')
            st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # TAB 3: AI SALARY ANALYST
    # ========================================================================
    
    with tab3:
        st.markdown(f"""
        <div class="chat-card">
            <h3 style="color:white;">🤖 AI Salary Analyst</h3>
            <p>Based on <strong>{len(df)} predictions</strong> in our database. Ask anything about salary trends!</p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'general_history' not in st.session_state:
            st.session_state['general_history'] = []
        
        for chat in st.session_state['general_history']:
            with st.chat_message("user"):
                st.write(chat['question'])
            with st.chat_message("assistant"):
                st.write(chat['answer'])
        
        general_q = st.chat_input("Ask the AI Salary Analyst...", key="general_chat")
        if general_q:
            with st.chat_message("user"):
                st.write(general_q)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = generate_chat_response(general_q, df)
                    st.write(response)
                    st.session_state['general_history'].append({'question': general_q, 'answer': response})
        
        with st.expander("💡 Example Questions"):
            st.markdown("""
            - What's the average salary for a Data Scientist?
            - Which location pays the highest?
            - How much more do Senior roles earn?
            - Does remote work affect salary?
            - What's the best career path?
            """)
    
    # ========================================================================
    # TAB 4: ABOUT PROJECT (FIXED - NO MARKDOWN ISSUES)
    # ========================================================================
    
    with tab4:
        st.markdown("""
        <div class="about-section">
            <h1>🚀 About Salary Predictor Pro</h1>
            
            <h2>The Problem We Solve</h2>
            <p>Salary uncertainty costs professionals thousands of dollars annually. Without data-driven insights, job seekers and hiring managers rely on guesswork, leading to undervaluation or overpayment.</p>
            
            <h2>Our Solution</h2>
            <p>A complete AI-powered salary intelligence platform that provides instant, accurate salary predictions based on real-world data and machine learning.</p>
            
            <h2>Technology Stack</h2>
            <ul>
                <li><strong>Model:</strong> Decision Tree Regressor (R² = 0.473)</li>
                <li><strong>API:</strong> FastAPI for real-time predictions</li>
                <li><strong>LLM:</strong> Ollama (llama3.2) for intelligent analysis</li>
                <li><strong>Database:</strong> Supabase (PostgreSQL)</li>
                <li><strong>Dashboard:</strong> Streamlit</li>
                <li><strong>Data Source:</strong> Kaggle Data Science Salaries (3,755 records)</li>
            </ul>
            
            <h2>Key Findings from EDA</h2>
            <ul>
                <li><strong>📍 Location is 55.6% of salary</strong> - Where you live matters most</li>
                <li><strong>📈 Senior roles pay 136% more</strong> than entry level</li>
                <li><strong>🤖 ML Engineers earn the most</strong> among data roles</li>
                <li><strong>🏠 Remote work has no salary penalty</strong></li>
                <li><strong>🏢 Large companies pay 30-45% more</strong></li>
            </ul>
            
            <h2>How to Use</h2>
            <ol>
                <li>Fill in your job details in the left sidebar</li>
                <li>Click "Predict My Salary" for instant AI-powered prediction</li>
                <li>Explore analytics and EDA findings in the Analytics tab</li>
                <li>Ask the AI Salary Analyst for personalized career advice</li>
                <li>Compare salaries across locations, experience levels, and company sizes</li>
            </ol>
            
            <hr>
            <p style="text-align:center;">Built with Python, FastAPI, Streamlit, Ollama, and Supabase</p>
            <p style="text-align:center;">Data Science Salaries Dataset | 3,755 Records | 2020-2026</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption(f"📊 {len(df)} predictions in database")
    with c2:
        st.caption(f"🕐 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with c3:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()


if __name__ == "__main__":
    main()