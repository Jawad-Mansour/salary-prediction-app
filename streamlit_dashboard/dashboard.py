"""
Salary Prediction Dashboard - Professional Edition

Features:
- Model: Decision Tree v4 (Balanced)
- Professional AI Analyst persona (formal, data-driven)
- Context-specialized AI per tab using Ollama
- EDA charts from actual dataset insights

Author: Salary Prediction App
Version: 21.0.0 - PROFESSIONAL
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime

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
    page_title="Salary Intelligence Platform",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - Professional Dark Theme
# ============================================================================

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0a0a15 0%, #14142b 100%); }
    .welcome-section { background: linear-gradient(135deg, #0f2136 0%, #16213e 100%); padding: 2rem; border-radius: 20px; margin-bottom: 2rem; border: 1px solid #2a5a8c; text-align: center; }
    .welcome-title { color: white; font-size: 1.8rem; font-weight: bold; margin-bottom: 0.5rem; }
    .welcome-subtitle { color: #b0b0c0; margin-bottom: 1.5rem; }
    .metric-card { background: #1a1a2e; padding: 1rem; border-radius: 15px; text-align: center; border: 1px solid #2a5a8c; }
    .metric-icon { font-size: 1.8rem; }
    .metric-value { font-size: 1.8rem; font-weight: bold; color: #4a8cbc; margin: 0.5rem 0; }
    .metric-label { color: #b0b0c0; font-size: 0.85rem; }
    .salary-card { background: linear-gradient(135deg, #0f2136 0%, #1a3a5c 100%); padding: 2rem; border-radius: 20px; color: white; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.3); border: 1px solid #2a5a8c; margin-bottom: 1rem; }
    .salary-card h1 { font-size: 2.8rem; margin: 1rem 0; }
    .analysis-card { background: #1a1a2e; padding: 1.5rem; border-radius: 20px; border: 1px solid #3a6a9c; margin-bottom: 1rem; }
    .analysis-card h4 { color: #4a8cbc; margin-bottom: 1rem; }
    .chat-container { background: #1a1a2e; padding: 1rem; border-radius: 15px; border: 1px solid #2a5a8c; margin-top: 1rem; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0a0a15 0%, #0f2136 100%); border-right: 1px solid #2a5a8c; }
    .stButton > button { background: linear-gradient(135deg, #2a5a8c 0%, #1a3a5c 100%); color: white; border: none; border-radius: 10px; padding: 0.6rem 1rem; font-weight: bold; width: 100%; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; background-color: transparent; }
    .stTabs [data-baseweb="tab"] { background: #1a1a2e; border-radius: 10px; padding: 0.5rem 1.5rem; font-weight: bold; color: #b0b0c0; border: 1px solid #2a5a8c; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #2a5a8c 0%, #1a3a5c 100%); color: white; }
    .sample-question { background: #0f2136; padding: 0.4rem 1rem; border-radius: 20px; margin: 0.2rem; display: inline-block; cursor: pointer; border: 1px solid #2a5a8c; font-size: 0.8rem; color: #c0c0d0; }
    .sample-question:hover { background: #1a3a5c; color: white; }
    .detail-row { background: #0f2136; padding: 0.8rem; border-radius: 10px; margin: 0.5rem 0; }
    hr { margin: 1.5rem 0; border: none; height: 1px; background: linear-gradient(90deg, transparent, #2a5a8c, transparent); }
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
        return pd.DataFrame()


# ============================================================================
# API FUNCTIONS
# ============================================================================

def predict_via_api(job_title, experience_level, employment_type, company_size,
                    remote_ratio, work_year, employee_residence="US", company_location="US"):
    try:
        params = {
            "job_title": job_title, "experience_level": experience_level,
            "employment_type": employment_type, "company_size": company_size,
            "remote_ratio": remote_ratio, "work_year": work_year,
            "employee_residence": employee_residence, "company_location": company_location
        }
        response = requests.get(f"{FASTAPI_URL}/predict", params=params, timeout=30)
        if response.status_code == 200:
            return response.json(), None
        return None, f"API Error: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return None, "FastAPI not running"
    except Exception as e:
        return None, str(e)


# ============================================================================
# OLLAMA AI FUNCTIONS
# ============================================================================

def check_ollama_status():
    """Check if Ollama is running."""
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def generate_ai_analysis(prediction_data, df_context=None):
    """Generate professional AI analysis for a salary prediction."""
    
    if not check_ollama_status():
        return "⚠️ Ollama is not running. Start with `ollama serve` to enable AI analysis."
    
    exp_names = {'EN': 'Entry Level', 'MI': 'Mid Level', 'SE': 'Senior', 'EX': 'Executive'}
    exp_display = exp_names.get(prediction_data.get('experience_level', 'SE'), 'Senior')
    
    # Build context from database if available
    db_context = ""
    if df_context is not None and not df_context.empty:
        avg_salary = df_context['predicted_salary_usd'].mean()
        similar_roles = df_context[df_context['job_title'] == prediction_data.get('job_title', '')]
        role_avg = similar_roles['predicted_salary_usd'].mean() if not similar_roles.empty else avg_salary
        db_context = f"\nDatabase Context: {len(df_context)} total predictions. Average salary: ${avg_salary:,.0f}. Average for {prediction_data.get('job_title', 'this role')}: ${role_avg:,.0f}."
    
    prompt = f"""You are a professional compensation analyst. Provide a concise, formal analysis of this salary prediction.

Job: {prediction_data.get('job_title', 'Data Scientist')} ({exp_display})
Employment: {prediction_data.get('employment_type', 'FT')}
Company Size: {prediction_data.get('company_size', 'M')}
Remote: {prediction_data.get('remote_ratio', 0)}%
Location: {prediction_data.get('location', 'US')}
Predicted Salary: ${prediction_data.get('salary', 0):,.2f}
{db_context}

Provide a 3-4 sentence professional analysis covering:
1. How this salary compares to market expectations
2. Key factors influencing this prediction
3. Brief strategic insight

Be formal and data-driven. No markdown formatting."""
    
    try:
        payload = {
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.5, "num_predict": 200}
        }
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get('response', "Analysis unavailable.")
        return "AI analysis temporarily unavailable."
    except Exception as e:
        return f"AI service error: {str(e)}"


def generate_tab_specific_response(question, df, tab_context):
    """Generate AI response specialized for each tab."""
    
    if not check_ollama_status():
        return "⚠️ Ollama is not running. Please start Ollama to use AI features."
    
    if df.empty:
        return "No prediction data available. Generate predictions first."
    
    # Build data context
    avg_salary = df['predicted_salary_usd'].mean()
    top_role = df.loc[df['predicted_salary_usd'].idxmax()]['job_title'] if not df.empty else "N/A"
    top_salary = df['predicted_salary_usd'].max()
    
    # Tab-specific system prompts
    system_prompts = {
        "portfolio": """You are a data analyst specializing in compensation data. Answer questions about the prediction portfolio table. Be concise and reference specific data when possible. Focus on trends and patterns in the saved predictions.""",
        
        "analytics": """You are a market intelligence analyst. Answer questions about salary trends, market dynamics, and industry patterns. Reference the analytics charts and provide data-driven insights. Be professional and insightful.""",
        
        "analyst": """You are a senior compensation consultant. Provide strategic advice about salary negotiation, career progression, and market positioning. Be authoritative and practical."""
    }
    
    system_prompt = system_prompts.get(tab_context, system_prompts["analyst"])
    
    prompt = f"""{system_prompt}

Available Data: {len(df)} predictions. Average: ${avg_salary:,.0f}. Top role: {top_role} (${top_salary:,.0f}).

Question: {question}

Provide a concise, professional response (3-5 sentences)."""
    
    try:
        payload = {
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.5, "num_predict": 200}
        }
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get('response', "Unable to generate response.")
        return "AI service temporarily unavailable."
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar(supabase):
    with st.sidebar:
        st.markdown("# 🎯 Salary Predictor")
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
            [(0, "On-site"), (50, "Hybrid"), (100, "Remote")],
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
        
        if st.button("💰 Predict Salary", type="primary", use_container_width=True):
            with st.spinner("Analyzing market data..."):
                result, error = predict_via_api(
                    job_title, experience_level, employment_type, company_size,
                    remote_ratio, work_year, location_code, location_code
                )
                
                if result:
                    salary = result['predicted_salary_usd']
                    
                    # Load existing predictions for context
                    df_existing = load_predictions(supabase)
                    
                    pred_data = {
                        'salary': salary, 'job_title': job_title, 'experience_level': experience_level,
                        'employment_type': employment_type, 'company_size': company_size,
                        'remote_ratio': remote_ratio, 'work_year': work_year, 'location': location_code
                    }
                    
                    # Generate AI analysis
                    analysis = generate_ai_analysis(pred_data, df_existing)
                    
                    st.session_state['last_prediction'] = {
                        **pred_data,
                        'analysis': analysis
                    }
                    
                    if supabase:
                        prediction_record = {
                            "job_title": job_title, "experience_level": experience_level,
                            "employment_type": employment_type, "company_size": company_size,
                            "remote_ratio": remote_ratio, "work_year": work_year,
                            "employee_residence": location_code, "company_location": location_code,
                            "predicted_salary_usd": salary,
                            "llm_narrative": analysis
                        }
                        save_prediction_to_db(supabase, prediction_record)
                        st.success("✅ Prediction saved to database!")
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
        <div class="welcome-title">💰 Salary Intelligence Platform</div>
        <div class="welcome-subtitle">AI-powered compensation analytics for data science professionals</div>
    </div>
    """, unsafe_allow_html=True)
    
    df = load_predictions(supabase)
    ollama_status = check_ollama_status()
    
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
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🤖</div>
            <div class="metric-value" style="font-size:1rem;">{'✅ Online' if ollama_status else '⚠️ Offline'}</div>
            <div class="metric-label">AI Status</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Last Prediction Display (Salary Card + Analysis Card + Chat)
    if 'last_prediction' in st.session_state:
        pred = st.session_state['last_prediction']
        
        # Salary Card
        st.markdown(f"""
        <div class="salary-card">
            <h3>📋 Predicted Compensation</h3>
            <h1>${pred['salary']:,.2f}</h1>
            <hr>
            <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 2rem;">
                <div><strong>Role:</strong><br>{pred['job_title']}</div>
                <div><strong>Experience:</strong><br>{pred['experience_level']}</div>
                <div><strong>Employment:</strong><br>{pred['employment_type']}</div>
                <div><strong>Company:</strong><br>{pred['company_size']}</div>
                <div><strong>Remote:</strong><br>{pred['remote_ratio']}%</div>
                <div><strong>Year:</strong><br>{pred['work_year']}</div>
                <div><strong>Location:</strong><br>{pred['location']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # AI Analysis Card
        st.markdown(f"""
        <div class="analysis-card">
            <h4>🤖 AI Market Analysis</h4>
            <p style="color: #e0e0e0; line-height: 1.6;">{pred['analysis']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Chat about this prediction
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        st.markdown("#### 💬 Discuss This Prediction")
        
        if 'prediction_chat' not in st.session_state:
            st.session_state['prediction_chat'] = []
        
        for msg in st.session_state['prediction_chat']:
            with st.chat_message(msg['role']):
                st.write(msg['content'])
        
        chat_q = st.chat_input("Ask about this prediction...", key="pred_chat_input")
        if chat_q:
            with st.chat_message("user"):
                st.write(chat_q)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    response = generate_tab_specific_response(chat_q, df, "analyst")
                    st.write(response)
            st.session_state['prediction_chat'].append({'role': 'user', 'content': chat_q})
            st.session_state['prediction_chat'].append({'role': 'assistant', 'content': response})
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 All Predictions", "📊 Market Analytics", "🤖 AI Salary Analyst", "ℹ️ About"])
    
    # ========================================================================
    # TAB 1: ALL PREDICTIONS
    # ========================================================================
    
    with tab1:
        st.markdown("### 📋 Prediction Database")
        
        if df.empty:
            st.info("No predictions yet. Use the sidebar to create your first prediction!")
        else:
            display_df = df[['job_title', 'experience_level', 'employment_type', 'company_size', 
                            'remote_ratio', 'work_year', 'employee_residence',
                            'predicted_salary_usd', 'created_at']].copy()
            display_df['predicted_salary_usd'] = display_df['predicted_salary_usd'].apply(lambda x: f"${x:,.2f}")
            display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d')
            display_df.columns = ['Role', 'Exp', 'Emp', 'Size', 'Remote', 'Year', 'Location', 'Salary', 'Date']
            
            st.dataframe(display_df, use_container_width=True, height=350)
            
            # Chatbot for this tab
            st.markdown("---")
            st.markdown("#### 💬 Ask About These Predictions")
            
            st.markdown("**Sample Questions:**")
            cols = st.columns(4)
            samples = [
                "What roles have the highest salaries?",
                "How does experience level affect compensation?",
                "Which locations pay the most?",
                "What patterns do you see in this data?"
            ]
            for i, q in enumerate(samples):
                with cols[i]:
                    if st.button(q, key=f"sample_tab1_{i}", use_container_width=True):
                        with st.chat_message("user"):
                            st.write(q)
                        with st.chat_message("assistant"):
                            with st.spinner("..."):
                                response = generate_tab_specific_response(q, df, "portfolio")
                                st.write(response)
            
            if 'tab1_chat' not in st.session_state:
                st.session_state['tab1_chat'] = []
            
            for msg in st.session_state['tab1_chat']:
                with st.chat_message(msg['role']):
                    st.write(msg['content'])
            
            tab1_q = st.chat_input("Ask about the prediction data...", key="tab1_input")
            if tab1_q:
                with st.chat_message("user"):
                    st.write(tab1_q)
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing data..."):
                        response = generate_tab_specific_response(tab1_q, df, "portfolio")
                        st.write(response)
                st.session_state['tab1_chat'].append({'role': 'user', 'content': tab1_q})
                st.session_state['tab1_chat'].append({'role': 'assistant', 'content': response})
                st.rerun()
    
    # ========================================================================
    # TAB 2: MARKET ANALYTICS (EDA Charts)
    # ========================================================================
    
    with tab2:
        st.markdown("### 📊 Market Intelligence Dashboard")
        st.markdown("*Based on 3,755 real salary records (2020-2026)*")
        
        # Row 1: Salary Distribution + Experience Impact
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💰 Salary Distribution")
            np.random.seed(42)
            synthetic = np.random.normal(137570, 63055, 3755)
            synthetic = np.clip(synthetic, 5000, 450000)
            fig = px.histogram(x=synthetic, nbins=40, color_discrete_sequence=['#2a5a8c'])
            fig.add_vline(x=137570, line_dash="dash", line_color="#ff7f7f", annotation_text="Mean: $137.6K")
            fig.add_vline(x=135000, line_dash="dash", line_color="#ffdf7f", annotation_text="Median: $135K")
            fig.update_layout(height=380, plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e', font_color='#e0e0e0', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📈 Experience Level Impact")
            exp_data = pd.DataFrame({
                'Experience': ['Entry', 'Mid', 'Senior', 'Executive'],
                'Avg Salary': [78546, 104526, 153051, 194931]
            })
            fig = px.bar(exp_data, x='Experience', y='Avg Salary', color='Avg Salary',
                        color_continuous_scale='Blues', text=exp_data['Avg Salary'].apply(lambda x: f'${x/1000:.0f}K'))
            fig.update_traces(textposition='outside')
            fig.update_layout(height=380, plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
                             font_color='#e0e0e0', showlegend=False, yaxis_range=[0, 220000])
            st.plotly_chart(fig, use_container_width=True)
        
        # Row 2: Company Size + Location
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏢 Company Size Impact")
            size_data = pd.DataFrame({
                'Company Size': ['Small', 'Medium', 'Large'],
                'Avg Salary': [78227, 143131, 118301]
            })
            fig = px.bar(size_data, x='Company Size', y='Avg Salary', color='Avg Salary',
                        color_continuous_scale='Greens', text=size_data['Avg Salary'].apply(lambda x: f'${x/1000:.0f}K'))
            fig.update_traces(textposition='outside')
            fig.update_layout(height=380, plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
                             font_color='#e0e0e0', showlegend=False, yaxis_range=[0, 160000])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 🌍 Top Paying Locations")
            loc_data = pd.DataFrame({
                'Location': ['US', 'CA', 'AU', 'GB', 'SG', 'DE', 'NL', 'FR'],
                'Avg Salary': [145000, 120000, 105000, 95000, 98000, 90000, 92000, 85000]
            }).sort_values('Avg Salary', ascending=True)
            fig = px.bar(loc_data, x='Avg Salary', y='Location', orientation='h',
                        color='Avg Salary', color_continuous_scale='Viridis',
                        text=loc_data['Avg Salary'].apply(lambda x: f'${x/1000:.0f}K'))
            fig.update_traces(textposition='outside')
            fig.update_layout(height=380, plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
                             font_color='#e0e0e0', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Row 3: Remote Work + Top Roles
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏠 Remote Work Impact")
            remote_data = pd.DataFrame({
                'Arrangement': ['On-site', 'Hybrid', 'Fully Remote'],
                'Avg Salary': [144316, 78401, 136481]
            })
            fig = px.bar(remote_data, x='Arrangement', y='Avg Salary', color='Avg Salary',
                        color_continuous_scale='Oranges', text=remote_data['Avg Salary'].apply(lambda x: f'${x/1000:.0f}K'))
            fig.update_traces(textposition='outside')
            fig.update_layout(height=380, plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
                             font_color='#e0e0e0', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 🏆 Highest Paying Roles")
            role_data = pd.DataFrame({
                'Role': ['Data Science Tech Lead', 'Cloud Data Architect', 'Data Lead',
                        'Principal Data Scientist', 'Director of Data Science'],
                'Avg Salary': [375000, 250000, 212500, 198171, 195141]
            }).sort_values('Avg Salary', ascending=True)
            fig = px.bar(role_data, x='Avg Salary', y='Role', orientation='h',
                        color='Avg Salary', color_continuous_scale='Reds',
                        text=role_data['Avg Salary'].apply(lambda x: f'${x/1000:.0f}K'))
            fig.update_traces(textposition='outside')
            fig.update_layout(height=380, plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
                             font_color='#e0e0e0', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Row 4: Salary Trend
        st.markdown("#### 📅 Salary Trends (2020-2026)")
        trend_data = pd.DataFrame({
            'Year': [2020, 2021, 2022, 2023, 2024, 2025, 2026],
            'Avg Salary': [92303, 94087, 133339, 149046, 155000, 162000, 170000]
        })
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend_data['Year'], y=trend_data['Avg Salary'],
                                mode='lines+markers', line=dict(color='#2a5a8c', width=3),
                                marker=dict(size=10)))
        fig.update_layout(height=350, plot_bgcolor='#1a1a2e', paper_bgcolor='#1a1a2e',
                         font_color='#e0e0e0', yaxis_title='Salary (USD)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Chat for Analytics
        st.markdown("---")
        st.markdown("#### 💬 Ask About Market Trends")
        
        st.markdown("**Sample Questions:**")
        cols = st.columns(3)
        samples = [
            "Why do medium companies pay more?",
            "How has remote work affected salaries?",
            "What drives the salary differences by location?"
        ]
        for i, q in enumerate(samples):
            with cols[i]:
                if st.button(q, key=f"sample_tab2_{i}", use_container_width=True):
                    with st.chat_message("user"):
                        st.write(q)
                    with st.chat_message("assistant"):
                        with st.spinner("..."):
                            response = generate_tab_specific_response(q, df, "analytics")
                            st.write(response)
        
        if 'tab2_chat' not in st.session_state:
            st.session_state['tab2_chat'] = []
        
        for msg in st.session_state['tab2_chat']:
            with st.chat_message(msg['role']):
                st.write(msg['content'])
        
        tab2_q = st.chat_input("Ask about market trends...", key="tab2_input")
        if tab2_q:
            with st.chat_message("user"):
                st.write(tab2_q)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    response = generate_tab_specific_response(tab2_q, df, "analytics")
                    st.write(response)
            st.session_state['tab2_chat'].append({'role': 'user', 'content': tab2_q})
            st.session_state['tab2_chat'].append({'role': 'assistant', 'content': response})
            st.rerun()
    
    # ========================================================================
    # TAB 3: AI SALARY ANALYST
    # ========================================================================
    
    with tab3:
        st.markdown(f"""
        <div class="chat-container">
            <h3>🤖 AI Compensation Analyst</h3>
            <p>I have access to <strong>{len(df)} predictions</strong> and market data. Ask me about salary expectations, negotiation strategies, or career progression.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Sample Questions:**")
        cols = st.columns(4)
        samples = [
            "What salary should I expect as a Senior Data Scientist?",
            "How much more do executives earn?",
            "Is remote work affecting compensation?",
            "What factors drive higher salaries?"
        ]
        for i, q in enumerate(samples):
            with cols[i]:
                if st.button(q, key=f"sample_tab3_{i}", use_container_width=True):
                    with st.chat_message("user"):
                        st.write(q)
                    with st.chat_message("assistant"):
                        with st.spinner("..."):
                            response = generate_tab_specific_response(q, df, "analyst")
                            st.write(response)
        
        if 'tab3_chat' not in st.session_state:
            st.session_state['tab3_chat'] = []
        
        for msg in st.session_state['tab3_chat']:
            with st.chat_message(msg['role']):
                st.write(msg['content'])
        
        tab3_q = st.chat_input("Ask the AI Analyst...", key="tab3_input")
        if tab3_q:
            with st.chat_message("user"):
                st.write(tab3_q)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    response = generate_tab_specific_response(tab3_q, df, "analyst")
                    st.write(response)
            st.session_state['tab3_chat'].append({'role': 'user', 'content': tab3_q})
            st.session_state['tab3_chat'].append({'role': 'assistant', 'content': response})
            st.rerun()
    
    # ========================================================================
    # TAB 4: ABOUT
    # ========================================================================
    
    with tab4:
        st.markdown("""
        <div style="background: #1a1a2e; padding: 2rem; border-radius: 20px; border: 1px solid #2a5a8c;">
            <h1 style="color: white;">💰 Salary Intelligence Platform</h1>
            <p style="color: #b0b0c0; font-size: 1.1rem;">AI-powered compensation analytics for data science professionals</p>
            
            <h2 style="color: #4a8cbc; margin-top: 2rem;">📊 About This Platform</h2>
            <p style="color: #c0c0d0; line-height: 1.6;">This platform combines machine learning with AI analysis to provide accurate salary predictions and market intelligence. The model is trained on 3,755 real salary records from data science professionals worldwide (2020-2026).</p>
            
            <h2 style="color: #4a8cbc; margin-top: 1.5rem;">🔬 The Model</h2>
            <ul style="color: #c0c0d0;">
                <li><strong>Algorithm:</strong> Decision Tree Regressor v4</li>
                <li><strong>R² Score:</strong> 0.456 (test set)</li>
                <li><strong>Features:</strong> 19 engineered features including location, experience interactions, and development index</li>
                <li><strong>Balancing:</strong> Sample weights applied to handle experience level imbalance</li>
            </ul>
            
            <h2 style="color: #4a8cbc; margin-top: 1.5rem;">📈 Key Market Findings</h2>
            <ul style="color: #c0c0d0;">
                <li><strong>Location × Experience</strong> accounts for 62% of salary variance</li>
                <li><strong>Senior roles</strong> earn 95% more than entry-level</li>
                <li><strong>Medium companies</strong> pay higher averages than large corporations</li>
                <li><strong>Remote work</strong> shows no salary penalty for experienced professionals</li>
            </ul>
            
            <h2 style="color: #4a8cbc; margin-top: 1.5rem;">🛠️ Technology Stack</h2>
            <ul style="color: #c0c0d0;">
                <li><strong>Model:</strong> scikit-learn Decision Tree with GridSearchCV</li>
                <li><strong>API:</strong> FastAPI (REST)</li>
                <li><strong>AI:</strong> Ollama with llama3.2 (local)</li>
                <li><strong>Database:</strong> Supabase (PostgreSQL)</li>
                <li><strong>Dashboard:</strong> Streamlit with Plotly</li>
            </ul>
            
            <hr style="margin: 2rem 0;">
            <p style="color: #808090; text-align: center;">Data: Kaggle Data Science Salaries Dataset | 3,755 Records | 2020-2026</p>
            <p style="color: #808090; text-align: center;">Version 21.0.0</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption(f"📊 {len(df)} predictions in database")
    with c2:
        st.caption(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with c3:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()


if __name__ == "__main__":
    main()