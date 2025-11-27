"""
ThreadX UI Styles
=================

Module managing custom CSS injection for a premium UI experience.
"""

import streamlit as st

def apply_custom_styles():
    """Injects custom CSS into the Streamlit app."""
    st.markdown(
        """
        <style>
        /* Global Font & Colors */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Main Background */
        .stApp {
            background-color: #0e1117;
        }

        /* Headers */
        h1, h2, h3 {
            color: #f0f2f6;
            font-weight: 600;
            letter-spacing: -0.5px;
        }
        
        h1 {
            font-size: 2.2rem;
            background: linear-gradient(90deg, #42a5f5, #ab47bc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }

        /* Cards / Containers */
        .stContainer, .stExpander {
            background-color: #1a1c24;
            border-radius: 12px;
            border: 1px solid #2b2d3e;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            font-size: 1.8rem;
            color: #42a5f5;
            font-weight: 700;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 0.9rem;
            color: #a8b2d1;
        }

        /* Buttons */
        .stButton > button {
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }

        /* Inputs */
        .stTextInput > div > div > input, .stSelectbox > div > div > div {
            background-color: #1a1c24;
            color: #f0f2f6;
            border-radius: 8px;
            border: 1px solid #2b2d3e;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #13151c;
            border-right: 1px solid #2b2d3e;
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0e1117;
        }
        ::-webkit-scrollbar-thumb {
            background: #2b2d3e;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #42a5f5;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
