#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Import agent classes and setup function from sustainable_farming_system.py
from sustainable_farming_system import (
    setup_database,
    FarmerInputAgent,
    EnvironmentalAnalysisAgent,
    CropRecommendationAgent,
    MarketAnalysisAgent,
    SustainabilityOptimizationAgent,
    DecisionIntegrationAgent
)

# Page configuration
st.set_page_config(
    page_title="Krishniti AI",
    page_icon="🌱",
    layout="wide"
)

# Database connection helper
def get_db_connection():
    return sqlite3.connect('farming_agents.db')

# Initialize database
setup_database()

# Load agents (cached for performance)
@st.cache_resource
def load_agents():
    return {
        'farmer_agent': FarmerInputAgent(),
        'env_agent': EnvironmentalAnalysisAgent('farmer_advisor_dataset.csv'),
        'crop_agent': CropRecommendationAgent('farmer_advisor_dataset.csv'),
        'market_agent': MarketAnalysisAgent('marketer_researcher_dataset.csv'),
        'sust_agent': SustainabilityOptimizationAgent('farmer_advisor_dataset.csv'),
        'decision_agent': DecisionIntegrationAgent()
    }

agents = load_agents()

# Load datasets for visualization
@st.cache_data
def load_data():
    advisor_data = pd.read_csv('farmer_advisor_dataset.csv')
    market_data = pd.read_csv('marketer_researcher_dataset.csv')
    return advisor_data, market_data

advisor_data, market_data = load_data()

# Fetch all registered farms from the database
def get_registered_farms():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT farm_id, farmer_name, location, total_area FROM farms")
    farms = cursor.fetchall()
    conn.close()
    return [{'farm_id': f[0], 'farmer_name': f[1], 'location': f[2], 'total_area': f[3]} for f in farms]

# Fetch recent recommendations from the database
def get_recent_recommendations():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT farm_id, recommendation_date, crop_type, expected_yield, sustainability_score
        FROM recommendations
        ORDER BY recommendation_date DESC
        LIMIT 5
    """)
    recs = cursor.fetchall()
    conn.close()
    return pd.DataFrame(recs, columns=['Farm_ID', 'Date', 'Recommended_Crop', 'Expected_Yield', 'Sustainability_Score'])

# AI-driven sustainability scoring system (simplified)
def calculate_sustainability_score(predicted_sustainability, fertilizer_kg, pesticide_kg, tips_followed):
    base_score = predicted_sustainability  # From CropRecommendationAgent
    # Reduce score for high input usage
    fertilizer_penalty = max(0, (fertilizer_kg - 100) * 0.1)  # Penalty for >100kg
    pesticide_penalty = max(0, (pesticide_kg - 5) * 0.5)     # Penalty for >5kg
    # Bonus for following sustainability tips
    tips_bonus = len(tips_followed) * 5  # 5 points per tip followed
    final_score = min(100, max(0, base_score - fertilizer_penalty - pesticide_penalty + tips_bonus))
    return final_score

# Blockchain-based smart contract reward (conceptual)
def generate_blockchain_reward(farm_id, sustainability_score):
    reward_tokens = sustainability_score * 0.1  # 0.1 token per sustainability point
    return {
        "farm_id": farm_id,
        "sustainability_score": sustainability_score,
        "reward_tokens": reward_tokens,
        "contract": f"SmartContract.deploy(farmId={farm_id}, score={sustainability_score}, tokens={reward_tokens})"
    }

# Main app title
st.title("🌱 Krishniti AI")
st.markdown("""
### 🌿 **Sustainable Farming. Smarter Future.**  
**🤖 Powered by Multi-Agent AI | 🌍 Driven by Data | 🌾 Rooted in Sustainability**  
<br>

Our intelligent farming assistant uses cutting-edge multi-agent AI to provide real-time, data-driven insights.  
It recommends the best crop choices by analyzing environmental factors, market trends, and sustainability goals — helping farmers grow smarter, greener, and more profitably.
""", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", ["Dashboard", "Farm Management", "Recommendations", "Data Analysis", "About"])

# Dashboard page
if page == "Dashboard":
    st.header("System Dashboard")
    
    farms = get_registered_farms()
    recent_recs = get_recent_recommendations()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Farms Registered", value=len(farms))
    with col2:
        st.metric(label="Recommendations Generated", value=len(recent_recs))
    with col3:
        avg_sust = recent_recs['Sustainability_Score'].mean() if not recent_recs.empty else 0
        st.metric(label="Avg. Sustainability Score", value=f"{avg_sust:.1f}")
    
    st.subheader("Farm Distribution")
    if farms:
        farm_data = pd.DataFrame(farms)
        farm_data = farm_data.merge(recent_recs[['Farm_ID', 'Recommended_Crop']], 
                                   left_on='farm_id', right_on='Farm_ID', how='left')
        farm_data['Recommended_Crop'] = farm_data['Recommended_Crop'].fillna('Unknown')
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(farm_data, x='farm_id', y='total_area', color='Recommended_Crop', title='Farm Sizes by ID')
            st.plotly_chart(fig)
        with col2:
            fig = px.pie(farm_data, names='Recommended_Crop', values='total_area', title='Crop Distribution')
            st.plotly_chart(fig)
    else:
        st.write("No farm data available yet.")
    
    st.subheader("Recent Recommendations")
    if not recent_recs.empty:
        st.dataframe(recent_recs)
    else:
        st.write("No recommendations generated yet.")
    
    st.subheader("7-Day Weather Forecast")
    forecast = pd.DataFrame({
        'Day': ['Today', 'Tomorrow', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'],
        'Temperature_C': [24.5, 25.3, 23.8, 22.1, 24.0, 26.2, 25.8],
        'Rainfall_mm': [0, 2.5, 15.3, 0, 0, 4.2, 3.1],
        'Humidity_%': [45, 52, 78, 62, 48, 53, 55]
    })
    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(forecast, x='Day', y='Temperature_C', markers=True, title='Temperature Forecast')
        st.plotly_chart(fig)
    with col2:
        fig = px.bar(forecast, x='Day', y='Rainfall_mm', title='Rainfall Forecast')
        st.plotly_chart(fig)

# Farm Management page
elif page == "Farm Management":
    st.header("Farm Management")
    
    tab1, tab2 = st.tabs(["Register New Farm", "Update Farm Data"])
    
    with tab1:
        st.subheader("Register a New Farm")
        
        col1, col2 = st.columns(2)
        with col1:
            farm_id = st.number_input("Farm ID", min_value=1, step=1, value=106)
            location = st.text_input("Location", "Enter location")
        with col2:
            total_area = st.number_input("Total Area (acres)", min_value=1.0, value=50.0)
            farmer_name = st.text_input("Farmer Name", "Enter name")
        
        if st.button("Register Farm"):
            try:
                success = agents['farmer_agent'].store_farm_data(farm_id, location, total_area, farmer_name)
                if success:
                    st.success(f"Farm {farm_id} ({farmer_name}) successfully registered!")
                else:
                    st.error("Failed to register farm.")
            except Exception as e:
                st.error(f"Error registering farm: {str(e)}")
    
    with tab2:
        st.subheader("Update Environmental Conditions")
        
        farms = get_registered_farms()
        farm_ids = [farm['farm_id'] for farm in farms]
        if not farm_ids:
            st.warning("No farms registered yet. Please register a farm first.")
        else:
            selected_farm = st.selectbox("Select Farm ID", farm_ids)
            
            col1, col2 = st.columns(2)
            with col1:
                soil_ph = st.slider("Soil pH", 5.0, 8.0, 6.5)
                soil_moisture = st.slider("Soil Moisture (%)", 20.0, 80.0, 40.0)
            with col2:
                temperature = st.slider("Temperature (°C)", 15.0, 35.0, 25.0)
                rainfall = st.slider("Rainfall (mm)", 0.0, 500.0, 200.0)
            
            if st.button("Update Farm Conditions"):
                try:
                    success = agents['farmer_agent'].store_environmental_data(
                        selected_farm, soil_ph, soil_moisture, temperature, rainfall
                    )
                    if success:
                        st.success(f"Conditions for Farm {selected_farm} updated successfully!")
                    else:
                        st.error("Failed to update conditions.")
                except Exception as e:
                    st.error(f"Error updating conditions: {str(e)}")

# Recommendations page
elif page == "Recommendations":
    st.header("Crop Recommendations")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Get Recommendations")
        
        farms = get_registered_farms()
        farm_ids = [farm['farm_id'] for farm in farms]
        
        if not farm_ids:
            st.warning("No farms registered yet. Please register a farm in 'Farm Management'.")
        else:
            selected_farm = st.selectbox("Select Farm ID", farm_ids)
            
            st.write("Or enter conditions manually:")
            manual_soil_ph = st.slider("Soil pH (Manual)", 5.0, 8.0, 6.5, key="manual_ph")
            manual_soil_moisture = st.slider("Soil Moisture (%) (Manual)", 20.0, 80.0, 40.0, key="manual_moisture")
            manual_temperature = st.slider("Temperature (°C) (Manual)", 15.0, 35.0, 25.0, key="manual_temp")
            manual_rainfall = st.slider("Rainfall (mm) (Manual)", 0.0, 500.0, 200.0, key="manual_rain")
            
            use_manual = st.checkbox("Use manual conditions instead of stored data")
            
            if st.button("Generate Recommendation"):
                with st.spinner("Generating recommendation..."):
                    try:
                        if use_manual:
                            agents['farmer_agent'].store_environmental_data(
                                selected_farm, manual_soil_ph, manual_soil_moisture, 
                                manual_temperature, manual_rainfall
                            )
                        
                        recommendation = agents['decision_agent'].generate_recommendation(
                            selected_farm,
                            agents['env_agent'],
                            agents['crop_agent'],
                            agents['market_agent'],
                            agents['sust_agent']
                        )
                        
                        if "error" in recommendation:
                            st.error(recommendation["error"])
                        else:
                            # Calculate enhanced sustainability score
                            tips_followed = recommendation['sustainability_tips'][:2]  # Assume first 2 tips followed
                            sustainability_score = calculate_sustainability_score(
                                recommendation['predicted_sustainability'],
                                recommendation['recommended_fertilizer_kg'],
                                recommendation['recommended_pesticide_kg'],
                                tips_followed
                            )
                            recommendation['enhanced_sustainability_score'] = sustainability_score
                            
                            # Generate blockchain reward
                            reward = generate_blockchain_reward(selected_farm, sustainability_score)
                            recommendation['blockchain_reward'] = reward
                            
                            st.session_state.recommendation = recommendation
                            st.success("Recommendation generated successfully!")
                    except Exception as e:
                        st.error(f"Error generating recommendation: {str(e)}")
    
    with col2:
        if 'recommendation' in st.session_state:
            rec = st.session_state.recommendation
            
            # Enhanced UI with expanders and better styling
            st.markdown(f"""
                <h2 style='color: #2E7D32; text-align: center;'>Recommended Crop: {rec['recommended_crop']}</h2>
                <p style='text-align: center; color: #555;'>Optimized for your farm's conditions</p>
            """, unsafe_allow_html=True)
            
            # Key Metrics Card
            with st.expander("🌍 Key Performance Metrics", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Environmental Compatibility", f"{rec['environmental_compatibility']:.1f}%", 
                             delta=f"+{rec['environmental_compatibility']-50:.1f}%", delta_color="normal")
                with col2:
                    st.metric("Expected Yield", f"{rec['predicted_yield']:.2f} tons", 
                             help="Predicted yield based on current conditions")
                with col3:
                    st.metric("Sustainability Score", f"{rec['enhanced_sustainability_score']:.1f}", 
                             delta=f"+{rec['enhanced_sustainability_score']-rec['predicted_sustainability']:.1f}", 
                             delta_color="normal")
            
            # Financial Insights Card
            with st.expander("💰 Financial Insights", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Market Price", f"${rec['market_price']:.2f}/ton", 
                             help="Average market price from dataset")
                with col2:
                    st.metric("Demand Index", f"{rec['demand_index']:.1f}", 
                             delta=f"{rec['demand_index']-100:.1f}", delta_color="normal")
                with col3:
                    st.metric("Estimated Profit", f"${rec['estimated_profit']:.2f}", 
                             help="Revenue minus estimated costs")
            
            # Recommended Inputs Card
            with st.expander("🌱 Recommended Inputs", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Fertilizer", f"{rec['recommended_fertilizer_kg']:.1f} kg", 
                             help="Optimized for sustainability")
                with col2:
                    st.metric("Pesticide", f"{rec['recommended_pesticide_kg']:.1f} kg", 
                             help="Minimized for eco-friendliness")
            
            # Crop Comparison Visualization
            with st.expander("📊 Crop Comparison", expanded=False):
                crop_scores = pd.DataFrame({
                    'Crop': list(rec['all_crop_scores'].keys()),
                    'Score': list(rec['all_crop_scores'].values())
                })
                fig = px.bar(crop_scores, x='Crop', y='Score', color='Score',
                            color_continuous_scale='Viridis', title='Crop Suitability Scores',
                            height=400)
                fig.update_layout(xaxis_title="Crop Type", yaxis_title="Suitability Score (0-1)")
                st.plotly_chart(fig, use_container_width=True)
            
            # Sustainability Tips
            with st.expander("🌿 Sustainability Tips", expanded=True):
                st.markdown("**Follow these tips to boost your sustainability score:**")
                for i, tip in enumerate(rec['sustainability_tips'], 1):
                    st.write(f"{i}. {tip}")
            
            # Blockchain Reward Bonus (Improved Styling)
            with st.expander("🏆 Blockchain Sustainability Reward (Beta)", expanded=True):
                reward = rec['blockchain_reward']
                st.markdown(f"""
                    <div style='padding: 15px; border-radius: 10px; border: 2px solid #2E7D32; text-align: center;'>
                        <h4 style='color: #2E7D32; margin-bottom: 10px;'>Sustainability Reward</h4>
                        <p style='margin: 5px 0;'><b>Farm ID:</b> {reward['farm_id']}</p>
                        <p style='margin: 5px 0;'><b>Enhanced Sustainability Score:</b> {reward['sustainability_score']:.1f}</p>
                        <p style='margin: 5px 0;'><b>Reward Tokens:</b> {reward['reward_tokens']:.2f} AgriCoins</p>
                        <p style='margin: 5px 0;'><b>Smart Contract:</b> <code style='background-color: #FFF; padding: 2px 5px; border-radius: 3px;'>{reward['contract']}</code></p>
                        <p style='margin-top: 10px; font-size: 0.9em; color: #555;'>Implement sustainable practices to earn AgriCoins on our blockchain network!</p>
                    </div>
                """, unsafe_allow_html=True)

# Data Analysis page
elif page == "Data Analysis":
    st.header("Data Analysis")
    
    tab1, tab2 = st.tabs(["Crop Analysis", "Market Analysis"])
    
    with tab1:
        st.subheader("Crop Performance Analysis")
        
        crop_analysis = advisor_data.groupby('Crop_Type').agg({
            'Crop_Yield_ton': 'mean',
            'Sustainability_Score': 'mean',
            'Fertilizer_Usage_kg': 'mean',
            'Pesticide_Usage_kg': 'mean'
        }).reset_index()
        
        fig = px.scatter(crop_analysis, x='Crop_Yield_ton', y='Sustainability_Score',
                        size='Fertilizer_Usage_kg', color='Crop_Type',
                        hover_name='Crop_Type', size_max=60,
                        title='Crop Yield vs Sustainability')
        st.plotly_chart(fig)
        
        st.subheader("Environmental Factors Analysis")
        selected_crop = st.selectbox("Select Crop", advisor_data['Crop_Type'].unique())
        
        crop_data = advisor_data[advisor_data['Crop_Type'] == selected_crop]
        
        st.write("Correlation between environmental factors and crop performance")
        corr = crop_data[['Soil_pH', 'Soil_Moisture', 'Temperature_C', 'Rainfall_mm', 
                         'Fertilizer_Usage_kg', 'Pesticide_Usage_kg', 
                         'Crop_Yield_ton', 'Sustainability_Score']].corr()
        
        fig = px.imshow(corr, text_auto=True, aspect="auto",
                       title=f"Correlation Heatmap for {selected_crop}")
        st.plotly_chart(fig)
    
    with tab2:
        st.subheader("Market Trends Analysis")
        
        market_summary = market_data.groupby('Product').agg({
            'Market_Price_per_ton': 'mean',
            'Demand_Index': 'mean',
            'Supply_Index': 'mean'
        }).reset_index()
        
        market_summary['Demand_Supply_Ratio'] = market_summary['Demand_Index'] / market_summary['Supply_Index']
        
        fig = px.bar(market_summary, x='Product', y=['Market_Price_per_ton', 'Demand_Index', 'Supply_Index'],
                    barmode='group', title='Market Overview by Crop')
        st.plotly_chart(fig)
        
        fig = px.scatter(market_summary, x='Demand_Supply_Ratio', y='Market_Price_per_ton',
                        color='Product', size='Market_Price_per_ton', 
                        hover_name='Product', size_max=60,
                        title='Price vs Demand/Supply Ratio')
        fig.update_layout(xaxis_title="Demand/Supply Ratio", yaxis_title="Price per Ton")
        st.plotly_chart(fig)
        
        st.subheader("Seasonal Price Trends")
        seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        crops = market_summary['Product'].unique()
        
        seasonal_data = []
        for crop in crops:
            base_price = market_summary[market_summary['Product'] == crop]['Market_Price_per_ton'].values[0]
            for season in seasons:
                if season == 'Winter':
                    factor = 1.1 if crop in ['Wheat', 'Rice'] else 0.9
                elif season == 'Spring':
                    factor = 0.9 if crop in ['Wheat', 'Rice'] else 1.0
                elif season == 'Summer':
                    factor = 0.8 if crop in ['Wheat', 'Rice'] else 1.2
                else:  # Fall
                    factor = 1.0
                seasonal_data.append({'Crop': crop, 'Season': season, 'Price': base_price * factor})
        
        seasonal_df = pd.DataFrame(seasonal_data)
        fig = px.line(seasonal_df, x='Season', y='Price', color='Crop', markers=True,
                     title='Seasonal Price Trends')
        st.plotly_chart(fig)

# About page
elif page == "About":
    st.header("About the System")
    
    st.markdown("""
    ## Multi-Agent Sustainable Farming System
    
    This system was developed for the AI Agents Hackathon 2025, addressing the challenge of data-driven approaches for sustainable farming.
    
    ### System Architecture
    
    The system uses a multi-agent approach with the following specialized agents:
    
    1. **Farmer Input Agent**: Manages farm-specific data and environmental conditions.
    2. **Environmental Analysis Agent**: Analyzes soil and weather conditions to determine optimal crops.
    3. **Crop Recommendation Agent**: Uses machine learning models to predict crop performance.
    4. **Market Analysis Agent**: Analyzes market trends to maximize profitability.
    5. **Sustainability Optimization Agent**: Recommends practices to improve environmental sustainability.
    6. **Decision Integration Agent**: Combines all inputs to generate final recommendations.
    
    ### Key Features
    
    - **Personalized recommendations** based on farm-specific conditions
    - **Balances sustainability and profitability** through multi-objective optimization
    - **Long-term memory** via SQLite database for tracking farm performance over time
    - **Data-driven insights** from historical farming data and market trends
    - **Sustainable farming practices** recommended based on crop-specific requirements
    
    ### Technical Implementation
    
    - Python-based backend with machine learning models for prediction
    - SQLite database for persistent storage
    - Streamlit web interface for user interaction
    - Multi-agent system architecture for specialized decision-making
    """)
    
    st.subheader("Team Information")
    st.write("This project was developed by Bir Vision for the AI Agents Hackathon 2025.")
    st.write("Team member: Dharambir Singh Sidhu (Team leader)")
    

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center'><p>© Krishniti AI - Sustainable Farming AI System | Developed for AI Agents Hackathon 2025</p></div>", unsafe_allow_html=True)

