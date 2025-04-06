Krishniti AI: Multi-Agent System for Sustainable Farming
Project Overview
Krishniti AI is an innovative multi-agent AI system designed to optimize sustainable farming. By integrating environmental data, crop suitability analysis, market trends, and sustainability metrics, it provides farmers with personalized, data-driven recommendations that balance profitability and environmental responsibility. Developed for the AI Agents Hackathon 2025, this system addresses the challenge of "Data-Driven AI for Sustainable Farming" by empowering smallholder farmers with actionable insights.

Key Features
Multi-Agent Architecture: Six specialized agents collaborate to deliver holistic farming recommendations.
Sustainable Practices: Optimizes resource use (e.g., fertilizer, pesticide) while enhancing sustainability scores.
Market-Driven Insights: Incorporates crop prices and demand for economic viability.
Data-Driven Predictions: Leverages datasets to forecast crop performance and yields.
Interactive Dashboard: Streamlit-based UI offers real-time metrics, visualizations, and farm management tools.
Blockchain Rewards (Beta): Incentivizes eco-friendly practices with conceptual "AgriCoin" tokens.



Project Structure

krishniti-ai/
├── app.py                           # Main Streamlit application with UI and logic
├── sustainable_farming_system.py    # Multi-agent system and database setup
├── farmer_advisor_dataset.csv       # Crop and environmental data source
├── marketer_researcher_dataset.csv  # Market trends data source
├── farming_agents.db               # SQLite database for persistent storage
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation

Installation & Setup
Clone the Repository:

git clone https://github.com/[yourusername]/krishniti-ai.git
cd krishniti-ai

Create a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install Dependencies:
pip install -r requirements.txt

Run the Application:
streamlit run app.py



System Architecture

Agent Components

Farmer Input Agent:
Collects farm data (ID, location, area, farmer name) and environmental inputs (soil pH, moisture, temperature, rainfall).
Stores data in SQLite for real-time access.

Environmental Analysis Agent:
Evaluates soil conditions (pH 5.0-8.0, moisture 20-80%) and weather (temperature 15-35°C, rainfall 0-500mm).
Assesses compatibility with crop requirements.

Crop Recommendation Agent:
Predicts crop suitability and expected yields (e.g., up to 5 tons/ha) using farmer_advisor_dataset.csv.
Balances environmental fit with farmer needs.

Market Analysis Agent:
Analyzes market data (prices, demand index) from marketer_researcher_dataset.csv.
Estimates profitability for recommended crops.

Sustainability Optimization Agent:
Scores sustainability (0-100) based on input usage (e.g., fertilizer <100kg, pesticide <5kg).
Suggests eco-friendly practices (e.g., "Use compost to reduce fertilizer by 20%").

Decision Integration Agent:
Synthesizes outputs from all agents into a final recommendation.
Weights factors: 40% environmental, 30% market, 30% sustainability.

Database Schema
SQLite (farming_agents.db) ensures long-term memory with:
farms: Stores farm details (farm_id, farmer_name, location, total_area).
recommendations: Logs recommendations (farm_id, date, crop_type, yield, sustainability_score).

Usage Guide
Register Your Farm: Input farm ID, location, area, and farmer name via the "Farm Management" tab.
Update Conditions: Add current soil pH, moisture, temperature, and rainfall data.
Generate Recommendations: Click "Generate Recommendation" to receive crop suggestions and sustainability scores.
Explore Dashboard: View farm distribution (bar/pie charts), recent recommendations, and 7-day weather forecasts.
Review Insights: Analyze crop performance and market trends in the "Data Analysis" section.

Technologies Used
Python: Core language for logic and agent development.
Streamlit: Web framework for interactive UI and dashboards.
SQLite: Lightweight database for farm and recommendation storage.
Pandas: Data manipulation for CSV datasets and analysis.
NumPy: Numerical computations for scoring and predictions.
Plotly: Interactive visualizations (bar, pie, line charts).
Git: Version control for team collaboration.


Contributors
Dharambir Singh Sidhu - Team Leader
