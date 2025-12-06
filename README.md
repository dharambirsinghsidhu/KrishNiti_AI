# KrishNiti AI: Data-Driven Agritech Intelligence
<br>

**Live Demo:  [Click Here for Live Demo](https://krishniti-ai.streamlit.app/)** <div id="top"></div>

<br>

## 📄 Overview

Krishniti AI is a multi-agent platform transforming Indian agriculture by tackling climate uncertainty, fragmented data, and sustainability challenges. It provides farmers with real-time, actionable insights to improve productivity and profitability.

By integrating environmental, crop, and market data, the system delivers intelligent recommendations that reduce climate risks, optimize resources, and support data-driven farming.

---

<br>

## ✨ Key Features

- **AI Crop Recommendations:** Identifies the best crops using farm conditions, yield potential, and sustainability metrics.
- **Multi-Agent Architecture:** Six coordinated agents handle environment analysis, market intelligence, and decision integration.
- **Environmental & Market Analysis:** Scores soil conditions, evaluates rainfall/temperature, and assesses price, demand, and competition.
- **Profit & Sustainability Insights:** Predicts yield, estimates revenue, and provides a 0–100 sustainability score with optimized input usage.
- **Streamlit Dashboard:** Clean, interactive interface with persistent farm profiles stored in SQLite.
- **ML-Powered Predictions:** Random Forest models drive accurate yield, resource, and impact predictions.
- **Scalable OOP Design:** Modular agent classes enable easy expansion with new crops, data sources, or logic.
- **Business Value:** Boosts yield, profit, and long-term sustainability through science-backed, data-driven farming.

---

<br>

## 🖼️ Visual Demonstration

Here are some visuals showcasing the application's interface and capabilities:


<div style="display: flex; flex-wrap: wrap; justify-content: space-around; gap: 10px;">
  <div style="flex: 1 1 300px; max-width: 48%; text-align: center; border: 1px solid #eee; padding: 5px;">
    <br>
    <p><b>Streamlit App Interface</b></p>
    <br>
    <img src="./images/home.png" alt="Upload Interface" style="max-width: 100%; height: auto; display: block; margin: 0 auto;">
  </div>
  <hr style="border: none; background-color: #ccc; height: 0.1px; margin: 20px 0;">
  
  <div style="flex: 1 1 300px; max-width: 48%; text-align: center; border: 1px solid #eee; padding: 5px;">
    <br>
    <p><b>Prediction Details</b></p>
    <br>
    <img src="./images/prediction.png" alt="Prediction Result" style="max-width: 100%; height: auto; display: block; margin: 0 auto;">
  </div>
</div>


---

<br>

## 📊 Dataset

**1. Farmer Advisor Dataset (`farmer_advisor_dataset.csv`)**

This dataset captures **farm-level environmental conditions, management practices, and outcomes** across **10,000 records**.  
It is used to train and evaluate the **yield prediction**, **crop advisory**, and **sustainability scoring** components.

#### **Key Columns**

| Column Name | Description |
|-------------|-------------|
| **Farm_ID** | Unique identifier for each farm record. |
| **Soil_pH** | Soil acidity/alkalinity level - used for crop–soil compatibility analysis. |
| **Soil_Moisture** | Soil moisture percentage - crucial for water stress assessment. |
| **Temperature_C** | Average temperature in °C at the farm. |
| **Rainfall_mm** | Recent or seasonal rainfall in millimeters. |
| **Crop_Type** | Crop grown (e.g., Wheat, Corn, Soybean, Rice). |
| **Fertilizer_Usage_kg** | Fertilizer applied per unit area (kg). |
| **Pesticide_Usage_kg** | Pesticide applied per unit area (kg). |
| **Crop_Yield_ton** | Observed yield in tons - **primary target variable** for prediction. |
| **Sustainability_Score** | Environmental/resource efficiency score - used for sustainability modeling. |

#### **How Krishniti AI Uses This Dataset**

- Trains ML models to **predict Crop_Yield_ton** and **Sustainability_Score** based on soil, weather, and farm inputs.  
- Powers the **Crop Recommendation Agent**, optimizing soil–crop matching.
- Supports the **Sustainability Optimization Agent**, advising on eco-friendly farming practices.
- Generates dynamic, realistic UI insights like:  
  - “Expected yield for given inputs”  
  - “Predicted sustainability score”  

<br>

**2. Market Researcher Dataset (`marketer_researcher_dataset.csv`)**

This dataset captures **market dynamics**, including prices, demand/supply indicators, seasonality, and economic influences across **10,000 market snapshots**.

#### **Key Columns**

| Column Name | Description |
|-------------|-------------|
| **Market_ID** | Unique identifier for each market snapshot. |
| **Product** | Crop/product name (e.g., Rice, Wheat, Corn). |
| **Market_Price_per_ton** | Current market selling price (per ton). |
| **Demand_Index** | Indicator of product demand (higher = stronger demand). |
| **Supply_Index** | Indicator of market supply (higher = more supply). |
| **Competitor_Price_per_ton** | Average competitor selling price. |
| **Economic_Indicator** | Macro-economic condition score (cost pressure, profitability, etc.). |
| **Weather_Impact_Score** | Effect of weather on market volatility and risk. |
| **Seasonal_Factor** | Categorical (Low/Medium/High) indicator of seasonality impact. |
| **Consumer_Trend_Index** | Measure of changing consumer preference for the product. |

#### **How Krishniti AI Uses This Dataset**

- Feeds the **Market Analysis Agent** to estimate:
  - Expected revenue  
  - Profitability  
  - Market risk  
- Helps rank crops by **economic viability**, not just agronomic suitability.  
- Enables real-time recommendations like:  
  - “Grow Crop X instead of Crop Y due to better demand and pricing this season.”  
  - “High volatility detected - diversify crop selection.”

<br>

## ⚙️ System Architecture & Logic Flow

The Krishniti AI platform is built around a sophisticated multi-agent system designed to process diverse agricultural data and provide comprehensive recommendations.

<br>
   
**Core Agents:**

1.  **Farmer Input Agent:** Gathers and processes farm-specific data from the farmer.
2.  **Environmental Analysis Agent:** Analyzes soil, temperature, and rainfall data to assess environmental suitability.
3.  **Crop Recommendation Agent:** Uses machine learning models to predict optimal crop performance based on environmental factors.
4.  **Market Analysis Agent:** Analyzes current market trends and pricing to maximize profitability.
5.  **Sustainability Optimization Agent:** Recommends practices to improve environmental sustainability and resource efficiency.
6.  **Decision Integration Agent:** Synthesizes information from all preceding agents to generate final, balanced recommendations.


<br>

<div style="display: flex; flex-wrap: wrap; justify-content: space-around; gap: 10px;">
    <div style="flex: 1 1 300px; max-width: 48%; text-align: center; border: 1px solid #eee; padding: 5px;">
        <div align="center">
          <img src="./images/ai-agents.png" alt="System Architecture Diagram" width="600"/>
        </div>    
    </div>
</div>

<br>

<div style="display: flex; flex-wrap: wrap; justify-content: space-around; gap: 10px;">
    <div style="flex: 1 1 300px; max-width: 48%; text-align: center; border: 1px solid #eee; padding: 5px;">
        <div align="center">
          <img src="./images/dataflow.png" alt="System Architecture Diagram" width="600"/>
        </div>    
    </div>
</div>

---

<br>

## 💻 Technologies Used

This project leverages a focused set of key technologies:

* **Streamlit:** Interactive web dashboard for farmer inputs and AI-driven recommendations.
* **Python 3.x:** Core programming language for agents, pipelines, and backend logic.
* **Scikit-learn:** Used for Random Forest models powering yield prediction and sustainability scoring.
* **Pandas:** Handles data processing, cleaning, and CSV dataset loading.
* **NumPy:** Performs numerical operations for environmental scoring and resource optimization.
* **SQLite:** Persistent local database for farm profiles, inputs, and recommendation history.
* **SQLAlchemy / sqlite3:** Manages database connection, queries, and ORM-like interactions.
* **OOP Python Classes:** Implements 6 modular AI agents following clean architecture and SRP principles.
* **Custom Data Pipelines:** Orchestrated through the `SustainableFarmingSystem` for multi-agent coordination.
* **Jupyter:** Used for dataset exploration, feature engineering, and model training.
* **Git & GitHub:** Version control, and project repository hosting.

---

<br>

## 📂 Project Structure

All core files for this project are organized in the root directory:

```bash
├── images/                           # Visual assets and system diagrams used in the README
│   ├── ai-agents.png                 # Diagram showing the multi-agent architecture
│   ├── dataflow.png                  # End-to-end system dataflow illustration
│   ├── home.png                      # Screenshot of the main interface/dashboard
│   └── prediction.png                # Screenshot of prediction/recommendation output
│
├── app.py                            # Main application script for the Krishniti AI system
├── app.ipynb                         # Notebook exploration of the application logic
├── sustainable_farming_system.py     # Core multi-agent architecture and orchestration
├── sustainable_farming_system.ipynb  # Notebook for system development, testing, and debugging
│
├── farmer_advisor_dataset.csv        # Dataset containing farm-level environmental and input data
├── marketer_researcher_dataset.csv   # Dataset containing market, demand, and economic indicators
├── farming_agents.db                 # SQLite database storing persistent farm profiles and history
│
├── requirements.txt                  # Python dependencies required to run the project
├── LICENSE                           # License information for legal usage and distribution
└── README.md                         # Main documentation and setup instructions
```

---

<br>

## 🚀 Getting Started

These instructions will give you a copy of the project up and running on
your local machine for development and testing purposes.

### Prerequisites

Requirements for the software and other tools to build, test and push 
- Python 3.8+ (preferably 3.11)

### Installation

1.  **Clone the Repository:**
   
    ```bash
    git clone https://github.com/dharambirsinghsidhu/Krishniti-AI.git
    cd KrishNiti_AI_Agritech_Intelligence
    ```

2.  **Create and Activate a Virtual Environment:**

    It's highly recommended to use a virtual environment to manage project dependencies.

    ```bash
    python -m venv venv
    ```

    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

3.  **Install Python Dependencies:**

    First, ensure your `pip` installer is up to date, then install the required Python libraries.

    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install streamlit
    ```

4.  **Run the Application:**

    Launch the Streamlit interface:

    ```bash
    streamlit run app.py
    ```

---

<br>

## 🤝 Contributing

We warmly welcome contributions to this project! If you're interested in improving the model, enhancing the user interface, or adding new functionalities, please follow these general steps:

1.  Fork the repository.
2.  Create a new branch for your feature (`git checkout -b feature/YourAwesomeFeature`).
3.  Commit your changes (`git commit -m 'Add a new feature'`).
4.  Push to your branch (`git push origin feature/YourAwesomeFeature`).
5.  Open a Pull Request, describing your changes in detail.

Please make sure your code adheres to good practices and includes relevant tests if applicable.

---

<br>

## 📧 Contact

For any questions or collaborations, feel free to reach out to the project maintainer:

* **Dharambir Singh Sidhu:** dharambirsinghsidhu.work@gmail.com

<br>

---

<br>

<div style="display: flex; flex-wrap: wrap; justify-content: space-around; gap: 10px;">
    <div style="flex: 1 1 300px; max-width: 48%; text-align: center; border: 1px solid #eee; padding: 5px;">
        <div align="center">
              <div>© 2025 Dharambir Singh Sidhu. Licensed under the <a href="./LICENSE">MIT License</a>.</div>
          <br>
          🔹<a href="#top"> Back to Top </a>🔹
        </div>    
    </div>
</div>

<br>

---
