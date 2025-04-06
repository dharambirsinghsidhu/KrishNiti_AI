#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sqlite3
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Database setup
def setup_database():
    """Create SQLite database for long-term memory storage"""
    conn = sqlite3.connect('farming_agents.db')
    cursor = conn.cursor()
    
    # Create farms table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS farms (
        farm_id INTEGER PRIMARY KEY,
        location TEXT,
        total_area REAL,
        farmer_name TEXT,
        last_updated TIMESTAMP
    )
    ''')
    
    # Create environmental_conditions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS environmental_conditions (
        record_id INTEGER PRIMARY KEY AUTOINCREMENT,
        farm_id INTEGER,
        soil_ph REAL,
        soil_moisture REAL,
        temperature_c REAL,
        rainfall_mm REAL,
        measurement_date TIMESTAMP,
        FOREIGN KEY (farm_id) REFERENCES farms (farm_id)
    )
    ''')
    
    # Create recommendations table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS recommendations (
        recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        farm_id INTEGER,
        crop_type TEXT,
        fertilizer_recommendation REAL,
        pesticide_recommendation REAL,
        expected_yield REAL,
        sustainability_score REAL,
        market_price REAL,
        profit_estimate REAL,
        recommendation_date TIMESTAMP,
        FOREIGN KEY (farm_id) REFERENCES farms (farm_id)
    )
    ''')
    
    # Create historical_data table
    # cursor.execute('''
    # CREATE TABLE IF NOT EXISTS historical_data (
    #     record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    #     farm_id INTEGER,
    #     crop_type TEXT,
    #     planting_date TIMESTAMP,
    #     harvest_date TIMESTAMP,
    #     actual_yield REAL,
    #     actual_sustainability REAL,
    #     notes TEXT,
    #     FOREIGN KEY (farm_id) REFERENCES farms (farm_id)
    # )
    # ''')
    
    conn.commit()
    conn.close()
    return True

# Agent classes
class FarmerInputAgent:
    """Agent to process farmer input data"""
    
    def __init__(self, db_path='farming_agents.db'):
        self.db_path = db_path
    
    def store_farm_data(self, farm_id, location, total_area, farmer_name):
        """Store basic farm information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT OR REPLACE INTO farms (farm_id, location, total_area, farmer_name, last_updated)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (farm_id, location, total_area, farmer_name))
        
        conn.commit()
        conn.close()
        return True
    
    def store_environmental_data(self, farm_id, soil_ph, soil_moisture, temperature_c, rainfall_mm):
        """Store current environmental conditions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO environmental_conditions 
        (farm_id, soil_ph, soil_moisture, temperature_c, rainfall_mm, measurement_date)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (farm_id, soil_ph, soil_moisture, temperature_c, rainfall_mm))
        
        conn.commit()
        conn.close()
        return True
    
    def get_latest_conditions(self, farm_id):
        """Retrieve latest environmental conditions for a farm"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT soil_ph, soil_moisture, temperature_c, rainfall_mm
        FROM environmental_conditions
        WHERE farm_id = ?
        ORDER BY measurement_date DESC
        LIMIT 1
        ''', (farm_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'soil_ph': result[0],
                'soil_moisture': result[1],
                'temperature_c': result[2],
                'rainfall_mm': result[3]
            }
        return None


class EnvironmentalAnalysisAgent:
    """Agent to analyze environmental conditions"""
    
    def __init__(self, advisor_data_path):
        # Load and process the farmer advisor dataset
        self.data = pd.read_csv(advisor_data_path)
        
        # Calculate optimal ranges for each crop
        self.crop_optimal_ranges = self._calculate_optimal_ranges()
    
    def _calculate_optimal_ranges(self):
        """Calculate optimal environmental ranges for each crop"""
        crops = self.data['Crop_Type'].unique()
        optimal_ranges = {}
        
        for crop in crops:
            crop_data = self.data[self.data['Crop_Type'] == crop]
            
            # Filter for high-performing instances (top 25% by yield and sustainability)
            yield_threshold = crop_data['Crop_Yield_ton'].quantile(0.75)
            sust_threshold = crop_data['Sustainability_Score'].quantile(0.75)
            top_performers = crop_data[(crop_data['Crop_Yield_ton'] >= yield_threshold) & 
                                      (crop_data['Sustainability_Score'] >= sust_threshold)]
            
            # Calculate ranges
            optimal_ranges[crop] = {
                'soil_ph': {
                    'min': top_performers['Soil_pH'].min(),
                    'max': top_performers['Soil_pH'].max(),
                    'mean': top_performers['Soil_pH'].mean()
                },
                'soil_moisture': {
                    'min': top_performers['Soil_Moisture'].min(),
                    'max': top_performers['Soil_Moisture'].max(),
                    'mean': top_performers['Soil_Moisture'].mean()
                },
                'temperature_c': {
                    'min': top_performers['Temperature_C'].min(),
                    'max': top_performers['Temperature_C'].max(),
                    'mean': top_performers['Temperature_C'].mean()
                },
                'rainfall_mm': {
                    'min': top_performers['Rainfall_mm'].min(),
                    'max': top_performers['Rainfall_mm'].max(),
                    'mean': top_performers['Rainfall_mm'].mean()
                },
                'fertilizer_usage': {
                    'min': top_performers['Fertilizer_Usage_kg'].min(),
                    'max': top_performers['Fertilizer_Usage_kg'].max(),
                    'mean': top_performers['Fertilizer_Usage_kg'].mean()
                },
                'pesticide_usage': {
                    'min': top_performers['Pesticide_Usage_kg'].min(),
                    'max': top_performers['Pesticide_Usage_kg'].max(),
                    'mean': top_performers['Pesticide_Usage_kg'].mean()
                }
            }
        
        return optimal_ranges
    
    def get_environmental_compatibility(self, conditions):
        """
        Assess compatibility of current conditions with each crop
        Returns a compatibility score (0-100) for each crop
        """
        compatibility_scores = {}
        
        for crop, optimal in self.crop_optimal_ranges.items():
            # Calculate how close current conditions are to optimal ranges
            ph_score = self._calculate_range_score(conditions['soil_ph'], 
                                                 optimal['soil_ph']['min'], 
                                                 optimal['soil_ph']['max'])
            
            moisture_score = self._calculate_range_score(conditions['soil_moisture'], 
                                                      optimal['soil_moisture']['min'], 
                                                      optimal['soil_moisture']['max'])
            
            temp_score = self._calculate_range_score(conditions['temperature_c'], 
                                                  optimal['temperature_c']['min'], 
                                                  optimal['temperature_c']['max'])
            
            rain_score = self._calculate_range_score(conditions['rainfall_mm'], 
                                                  optimal['rainfall_mm']['min'], 
                                                  optimal['rainfall_mm']['max'])
            
            # Overall compatibility score (average of individual scores)
            compatibility_scores[crop] = (ph_score + moisture_score + temp_score + rain_score) / 4
        
        return compatibility_scores
    
    def _calculate_range_score(self, value, min_val, max_val):
        """Calculate how close a value is to being within an optimal range (0-100)"""
        if min_val <= value <= max_val:
            return 100  # Perfect score if within range
        
        # Calculate distance from range as percentage of range width
        range_width = max_val - min_val
        if value < min_val:
            distance = min_val - value
        else:
            distance = value - max_val
            
        # Convert to a score (0-100)
        score = max(0, 100 - (distance / (range_width * 0.5) * 100))
        return score


class CropRecommendationAgent:
    """Agent to recommend optimal crops based on environmental conditions"""
    
    def __init__(self, advisor_data_path):
        self.data = pd.read_csv(advisor_data_path)
        self.models = self._train_crop_models()
    
    def _train_crop_models(self):
        """Train models to predict yield and sustainability for each crop"""
        crops = self.data['Crop_Type'].unique()
        models = {}
        
        for crop in crops:
            crop_data = self.data[self.data['Crop_Type'] == crop]
            
            # Features and targets
            X = crop_data[['Soil_pH', 'Soil_Moisture', 'Temperature_C', 'Rainfall_mm', 
                         'Fertilizer_Usage_kg', 'Pesticide_Usage_kg']]
            y_yield = crop_data['Crop_Yield_ton']
            y_sust = crop_data['Sustainability_Score']
            
            # Train yield model
            X_train, X_test, y_train, y_test = train_test_split(X, y_yield, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            yield_model = RandomForestRegressor(n_estimators=100, random_state=42)
            yield_model.fit(X_train_scaled, y_train)
            
            # Train sustainability model
            X_train, X_test, y_train, y_test = train_test_split(X, y_sust, test_size=0.2, random_state=42)
            X_train_scaled = scaler.fit_transform(X_train)
            
            sust_model = RandomForestRegressor(n_estimators=100, random_state=42)
            sust_model.fit(X_train_scaled, y_train)
            
            models[crop] = {
                'yield_model': yield_model,
                'sustainability_model': sust_model,
                'scaler': scaler
            }
        
        return models
    
    def predict_performance(self, conditions, fertilizer_kg, pesticide_kg):
        """Predict yield and sustainability for each crop with given conditions"""
        predictions = {}
        
        for crop, model_set in self.models.items():
            # Prepare input features
            features = np.array([[
                conditions['soil_ph'],
                conditions['soil_moisture'],
                conditions['temperature_c'],
                conditions['rainfall_mm'],
                fertilizer_kg,
                pesticide_kg
            ]])
            
            # Scale features
            features_scaled = model_set['scaler'].transform(features)
            
            # Make predictions
            yield_pred = model_set['yield_model'].predict(features_scaled)[0]
            sust_pred = model_set['sustainability_model'].predict(features_scaled)[0]
            
            predictions[crop] = {
                'predicted_yield': yield_pred,
                'predicted_sustainability': sust_pred,
                'combined_score': (yield_pred / 10) + (sust_pred / 100)  # Normalize and combine
            }
        
        return predictions

    def get_best_crop(self, predictions, env_compatibility, market_data):
        """Determine the best crop based on predictions, environmental compatibility, and market data"""
        crop_scores = {}
        
        for crop in predictions:
            # Calculate overall score (weighted combination of factors)
            yield_score = predictions[crop]['predicted_yield'] / 10  # Normalize yield (0-10)
            sust_score = predictions[crop]['predicted_sustainability'] / 100  # Normalize sustainability (0-1)
            env_score = env_compatibility[crop] / 100  # Normalize compatibility (0-1)
            
            # Get market score (if available)
            market_score = 0
            if crop in market_data:
                # Normalize market indicators (0-1)
                price_score = market_data[crop]['price'] / 1000  # Assuming max price around 1000
                demand_score = market_data[crop]['demand_index'] / 200  # Assuming max demand index around 200
                market_score = (price_score + demand_score) / 2
            
            # Calculate weighted final score (adjust weights as needed)
            crop_scores[crop] = (
                yield_score * 0.3 +
                sust_score * 0.3 +
                env_score * 0.2 +
                market_score * 0.2
            )
        
        # Find the best crop (highest score)
        best_crop = max(crop_scores.items(), key=lambda x: x[1])
        
        return {
            'best_crop': best_crop[0],
            'score': best_crop[1],
            'all_scores': crop_scores
        }


class MarketAnalysisAgent:
    """Agent to analyze market trends and pricing"""
    
    def __init__(self, market_data_path):
        self.data = pd.read_csv(market_data_path)
        
    def get_crop_market_data(self):
        """Get market data for each crop type"""
        market_data = {}
        
        for product in self.data['Product'].unique():
            product_data = self.data[self.data['Product'] == product]
            
            # Calculate average price and demand
            avg_price = product_data['Market_Price_per_ton'].mean()
            avg_demand = product_data['Demand_Index'].mean()
            avg_supply = product_data['Supply_Index'].mean()
            
            # Calculate price trend (positive means increasing)
            price_trend = 0
            if len(product_data) > 1:
                price_trend = product_data['Market_Price_per_ton'].pct_change().mean()
            
            market_data[product] = {
                'price': avg_price,
                'demand_index': avg_demand,
                'supply_index': avg_supply,
                'price_trend': price_trend,
                'demand_supply_ratio': avg_demand / avg_supply if avg_supply > 0 else 0
            }
        
        return market_data
    
    def get_profit_estimate(self, crop, predicted_yield, market_data):
        """Estimate profit for a specific crop based on predicted yield and market data"""
        if crop not in market_data:
            return 0
        
        # Basic profit calculation
        market_price = market_data[crop]['price']
        revenue = predicted_yield * market_price
        
        # Estimated production cost (simplified calculation)
        # In a real scenario, this would be more complex and include many factors
        base_cost_per_ton = market_price * 0.6  # Assume 60% of price goes to production costs
        production_cost = predicted_yield * base_cost_per_ton
        
        profit = revenue - production_cost
        return profit


class SustainabilityOptimizationAgent:
    """Agent to recommend sustainable farming practices"""
    
    def __init__(self, advisor_data_path):
        self.data = pd.read_csv(advisor_data_path)
        
    def optimize_inputs(self, crop, conditions):
        """Optimize fertilizer and pesticide usage for sustainability"""
        crop_data = self.data[self.data['Crop_Type'] == crop]
        
        # Find similar environmental conditions
        similar_conditions = crop_data[
            (crop_data['Soil_pH'].between(conditions['soil_ph'] - 0.5, conditions['soil_ph'] + 0.5)) &
            (crop_data['Soil_Moisture'].between(conditions['soil_moisture'] - 5, conditions['soil_moisture'] + 5)) &
            (crop_data['Temperature_C'].between(conditions['temperature_c'] - 3, conditions['temperature_c'] + 3)) &
            (crop_data['Rainfall_mm'].between(conditions['rainfall_mm'] - 30, conditions['rainfall_mm'] + 30))
        ]
        
        if len(similar_conditions) == 0:
            # If no similar conditions found, use the whole crop dataset
            similar_conditions = crop_data
        
        # Sort by sustainability score (descending)
        top_sustainable = similar_conditions.sort_values('Sustainability_Score', ascending=False).head(10)
        
        # Get average input usage from top sustainable practices
        recommended_fertilizer = top_sustainable['Fertilizer_Usage_kg'].mean()
        recommended_pesticide = top_sustainable['Pesticide_Usage_kg'].mean()
        
        return {
            'recommended_fertilizer_kg': recommended_fertilizer,
            'recommended_pesticide_kg': recommended_pesticide,
            'expected_sustainability': top_sustainable['Sustainability_Score'].mean()
        }
    
    def get_sustainability_tips(self, crop, conditions):
        """Generate sustainability tips based on conditions"""
        tips = []
        
        # General sustainability tips
        tips.append("Practice crop rotation to improve soil health and reduce pest pressures.")
        tips.append("Consider using cover crops during off-seasons to prevent soil erosion.")
        tips.append("Implement precision farming techniques to optimize resource usage.")
        
        # Crop-specific tips
        if crop == 'Wheat':
            tips.append("Consider using drought-resistant wheat varieties to reduce water usage.")
            tips.append("Implement conservation tillage to reduce soil erosion and improve water retention.")
        elif crop == 'Rice':
            tips.append("Consider alternate wetting and drying technique to reduce water usage.")
            tips.append("Use organic fertilizers when possible to improve soil health.")
        elif crop == 'Corn':
            tips.append("Plant nitrogen-fixing cover crops to reduce fertilizer needs.")
            tips.append("Consider strip-tillage to reduce soil disturbance while maintaining yields.")
        elif crop == 'Soybean':
            tips.append("Soybeans fix nitrogen naturally - reduce fertilizer application accordingly.")
            tips.append("Use integrated pest management to minimize pesticide usage.")
        
        # Condition-specific tips
        if conditions['soil_ph'] < 6.0:
            tips.append("Consider applying lime to raise soil pH to optimal levels.")
        elif conditions['soil_ph'] > 7.5:
            tips.append("Consider applying sulfur or organic matter to lower soil pH.")
        
        if conditions['soil_moisture'] < 30:
            tips.append("Implement water conservation techniques such as mulching.")
            tips.append("Consider drip irrigation to optimize water usage.")
        elif conditions['soil_moisture'] > 70:
            tips.append("Improve drainage to prevent waterlogging and root diseases.")
        
        return tips


class DecisionIntegrationAgent:
    """Agent to integrate all recommendations and provide final advice"""
    
    def __init__(self, db_path='farming_agents.db'):
        self.db_path = db_path
    
    def generate_recommendation(self, farm_id, environmental_agent, crop_agent, market_agent, sustainability_agent):
        """Generate comprehensive recommendations for the farmer"""
        # Get farm's latest conditions
        farmer_agent = FarmerInputAgent(self.db_path)
        conditions = farmer_agent.get_latest_conditions(farm_id)
        
        if not conditions:
            return {"error": "No environmental data found for this farm"}
        
        # Get environmental compatibility scores
        env_compatibility = environmental_agent.get_environmental_compatibility(conditions)
        
        # Get market data
        market_data = market_agent.get_crop_market_data()
        
        # For each crop, get optimized inputs and predict performance
        crop_predictions = {}
        for crop in env_compatibility.keys():
            # Get optimized inputs
            optimized_inputs = sustainability_agent.optimize_inputs(crop, conditions)
            
            # Predict performance with optimized inputs
            performance = crop_agent.predict_performance(
                conditions, 
                optimized_inputs['recommended_fertilizer_kg'],
                optimized_inputs['recommended_pesticide_kg']
            )
            
            crop_predictions[crop] = {
                **performance[crop],
                **optimized_inputs
            }
        
        # Get best crop recommendation
        recommendation = crop_agent.get_best_crop(
            {crop: data for crop, data in crop_predictions.items()},
            env_compatibility,
            market_data
        )
        
        best_crop = recommendation['best_crop']
        
        # Calculate profit estimate
        profit_estimate = market_agent.get_profit_estimate(
            best_crop,
            crop_predictions[best_crop]['predicted_yield'],
            market_data
        )
        
        # Get sustainability tips
        sustainability_tips = sustainability_agent.get_sustainability_tips(best_crop, conditions)
        
        # Store recommendation in database
        self._store_recommendation(
            farm_id,
            best_crop,
            crop_predictions[best_crop]['recommended_fertilizer_kg'],
            crop_predictions[best_crop]['recommended_pesticide_kg'],
            crop_predictions[best_crop]['predicted_yield'],
            crop_predictions[best_crop]['predicted_sustainability'],
            market_data.get(best_crop, {}).get('price', 0),
            profit_estimate
        )
        
        # Format final recommendation
        final_recommendation = {
            'recommended_crop': best_crop,
            'environmental_compatibility': env_compatibility[best_crop],
            'predicted_yield': crop_predictions[best_crop]['predicted_yield'],
            'predicted_sustainability': crop_predictions[best_crop]['predicted_sustainability'],
            'recommended_fertilizer_kg': crop_predictions[best_crop]['recommended_fertilizer_kg'],
            'recommended_pesticide_kg': crop_predictions[best_crop]['recommended_pesticide_kg'],
            'market_price': market_data.get(best_crop, {}).get('price', 0),
            'demand_index': market_data.get(best_crop, {}).get('demand_index', 0),
            'estimated_profit': profit_estimate,
            'sustainability_tips': sustainability_tips,
            'all_crop_scores': recommendation['all_scores']
        }
        
        return final_recommendation
    
    def _store_recommendation(self, farm_id, crop_type, fertilizer, pesticide, 
                             expected_yield, sustainability, market_price, profit):
        """Store recommendation in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO recommendations 
        (farm_id, crop_type, fertilizer_recommendation, pesticide_recommendation, 
         expected_yield, sustainability_score, market_price, profit_estimate, recommendation_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (farm_id, crop_type, fertilizer, pesticide, expected_yield, 
              sustainability, market_price, profit))
        
        conn.commit()
        conn.close()
        return True
    
    def get_historical_recommendations(self, farm_id):
        """Retrieve historical recommendations for a farm"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT recommendation_id, crop_type, fertilizer_recommendation, pesticide_recommendation,
               expected_yield, sustainability_score, market_price, profit_estimate, recommendation_date
        FROM recommendations
        WHERE farm_id = ?
        ORDER BY recommendation_date DESC
        ''', (farm_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        recommendations = []
        for row in results:
            recommendations.append({
                'recommendation_id': row[0],
                'crop_type': row[1],
                'fertilizer_recommendation': row[2],
                'pesticide_recommendation': row[3],
                'expected_yield': row[4],
                'sustainability_score': row[5],
                'market_price': row[6],
                'profit_estimate': row[7],
                'recommendation_date': row[8]
            })
        
        return recommendations


# In[5]:


# get_ipython().system('pip install streamlit')


# In[2]:


# Main class that brings all the agents together
class SustainableFarmingSystem:
    """Main class that coordinates all agents"""
    
    def __init__(self, advisor_data_path, market_data_path, db_path='farming_agents.db'):
        # Initialize database
        setup_database()
        
        # Initialize all agents
        self.farmer_agent = FarmerInputAgent(db_path)
        self.environmental_agent = EnvironmentalAnalysisAgent(advisor_data_path)
        self.crop_agent = CropRecommendationAgent(advisor_data_path)
        self.market_agent = MarketAnalysisAgent(market_data_path)
        self.sustainability_agent = SustainabilityOptimizationAgent(advisor_data_path)
        self.decision_agent = DecisionIntegrationAgent(db_path)
    
    def register_farm(self, farm_id, location, total_area, farmer_name):
        """Register a new farm in the system"""
        return self.farmer_agent.store_farm_data(farm_id, location, total_area, farmer_name)
    
    def update_farm_conditions(self, farm_id, soil_ph, soil_moisture, temperature_c, rainfall_mm):
        """Update environmental conditions for a farm"""
        return self.farmer_agent.store_environmental_data(
            farm_id, soil_ph, soil_moisture, temperature_c, rainfall_mm
        )
    
    def get_recommendation(self, farm_id):
        """Get comprehensive recommendation for a farm"""
        return self.decision_agent.generate_recommendation(
            farm_id,
            self.environmental_agent,
            self.crop_agent,
            self.market_agent,
            self.sustainability_agent
        )
    
    def get_historical_recommendations(self, farm_id):
        """Get historical recommendations for a farm"""
        return self.decision_agent.get_historical_recommendations(farm_id)


# Example usage
def main():
    # Set up the system
    system = SustainableFarmingSystem(
        advisor_data_path='farmer_advisor_dataset.csv',
        market_data_path='marketer_researcher_dataset.csv'
    )
    
    # Register a farm
    system.register_farm(
        farm_id=101,
        location="Midwest Region",
        total_area=50.5,  # acres
        farmer_name="John Smith"
    )
    
    # Update environmental conditions
    system.update_farm_conditions(
        farm_id=101,
        soil_ph=6.8,
        soil_moisture=45.2,
        temperature_c=24.5,
        rainfall_mm=210.3
    )
    
    # Get recommendation
    recommendation = system.get_recommendation(farm_id=101)
    
    # Print recommendation
    print(f"Recommended crop: {recommendation['recommended_crop']}")
    print(f"Expected yield: {recommendation['predicted_yield']:.2f} tons")
    print(f"Sustainability score: {recommendation['predicted_sustainability']:.2f}")
    print(f"Fertilizer recommendation: {recommendation['recommended_fertilizer_kg']:.2f} kg")
    print(f"Pesticide recommendation: {recommendation['recommended_pesticide_kg']:.2f} kg")
    print(f"Estimated profit: ${recommendation['estimated_profit']:.2f}")
    print("\nSustainability tips:")
    for tip in recommendation['sustainability_tips']:
        print(f"- {tip}")


if __name__ == "__main__":
    main()


# In[ ]:




