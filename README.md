# Accident-Severity-Prediction-System
A machine learning–based system that predicts road accident severity using key accident factors and an interactive Streamlit interface.

## Features
- Predicts accident severity as Slight, Serious, or Fatal using machine learning
- Interactive Streamlit-based user interface
- Real-time prediction based on user inputs
- Visual feedback using emojis and icons for better understanding
- Data visualization using graphs and charts
- Easy to run and deploy locally

## Dataset Information
The dataset used in this project contains road accident-related data with features
such as day of week, light conditions, vehicle type, speed limit, road type,
pedestrian crossing, and number of passengers. These features are used to predict
the severity of road accidents.

## Working Flow
1. User enters accident-related inputs through the web interface
2. Input data is preprocessed and encoded
3. Trained machine learning model processes the input
4. Accident severity is predicted
5. Result is displayed along with emoji, image, and visual insights

## Machine Learning Model
A Random Forest Classifier is used to train the accident severity prediction model.
The model is trained after data cleaning, preprocessing, and feature engineering.
Joblib is used to save and load the trained model for real-time predictions.

## Data Visualization
The project includes multiple visualizations such as:
- Accident distribution by day of week
- Effect of speed limit on accident severity
- Road type vs accident severity
- Impact of light conditions on accidents
These graphs help in understanding accident patterns and trends.

## Future Scope
- Integration of real-time traffic and weather data
- Use of advanced deep learning models
- Deployment on cloud platforms
- Mobile application development
- Accident risk alert system

## Conclusion
This project demonstrates the practical use of machine learning in predicting
road accident severity. The system combines predictive modeling, data visualization,
and an interactive interface to provide meaningful insights for road safety analysis.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## Project Structure
accident-severity-prediction/
│
├── app.py
├── accident_model.pkl
├── accidents_india.csv
├── requirements.txt
└── README.md
