import streamlit as st
import pandas as pd
import numpy as np
import pickle
import math

# Load models
pipe = pickle.load(open('pipe.pkl', 'rb'))
model = pickle.load(open('ml_model.pkl', 'rb'))

# Title of the app
st.title("IPL Prediction App")

# Main menu
st.sidebar.title("Choose Feature")
feature = st.sidebar.selectbox("Select a feature", ["Win Probability", "Score Prediction"])

if feature == "Win Probability":
    st.header("IPL Win Predictor")

    # Team selection
    teams = sorted([
        'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
        'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
        'Rajasthan Royals', 'Delhi Capitals'
    ])

    col1, col2 = st.columns(2)
    with col1:
        batting_team = st.selectbox('Select the batting team', teams)
    with col2:
        bowling_team = st.selectbox('Select the bowling team', teams)

    # City selection
    cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
              'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
              'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
              'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
              'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
              'Sharjah', 'Mohali', 'Bengaluru']
    selected_city = st.selectbox('Select City', sorted(cities))

    # Match details
    target = st.number_input('Target', min_value=0)
    col3, col4, col5 = st.columns(3)
    with col3:
        score = st.number_input('Score', min_value=0)
    with col4:
        wickets = st.number_input('Wickets', min_value=0, max_value=9)
    with col5:
        overs = st.number_input('Overs completed', min_value=0, max_value=20)

    if st.button('Predict Win Probability'):
        runs_left = target - score
        balls_left = 120 - overs * 6
        wickets_left = 10 - wickets
        crr = score / overs
        rrr = runs_left * 6 / balls_left if balls_left > 0 else 0

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_left],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        result = pipe.predict_proba(input_df)
        st.write(f"Win Probability for {batting_team}: {result[0][1] * 100:.2f}%")
        st.write(f"Win Probability for {bowling_team}: {result[0][0] * 100:.2f}%")

elif feature == "Score Prediction":
    st.header("IPL Score Predictor")

    teams = [
        'Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab',
        'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals',
        'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
    ]

    # Team selections
    batting_team = st.selectbox('Select the Batting Team', teams)
    bowling_team = st.selectbox('Select the Bowling Team', teams)

    # Validating team selection
    if batting_team == bowling_team:
        st.error("Batting and Bowling teams should be different")

    # Inputs for prediction
    col1, col2 = st.columns(2)
    with col1:
        overs = st.number_input('Enter the Current Over', min_value=5.1, max_value=19.5, value=5.1, step=0.1)
    with col2:
        runs = st.number_input('Enter Current runs', min_value=0, max_value=354, step=1, format='%i')

    wickets = st.slider('Enter Wickets fallen till now', 0, 9)
    col3, col4 = st.columns(2)
    with col3:
        runs_in_prev_5 = st.number_input('Runs scored in the last 5 overs', min_value=0, max_value=runs, step=1,
                                         format='%i')
    with col4:
        wickets_in_prev_5 = st.number_input('Wickets taken in the last 5 overs', min_value=0, max_value=wickets, step=1,
                                            format='%i')

    # Prepare input for the model
    prediction_array = []
    for team in teams:
        prediction_array.append(1 if team == batting_team else 0)
    for team in teams:
        prediction_array.append(1 if team == bowling_team else 0)
    prediction_array += [runs, wickets, overs, runs_in_prev_5, wickets_in_prev_5]

    if st.button('Predict Score'):
        predict = model.predict(np.array([prediction_array]))
        my_prediction = int(round(predict[0]))
        st.success(f"Predicted Score Range: {my_prediction - 5} to {my_prediction + 5}")
