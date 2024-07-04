import streamlit as st
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd
import json
import os
import joblib
import pickle
import numpy as np
import xgboost as xgb

# Load data

phone_dict_path = "/mount/src/sri-lankan-used-iphones-price-prediction-app-using-youtube-comments-mining/Dictionaries_TextBlob/Sort_Encodes_Phones_dict.json"
full_data_path = "/mount/src/sri-lankan-used-iphones-price-prediction-app-using-youtube-comments-mining/Full_Data_FbProphet_Sentiments_v2.xlsx"
cleaned_phone_options_path = "/mount/src/sri-lankan-used-iphones-price-prediction-app-using-youtube-comments-mining/Phone_Default_Details/Cleaned_phone_options_df.xlsx"
scaler_path = '/mount/src/sri-lankan-used-iphones-price-prediction-app-using-youtube-comments-mining/Price_Predict_XGBBOOST/scaler_X.pkl'
model_path = '/mount/src/sri-lankan-used-iphones-price-prediction-app-using-youtube-comments-mining/Price_Predict_XGBBOOST/best_xgb_params.pkl'
onehot_path = '/mount/src/sri-lankan-used-iphones-price-prediction-app-using-youtube-comments-mining/Price_Predict_XGBBOOST/onehotencoder.pkl'
phone_storages_path = '/mount/src/sri-lankan-used-iphones-price-prediction-app-using-youtube-comments-mining/Phone_Default_Details/F_Phone_storages.xlsx'
phone_colours_path = '/mount/src/sri-lankan-used-iphones-price-prediction-app-using-youtube-comments-mining/Phone_Default_Details/F_Phone_colours.xlsx'

phone_options_df = pd.read_excel(cleaned_phone_options_path)
phone_storages =  pd.read_excel(phone_storages_path)
phone_colours =  pd.read_excel(phone_colours_path)


with open(phone_dict_path, 'r') as f:
    phone_dict = json.load(f)

# Streamlit UI
#phones UI
st.title('Sri Lankan Used IPhones Price Prediction App using Youtube Comments mining')
phones = phone_options_df.Phone
selected_phone = st.selectbox('Select an Apple phone for Price prediction :', phones, index=0)
selected_phone_code = phone_dict[selected_phone]
# st.write(f"Selected phone code: {selected_phone}")

selected_storage = st.selectbox('Select Phone Storage (GB) :', phone_storages[selected_phone][phone_storages[selected_phone]!=0])
# st.write(f"Selected phone Storage : {selected_storage}")

selected_colour = st.selectbox('Select Phone Colour :', phone_colours[selected_phone][phone_colours[selected_phone].notna()])
# st.write(f"Selected phone Colour : {selected_colour}")

#date UI
full_df = pd.read_excel(full_data_path)
end_date = full_df["Date"].max()
st.write("### Select The DATE To Forecast")

# Date selection sliders
year, month, day, hour, minute = [end_date.year, end_date.month, end_date.day, end_date.hour, end_date.minute]
year = st.slider("Select a year", min_value=2024, max_value=2030, value=year)
month = st.slider("Select a month", min_value=1, max_value=12, value=month)
day = st.slider("Select a day", min_value=1, max_value=31, value=day)
hour = st.slider("Select an hour", min_value=0, max_value=23, value=hour)
minute = st.slider("Select a minute", min_value=0, max_value=59, value=minute)

selected_date = pd.Timestamp(year, month, day, hour, minute)
if selected_date < end_date:
    st.warning("Selected date cannot be earlier than the initial date. Resetting to the initial date.")
    selected_date = end_date

# Generate future dates
def auto_generate_points(start, end):
    delta = (end - start).total_seconds()
    if delta <= 3600:
        freq = 'T'
    elif delta <= 86400:
        freq = 'H'
    elif delta <= 2678400:
        freq = 'D'
    elif delta <= 31536000:
        freq = 'D'
    else:
        freq = 'MS'
    return pd.date_range(start=start, end=end, freq=freq)

date_range = auto_generate_points(end_date, selected_date)
df = pd.DataFrame(date_range, columns=['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Minute'] = df['Date'].dt.minute
df['Hour'] = df['Date'].dt.hour

cleaned_phone_options_df = pd.read_excel(cleaned_phone_options_path)
filtered_rows = cleaned_phone_options_df[cleaned_phone_options_df.Phone_Encode == selected_phone_code]

df["Minute_Difference"] = df["Date"].apply(lambda date: (date-filtered_rows.Released_Date).dt.total_seconds() / 60)

df['Phone_Storage'] = selected_storage
df['Phone_Encode'] =selected_phone_code


# Predict sentiment
def predict_sentiment(df, phone):
    df['Date'] = pd.to_datetime(df['Date'])
    future_dates = df[['Date']].rename(columns={'Date': 'ds'})

    models_path = f'/mount/src/sri-lankan-used-iphones-price-prediction-app-using-youtube-comments-mining/FbProphet_models_Sentiment_forcasting/{phone}'
    positive_model = joblib.load(os.path.join(models_path, f'{phone}_Positive.joblib'))
    negative_model = joblib.load(os.path.join(models_path, f'{phone}_Negative.joblib'))

    positive_forecast = positive_model.predict(future_dates)
    negative_forecast = negative_model.predict(future_dates)

    df['Positive_Predicted'] = positive_forecast['yhat']
    df['Negative_Predicted'] = negative_forecast['yhat']
    return df

df = predict_sentiment(df, selected_phone_code)

# Applying one-hot Encoding
with open(onehot_path, 'rb') as oh:
    OneHot = pickle.load(oh)
# np.array(selected_colour).reshape(-1, 1)
# st.write(selected_colour)
df["Phone_Colour"] = selected_colour
Colours_names = [
    'Phone_Colour_BLACK', 'Phone_Colour_BLACK TITANIUM',
    'Phone_Colour_BLACK/SLATE', 'Phone_Colour_BLUE',
    'Phone_Colour_DEEP PURPLE', 'Phone_Colour_GOLD',
    'Phone_Colour_GRAPHITE', 'Phone_Colour_GREEN',
    'Phone_Colour_JET BLACK', 'Phone_Colour_MATTE SPACE GRAY',
    'Phone_Colour_MIDNIGHT', 'Phone_Colour_PACIFIC BLUE',
    'Phone_Colour_PURPLE', 'Phone_Colour_RED',
    'Phone_Colour_ROSE GOLD', 'Phone_Colour_SIERRA BLUE',
    'Phone_Colour_SILVER', 'Phone_Colour_SPACE BLACK',
    'Phone_Colour_SPACE GRAY', 'Phone_Colour_STARLIGHT',
    'Phone_Colour_WHITE'
]
# st.write(df["Phone_Colour"])
# st.write(OneHot.transform(df[['Phone_Colour']]))
# Create a DataFrame with the encoded data
encoded_df = pd.DataFrame(OneHot.transform(df[['Phone_Colour']]), columns=Colours_names)

# Concatenate the original DataFrame and the one-hot encoded DataFrame
df = pd.concat([df, encoded_df], axis=1)

# st.write(df)
# st.write(df[['Minute_Difference', 'Phone_Storage', 'Phone_Encode',
#        'Positive_Predicted', 'Negative_Predicted', 'Year', 'Month', 'Day',
#        'Minute', 'Hour', 'Phone_Colour_BLACK', 'Phone_Colour_BLACK TITANIUM',
#        'Phone_Colour_BLACK/SLATE', 'Phone_Colour_BLUE',
#        'Phone_Colour_DEEP PURPLE', 'Phone_Colour_GOLD',
#        'Phone_Colour_GRAPHITE', 'Phone_Colour_GREEN', 'Phone_Colour_JET BLACK',
#        'Phone_Colour_MATTE SPACE GRAY', 'Phone_Colour_MIDNIGHT',
#        'Phone_Colour_PACIFIC BLUE', 'Phone_Colour_PURPLE', 'Phone_Colour_RED',
#        'Phone_Colour_ROSE GOLD', 'Phone_Colour_SIERRA BLUE',
#        'Phone_Colour_SILVER', 'Phone_Colour_SPACE BLACK',
#        'Phone_Colour_SPACE GRAY', 'Phone_Colour_STARLIGHT',
#        'Phone_Colour_WHITE']])
# Predict prices
def predict_phone_prices(df,scaler_X,loaded_model):

    X=df[['Minute_Difference', 'Phone_Storage', 'Phone_Encode',
       'Positive_Predicted', 'Negative_Predicted', 'Year', 'Month', 'Day',
       'Minute', 'Hour', 'Phone_Colour_BLACK', 'Phone_Colour_BLACK TITANIUM',
       'Phone_Colour_BLACK/SLATE', 'Phone_Colour_BLUE',
       'Phone_Colour_DEEP PURPLE', 'Phone_Colour_GOLD',
       'Phone_Colour_GRAPHITE', 'Phone_Colour_GREEN', 'Phone_Colour_JET BLACK',
       'Phone_Colour_MATTE SPACE GRAY', 'Phone_Colour_MIDNIGHT',
       'Phone_Colour_PACIFIC BLUE', 'Phone_Colour_PURPLE', 'Phone_Colour_RED',
       'Phone_Colour_ROSE GOLD', 'Phone_Colour_SIERRA BLUE',
       'Phone_Colour_SILVER', 'Phone_Colour_SPACE BLACK',
       'Phone_Colour_SPACE GRAY', 'Phone_Colour_STARLIGHT',
       'Phone_Colour_WHITE']]



    # loaded_model = joblib.load(model_path)
    X_scaled = scaler_X.transform(X)
    predictions = loaded_model.predict(X_scaled)
    df["Price"] = predictions
    return df

with open(scaler_path, 'rb') as f:
    scaler_X = pickle.load(f)

with open(model_path, 'rb') as m:
    loaded_model = joblib.load(m)

df = predict_phone_prices(df,scaler_X,loaded_model)
st.subheader(f'Generated Future Data Set till {selected_date}')
st.write(df)

# Plot Sentiments
st.subheader(f'Forecasting Sentiments till {selected_date}')
original_positive_predictions = full_df[["Date", "Positive_Predicted"]].sort_values(by="Date")[full_df.Phone_Encode == selected_phone_code]
original_negative_predictions = full_df[["Date", "Negative_Predicted"]].sort_values(by="Date")[full_df.Phone_Encode == selected_phone_code]
forecast_positive_predictions = df[["Date", "Positive_Predicted"]]
forecast_negative_predictions = df[["Date", "Negative_Predicted"]]
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=original_positive_predictions["Date"], y=original_positive_predictions["Positive_Predicted"], mode='lines', name='Original Positive Predicted',line=dict(color='green')))
fig2.add_trace(go.Scatter(x=original_negative_predictions["Date"], y=original_negative_predictions["Negative_Predicted"], mode='lines', name='Original Negative Predicted',line=dict(color='red')))
fig2.add_trace(go.Scatter(x=forecast_positive_predictions["Date"], y=forecast_positive_predictions["Positive_Predicted"], mode='lines+markers', name='Forecast Positive Predicted', line=dict(color='green')))
fig2.add_trace(go.Scatter(x=forecast_negative_predictions["Date"], y=forecast_negative_predictions["Negative_Predicted"], mode='lines+markers', name='Forecast Negative Predicted', line=dict(color='red')))
fig2.add_trace(go.Scatter(
    x=[original_positive_predictions.iloc[-1]["Date"], forecast_positive_predictions.iloc[0]["Date"]],
    y=[original_positive_predictions.iloc[-1]["Positive_Predicted"], forecast_positive_predictions.iloc[0]["Positive_Predicted"]],
    mode='lines',
    name='Connecting Line',
    line=dict(color='green')
))
fig2.add_trace(go.Scatter(
    x=[original_negative_predictions.iloc[-1]["Date"], forecast_negative_predictions.iloc[0]["Date"]],
    y=[original_negative_predictions.iloc[-1]["Negative_Predicted"], forecast_negative_predictions.iloc[0]["Negative_Predicted"]],
    mode='lines',
    name='Connecting Line',
    line=dict(color="red")
))

st.plotly_chart(fig2)



# Plot forecast
st.subheader(f'Forecasting Fair Price till {selected_date}')
original = full_df[["Date", "Price"]].sort_values(by="Date")[full_df.Phone_Encode == selected_phone_code]
forecast = df[["Date", "Price"]]
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=original["Date"], y=original["Price"], mode='lines', name='Original'))
fig1.add_trace(go.Scatter(x=forecast["Date"], y=forecast["Price"], mode='lines', name='Forecast', line=dict(color='red')))
fig1.add_trace(go.Scatter(
    x=[original.iloc[-1]["Date"], forecast.iloc[0]["Date"]],
    y=[original.iloc[-1]["Price"], forecast.iloc[0]["Price"]],
    mode='lines',
    name='Connecting Line',
    line=dict(color='red')
))
st.plotly_chart(fig1)
# st.subheader(f'Fair Predicted Price: RS. {forecast.loc[forecast["Date"].idxmax(), "Price"]}')

# Find the max date's price
max_date_price = round(float(forecast.loc[forecast["Date"].idxmax(), "Price"]),2)

# HTML string with inline CSS for yellow color
html_str = f"""
    <h3>Fair Predicted Price on {selected_date}: <span style='color: yellow;'>RS. {max_date_price}</span></h3>
"""

# Render the HTML with Streamlit
st.markdown(html_str, unsafe_allow_html=True)
