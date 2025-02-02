# -*- coding: utf-8 -*-
"""Untitled18.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1jtgSuor4-rCjovSyG3JieqEKdkfAATWh
"""

#! pip install gradio

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import requests
import spacy
import pandas as pd
import folium
import gradio as gr
import os
from io import BytesIO

from config import geoapify_key, open_weather_key

csv_file = "Resources/queries.csv"
df = pd.read_csv(csv_file)

# Check columns
print(df.columns)

# Prepare the data
X = df['Query']
y = df['Category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

# Predict intent for a new sentence
def predict_intent(user_input):
    user_input = user_input.lower()
    return model.predict([user_input])[0]

# Load pre-trained spaCy model
nlp = spacy.load('en_core_web_sm')

# Function to extract entities from user input
def extract_entities(user_input):
    doc = nlp(user_input)
    entities = {ent.label_: ent.text for ent in doc.ents}  # Return as a dictionary
    return entities

# Function to get weather data from an API
def get_weather(location):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={open_weather_key}&units=imperial"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

# Function to get restaurant data from an API
def get_restaurants(location):
    geocoding_url = f"https://api.geoapify.com/v1/geocode/search?text={location}&apiKey={geoapify_key}"
    geo_response = requests.get(geocoding_url)
    if geo_response.status_code == 200:
        geo_data = geo_response.json()
        if geo_data['features']:
            coordinates = geo_data['features'][0]['geometry']['coordinates']
            longitude, latitude = coordinates[0], coordinates[1]
        else:
            return [], "Location not found."
    else:
        return [], f"Failed to fetch location data. Error code: {geo_response.status_code}"

    endpoint_url = "https://api.geoapify.com/v2/places"
    radius = 1000
    filters = f"circle:{longitude},{latitude},{radius}"
    bias = f"proximity:{longitude},{latitude}"
    categories = "catering.restaurant"

    params = {
        "filter": filters,
        "limit": 20,
        "bias": bias,
        "categories": categories,
        "apiKey": geoapify_key
    }

    response = requests.get(endpoint_url, params=params)
    if response.status_code == 200:
        results = response.json().get('features', [])
        if results:
            restaurants = [(result['properties'].get('name', 'Unnamed'),
                            result['geometry']['coordinates'][1],
                            result['geometry']['coordinates'][0]) for result in results]
            return restaurants, None
        return [], "No restaurants found."
    return [], f"Failed to fetch restaurant data. Error code: {response.status_code}"

# Function to get hotel data from an API
def get_hotels(location):
    geocoding_url = f"https://api.geoapify.com/v1/geocode/search?text={location}&apiKey={geoapify_key}"
    geo_response = requests.get(geocoding_url)
    if geo_response.status_code == 200:
        geo_data = geo_response.json()
        if geo_data['features']:
            coordinates = geo_data['features'][0]['geometry']['coordinates']
            longitude, latitude = coordinates[0], coordinates[1]
        else:
            return [], "Location not found."
    else:
        return [], f"Failed to fetch location data. Error code: {geo_response.status_code}"

    endpoint_url = "https://api.geoapify.com/v2/places"
    radius = 1000
    filters = f"circle:{longitude},{latitude},{radius}"
    bias = f"proximity:{longitude},{latitude}"
    categories = "accommodation.hotel"

    params = {
        "filter": filters,
        "limit": 20,
        "bias": bias,
        "categories": categories,
        "apiKey": geoapify_key
    }

    response = requests.get(endpoint_url, params=params)
    if response.status_code == 200:
        results = response.json().get('features', [])
        if results:
            hotels = [(result['properties'].get('name', 'Unnamed'),
                       result['geometry']['coordinates'][1],
                       result['geometry']['coordinates'][0]) for result in results]
            return hotels, None
        return [], "No hotels found."
    return [], f"Failed to fetch hotel data. Error code: {response.status_code}"

# Function to create a map
def create_map(location):
    # Geoapify Geocoding to get coordinates
    geocoding_url = f"https://api.geoapify.com/v1/geocode/search?text={location}&apiKey={geoapify_key}"
    geo_response = requests.get(geocoding_url)
    if geo_response.status_code == 200:
        geo_data = geo_response.json()
        if geo_data['features']:
            coordinates = geo_data['features'][0]['geometry']['coordinates']
            longitude, latitude = coordinates[0], coordinates[1]
        else:
            return "Location not found."
    else:
        return f"Failed to fetch location data. Error code: {geo_response.status_code}"

    # Fetch custom map style from Geoapify
    style_url = "https://maps.geoapify.com/v1/styles/osm-carto/style.json?apiKey=" + geoapify_key
    style_response = requests.get(style_url)
    if style_response.status_code == 200:
        style_json = style_response.json()
    else:
        return f"Failed to fetch map style. Error code: {style_response.status_code}"

    # Create the map with the custom style
    map_ = folium.Map(location=[latitude, longitude], zoom_start=14, tiles=None)
    
    # Add custom tiles layer using Geoapify style URL
    folium.TileLayer(
        tiles=f"https://{style_json['tiles'][0]}",
        attr="Map data © OpenStreetMap contributors",
        name="Custom Style",
        overlay=False,
        control=True
    ).add_to(map_)

    # Add markers for hotels
    hotels, hotel_error = get_hotels(location)
    if hotel_error is None:
        for name, lat, lon in hotels:
            folium.Marker(
                location=[lat, lon],
                popup=name,
                icon=folium.Icon(icon='bed', color='blue')
            ).add_to(map_)

    # Add markers for restaurants
    restaurants, restaurant_error = get_restaurants(location)
    if restaurant_error is None:
        for name, lat, lon in restaurants:
            folium.Marker(
                location=[lat, lon],
                popup=name,
                icon=folium.Icon(icon='cutlery', color='red')
            ).add_to(map_)

    # Convert the map to HTML string
    map_html = map_._to_html()
    return map_html

# Updated function to generate responses
def generate_response(intent, entities, user_response=None, follow_up_stage=0):
    location = entities.get('GPE', 'unknown location')
    if follow_up_stage == 1:
        if user_response.lower() == "yes":
            weather_data = get_weather(location)
            if weather_data:
                description = weather_data['weather'][0]['description']
                temp_fahrenheit = weather_data['main']['temp']
                weather_response = f"The temperature in {location} is currently {temp_fahrenheit}°F with {description}."
                return f"{weather_response} Would you like to see hotels in {location}? (yes/no)", 2, create_map(location)
            else:
                return "Sorry, I couldn't retrieve the weather information right now.", 0, None
        return f"Would you like to see hotels in {location}? (yes/no)", 2, None

    elif follow_up_stage == 2:
        if user_response.lower() == "yes":
            hotels_data, hotel_error = get_hotels(location)
            if hotel_error is None:
                hotel_list = ", ".join([name for name, _, _ in hotels_data])
                return f"Here are some hotels in {location}: {hotel_list}. Would you like to see restaurants in {location}? (yes/no)", 3, create_map(location)
            return hotel_error, 0, None
        return "Okay, let me know if there's anything else you'd like to know!", 0, None

    elif follow_up_stage == 3:
        if user_response.lower() == "yes":
            restaurants_data, restaurant_error = get_restaurants(location)
            if restaurant_error is None:
                restaurant_list = ", ".join([name for name, _, _ in restaurants_data])
                return f"Here are some restaurants in {location}: {restaurant_list}.", 0, create_map(location)
            return restaurant_error, 0, None
        return "Okay, let me know if there's anything else you'd like to know!", 0, None

    if intent == "city_query":
        if location:
            return f"Do you want to know the weather for {location}? (yes/no)", 1, create_map(location)
        return "I didn't catch the location. Could you please specify?", 0, None

    elif intent == "weather_query":
        weather_data = get_weather(location)
        if weather_data:
            description = weather_data['weather'][0]['description']
            temp_fahrenheit = weather_data['main']['temp']
            return f"The temperature in {location} is currently {temp_fahrenheit}°F with {description}.", 0, create_map(location)
        return "Sorry, I couldn't retrieve the weather information right now.", 0, create_map(location)

    elif intent == "restaurant_query":
        restaurants_data, restaurant_error = get_restaurants(location)
        if restaurant_error is None:
            restaurant_list = ", ".join([name for name, _, _ in restaurants_data])
            return f"Here are some restaurants in {location}: {restaurant_list}.", 0, create_map(location)
        return restaurant_error, 0, None

    elif intent == "hotel_query":
        hotels_data, hotel_error = get_hotels(location)
        if hotel_error is None:
            hotel_list = ", ".join([name for name, _, _ in hotels_data])
            return f"Here are some hotels in {location}: {hotel_list}.", 0, create_map(location)
        return hotel_error, 0, None

    return "I'm not sure how to help with that.", 0, None

def chatbot_response(user_input, follow_up_stage=0):
    intent = predict_intent(user_input)
    entities = extract_entities(user_input)
    response, next_stage, map_html = generate_response(intent, entities, user_input, follow_up_stage)
    return response, next_stage, map_html

def gradio_interface(destination, weather_input, restaurant_input, hotel_input, follow_up_stage=0):
    weather_response, next_stage_weather = "", 0
    restaurant_response, next_stage_restaurant = "", 0
    hotel_response, next_stage_hotel = "", 0
    map_html = ""

    if destination.strip():
        intent = predict_intent(destination)
        entities = extract_entities(destination)
        destination_response, next_stage, map_html = generate_response(intent, entities, destination, follow_up_stage)

    if weather_input.strip():
        weather_response, next_stage_weather, map_html = chatbot_response(weather_input, follow_up_stage)

    if restaurant_input.strip():
        restaurant_response, next_stage_restaurant, map_html = chatbot_response(restaurant_input, follow_up_stage)

    if hotel_input.strip():
        hotel_response, next_stage_hotel, map_html = chatbot_response(hotel_input, follow_up_stage)

    return weather_response, restaurant_response, hotel_response, map_html

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="Enter your destination"),
        gr.Textbox(label="Ask about your destination's weather"),
        gr.Textbox(label="Ask about restaurants near your destination"),
        gr.Textbox(label="Ask about hotels near your destination")
    ],
    outputs=[
        gr.Textbox(label="Destination's Weather"),
        gr.Textbox(label="Destination's Restaurants"),
        gr.Textbox(label="Destination's Hotels"),
        gr.HTML(label="Destination Map")
    ],
    live=True,
    theme="dark",
    title="Voyage Vibes"
)

iface.launch()
