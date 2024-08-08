from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import requests
import spacy
import pandas as pd

from config import geoapify_key, open_weather_key

csv_file = "Resources/queries.csv"
df = pd.read_csv(csv_file)

X = df['Query']
y = df['Category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a text classification pipeline
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

# Predict the test set results
y_pred = model.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
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
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={open_weather_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return None

# Function to get restaurant data from an API
def get_restaurants(location):
    # Geocoding API to get latitude and longitude from location name
    geocoding_url = f"https://api.geoapify.com/v1/geocode/search?text={location}&apiKey={geoapify_key}"
    
    geo_response = requests.get(geocoding_url)
    if geo_response.status_code == 200:
        geo_data = geo_response.json()
        if geo_data['features']:
            coordinates = geo_data['features'][0]['geometry']['coordinates']
            longitude, latitude = coordinates[0], coordinates[1]
        else:
            return ["Location not found."]
    else:
        return [f"Failed to fetch location data. Error code: {geo_response.status_code}"]

    # Now use the latitude and longitude for restaurant search
    endpoint_url = "https://api.geoapify.com/v2/places"
    radius = 1000 
    filters = f"circle:{longitude},{latitude},{radius}"
    bias = f"proximity:{longitude},{latitude}"
    categories = "catering.restaurant"  # Specify the category for restaurants

    params = {
        "filter": filters,
        "limit": 20,  # Limit the number of results
        "bias": bias,
        "categories": categories,
        "apiKey": geoapify_key  
    }
   
    response = requests.get(endpoint_url, params=params)
    if response.status_code == 200:
        results = response.json().get('features', [])
        if results:
            # Extract restaurant names from the results
            restaurant_names = [result['properties'].get('name', 'Unnamed') for result in results]
            return restaurant_names
        else:
            return ["No restaurants found."]
    else:
        # Debugging: Print full response for troubleshooting
        print(f"Response JSON: {response.json()}")
        return [f"Failed to fetch restaurant data. Error code: {response.status_code}"]

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
            return ["Location not found."]
    else:
        return [f"Failed to fetch location data. Error code: {geo_response.status_code}"]

    # Now use the latitude and longitude for hotel search
    endpoint_url = "https://api.geoapify.com/v2/places"
    radius = 1000 
    filters = f"circle:{longitude},{latitude},{radius}"
    bias = f"proximity:{longitude},{latitude}"
    categories = "accommodation.hotel"  # Specify the category for hotels

    params = {
        "filter": filters,
        "limit": 20,  # Limit the number of results
        "bias": bias,
        "categories": categories,
        "apiKey": geoapify_key  
    }
   
    response = requests.get(endpoint_url, params=params)
    if response.status_code == 200:
        results = response.json().get('features', [])
        if results:
            # Extract hotel names from the results
            hotel_names = [result['properties'].get('name', 'Unnamed') for result in results]
            return hotel_names
        else:
            return ["No hotels found."]
    else:
        # Debugging: Print full response for troubleshooting
        print(f"Response JSON: {response.json()}")
        return [f"Failed to fetch hotel data. Error code: {response.status_code}"]

# Updated function to generate responses
def generate_response(intent, entities, user_response=None, follow_up_stage=0):
    if follow_up_stage == 1: 
        location = entities.get('GPE', 'unknown location')
        if user_response.lower() == "yes":
            weather_data = get_weather(location)
            if weather_data:
                description = weather_data['weather'][0]['description']
                temp_kelvin = weather_data['main']['temp']
                temp_fahrenheit = round((temp_kelvin - 273.15) * 9/5 + 32)
                weather_response = f"The temperature in {location} is currently {temp_fahrenheit}°F with {description}."
                return f"{weather_response} Would you like to see hotels in {location}? (yes/no)", 2
            else:
                return "Sorry, I couldn't retrieve the weather information right now.", 0
        else:
            return f"Would you like to see hotels in {location}? (yes/no)", 2

    elif follow_up_stage == 2:  # After asking about hotels
        if user_response.lower() == "yes":
            location = entities.get('GPE', 'unknown location')
            hotels_data = get_hotels(location)
            if hotels_data:
                hotel_list = ", ".join(hotels_data)
                return f"Here are some hotels in {location}: {hotel_list}. Would you like to see restaurants in {location}? (yes/no)", 3
            else:
                return "Sorry, I couldn't find any hotels at that location right now.", 0
        else:
            return "Okay, let me know if there's anything else you'd like to know!", 0

    elif follow_up_stage == 3:  # After asking about restaurants
        if user_response.lower() == "yes":
            location = entities.get('GPE', 'unknown location')
            restaurants_data = get_restaurants(location)
            if restaurants_data:
                restaurant_list = ", ".join(restaurants_data)
                return f"Here are some restaurants in {location}: {restaurant_list}.", 0
            else:
                return "Sorry, I couldn't find any restaurants at that location right now.", 0
        else:
            return "Okay, let me know if there's anything else you'd like to know!", 0

    # Initial user input processing
    if intent == "city_query":
        location = entities.get('GPE')
        if location:
            return f"Do you want to know the weather for {location}? (yes/no)", 1
        else:
            return "I didn't catch the location. Could you please specify?", 0

    elif intent == "weather_query":
        location = entities.get('GPE')
        weather_data = get_weather(location)
        if weather_data:
            description = weather_data['weather'][0]['description']
            temp_kelvin = weather_data['main']['temp']
            temp_fahrenheit = round((temp_kelvin - 273.15) * 9/5 + 32)
            return f"The temperature in {location} is currently {temp_fahrenheit}°F with {description}.", 0
        else:
            return "Sorry, I couldn't retrieve the weather information right now.", 0

    elif intent == "restaurant_query":
        location = entities.get('GPE')
        restaurants_data = get_restaurants(location)
        if restaurants_data:
            restaurant_list = ", ".join(restaurants_data)
            return f"Here are some restaurants in {location}: {restaurant_list}.", 0
        else:
            return "Sorry, I couldn't find any restaurants at that location right now.", 0

    elif intent == "hotel_query":
        location = entities.get('GPE')
        hotels_data = get_hotels(location)
        if hotels_data:
            hotel_list = ", ".join(hotels_data)
            return f"Here are some hotels in {location}: {hotel_list}.", 0
        else:
            return "Sorry, I couldn't find any hotels at that location right now.", 0

    else:
        return "I'm not sure how to help with that.", 0

def chatbot():
    print("Hello! Where would you like to travel?")
    follow_up_stage = 0
    entities = {}

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        if follow_up_stage:
            response, follow_up_stage = generate_response(None, entities, user_input, follow_up_stage)
        else:
            intent = predict_intent(user_input)
            entities = extract_entities(user_input)
            response, follow_up_stage = generate_response(intent, entities)

        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot()