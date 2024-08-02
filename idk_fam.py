from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import requests
import spacy

from config import geoapify_key, open_weather_key

data = [
    ("I want to travel to Houston", "city_query"),
    ("I wanna go to Salt Lake City", "city_query"),
    ("I want to visit Chicago", "city_query"),
    ("Oregon", "city_query"),
    ("San Diego", "city_query"),
    ("Boston", "city_query"),
    ("Las Vegas", "city_query"),
    ("Tucson", "city_query"),
    ("El Paso", "city_query"),
    ("Boise", "city_query"),
    ("What's the weather like?", "weather_query"),
    ("Can you tell me the weather in Chicago?", "weather_query"),
    ("What's the temperature in Seattle?", "weather_query"),
    ("Help me with the weather forecast", "weather_query"),
    ("How hot is it in San Francisco?", "weather_query"),
    ("Food in San Jose", "restaurant_query"),
    ("Restaurants in Pheonix", "restaurant_query"),
    ("Are there any restaurants in Baltimore?", "restaurant_query"),
    ("I wanna grab food in Detroit", "restaurant_query"),
    ("Places to eat in Austin", "restaurant_query"),
    # Add more examples as needed
]

X, y = zip(*data)

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
print(f"Model Accuracy: {accuracy * 100:.2f}%")

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

# Function to generate a response based on intent and entities
import requests

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
    radius = 1000  # Reduced radius for testing
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

def generate_response(intent, entities, user_response=None, follow_up=False):
    if follow_up:
        if intent == "city_query":
            if user_response.lower() == "yes":
                location = entities.get('GPE', 'unknown location')
                weather_data = get_weather(location)
                if weather_data:
                    description = weather_data['weather'][0]['description']
                    temp_kelvin = weather_data['main']['temp']
                    temp_fahrenheit = round((temp_kelvin - 273.15) * 9/5 + 32)
                    weather_response = f"The temperature in {location} is currently {temp_fahrenheit}°F with {description}."
                    # Follow-up for restaurant query
                    return f"{weather_response} Would you like to see restaurants in {location}? (yes/no)"
                else:
                    return "Sorry, I couldn't retrieve the weather information right now."
            elif user_response.lower() == "no":
                location = entities.get('GPE', 'unknown location')
                return f"Okay, let me know if there's anything else you'd like to know about {location}!"
            else:
                return "I didn't understand that. Please respond with 'yes' or 'no'."
        
        elif intent == "restaurant_query":
            if user_response.lower() == "yes":
                location = entities.get('GPE', 'unknown location')
                restaurants_data = get_restaurants(location)
                if restaurants_data:
                    restaurant_list = ", ".join(restaurants_data)
                    return f"Here are some restaurants in {location}: {restaurant_list}."
                else:
                    return "Sorry, I couldn't find any restaurants at that location right now."
            else:
                return "Okay, let me know if there's anything else you'd like to know!"
    
    if intent == "city_query":
        location = entities.get('GPE')
        return f"Do you want to know the weather for {location}? (yes/no)"

    elif intent == "weather_query":
        location = entities.get('GPE')
        weather_data = get_weather(location)
        if weather_data:
            description = weather_data['weather'][0]['description']
            temp_kelvin = weather_data['main']['temp']
            temp_fahrenheit = round((temp_kelvin - 273.15) * 9/5 + 32)
            return f"The temperature in {location} is currently {temp_fahrenheit}°F with {description}."
        else:
            return "Sorry, I couldn't retrieve the weather information right now."

    elif intent == "restaurant_query":
        location = entities.get('GPE')
        restaurants_data = get_restaurants(location)
        if restaurants_data:
            restaurant_list = ", ".join(restaurants_data)
            return f"Here are some restaurants in {location}: {restaurant_list}."
        else:
            return "Sorry, I couldn't find any restaurants at that location right now."

    else:
        return "I'm not sure how to help with that."

def chatbot():
    print("Hello! Where would you like to travel?")
    follow_up_intent = None
    entities = {}

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        if follow_up_intent:
            intent = follow_up_intent
            response = generate_response(intent, entities, user_input, follow_up=True)
        else:
            intent = predict_intent(user_input)
            entities = extract_entities(user_input)
            response = generate_response(intent, entities)

        print(f"Chatbot: {response}")

        if "Do you want to know the weather" in response:
            follow_up_intent = "city_query"
        elif "Would you like to see restaurants" in response:
            follow_up_intent = "restaurant_query"
        else:
            follow_up_intent = None

# Run the chatbot
if __name__ == "__main__":
    chatbot()