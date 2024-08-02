from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import requests
import spacy

data = [
    ("What's the weather like?", "weather_query"),
    ("Book a flight to New York", "flight_booking"),
    ("I need a flight to Los Angeles", "flight_booking"),
    ("Can you tell me the weather in Chicago?", "weather_query"),
    ("I want to fly to San Francisco", "flight_booking"),
    ("What's the temperature in Seattle?", "weather_query"),
    ("I would like to book a flight", "flight_booking"),
    ("Help me with the weather forecast", "weather_query"),
    ("I am looking for restaurants", "food_query")
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
    api_key = "cb84bcfbcbd5a92415a0c484b57886c8"  # Replace with your actual API key
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return None

# Function to generate a response based on intent and entities
def generate_response(intent, entities):
    if intent == "weather_query":
        location = entities.get('GPE', 'unknown location')  # Use 'GPE' or appropriate entity label
        weather_data = get_weather(location)
        if weather_data:
            # main = weather_data['weather'][0]['main']
            description = weather_data['weather'][0]['description']
            temp_kelvin = weather_data['main']['temp']
            temp_fahrenheit = round((temp_kelvin - 273.15) * 9/5 + 32)
            return f"The temperature in {location} is currently {temp_fahrenheit}°F with {description}."
        else:
            return "Sorry, I couldn't retrieve the weather information right now."
    else:
        return "I'm not sure how to help with that."

# Chatbot function
def chatbot():
    print("Hello! How can I assist you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break
        
        intent = predict_intent(user_input)
        entities = extract_entities(user_input)
        response = generate_response(intent, entities)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot()
