import streamlit as st
import requests
import openai
import json
from typing import Dict, List

st.set_page_config(page_title="Lab 5 - What to Wear Bot", layout="centered")

OPENWEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")

# --- Get city suggestions via OpenWeather Geocoding API ---
def get_city_suggestions(query: str) -> List[Dict]:
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={requests.utils.requote_uri(query)}&limit=10&appid={OPENWEATHER_API_KEY}"
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        return []
    return resp.json()

# --- Weather retrieval by coordinates ---
def get_current_weather_by_coords(lat: float, lon: float, API_key: str) -> Dict:
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_key}&units=metric"
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        raise ValueError(f"Error fetching weather: {resp.text}")
    data = resp.json()
    return {
        "location": f"{data.get('name')}, {data.get('sys', {}).get('country')}",
        "temperature_c": round(data['main']['temp'], 2),
        "feels_like_c": round(data['main']['feels_like'], 2),
        "temp_min_c": round(data['main']['temp_min'], 2),
        "temp_max_c": round(data['main']['temp_max'], 2),
        "humidity": data['main']['humidity'],
        "weather_description": data['weather'][0]['description'] if data.get('weather') else '',
        "wind_speed_m_s": data.get('wind', {}).get('speed'),
        "_raw": data
    }

# --- Clothing suggestion via OpenAI ---
def get_clothing_suggestion_with_openai(weather: Dict) -> Dict:
    openai.api_key = OPENAI_API_KEY
    prompt = (
        f"The weather in {weather['location']} is {weather['weather_description']} with "
        f"temperature {weather['temperature_c']}°C. Suggest clothing and picnic advice."
    )
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=400
    )
    return {"success": True, "output": response['choices'][0]['message']['content'], "weather": weather}

# --- Streamlit UI ---
st.title("Lab 5 — What to Wear Bot 🌤️👕")
st.markdown("Enter a city to get clothing suggestions.")

query = st.text_input("Enter city (e.g. Syracuse):")
suggestions = get_city_suggestions(query) if query else []

selected_country = selected_state = selected_city = None
selected_lat = selected_lon = None

if suggestions:
    countries = sorted(set(item['country'] for item in suggestions))
    selected_country = st.selectbox("Select Country", countries)

    states = sorted(set(item.get('state', '') for item in suggestions if item['country'] == selected_country))
    if states:
        selected_state = st.selectbox("Select State", states)

    cities = [
        item for item in suggestions
        if item['country'] == selected_country and (item.get('state', '') == selected_state if selected_state else True)
    ]
    selected_city_name = st.selectbox("Select City", [item['name'] for item in cities])

    # Store lat/lon
    for item in cities:
        if item['name'] == selected_city_name:
            selected_lat = item['lat']
            selected_lon = item['lon']
            break

if st.button("Get suggestion"):
    if not OPENWEATHER_API_KEY or not OPENAI_API_KEY:
        st.error("Missing API keys in Streamlit secrets.")
    elif not selected_lat or not selected_lon:
        st.error("Please select a city.")
    else:
        with st.spinner("Fetching weather and suggestion..."):
            try:
                weather = get_current_weather_by_coords(selected_lat, selected_lon, OPENWEATHER_API_KEY)
                result = get_clothing_suggestion_with_openai(weather)
            except Exception as e:
                st.error(str(e))
                st.stop()

        if not result["success"]:
            st.error(result.get("error"))
        else:
            st.subheader("Recommendation")
            st.markdown(result["output"])
            if result.get("weather"):
                w = result["weather"]
                st.subheader("Weather Data")
                st.write(f"**Location:** {w.get('location')}")
                st.write(f"**Temperature:** {w.get('temperature_c')} °C (feels like {w.get('feels_like_c')} °C)")
                st.write(f"**Weather:** {w.get('weather_description')} | **Humidity:** {w.get('humidity')}%")
                if w.get('wind_speed_m_s') is not None:
                    st.write(f"**Wind speed:** {w.get('wind_speed_m_s')} m/s")
                with st.expander("Raw weather JSON"):
                    st.json(w.get('_raw'))
