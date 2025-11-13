# pip install streamlit requests openai anthropic google-generativeai
import os
import re
import json
import math
import time
import typing as t
from datetime import datetime, timedelta, date

import requests
import streamlit as st

############################
# Helper: Secrets & Config #
############################

def _get_secret(key: str, default: str | None = None) -> str | None:
    # Prefer st.secrets, then environment
    try:
        return st.secrets.get(key) or os.environ.get(key) or default
    except Exception:
        return os.environ.get(key) or default

OWM_KEY = _get_secret("api_key") or _get_secret("OPENWEATHERMAP_API_KEY")
OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")
ANTHROPIC_API_KEY = _get_secret("ANTHROPIC_API_KEY")
GEMINI_API_KEY = _get_secret("GEMINI_API_KEY")

SUPPORTED_LLM_PROVIDERS = {
    "OpenAI": bool(OPENAI_API_KEY),
    "Claude": bool(ANTHROPIC_API_KEY),
    "Gemini": bool(GEMINI_API_KEY),
}

#########################
# LLM Client Abstraction #
#########################

class LLMClient:
    def __init__(self, provider: str, model: str | None = None, temperature: float = 0.3):
        self.provider = provider
        self.temperature = temperature
        self.model = model or (
            "gpt-4o-mini" if provider == "OpenAI" else
            "claude-3-5-sonnet-20241022" if provider == "Claude" else
            "gemini-1.5-pro" if provider == "Gemini" else None
        )
        # Lazy imports to keep startup fast
        if provider == "OpenAI":
            try:
                from openai import OpenAI  # type: ignore
            except Exception as e:
                raise RuntimeError("openai package missing. Run: pip install openai") from e
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        elif provider == "Claude":
            try:
                import anthropic  # type: ignore
            except Exception as e:
                raise RuntimeError("anthropic package missing. Run: pip install anthropic") from e
            self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        elif provider == "Gemini":
            try:
                import google.generativeai as genai  # type: ignore
            except Exception as e:
                raise RuntimeError("google-generativeai package missing. Run: pip install google-generativeai") from e
            genai.configure(api_key=GEMINI_API_KEY)
            self.client = genai
        else:
            raise ValueError("Unsupported provider")

    def call_json(self, system_prompt: str, user_prompt: str, schema_hint: dict | None = None) -> dict:
        """Call the selected LLM and return parsed JSON (robustly)."""
        text = self._call_text(system_prompt, user_prompt)
        return parse_json_safely(text, schema_hint=schema_hint)

    def _call_text(self, system_prompt: str, user_prompt: str) -> str:
        if self.provider == "OpenAI":
            # Newer OpenAI SDK
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content or "{}"
        elif self.provider == "Claude":
            import anthropic  # type: ignore
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            # Claude returns a list of content blocks
            parts = []
            for blk in resp.content:
                if getattr(blk, "type", "") == "text":
                    parts.append(blk.text)
            return "\n".join(parts) or "{}"
        elif self.provider == "Gemini":
            model = self.client.GenerativeModel(self.model)
            resp = model.generate_content([
                {"role": "user", "parts": [system_prompt + "\n\n" + user_prompt]}
            ], generation_config={"temperature": self.temperature})
            return resp.text or "{}"
        else:
            return "{}"


def parse_json_safely(text: str, schema_hint: dict | None = None) -> dict:
    """Extract JSON from raw LLM text reliably. Accepts fenced blocks or loose objects."""
    if not text:
        return schema_hint or {}
    # Try to find fenced code blocks first
    fence = re.search(r"```(?:json)?\n(.*?)```", text, flags=re.S)
    if fence:
        snippet = fence.group(1)
        try:
            return json.loads(snippet)
        except Exception:
            pass
    # Fallback: find first JSON object
    brace = re.search(r"\{[\s\S]*\}$", text.strip())
    if brace:
        try:
            return json.loads(brace.group(0))
        except Exception:
            pass
    # Last resort: return hint or empty
    return schema_hint or {}

###############################
# OpenWeatherMap API Utilities #
###############################

OWM_GEOCODE = "https://api.openweathermap.org/geo/1.0/direct"
OWM_FORECAST = "https://api.openweathermap.org/data/2.5/forecast"  # 5-day / 3-hour

@st.cache_data(show_spinner=False, ttl=3600)
def geocode_city(q: str, limit: int = 1) -> dict | None:
    if not OWM_KEY:
        return None
    try:
        r = requests.get(OWM_GEOCODE, params={"q": q, "limit": limit, "appid": OWM_KEY}, timeout=20)
        r.raise_for_status()
        data = r.json()
        if data:
            item = data[0]
            return {
                "name": item.get("name"),
                "lat": item.get("lat"),
                "lon": item.get("lon"),
                "country": item.get("country"),
                "state": item.get("state"),
            }
        return None
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=900)
def fetch_forecast(lat: float, lon: float, units: str = "metric") -> dict | None:
    if not OWM_KEY:
        return None
    try:
        r = requests.get(OWM_FORECAST, params={"lat": lat, "lon": lon, "appid": OWM_KEY, "units": units}, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

###########################
# Geography & Time Helpers #
###########################

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

########################
# Weather Agent & Wrap #
########################

def summarize_forecast_for_date(forecast_json: dict, target: datetime) -> dict:
    """Aggregate 3-hour forecasts around target date into a simple summary."""
    if not forecast_json or not forecast_json.get("list"):
        return {}
    items = forecast_json["list"]
    # Choose entries within +/- 12 hours of local noon of target day (approx)
    t0 = datetime(target.year, target.month, target.day, 12, 0)
    lo, hi = t0 - timedelta(hours=12), t0 + timedelta(hours=12)
    samples = []
    for it in items:
        ts = datetime.fromtimestamp(it.get("dt", 0))
        if lo <= ts <= hi:
            samples.append(it)
    if not samples:
        # Fallback: take the closest
        closest = min(items, key=lambda it: abs(datetime.fromtimestamp(it.get("dt", 0)) - t0))
        samples = [closest]
    temps = [s.get("main", {}).get("temp") for s in samples if s.get("main")]
    conds = [s.get("weather", [{}])[0].get("main") for s in samples if s.get("weather")]
    wind = [s.get("wind", {}).get("speed") for s in samples if s.get("wind")]
    return {
        "avg_temp": round(sum(temps) / len(temps), 1) if temps else None,
        "conditions": max(set(conds), key=conds.count) if conds else None,
        "wind_kps": round(sum(wind) / len(wind), 1) if wind else None,
    }


def weather_agent(origin_geo: dict, dest_geo: dict, depart_dt: datetime, duration_days: int, llm: LLMClient | None) -> dict:
    """Compare weather at origin vs destination for the departure day and overall guidance."""
    units = "metric"
    origin_fc = fetch_forecast(origin_geo["lat"], origin_geo["lon"], units)
    dest_fc = fetch_forecast(dest_geo["lat"], dest_geo["lon"], units)

    trip_dates = [depart_dt + timedelta(days=i) for i in range(duration_days)]

    o_summaries, d_summaries = [], []
    for dt_ in trip_dates:
        # OWM gives 5-day horizon. If target beyond ~4 days, ask LLM to extrapolate.
        too_far = (dt_ - datetime.now()).days > 4
        if not too_far:
            o_summaries.append(summarize_forecast_for_date(origin_fc, dt_))
            d_summaries.append(summarize_forecast_for_date(dest_fc, dt_))
        else:
            # LLM extrapolation fallback
            if llm:
                schema = {"avg_temp": None, "conditions": None, "wind_kps": None}
                prompt = f"""
                Extrapolate a plausible weather summary for {dest_geo['name']}, {dest_geo.get('country','')} on {dt_.date()}.
                If you lack data, infer typical seasonal conditions. Output JSON with keys: avg_temp (°C), conditions, wind_kps.
                Keep it conservative and realistic for travelers.
                """
                d_summaries.append(llm.call_json("You are a careful travel meteorology assistant.", prompt, schema))
                prompt2 = f"""
                Extrapolate a plausible weather summary for {origin_geo['name']}, {origin_geo.get('country','')} on {dt_.date()}.
                Output JSON with keys: avg_temp (°C), conditions, wind_kps.
                """
                o_summaries.append(llm.call_json("You are a careful travel meteorology assistant.", prompt2, schema))
            else:
                o_summaries.append({})
                d_summaries.append({})

    # Departure day comparison (index 0)
    dep_o = o_summaries[0] if o_summaries else {}
    dep_d = d_summaries[0] if d_summaries else {}

    return {
        "trip_dates": [d.strftime("%Y-%m-%d") for d in trip_dates],
        "origin_daily": o_summaries,
        "dest_daily": d_summaries,
        "departure_comparison": {
            "origin": dep_o,
            "destination": dep_d,
        },
    }

#######################
# Logistics & Packing #
#######################

def logistics_agent(origin: str, destination: str, origin_geo: dict, dest_geo: dict, depart_dt: datetime, duration_days: int, llm: LLMClient | None) -> dict:
    distance_km = None
    if origin_geo and dest_geo:
        distance_km = round(haversine_km(origin_geo["lat"], origin_geo["lon"], dest_geo["lat"], dest_geo["lon"]), 1)
    cross_border = (origin_geo.get("country") != dest_geo.get("country")) if origin_geo and dest_geo else False

    base = {
        "distance_km": distance_km,
        "cross_border": cross_border,
        "precheck": "For international routes, avoid suggesting driving across oceans; prefer flights.",
    }

    if not llm:
        # Heuristic fallback
        mode = "flight" if cross_border or (distance_km and distance_km > 900) else ("train" if distance_km and distance_km < 900 and distance_km > 200 else "drive")
        est_time_h = None
        if distance_km:
            if mode == "drive":
                est_time_h = round(distance_km / 80, 1)
            elif mode == "train":
                est_time_h = round(distance_km / 120, 1)
            else:
                est_time_h = round(2.5 + distance_km / 750, 1)  # block time
        base.update({
            "recommended_mode": mode,
            "estimated_time_h": est_time_h,
            "tips": [
                "Check visas/passport validity if cross-border.",
                "Arrive 2 hours early for domestic flights; 3 hours for international.",
                "Consider off-peak departures to reduce delays.",
            ],
        })
        return base

    schema = {
        "recommended_mode": "",
        "estimated_time_h": 0.0,
        "rationale": "",
        "tips": [],
    }
    sys = """
    You are a logistics planner. Choose sensible transport modes.
    Never suggest driving across oceans or between non-contiguous countries.
    Prefer flights for intercontinental trips; consider trains for 200–900km inside Europe/Asia.
    Output valid JSON with: recommended_mode (one of: drive/train/flight/bus/mix), estimated_time_h (float), rationale (short), tips (list).
    """
    user = f"""
    ORIGIN: {origin} ({origin_geo.get('country') if origin_geo else 'N/A'})
    DESTINATION: {destination} ({dest_geo.get('country') if dest_geo else 'N/A'})
    DEPARTURE_DATE: {depart_dt.date()}
    DURATION_DAYS: {duration_days}
    STRAIGHT_LINE_DISTANCE_KM: {distance_km}
    CROSS_BORDER: {cross_border}

    Return JSON only.
    """
    out = llm.call_json(sys, user, schema_hint=schema)
    out.update(base)
    return out


def packing_agent(duration_days: int, dest_daily: list[dict], activities_hint: str, llm: LLMClient | None) -> dict:
    if not llm:
        # Simple heuristic packing list
        avg_temp = None
        temps = [d.get("avg_temp") for d in dest_daily if d.get("avg_temp") is not None]
        if temps:
            avg_temp = sum(temps) / len(temps)
        base = ["Passport/ID", "Phone + charger", "Medications", "Water bottle", "Travel insurance"]
        if avg_temp is not None and avg_temp < 10:
            base += ["Thermal layer", "Warm jacket", "Beanie", "Gloves"]
        elif avg_temp is not None and avg_temp > 24:
            base += ["T-shirts", "Shorts", "Sunscreen", "Hat"]
        else:
            base += ["Light jacket", "Jeans", "Umbrella"]
        base += [f"{max(3, duration_days//2)} pairs socks", f"{max(3, duration_days//2)} pairs underwear"]
        return {"items": base, "notes": "Heuristic list based on average temperature and trip length."}

    schema = {"items": [], "notes": ""}
    sys = """
    You create concise packing lists tailored to temperature, precipitation and trip length.
    Balance minimalism with practicality. Pack for laundry every ~4 days. Output JSON with keys: items (list), notes (string).
    """
    climate_hint = {
        "temps": [d.get("avg_temp") for d in dest_daily],
        "conditions": [d.get("conditions") for d in dest_daily],
    }
    user = f"""
    TRIP_LENGTH_DAYS: {duration_days}
    CLIMATE_HINT: {json.dumps(climate_hint)}
    ACTIVITIES_HINT: {activities_hint}
    Return JSON only.
    """
    return llm.call_json(sys, user, schema_hint=schema)


def activity_agent(destination: str, dest_geo: dict, duration_days: int, interests: list[str], llm: LLMClient | None) -> dict:
    schema = {"overview": "", "days": []}
    if not llm:
        # Basic stub itinerary
        days = []
        for i in range(duration_days):
            days.append({"day": i+1, "title": f"Explore {destination}", "items": ["Morning: city center", "Afternoon: museum/park", "Evening: local cuisine"]})
        return {"overview": f"Self-guided highlights of {destination}", "days": days}

    sys = """
    You are a local trip designer. Craft a realistic day-wise itinerary with walkable clusters and time-aware sequencing.
    Prefer neighborhoods and signature experiences over generic lists. Output JSON with: overview (string) and days (list of {day, title, items}).
    Keep each day to 3–5 items, each item a short action with a neighborhood.
    """
    user = f"""
    DESTINATION: {destination} ({dest_geo.get('country') if dest_geo else 'N/A'})
    TRIP_DAYS: {duration_days}
    INTERESTS: {", ".join(interests) if interests else "general sights"}
    Return JSON only.
    """
    return llm.call_json(sys, user, schema_hint=schema)

#########################
# UI & App Orchestration #
#########################

def run_agents(origin: str, destination: str, depart_dt: datetime, duration_days: int, interests: list[str], activities_hint: str, llm: LLMClient | None):
    # Geocode first and validate
    origin_geo = geocode_city(origin) if origin else None
    dest_geo = geocode_city(destination) if destination else None

    if not origin_geo or not dest_geo:
        st.error("Invalid city names or geocoding failed. Please try 'City, Country' format (e.g., 'Paris, FR').")
        return

    # Weather compare
    with st.status("Fetching weather & comparing origin vs destination…", expanded=False) as s:
        wx = weather_agent(origin_geo, dest_geo, depart_dt, duration_days, llm)
        s.update(label="Weather ready", state="complete")

    # Logistics
    with st.status("Planning logistics & timing…", expanded=False) as s:
        lg = logistics_agent(origin, destination, origin_geo, dest_geo, depart_dt, duration_days, llm)
        s.update(label="Logistics ready", state="complete")

    # Packing
    with st.status("Assembling packing list…", expanded=False) as s:
        pk = packing_agent(duration_days, wx["dest_daily"], activities_hint, llm)
        s.update(label="Packing list ready", state="complete")

    # Activities
    with st.status("Designing day‑wise itinerary…", expanded=False) as s:
        it = activity_agent(destination, dest_geo, duration_days, interests, llm)
        s.update(label="Itinerary ready", state="complete")

    # Display
    st.subheader("Overview 🧭")
    st.write(
        {
            "origin": origin_geo,
            "destination": dest_geo,
            "depart_date": depart_dt.strftime("%Y-%m-%d"),
            "duration_days": duration_days,
            "llm_provider": llm.provider if llm else "Heuristic fallback",
        }
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Weather ☁️")
        st.write({
            "dates": wx["trip_dates"],
            "origin_daily": wx["origin_daily"],
            "destination_daily": wx["dest_daily"],
            "departure_comparison": wx["departure_comparison"],
        })
    with col2:
        st.markdown("### Logistics 🚆✈️🚗")
        st.write(lg)

    st.markdown("### Packing 🎒")
    st.write(pk)

    st.markdown("### Itinerary 📅")
    if it.get("days"):
        for d in it["days"]:
            with st.expander(f"Day {d.get('day')}: {d.get('title')}"):
                for item in d.get("items", []):
                    st.write("• ", item)
    else:
        st.write(it)

    # Basic validation messages per rubric
    st.markdown("---")
    st.markdown("#### Validation Checks")
    checks = []
    if not OWM_KEY:
        checks.append("❌ Missing OpenWeatherMap API key")
    else:
        checks.append("✅ OpenWeatherMap API key detected")
    if llm:
        checks.append(f"✅ LLM provider ready: {llm.provider} ({llm.model})")
    else:
        checks.append("⚠️ No LLM API key found — using heuristics (OK for development, not for final submission)")

    # Cross-border driving sanity check
    if lg.get("cross_border") and lg.get("recommended_mode") == "drive":
        checks.append("❌ Driving suggested for cross‑border trip — please re‑generate or adjust inputs")
    else:
        checks.append("✅ Transport mode passes sanity check")

    st.write("\n".join(checks))


def main():
    st.set_page_config(
        page_title="Multi‑Agent Travel Planner",
        page_icon="🗺️",
        layout="wide",
    )
    st.title("Lab 9 — Multi‑Agent Travel Planning System 🧠🗺️")
    st.caption("Weather Agent · Logistics Agent · Packing Agent · Activity Agent")

    with st.sidebar:
        st.header("Configuration")
        provider = st.selectbox("LLM Provider", options=["OpenAI", "Claude", "Gemini"], index=0)
        model_override = st.text_input("Model (optional)", value="")
        temp = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1, help="Use 0.3 for consistent JSON output, per rubric.")
        st.divider()
        st.markdown("**API Keys** (loaded from `.streamlit/secrets.toml` or environment)")
        st.write({
            "OpenWeatherMap": bool(OWM_KEY),
            "OpenAI": bool(OPENAI_API_KEY),
            "Claude": bool(ANTHROPIC_API_KEY),
            "Gemini": bool(GEMINI_API_KEY),
        })

    st.markdown("### Trip Inputs")
    c1, c2, c3 = st.columns([2,2,1])
    with c1:
        origin = st.text_input("Origin", placeholder="e.g., Syracuse, US")
    with c2:
        destination = st.text_input("Destination", placeholder="e.g., Boston, US")
    with c3:
        duration_days = st.number_input("Duration (days)", min_value=1, max_value=30, value=3)

    c4, c5 = st.columns([2,1])
    with c4:
        depart_date = st.date_input("Departure date", value=date.today() + timedelta(days=1))
    with c5:
        interests = st.multiselect("Interests (optional)", ["food", "museums", "outdoors", "architecture", "nightlife", "family"])

    activities_hint = st.text_input("Activities hint (optional)", placeholder="e.g., 1 business day + 2 leisure days")

    if st.button("Plan My Trip", type="primary"):
        llm: LLMClient | None = None
        has_key = (provider == "OpenAI" and OPENAI_API_KEY) or (provider == "Claude" and ANTHROPIC_API_KEY) or (provider == "Gemini" and GEMINI_API_KEY)
        if has_key:
            try:
                llm = LLMClient(provider, model_override or None, temperature=temp)
            except Exception as e:
                st.warning(f"Failed to initialize LLM ({provider}): {e}\nFalling back to heuristics.")
                llm = None
        else:
            st.info("No API key detected for selected provider — running heuristic fallback.")
        run_agents(origin.strip(), destination.strip(), datetime.combine(depart_date, datetime.min.time()), int(duration_days), interests, activities_hint, llm)

    st.markdown("---")
    st.markdown("""
    **Setup Notes**  
    1) Create `.streamlit/secrets.toml` with:
    
    ```toml
    api_key = "OpenWeatherMap_KEY"
    OPENAI_API_KEY = "..."
    ANTHROPIC_API_KEY = "..."
    GEMINI_API_KEY = "..."
    ```
    2) Install: `pip install streamlit requests openai anthropic google-generativeai`  
    3) Run: `streamlit run streamlit_app.py`
    """)


if __name__ == "__main__":
    main()

