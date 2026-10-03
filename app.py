import os
import ast
import operator
import requests
import streamlit as st

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error(
        "Hugging Face API key not found. "
        "Please create a .env file and add HF_TOKEN=your_token_here"
    )
    st.stop()


# Current Hugging Face chat model.
# You can change this in .env with HF_CHAT_MODEL=...
CHAT_MODEL = os.getenv(
    "HF_CHAT_MODEL",
    "Qwen/Qwen3-4B-Instruct-2507"
)

# Image model.
IMAGE_MODEL = os.getenv(
    "HF_IMAGE_MODEL",
    "black-forest-labs/FLUX.1-schnell"
)

client = InferenceClient(
    provider="auto",
    api_key=HF_TOKEN
)


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="Genius AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(80, 30, 150, 0.18),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(0, 180, 255, 0.12),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #020204 0%,
                #050507 45%,
                #000000 100%
            );

        color: #ffffff;
    }

    .main .block-container {
        max-width: 1400px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #050509 0%,
                #090912 50%,
                #030305 100%
            );

        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* ---------- HEADER ---------- */

    .genius-header {
        text-align: center;
        padding: 20px 10px 15px 10px;
    }

    .genius-title {
        font-size: 58px;
        font-weight: 900;
        letter-spacing: -3px;
        margin-bottom: 0;

        background:
            linear-gradient(
                90deg,
                #7c3aed,
                #00e5ff,
                #ffffff,
                #7c3aed
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;

        text-shadow:
            0 0 35px rgba(124,58,237,0.25);
    }

    .genius-subtitle {
        color: #9292a5;
        font-size: 16px;
        margin-top: 5px;
        letter-spacing: 2px;
    }

    /* ---------- FEATURE CARDS ---------- */

    .feature-card {
        background:
            linear-gradient(
                145deg,
                rgba(255,255,255,0.055),
                rgba(255,255,255,0.015)
            );

        border: 1px solid rgba(255,255,255,0.08);

        border-radius: 18px;

        padding: 20px;

        margin-bottom: 15px;

        box-shadow:
            0 10px 40px rgba(0,0,0,0.25);

        transition: 0.25s ease;
    }

    .feature-card:hover {
        border-color: rgba(124,58,237,0.5);

        box-shadow:
            0 0 35px rgba(124,58,237,0.12);
    }

    .feature-title {
        font-size: 18px;
        font-weight: 800;
        color: #ffffff;
    }

    .feature-description {
        color: #8d8d9d;
        font-size: 13px;
    }

    /* ---------- BUTTONS ---------- */

    .stButton > button {
        width: 100%;

        border-radius: 12px;

        border: 1px solid rgba(255,255,255,0.09);

        background:
            linear-gradient(
                135deg,
                rgba(124,58,237,0.22),
                rgba(0,229,255,0.08)
            );

        color: white;

        font-weight: 700;

        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #7c3aed;

        box-shadow:
            0 0 25px rgba(124,58,237,0.25);

        transform: translateY(-1px);
    }

    /* ---------- INPUTS ---------- */

    .stTextInput input,
    .stTextArea textarea {
        background: #08080d !important;
        color: white !important;

        border-radius: 12px !important;

        border: 1px solid rgba(255,255,255,0.1) !important;
    }

    /* ---------- CHAT ---------- */

    [data-testid="stChatMessage"] {
        background:
            rgba(255,255,255,0.025);

        border:
            1px solid rgba(255,255,255,0.055);

        border-radius: 18px;

        margin-bottom: 10px;
    }

    /* ---------- METRIC ---------- */

    .metric-box {
        background:
            rgba(255,255,255,0.035);

        border:
            1px solid rgba(255,255,255,0.07);

        border-radius: 15px;

        padding: 15px;

        text-align: center;
    }

    /* ---------- STATUS ---------- */

    .status {
        display: inline-flex;

        align-items: center;

        gap: 7px;

        color: #9ca3af;

        font-size: 13px;
    }

    .status-dot {
        width: 8px;
        height: 8px;

        background: #22c55e;

        border-radius: 50%;

        box-shadow:
            0 0 12px #22c55e;
    }

    /* ---------- DIVIDER ---------- */

    hr {
        border-color:
            rgba(255,255,255,0.07);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "mode" not in st.session_state:
    st.session_state.mode = "Chat"


# ============================================================
# SAFE CALCULATOR
# ============================================================

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_calculate(expression):
    """
    Safely evaluate basic mathematical expressions.
    No eval() is used.
    """

    expression = expression.replace("^", "**")

    if len(expression) > 200:
        raise ValueError("Expression is too long.")

    def evaluate(node):

        if isinstance(node, ast.Constant):

            if isinstance(node.value, (int, float)):
                return node.value

            raise ValueError("Invalid number.")

        if isinstance(node, ast.BinOp):

            left = evaluate(node.left)
            right = evaluate(node.right)

            operation = OPERATORS.get(type(node.op))

            if operation is None:
                raise ValueError("Operator not allowed.")

            return operation(left, right)

        if isinstance(node, ast.UnaryOp):

            value = evaluate(node.operand)

            operation = OPERATORS.get(type(node.op))

            if operation is None:
                raise ValueError("Operator not allowed.")

            return operation(value)

        raise ValueError("Invalid mathematical expression.")

    tree = ast.parse(expression, mode="eval")

    return evaluate(tree.body)


# ============================================================
# WEB SEARCH
# ============================================================

def web_search(query, max_results=6):

    """
    Uses DuckDuckGo's HTML search endpoint.

    This does not require another API key.
    """

    url = "https://html.duckduckgo.com/html/"

    headers = {
        "User-Agent":
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/130 Safari/537.36"
    }

    try:

        response = requests.get(
            url,
            params={"q": query},
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        results = []

        for result in soup.select(
            ".result"
        )[:max_results]:

            title = result.select_one(
                ".result__title"
            )

            link = result.select_one(
                ".result__a"
            )

            snippet = result.select_one(
                ".result__snippet"
            )

            if not title or not link:
                continue

            results.append(
                {
                    "title": title.get_text(
                        " ",
                        strip=True
                    ),
                    "url": link.get("href", ""),
                    "snippet":
                        snippet.get_text(
                            " ",
                            strip=True
                        )
                        if snippet
                        else ""
                }
            )

        return results

    except Exception as e:

        return [
            {
                "title": "Search error",
                "url": "",
                "snippet": str(e)
            }
        ]


# ============================================================
# WEATHER
# ============================================================

def get_weather(city):

    """
    Open-Meteo weather service.
    No weather API key required.
    """

    try:

        geo_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": city,
                "count": 1,
                "language": "en",
                "format": "json"
            },
            timeout=10
        )

        geo_response.raise_for_status()

        geo_data = geo_response.json()

        if not geo_data.get("results"):
            return None

        location = geo_data["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current":
                    "temperature_2m,"
                    "relative_humidity_2m,"
                    "apparent_temperature,"
                    "precipitation,"
                    "weather_code,"
                    "wind_speed_10m"
            },
            timeout=10
        )

        weather_response.raise_for_status()

        weather = weather_response.json()

        current = weather["current"]

        return {
            "city":
                location.get("name", city),

            "country":
                location.get("country", ""),

            "temperature":
                current.get("temperature_2m"),

            "feels_like":
                current.get("apparent_temperature"),

            "humidity":
                current.get("relative_humidity_2m"),

            "precipitation":
                current.get("precipitation"),

            "wind":
                current.get("wind_speed_10m"),

            "weather_code":
                current.get("weather_code")
        }

    except Exception:
        return None


# ============================================================
# AI CHAT
# ============================================================

def ask_ai(user_message, mode="Chat"):

    system_prompt = """
You are Genius AI, a highly capable, friendly and accurate AI assistant.

Rules:
- Answer clearly and accurately.
- Do not invent facts.
- If information may be outdated, say so.
- Explain technical topics step-by-step when useful.
- For programming questions, provide working code.
- Use Markdown when it improves readability.
"""

    if mode == "Thinker":

        system_prompt += """
You are currently in Thinker mode.

Before answering, carefully analyze the problem internally.
Check assumptions, edge cases and contradictions.
Give the user a clear, well-structured final answer.
Do not expose private chain-of-thought or hidden reasoning.
Provide concise reasoning summaries, conclusions and useful steps.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    # Keep conversation context manageable.
    messages.extend(
        st.session_state.messages[-12:]
    )

    messages.append(
        {
            "role": "user",
            "content": user_message
        }
    )

    try:

        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            max_tokens=1200,
            temperature=0.7
        )

        return completion.choices[0].message.content

    except Exception as e:

        return (
            "I couldn't connect to the Hugging Face model.\n\n"
            f"**Error:** `{str(e)}`\n\n"
            "Check your HF token and model/provider availability."
        )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="genius-header">

        <div class="genius-title">
            GENIUS AI
        </div>

        <div class="genius-subtitle">
            THINK • CREATE • SEARCH • EXPLORE
        </div>

        <br>

        <div class="status">
            <span class="status-dot"></span>
            Genius AI online
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🧠 Genius AI"
    )

    st.caption(
        "Your intelligent multi-tool AI workspace."
    )

    st.divider()

    mode = st.radio(
        "AI Mode",
        [
            "Chat",
            "Thinker"
        ],
        index=0
    )

    st.session_state.mode = mode

    st.divider()

    st.markdown("### ✨ Features")

    st.markdown(
        """
        🤖 AI Chatbot  
        🧠 Thinker  
        🎨 Image Generator  
        🌐 Web Search  
        🎙️ Voice Recorder  
        🌤️ Weather  
        🧮 Calculator  
        🌍 3D Globe  
        """
    )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation"
    ):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.caption(
        "Powered by Hugging Face"
    )


# ============================================================
# FEATURE NAVIGATION
# ============================================================

tabs = st.tabs(
    [
        "💬 Chat",
        "🧠 Thinker",
        "🎨 Image",
        "🌐 Web Search",
        "🎙️ Voice",
        "🌤️ Weather",
        "🧮 Calculator",
        "🌍 3D Globe"
    ]
)


# ============================================================
# CHAT TAB
# ============================================================

with tabs[0]:

    st.markdown(
        "## 💬 Genius Chat"
    )

    st.caption(
        "Ask anything. Genius AI can help with coding, writing, "
        "learning, ideas and general questions."
    )

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    prompt = st.chat_input(
        "Ask Genius AI anything..."
    )

    if prompt:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            with st.spinner(
                "Genius is thinking..."
            ):

                answer = ask_ai(
                    prompt,
                    st.session_state.mode
                )

            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            })

# ============================================================
# THINKER
# ============================================================

with tabs[1]:

    st.markdown(
        "## 🧠 Thinker Mode"
    )

    st.caption(
        "Designed for difficult problems, planning, coding "
        "and careful analysis."
    )

    thinker_prompt = st.text_area(
        "Problem",
        placeholder=(
            "Example: Design a scalable architecture "
            "for an AI chatbot..."
        ),
        height=180
    )

    if st.button(
        "🧠 Analyze Problem",
        key="think_button"
    ):

        if thinker_prompt.strip():

            with st.spinner(
                "Analyzing..."
            ):

                answer = ask_ai(
                    thinker_prompt,
                    "Thinker"
                )

            st.markdown(answer)

        else:

            st.warning(
                "Please enter a problem first."
            )


# ============================================================
# IMAGE GENERATOR
# ============================================================

# ============================================================
# 🎨 AI IMAGE GENERATOR
# ============================================================

with tabs[2]:

    st.markdown("## 🎨 AI Image Generator")
    st.caption("Create beautiful AI images using Hugging Face.")

    # Image prompt
    prompt = st.text_area(
        "Describe your image",
        placeholder=(
            "Example: A futuristic AI robot standing in a "
            "neon city at night, cinematic lighting, ultra detailed"
        ),
        height=150,
        key="image_prompt"
    )

    # Options
    col1, col2 = st.columns(2)

    with col1:
        image_style = st.selectbox(
            "🎨 Art Style",
            [
                "Photorealistic",
                "Cinematic",
                "3D Render",
                "Digital Art",
                "Anime",
                "Fantasy",
                "Cyberpunk",
                "Oil Painting"
            ],
            key="image_style"
        )

    with col2:
        image_quality = st.selectbox(
            "✨ Quality",
            [
                "High Quality",
                "Ultra Detailed",
                "Professional",
                "Cinematic"
            ],
            key="image_quality"
        )

    st.write("")

    # Generate button
    generate = st.button(
        "✨ Generate Image",
        type="primary",
        use_container_width=True,
        key="generate_image_button"
    )

    if generate:

        # ----------------------------------------------------
        # IMPORTANT:
        # st.text_area normally returns a string, but checking
        # for None prevents the .strip() AttributeError.
        # ----------------------------------------------------

        if prompt is None:
            prompt = ""

        prompt = str(prompt).strip()

        if not prompt:

            st.warning(
                "⚠️ Please enter a description for your image."
            )

        else:

            # Create final prompt
            final_prompt = (
                f"{prompt}. "
                f"Style: {image_style}. "
                f"Quality: {image_quality}. "
                "Highly detailed, visually stunning, "
                "professional composition."
            )

            # Show prompt
            with st.expander("🔍 View generated prompt"):

                st.code(
                    final_prompt
                )

            try:

                # ------------------------------------------------
                # Generate image with Hugging Face
                # ------------------------------------------------

                with st.spinner(
                    "🎨 Genius AI is creating your image..."
                ):

                    image = client.text_to_image(
                        prompt=final_prompt,
                        model=IMAGE_MODEL
                    )

                # ------------------------------------------------
                # Display result
                # ------------------------------------------------

                st.success(
                    "✨ Image generated successfully!"
                )

                st.markdown(
                    "## ✨ Your AI Creation"
                )

                st.image(
                    image,
                    caption="Generated by Genius AI",
                    width="stretch"
                )

                # ------------------------------------------------
                # Download button
                # ------------------------------------------------

                import io

                image_bytes = io.BytesIO()

                image.save(
                    image_bytes,
                    format="PNG"
                )

                st.download_button(
                    label="⬇️ Download Image",
                    data=image_bytes.getvalue(),
                    file_name="genius_ai_image.png",
                    mime="image/png",
                    key="download_generated_image"
                )

            except Exception as e:

                st.error(
                    "❌ Image generation failed."
                )

                st.markdown(
                    "### 🔧 Error details"
                )

                st.code(
                    str(e)
                )

                st.info(
                    "Check that your Hugging Face token is valid "
                    "and that the selected image model is available "
                    "through your Hugging Face account/provider."
                )



# ============================================================
# WEB SEARCH
# ============================================================

with tabs[3]:
    st.markdown(
        "## 🌐 Web Search"
    )

    st.caption(
        "Search the public web and inspect the returned results."
    )

    search_query = st.text_input(
        "Search query",
        placeholder="Latest AI news, Python tutorials, etc."
    )

    search_button = st.button(
        "🔎 Search Web",
        key="web_search_button"
    )

    if search_button:

        if not search_query.strip():

            st.warning(
                "Enter something to search."
            )

        else:

            with st.spinner(
                "Searching the web..."
            ):

                results = web_search(
                    search_query
                )

            if not results:

                st.info(
                    "No results found."
                )

            else:

                for result in results:

                    st.markdown(
                        f"""
                        ### [{result["title"]}]({result["url"]})

                        {result["snippet"]}

                        ---
                        """
                    )


# ============================================================
# VOICE RECORDER
# ============================================================

with tabs[4]:

    st.markdown(
        "## 🎙️ Voice Recorder"
    )


    st.caption(
        "Record audio directly from your browser."
    )

    audio = st.audio_input(
        "Record your voice"
    )

    if audio:

        st.audio(
            audio
        )

        st.download_button(
            label="⬇️ Download Recording",
            data=audio.getvalue(),
            file_name="genius_ai_recording.wav",
            mime="audio/wav"
        )

        st.info(
            "The recording feature stores/plays the audio. "
            "Speech-to-text is not enabled in this two-file version."
        )


# ============================================================
# WEATHER
# ============================================================

with tabs[5]:

    st.markdown(
        "## 🌤️ Weather Checker"
    )

    st.caption(
        "Current weather using Open-Meteo."
    )

    city = st.text_input(
        "City",
        placeholder="Islamabad"
    )

    weather_button = st.button(
        "🌤️ Check Weather",
        key="weather_button"
    )

    if weather_button:

        if not city.strip():

            st.warning(
                "Enter a city."
            )

        else:

            with st.spinner(
                "Getting weather..."
            ):

                weather = get_weather(
                    city
                )

            if weather is None:

                st.error(
                    "Could not find weather for that city."
                )

            else:

                st.markdown(
                    f"""
                    ### 📍 {weather["city"]},
                    {weather["country"]}
                    """
                )

                c1, c2, c3, c4 = st.columns(4)

                with c1:

                    st.metric(
                        "Temperature",
                        f'{weather["temperature"]} °C'
                    )

                with c2:

                    st.metric(
                        "Feels Like",
                        f'{weather["feels_like"]} °C'
                    )

                with c3:

                    st.metric(
                        "Humidity",
                        f'{weather["humidity"]}%'
                    )

                with c4:

                    st.metric(
                        "Wind",
                        f'{weather["wind"]} km/h'
                    )


# ============================================================
# CALCULATOR
# ============================================================

with tabs[6]:

    st.markdown(
        "## 🧮 Calculator"
    )

    st.caption(
        "Safe calculator for common mathematical expressions."
    )

    expression = st.text_input(
        "Expression",
        placeholder="25 * (12 + 8) / 4"
    )

    calculator_button = st.button(
        "🧮 Calculate",
        key="calculator_button"
    )

    if calculator_button:

        if not expression.strip():

            st.warning(
                "Enter a mathematical expression."
            )

        else:

            try:

                result = safe_calculate(
                    expression
                )

                st.success(
                    f"Result: {result}"
                )

            except Exception as e:

                st.error(
                    f"Invalid expression: {e}"
                )


# ============================================================
# 3D GLOBE
# ============================================================

with tabs[7]:

    st.markdown(
        "## 🌍 3D Globe"
    )

    st.caption(
        "Interactive 3D Earth visualization."
    )

    try:

        import pydeck as pdk

        globe_layer = pdk.Layer(
            "ScatterplotLayer",

            data=[
                {
                    "lat": 20,
                    "lon": 0,
                    "size": 500000
                }
            ],

            get_position=[
                "lon",
                "lat"
            ],

            get_radius="size",

            get_fill_color=[
                0,
                200,
                255,
                160
            ],

            pickable=True
        )

        globe_view = pdk.ViewState(
            latitude=20,
            longitude=0,
            zoom=1.2,
            pitch=35,
            bearing=0
        )

        deck = pdk.Deck(
            layers=[
                globe_layer
            ],

            initial_view_state=globe_view,

            map_style=None,

            tooltip={
                "text":
                    "Genius AI Earth Explorer"
            }
        )

        st.pydeck_chart(
            deck,
            width=True
        )

        st.info(
            "Drag to rotate and zoom the globe."
        )

    except ImportError:

        st.error(
            "PyDeck is not installed."
        )

        st.code(
            "pip install pydeck"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <br>
    <hr>

    <div style="
        text-align:center;
        color:#666675;
        font-size:13px;
        padding:15px;
    ">

        🧠 <b style="color:#9b7cff;">
        GENIUS AI
        </b>
        &nbsp; • &nbsp;
        Your intelligent AI workspace

    </div>
    """,
    unsafe_allow_html=True
)
# ============================================================
# 🎬 AI VIDEO GENERATOR
# ============================================================

with tabs[3]:

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:10px 0 20px 0;
        ">
            <h1 style="
                font-size:42px;
                font-weight:900;
                background:linear-gradient(
                    90deg,
                    #7c3aed,
                    #00e5ff,
                    #ffffff,
                    #7c3aed
                );
                -webkit-background-clip:text;
                -webkit-text-fill-color:transparent;
            ">
                🎬 Genius AI Video
            </h1>

            <p style="
                color:#888899;
                font-size:15px;
            ">
                Turn your imagination into AI-generated video.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # VIDEO MODEL
    # --------------------------------------------------------

    VIDEO_MODEL = os.getenv(
        "HF_VIDEO_MODEL",
        "Lightricks/LTX-Video-0.9.8-13B-distilled"
    )

    # --------------------------------------------------------
    # VIDEO PROMPT
    # --------------------------------------------------------

    video_prompt = st.text_area(
        "🎥 Describe your video",
        placeholder=(
            "Example: A futuristic robot walking through "
            "a neon cyberpunk city at night, cinematic camera "
            "movement, rain, realistic lighting, highly detailed."
        ),
        height=160,
        key="video_prompt"
    )

    # --------------------------------------------------------
    # OPTIONS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        video_style = st.selectbox(
            "🎨 Video Style",
            [
                "Cinematic",
                "Photorealistic",
                "3D Animation",
                "Anime",
                "Cyberpunk",
                "Fantasy",
                "Documentary",
                "Sci-Fi"
            ],
            key="video_style"
        )

    with col2:

        camera_motion = st.selectbox(
            "🎥 Camera Movement",
            [
                "Slow cinematic camera movement",
                "Camera slowly zooms in",
                "Camera slowly zooms out",
                "Smooth tracking shot",
                "Dynamic handheld camera",
                "Aerial drone shot",
                "Static camera"
            ],
            key="camera_motion"
        )

    # --------------------------------------------------------
    # ADVANCED SETTINGS
    # --------------------------------------------------------

    with st.expander("⚙️ Advanced Video Settings"):

        negative_prompt = st.text_input(
            "🚫 Negative Prompt",
            value=(
                "blurry, low quality, distorted, "
                "deformed, flickering, bad anatomy"
            ),
            key="video_negative_prompt"
        )

        col_a, col_b = st.columns(2)

        with col_a:

            num_frames = st.slider(
                "🎞️ Number of Frames",
                min_value=8,
                max_value=64,
                value=16,
                step=8,
                key="video_frames"
            )

        with col_b:

            inference_steps = st.slider(
                "✨ Quality Steps",
                min_value=4,
                max_value=30,
                value=8,
                step=1,
                key="video_steps"
            )

        guidance_scale = st.slider(
            "🎯 Guidance",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
            key="video_guidance"
        )

        seed = st.number_input(
            "🔢 Seed",
            min_value=0,
            max_value=999999999,
            value=42,
            step=1,
            key="video_seed"
        )

    st.write("")

    # --------------------------------------------------------
    # GENERATE BUTTON
    # --------------------------------------------------------

    generate_video = st.button(
        "🎬 Generate Video",
        type="primary",
        use_container_width=True,
        key="generate_video_button"
    )

    if generate_video:

        # ----------------------------------------------------
        # FIX FOR NONE .strip() ERROR
        # ----------------------------------------------------

        if video_prompt is None:
            video_prompt = ""

        video_prompt = str(
            video_prompt
        ).strip()

        # ----------------------------------------------------
        # CHECK PROMPT
        # ----------------------------------------------------

        if not video_prompt:

            st.warning(
                "⚠️ Please describe the video you want to create."
            )

        else:

            # ------------------------------------------------
            # BUILD FINAL PROMPT
            # ------------------------------------------------

            final_video_prompt = (
                f"{video_prompt}. "
                f"Visual style: {video_style}. "
                f"Camera movement: {camera_motion}. "
                "High quality, detailed visuals, "
                "smooth motion, coherent scene, "
                "cinematic composition."
            )

            # ------------------------------------------------
            # SHOW PROMPT
            # ------------------------------------------------

            with st.expander(
                "🔍 View final video prompt"
            ):

                st.code(
                    final_video_prompt
                )

            # ------------------------------------------------
            # GENERATE
            # ------------------------------------------------

            try:

                with st.spinner(
                    "🎬 Genius AI is generating your video... "
                    "This may take some time."
                ):

                    video_bytes = client.text_to_video(
                        prompt=final_video_prompt,
                        model=VIDEO_MODEL,
                        negative_prompt=[
                            negative_prompt
                        ],
                        num_frames=num_frames,
                        num_inference_steps=inference_steps,
                        guidance_scale=guidance_scale,
                        seed=int(seed)
                    )

                # ------------------------------------------------
                # VALIDATE RESPONSE
                # ------------------------------------------------

                if not video_bytes:

                    st.error(
                        "❌ The video model returned no video."
                    )

                else:

                    st.success(
                        "🎉 Video generated successfully!"
                    )

                    # ------------------------------------------------
                    # VIDEO TITLE
                    # ------------------------------------------------

                    st.markdown(
                        "## 🎬 Your AI Video"
                    )

                    # ------------------------------------------------
                    # VIDEO PLAYER
                    # ------------------------------------------------

                    st.video(
                        video_bytes
                    )

                    # ------------------------------------------------
                    # DOWNLOAD
                    # ------------------------------------------------

                    st.download_button(
                        label="⬇️ Download Video",
                        data=video_bytes,
                        file_name="genius_ai_video.mp4",
                        mime="video/mp4",
                        type="primary",
                        use_container_width=True,
                        key="download_video_button"
                    )

                    st.caption(
                        "✨ Generated by Genius AI using Hugging Face."
                    )

            # ------------------------------------------------
            # ERROR HANDLING
            # ------------------------------------------------

            except Exception as e:

                st.error(
                    "❌ Video generation failed."
                )

                with st.expander(
                    "🔧 Show technical error"
                ):

                    st.code(
                        str(e)
                    )

                st.info(
                    "Check your Hugging Face token, model name, "
                    "Inference Provider availability, and account "
                    "permissions."
                )
