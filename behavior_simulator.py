import json
import math
import uuid
import streamlit as st
from typing import Literal
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# ==========================================
# PYDANTIC SCHEMAS (FORCES JSON OUTPUTS)
# ==========================================

# --- Schemas for Forward Predictor ---
class PredictedAction(BaseModel):
    action: str = Field(description="A highly specific, plausible action this hypothetical profile might take.")
    raw_weight: int = Field(ge=1, le=100, description="Relative plausibility score (1-100) of this outcome occurring based on heuristic model weights, not statistically calibrated probability.")
    rationale: str = Field(description="Explanation of why, detailing mechanisms and specific trait/state/context/sensory interactions. Acknowledge unknown factors if relevant.")

class ForwardPrediction(BaseModel):
    modifier_relevance: str = Field(description="Analysis of which specific modifiers actually mattered here, and which were irrelevant and had negligible influence.")
    uncertainty_level: Literal["Low", "Moderate", "High"] = Field(description="Rate the uncertainty of this generation.")
    uncertainty_reason: str = Field(description="Explanation of why the generation carries this level of uncertainty.")
    predictions: list[PredictedAction] = Field(min_length=3, max_length=3, description="Exactly 3 plausible actions, ranked by relative plausibility.")

class CounterfactualResponse(BaseModel):
    identified_variable_changed: str = Field(description="The specific variable, state, or context detail that was altered from the baseline configuration.")
    comparison_summary: str = Field(description="Explanation of how and why this specific change alters the behavioral/engagement landscape compared to the baseline.")
    new_predictions: list[PredictedAction] = Field(min_length=3, max_length=3, description="Exactly 3 new plausible actions based on the modified scenario.")


# --- Schemas for Competing Explanations (Tab 2) ---
class HexacoScores(BaseModel):
    Honesty_Humility: str = Field(description="Qualitative range or 'Insufficient information'.")
    Emotionality: str = Field(description="Qualitative range or 'Insufficient information'.")
    Extraversion: str = Field(description="Qualitative range or 'Insufficient information'.")
    Agreeableness: str = Field(description="Qualitative range or 'Insufficient information'.")
    Conscientiousness: str = Field(description="Qualitative range or 'Insufficient information'.")
    Openness: str = Field(description="Qualitative range or 'Insufficient information'.")

class EvidenceBreakdown(BaseModel):
    directly_supported: str = Field(description="What we actually observed in the behavior.")
    interpretation: str = Field(description="What could plausibly explain it.")
    speculation: str = Field(description="What we are assuming because information is missing.")

class CompetingExplanation(BaseModel):
    explanation_name: str = Field(description="Short title.")
    compatibility: Literal["Strong", "Moderate", "Weak"] = Field(description="Compatibility with the observed behavior.")
    primary_mechanism: str = Field(description="Primary driver of the behavior.")
    situational_factors: str = Field(description="What external factors could explain the behavior?")
    temporary_state: str = Field(description="What temporary internal state might matter right now?")
    possible_trait_contribution: str = Field(description="How traits might contribute, IF relevant.")
    hexaco: HexacoScores
    evidence_breakdown: EvidenceBreakdown

class BehaviorAnalysisResult(BaseModel):
    behavioral_ambiguity: str = Field(description="How ambiguous the behavior is.")
    specific_uncertainty: str = Field(description="Why uncertain: specific factors we don't know.")
    cannot_be_inferred: list[str] = Field(description="Specific broad traits that CANNOT be inferred.")
    missing_information: list[str] = Field(description="Specific pieces of missing context.")
    explanations: list[CompetingExplanation] = Field(min_length=3, max_length=3, description="Exactly 3 genuinely competing explanations.")


# ==========================================
# STREAMLIT APP CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="Behavioral & Outreach Lab", layout="wide")

premium_css = """
<style>
    .stApp {
        background-color: #0d0d0e;
        color: #e5e5ea;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
        font-weight: 500;
        letter-spacing: -0.015em;
    }
    .stTextInput input, .stSelectbox div[data-baseweb="select"], .stTextArea textarea, .stMultiSelect div[data-baseweb="select"] {
        background-color: #151517 !important;
        border: 1px solid #2a2a2e !important;
        color: #e5e5ea !important;
        border-radius: 8px !important;
        box-shadow: none !important;
        transition: border-color 0.2s ease;
    }
    .stTextInput input:focus, .stSelectbox div[data-baseweb="select"]:focus-within, .stTextArea textarea:focus {
        border-color: #4F83F5 !important;
    }
    .stSlider [data-baseweb="slider"] div {
        background-color: #4F83F5 !important;
    }
    .stButton button {
        background-color: #1a1a1c;
        border: 1px solid #333336;
        color: #ffffff;
        border-radius: 8px;
        transition: all 0.2s ease;
        font-weight: 500;
        letter-spacing: 0.01em;
    }
    .stButton button:hover {
        border-color: #4F83F5;
        color: #4F83F5;
    }
    button[data-testid="baseButton-primary"] {
        background-color: #4F83F5 !important;
        border: 1px solid #3d6acc !important;
        color: #ffffff !important;
        padding: 8px 24px !important;
    }
    button[data-testid="baseButton-primary"]:hover {
        background-color: #3d6acc !important;
        box-shadow: 0 4px 15px rgba(79, 131, 245, 0.3) !important;
        color: #ffffff !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 32px;
        margin-bottom: 24px;
        border-bottom: 1px solid #1a1a1c;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #8e8e93 !important;
        border-bottom-color: transparent !important;
        padding: 12px 0;
        font-size: 1.05rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        border-bottom: 2px solid #4F83F5 !important;
    }
    div[data-testid="stAlert"] {
        background-color: #151517 !important;
        border: 1px solid #2a2a2e !important;
        border-radius: 8px !important;
        color: #e5e5ea !important;
    }
    .section-title {
        color: #ffffff;
        font-size: 1.4rem;
        font-weight: 400;
        letter-spacing: -0.01em;
        margin-bottom: 24px;
        border-bottom: 1px solid #2a2a2e;
        padding-bottom: 12px;
    }
    .premium-card {
        background-color: #151517;
        border: 1px solid #26262a;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .premium-card:hover {
        transform: translateY(-2px);
        border-color: #3a3a40;
    }
    .card-title {
        color: #ffffff;
        font-size: 1.2rem;
        font-weight: 500;
        margin-bottom: 12px;
        margin-top: 0;
        line-height: 1.4;
    }
    .card-weight {
        display: inline-block;
        background-color: #111112;
        color: #4F83F5;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 16px;
        border: 1px solid #1a2744;
    }
    .card-content {
        color: #a1a1aa;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    .hypothesis-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 16px;
        letter-spacing: 0.02em;
    }
    .badge-strong { background-color: #0b2216; color: #4ade80; border: 1px solid #144029; }
    .badge-moderate { background-color: #2b1d0c; color: #fbbf24; border: 1px solid #4a3416; }
    .badge-weak { background-color: #261012; color: #f87171; border: 1px solid #451b1f; }
    .evidence-block {
        background-color: #111112;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 3px solid;
    }
    .evidence-observed { border-left-color: #a78bfa; }
    .evidence-interpretation { border-left-color: #4F83F5; }
    .evidence-speculation { border-left-color: #fb923c; }
    .evidence-header {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
        font-weight: 600;
    }
    .evidence-observed .evidence-header { color: #a78bfa; }
    .evidence-interpretation .evidence-header { color: #4F83F5; }
    .evidence-speculation .evidence-header { color: #fb923c; }
    .evidence-body {
        color: #d4d4d8;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .custom-progress-bg {
        background-color: #26262a;
        border-radius: 4px;
        height: 6px;
        width: 100%;
        margin-bottom: 16px;
        overflow: hidden;
    }
    .custom-progress-fill {
        background-color: #4F83F5;
        height: 100%;
        border-radius: 4px;
        transition: width 1s ease-out;
    }
</style>
"""
st.markdown(premium_css, unsafe_allow_html=True)
st.markdown("<h1 style='font-weight: 300; margin-bottom: 0;'>Behavioral & Outreach Scenario Lab</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #8e8e93; font-size: 1.05rem; margin-bottom: 30px;'>Predictive modeling for human behavior & astro-acoustic science communication. <em>(ninolades.com x Astro Accel)</em></p>", unsafe_allow_html=True)

# Initialize Session State
for key in ['parsed_predictions', 'reverse_parsed_predictions', 'last_sim', 'last_situation', 'last_chat_response', 'last_config', 'simulation_id']:
    if key not in st.session_state:
        st.session_state[key] = None

# Sidebar Configuration
with st.sidebar:
    st.markdown("<h3 style='margin-bottom: 20px;'>Lab Configurations</h3>", unsafe_allow_html=True)
    
    model_choice = st.selectbox(
        "Primary AI Engine",
        [
            "Gemini 3.6 Flash (Fast & Capable - Default)", 
            "Gemini 3.1 Pro (Heavy Reasoning)", 
            "Gemini 3.5 Flash-Lite (Ultra Fast)"
        ],
        index=0
    )

    if "3.1 Pro" in model_choice:
        primary_model = "gemini-3.1-pro"
        backup_model = "gemini-3.6-flash"
    elif "3.5 Flash-Lite" in model_choice:
        primary_model = "gemini-3.5-flash-lite"
        backup_model = "gemini-3.6-flash"
    else: 
        primary_model = "gemini-3.6-flash"
        backup_model = "gemini-3.5-flash-lite"

    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Enter Gemini API Key:", type="password")

if not api_key:
    st.warning("Please enter your Gemini API Key in the sidebar or configure your secrets to begin.")
    st.stop()

client = genai.Client(api_key=api_key)

def calculate_normalized_percentages(predictions_list):
    if not predictions_list: 
        return [] 
    raw_weights = [action['raw_weight'] for action in predictions_list]
    total_weight = sum(raw_weights) 
    if total_weight <= 0: 
        return [round(100 / len(predictions_list))] * len(predictions_list)
    exact = [(w / total_weight) * 100 for w in raw_weights]
    percentages = [math.floor(x) for x in exact]
    remainder = int(100 - sum(percentages))
    order = sorted(range(len(exact)), key=lambda i: exact[i] - percentages[i], reverse=True)
    for i in order[:remainder]:
        percentages[i] += 1
    return percentages


tab1, tab2, tab3 = st.tabs(["Forward Predictor", "Competing Explanations", "Science & Methodology"])


# ==========================================
# TAB 1: FORWARD PREDICTOR
# ==========================================
with tab1:
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='section-title'>Profile Configuration</div>", unsafe_allow_html=True)
        st.markdown("<div style='color: #8e8e93; font-size: 0.9rem; margin-bottom: 20px;'>HEXACO Parameters (0–100). Hypothetical parameters for simulation.</div>", unsafe_allow_html=True)
        
        h = st.slider("Honesty-Humility", 0, 100, 50)
        e = st.slider("Emotionality", 0, 100, 60)
        x = st.slider("Extraversion", 0, 100, 50)
        a = st.slider("Agreeableness", 0, 100, 60)
        c = st.slider("Conscientiousness", 0, 100, 70)
        o = st.slider("Openness to Experience", 0, 100, 85)

    with col2:
        st.markdown("<div class='section-title'>States & Modifiers</div>", unsafe_allow_html=True)
        sensory = st.selectbox("Sensory Responsiveness", ["Lower hypothetical sensory reactivity", "Moderate hypothetical sensory reactivity", "Higher hypothetical sensory reactivity"], index=2)
        sensory_domains = st.multiselect("Relevant Sensory Domains", ["Auditory", "Visual", "Tactile", "Olfactory", "Gustatory", "Vestibular"], default=["Auditory", "Visual"])
        masking = st.selectbox("Behavioral Masking Tendency", ["None (Natural expression)", "Moderate", "High (Heavy camouflage)"])
        stimming = st.multiselect("Self-Regulation Tendency", ["None", "Fidgeting", "Pacing", "Auditory stimming", "Tactile stimming", "Vocal scripting"], default=["None"])
        reward_sensitivity = st.selectbox("Reward Sensitivity / Novelty Seeking", ["Low (Prefers predictable/stable options)", "Medium (Balanced)", "High (Sensitive to novelty, reward, stimulation)"], index=2)
        state_trait = st.selectbox("Current Internal State", ["Baseline", "Relaxed/Calm", "High stress/arousal", "Fatigued/Burnout"])
        cognitive_load = st.selectbox("Cognitive Load", ["Low (Clear headed)", "Medium (Busy)", "High (Distracted/Overwhelmed)"])
        
    st.markdown("<br><div class='section-title'>Outreach & Context Configuration</div>", unsafe_allow_html=True)
    
    preset = st.selectbox(
        "Load Outreach Preset:",
        [
            "Custom / General Behavioral",
            "Astro-Acoustic Stargazing Night (Telescope + Live Ukulele)",
            "Eco-Astronomy & Dark Sky Nature Walk",
            "School & Youth Astronomy Communication"
        ]
    )
    
    # Preset logic
    if preset == "Astro-Acoustic Stargazing Night (Telescope + Live Ukulele)":
        default_target = "Saturn's Rings & Deep-Sky Nebulae"
        default_music = "Live acoustic ukulele melodies, rhythmic stargazing storytelling"
        default_env = "Outdoor dark-sky site, open lawn under clear night sky"
        default_bg = "Public stargazing event combining high-magnification telescope viewing with live soundscapes."
        default_sit = "Participants are waiting in line at the telescope while acoustic music plays in the background."
    elif preset == "Eco-Astronomy & Dark Sky Nature Walk":
        default_target = "Milky Way core & seasonal constellations"
        default_music = "Ambient night sounds, subtle acoustic drone, storytelling"
        default_env = "Nocturnal nature trail, dark sky preserve"
        default_bg = "Guided night trail focusing on dark-sky preservation and nocturnal wildlife."
        default_sit = "Group pauses at an unlit clearing for naked-eye celestial navigation."
    elif preset == "School & Youth Astronomy Communication":
        default_target = "Moon craters & Jupiter's Galilean moons"
        default_music = "Upbeat rhythmic ukulele strumming, interactive call-and-response"
        default_env = "School courtyard / playground at twilight"
        default_bg = "Interactive educational session designed to demystify astrophysics for students."
        default_sit = "Students gather around the telescope during a live musical transition."
    else:
        default_target = "None / General Context"
        default_music = "None"
        default_env = "Loud, crowded subway station"
        default_bg = "Late for an important job interview."
        default_sit = "Finds a wallet with $500 cash."

    col_a, col_b = st.columns(2)
    with col_a:
        celestial_focus = st.text_input("Celestial Focus (Astro Target)", default_target)
        musical_element = st.text_input("Acoustic / Musical Element", default_music)
    with col_b:
        env_setting = st.text_input("Environmental Setting", default_env)
        audience_type = st.selectbox("Audience Profile", ["General Public", "Students / Youth", "Eco-Tourists", "Astronomy Enthusiasts"])

    extra_details = st.text_input("Background Context", default_bg)
    situation = st.text_area("Immediate Situation", default_sit, height=80)
    
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Run Simulation Engine", type="primary", use_container_width=True):
        st.session_state["parsed_predictions"] = None
        st.session_state["last_chat_response"] = None
        st.session_state["simulation_id"] = str(uuid.uuid4())[:8]
        
        st.session_state['last_config'] = {
            "HEXACO": {"H": h, "E": e, "X": x, "A": a, "C": c, "O": o},
            "Sensory Responsiveness": sensory,
            "Sensory Domains": sensory_domains,
            "Masking Tendency": masking,
            "Stimming Tendency": stimming,
            "Reward Sensitivity": reward_sensitivity,
            "Current State": state_trait,
            "Cognitive Load": cognitive_load,
            "Celestial Focus": celestial_focus,
            "Musical Element": musical_element,
            "Environmental Setting": env_setting,
            "Audience Type": audience_type,
            "Background Details": extra_details,
            "Situation": situation
        }
        
        prompt = f"""
        You are an AI generating plausible behavioral and science outreach engagement outcomes given hypothetical traits, sensory states, environmental contexts, and astro-acoustic parameters.
        
        CRITICAL RULES:
        1. Treat all parameters strictly as hypothetical simulation inputs.
        2. Never infer clinical diagnoses.
        3. Evaluate the interaction between ASTRO-ACOUSTIC & SENSORY variables: Analyze how live acoustic music (e.g., ukulele melody) combined with telescope visuals impacts audience attention, emotional awe, and cognitive retention.
        4. In outreach/stargazing contexts, model how multisensory inputs (starlight, acoustic melody, night air) lower cognitive load and increase retention of complex scientific concepts.
        5. Separate Traits, States, Sensory Inputs, and Context.
        6. Do not force every modifier to matter if irrelevant.
        7. The 3 predicted actions MUST be meaningfully different from each other.
        
        TRAITS:
        - HEXACO: H:{h}, E:{e}, X:{x}, A:{a}, C:{c}, O:{o}
        - Reward Sensitivity: {reward_sensitivity}
        - Sensory Responsiveness: {sensory}
        
        ASTRO-ACOUSTIC & SENSORY INPUTS:
        - Celestial Focus: {celestial_focus}
        - Musical/Acoustic Element: {musical_element}
        - Environmental Setting: {env_setting}
        - Audience Profile: {audience_type}
        - Sensory Domains: {sensory_domains}
        
        STATES & CONTEXT:
        - Current State: {state_trait} | Cognitive Load: {cognitive_load}
        - Background: {extra_details}
        - Situation: {situation}
        """
        
        def run_forward_generation(model_name):
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ForwardPrediction,
                    temperature=0.4
                )
            )
            return ForwardPrediction.model_validate_json(response.text), response.text
            
        with st.spinner("Executing simulation models..."):
            try:
                result_obj, raw_text = run_forward_generation(primary_model)
                st.session_state['parsed_predictions'] = result_obj.model_dump()
                st.session_state['last_sim'] = raw_text
                st.session_state['last_situation'] = situation
            except Exception as primary_error:
                st.warning("Primary engine failed. Retrying with backup engine...")
                try:
                    result_obj, raw_text = run_forward_generation(backup_model)
                    st.session_state['parsed_predictions'] = result_obj.model_dump()
                    st.session_state['last_sim'] = raw_text
                    st.session_state['last_situation'] = situation
                except Exception as backup_error:
                    st.error("System failure: Engines unavailable.")
                    st.stop()

    if st.session_state['parsed_predictions']:
        result = st.session_state['parsed_predictions']
        st.markdown("<br><hr style='border-color: #1a1a1c;'><br>", unsafe_allow_html=True)
        st.markdown(f"<div style='color: #4F83F5; font-size: 0.9rem; letter-spacing: 0.05em; margin-bottom: 24px; text-transform: uppercase;'>Simulation ID: {st.session_state['simulation_id']}</div>", unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.info(f"**Modifier Relevance:**\n\n{result['modifier_relevance']}")
        with col_b:
            st.warning(f"**Uncertainty Level:** {result['uncertainty_level']}\n\n{result['uncertainty_reason']}")
    
        st.markdown("<br><div class='section-title'>Predicted Engagement Pathways</div>", unsafe_allow_html=True)
        
        predictions = result.get('predictions', [])
        percentages = calculate_normalized_percentages(predictions)
        
        for idx, action in enumerate(predictions):
            calculated_pct = percentages[idx]
            card_html = f"""
            <div class="premium-card">
                <h4 class="card-title">{idx+1}. {action['action']}</h4>
                <div class="card-weight">Relative Plausibility: {calculated_pct}%</div>
                <div class="custom-progress-bg">
                    <div class="custom-progress-fill" style="width: {calculated_pct}%;"></div>
                </div>
                <div class="card-content"><strong>Rationale:</strong> {action['rationale']}</div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

        # Counterfactual Lab
        st.markdown("<br><div class='section-title'>Counterfactual Lab</div>", unsafe_allow_html=True)
        st.markdown("<div style='color: #8e8e93; font-size: 0.95rem; margin-bottom: 20px;'>Isolate and modify a single variable to observe cascading changes. <em>(e.g., 'What if we add live ukulele music during the telescope waiting period?' or 'What if lighting is added?')</em></div>", unsafe_allow_html=True)

        with st.form("chat_form"):
            query = st.text_input("Modify Variable", placeholder="e.g., Introduce live acoustic storytelling during viewing")
            submit_q = st.form_submit_button("Run Counterfactual Simulation")
    
            if submit_q and query:
                if not st.session_state.get('last_config'):
                    st.error("Please run a baseline simulation first.")
                else:
                    config_str = json.dumps(st.session_state['last_config'], indent=2)
                    chat_prompt = f"""
                    You are continuing a behavioral & science outreach simulation.
                    ORIGINAL CONFIGURATION: {config_str}
                    PREVIOUS AI PREDICTIONS: {st.session_state['last_sim']}
                    USER HYPOTHESIS: {query}
                    
                    Identify the exact variable changed, keep others constant, and project 3 new outcomes.
                    """
            
                    def run_counterfactual_generation(model_name):
                        resp = client.models.generate_content(
                            model=model_name,
                            contents=chat_prompt,
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=CounterfactualResponse,
                                temperature=0.5
                            )
                        )
                        return CounterfactualResponse.model_validate_json(resp.text)

                    with st.spinner("Processing counterfactual matrix..."):
                        try:
                            cf_obj = run_counterfactual_generation(primary_model)
                            st.session_state['last_chat_response'] = cf_obj.model_dump()
                        except Exception:
                            try:
                                cf_obj = run_counterfactual_generation(backup_model)
                                st.session_state['last_chat_response'] = cf_obj.model_dump()
                            except Exception:
                                st.error("Counterfactual engine failed.")

        if st.session_state['last_chat_response']:
            cf_data = st.session_state['last_chat_response']
            st.markdown(f"""
            <div style="background-color: #1a1a2e; border: 1px solid #28304a; padding: 20px; border-radius: 12px; margin-top: 20px; margin-bottom: 24px;">
                <div style="color: #4F83F5; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; margin-bottom: 8px;">Identified Modification</div>
                <div style="color: #ffffff; font-size: 1.1rem; margin-bottom: 16px;">{cf_data['identified_variable_changed']}</div>
                <div style="color: #4F83F5; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; margin-bottom: 8px;">Divergence from Baseline</div>
                <div style="color: #d4d4d8; font-size: 0.95rem; line-height: 1.5;">{cf_data['comparison_summary']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            cf_predictions = cf_data.get('new_predictions', [])
            cf_percentages = calculate_normalized_percentages(cf_predictions)
            
            st.markdown("<div class='section-title' style='font-size: 1.1rem;'>Counterfactual Projections</div>", unsafe_allow_html=True)
            for idx, action in enumerate(cf_predictions):
                calculated_pct = cf_percentages[idx]
                card_html = f"""
                <div class="premium-card" style="background-color: #121214;">
                    <h4 class="card-title" style="font-size: 1.1rem;">{idx+1}. {action['action']}</h4>
                    <div class="card-weight" style="background-color: #0d121c;">Relative Plausibility: {calculated_pct}%</div>
                    <div class="custom-progress-bg" style="background-color: #1e1e24;">
                        <div class="custom-progress-fill" style="width: {calculated_pct}%;"></div>
                    </div>
                    <div class="card-content"><strong>Rationale:</strong> {action['rationale']}</div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)


# ==========================================
# TAB 2: COMPETING EXPLANATIONS
# ==========================================
with tab2:
    st.markdown("<div class='section-title' style='border: none;'>Behavioral Equifinality Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div style='color: #8e8e93; font-size: 1rem; margin-bottom: 30px; line-height: 1.5;'>Reverse-engineers multiple distinct psychological profiles and contextual configurations that could produce a single observed reaction.</div>", unsafe_allow_html=True)
    
    rev_situation = st.text_area("Environmental Context", "Observing Saturn through a telescope at a dark sky lawn with live acoustic music.", height=80)
    observed_action = st.text_area("Observed Behavior", "Participant remained completely silent for 30 seconds after stepping away, then asked a deep question about planetary formation.", height=80)
    known_context = st.text_input("Verified Modifiers (Optional)", "First time viewing through a telescope")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔬 Generate Competing Hypotheses", type="primary", use_container_width=True):
        st.session_state["reverse_parsed_predictions"] = None
        
        prompt = f"""
        Perform a behavioral equifinality analysis.
        SITUATION: {rev_situation}
        OBSERVED ACTION: {observed_action}
        KNOWN CONTEXT: {known_context}
        """
        
        def run_reverse_generation(model_name):
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=BehaviorAnalysisResult,
                    temperature=0.6 
                )
            )
            return BehaviorAnalysisResult.model_validate_json(response.text)

        with st.spinner("Synthesizing behavioral pathways..."):
            try:
                rev_result_obj = run_reverse_generation(primary_model)
                st.session_state['reverse_parsed_predictions'] = rev_result_obj.model_dump()
            except Exception:
                try:
                    rev_result_obj = run_reverse_generation(backup_model)
                    st.session_state['reverse_parsed_predictions'] = rev_result_obj.model_dump()
                except Exception:
                    st.error("System failure.")

    if st.session_state['reverse_parsed_predictions']:
        result = st.session_state['reverse_parsed_predictions']
        st.markdown("<br><hr style='border-color: #1a1a1c;'><br>", unsafe_allow_html=True)
        st.info(f"**Structural Ambiguity:** {result['behavioral_ambiguity']}")
        st.warning(f"**Specific Epistemic Gaps:** {result['specific_uncertainty']}")
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_missing, col_cannot = st.columns(2)
        with col_missing:
            st.markdown("<div class='section-title' style='font-size: 1.1rem;'>Missing Variables</div>", unsafe_allow_html=True)
            for item in result.get('missing_information', []):
                st.markdown(f"<div style='color: #a1a1aa; margin-bottom: 8px;'>• {item}</div>", unsafe_allow_html=True)
        with col_cannot:
            st.markdown("<div class='section-title' style='font-size: 1.1rem;'>Unverifiable Traits</div>", unsafe_allow_html=True)
            for item in result.get('cannot_be_inferred', []):
                st.markdown(f"<div style='color: #a1a1aa; margin-bottom: 8px;'>• {item}</div>", unsafe_allow_html=True)
                
        st.markdown("<br><br><div class='section-title'>Divergent Hypotheses</div>", unsafe_allow_html=True)
        
        hypotheses = result.get('explanations', [])[:3]
        cols = st.columns(min(len(hypotheses), 3)) if hypotheses else []
        for idx, (col, hyp) in enumerate(zip(cols, hypotheses)):
            with col:
                comp = hyp['compatibility']
                badge_class = "badge-strong" if comp == "Strong" else "badge-moderate" if comp == "Moderate" else "badge-weak"
                ed = hyp['evidence_breakdown']
                
                card_html = f"""
                <div class="premium-card" style="height: 100%; padding: 20px;">
                    <h4 class="card-title" style="font-size: 1.1rem; color: #ffffff;">{idx+1}. {hyp['explanation_name']}</h4>
                    <div class="hypothesis-badge {badge_class}">Compatibility: {comp}</div>
                    
                    <div style="margin-bottom: 24px; font-size: 0.95rem; color: #e5e5ea; line-height: 1.5;">
                        <span style="color: #4F83F5; font-weight: 600;">MECHANISM:</span> {hyp['primary_mechanism']}
                    </div>
                    
                    <div class="evidence-block evidence-observed">
                        <div class="evidence-header">Observed Action</div>
                        <div class="evidence-body">{ed['directly_supported']}</div>
                    </div>
                    
                    <div class="evidence-block evidence-interpretation">
                        <div class="evidence-header">Plausible Interpretation</div>
                        <div class="evidence-body">{ed['interpretation']}</div>
                    </div>
                    
                    <div class="evidence-block evidence-speculation">
                        <div class="evidence-header">Required Speculation</div>
                        <div class="evidence-body">{ed['speculation']}</div>
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)


# ==========================================
# TAB 3: SCIENCE & METHODOLOGY
# ==========================================
with tab3:
    st.markdown("<div class='section-title' style='border: none;'>Theoretical Frameworks</div>", unsafe_allow_html=True)
    st.markdown("<div style='color: #8e8e93; font-size: 1rem; margin-bottom: 30px;'>Explore the sociological, psychological, and astro-acoustic principles anchoring this predictive engine.</div>", unsafe_allow_html=True)
    
    concept_choice = st.selectbox(
        "Select an investigative concept:",
        [
            "Astro-Acoustics & Multisensory Science Communication",
            "Equifinality (The One-to-Many Problem)", 
            "The HEXACO Personality Model", 
            "Cognitive Load & Working Memory", 
            "Sensory Processing & Overload", 
            "Custom (Ask your own question)"
        ]
    )
    
    custom_concept = ""
    if concept_choice == "Custom (Ask your own question)":
        custom_concept = st.text_input("Enter a custom concept:")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Synthesize Knowledge Brief", type="primary"):
        target_concept = custom_concept if concept_choice == "Custom (Ask your own question)" else concept_choice
        
        if not target_concept:
            st.warning("Input required to generate synthesis.")
        else:
            explain_prompt = f"""
            You are an expert sociologist and science communicator specializing in human resilience, astronomy communication, and astro-acoustics.
            Explain the concept of "{target_concept}" comprehensively and accessibly.
            Structure:
            1. **Core Definition**
            2. **Mechanics in Real-World Behavior / Outreach**
            3. **Practical Application in Science Engagement**
            """
            
            with st.spinner(f"Compiling research synthesis..."):
                try:
                    explanation = client.models.generate_content(model=primary_model, contents=explain_prompt)
                    st.markdown(f'<div class="premium-card" style="margin-top: 20px; padding: 40px; line-height: 1.7; font-size: 1.05rem;">{explanation.text}</div>', unsafe_allow_html=True)
                except Exception:
                    try:
                        explanation = client.models.generate_content(model=backup_model, contents=explain_prompt)
                        st.markdown(f'<div class="premium-card" style="margin-top: 20px; padding: 40px; line-height: 1.7; font-size: 1.05rem;">{explanation.text}</div>', unsafe_allow_html=True)
                    except Exception:
                        st.error("Knowledge retrieval failed.")


# --- PERMANENT FOOTER ---
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; color: #52525b; font-size: 0.85em; border-top: 1px solid #1a1a1c; padding-top: 30px; margin-top: 20px;'>
        <p style='margin-bottom: 8px; color: #71717a;'><b>Behavioral Scenario Lab</b> • Predictive Generative Architecture</p>
        <p style='max-width: 800px; margin: 0 auto; line-height: 1.5; margin-bottom: 16px;'><em>Strictly for exploratory, theoretical, and educational simulation. Predictions are synthetic generative estimates based on heuristic parameters and do not establish causal, clinical, or psychological facts regarding actual individuals. Behavioral equifinality guarantees that multiple distinct psychological architectures can output identical behaviors.</em></p>
        <p style='color: #4a4a52; font-size: 0.95em; font-weight: 500; letter-spacing: 0.02em;'>
            Designed & Engineered by Nikolai de Silva &nbsp;&nbsp;•&nbsp;&nbsp; &copy; 2026 <a href="https://ninolades.com" target="_blank" style="color: #4a4a52; text-decoration: none; border-bottom: 1px solid transparent; padding-bottom: 1px; transition: all 0.2s ease;" onmouseover="this.style.color='#4F83F5'; this.style.borderBottom='1px solid #4F83F5';" onmouseout="this.style.color='#4a4a52'; this.style.borderBottom='1px solid transparent';">ninolades.com</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
