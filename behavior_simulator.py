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
    rationale: str = Field(description="Explanation of why, detailing mechanisms and specific trait/state/context interactions. Acknowledge unknown factors (e.g., learning history, culture) if relevant.")

class ForwardPrediction(BaseModel):
    modifier_relevance: str = Field(description="Analysis of which specific modifiers actually mattered here, and which were irrelevant and had negligible influence.")
    uncertainty_level: Literal["Low", "Moderate", "High"] = Field(description="Rate the uncertainty of this generation.")
    uncertainty_reason: str = Field(description="Explanation of why the generation carries this level of uncertainty (e.g., missing learning history, ambiguous context).")
    predictions: list[PredictedAction] = Field(min_length=3, max_length=3, description="Exactly 3 plausible actions, ranked by relative plausibility.")

class CounterfactualResponse(BaseModel):
    identified_variable_changed: str = Field(description="The specific variable, state, or context detail that was altered from the baseline configuration.")
    comparison_summary: str = Field(description="Explanation of how and why this specific change alters the behavioral landscape compared to the baseline.")
    new_predictions: list[PredictedAction] = Field(min_length=3, max_length=3, description="Exactly 3 new plausible actions based on the modified scenario.")


# --- Schemas for Competing Explanations (Tab 2) ---
class HexacoScores(BaseModel):
    Honesty_Humility: str = Field(description="Qualitative range (e.g., Low, Moderate, High) or 'Insufficient information'.")
    Emotionality: str = Field(description="Qualitative range (e.g., Low, Moderate, High) or 'Insufficient information'.")
    Extraversion: str = Field(description="Qualitative range (e.g., Low, Moderate, High) or 'Insufficient information'.")
    Agreeableness: str = Field(description="Qualitative range (e.g., Low, Moderate, High) or 'Insufficient information'.")
    Conscientiousness: str = Field(description="Qualitative range (e.g., Low, Moderate, High) or 'Insufficient information'.")
    Openness: str = Field(description="Qualitative range (e.g., Low, Moderate, High) or 'Insufficient information'.")

class EvidenceBreakdown(BaseModel):
    directly_supported: str = Field(description="What we actually observed in the behavior.")
    interpretation: str = Field(description="What could plausibly explain it.")
    speculation: str = Field(description="What we are assuming because information is missing.")

class CompetingExplanation(BaseModel):
    explanation_name: str = Field(description="Short title (e.g., 'Incentive-Driven', 'Situational Pressure', 'Contextual Misunderstanding').")
    compatibility: Literal["Strong", "Moderate", "Weak"] = Field(description="Compatibility with the observed behavior. No numerical scores.")
    primary_mechanism: str = Field(description="Primary driver of the behavior.")
    situational_factors: str = Field(description="What external factors could explain the behavior?")
    temporary_state: str = Field(description="What temporary internal state might matter right now?")
    possible_trait_contribution: str = Field(description="How traits might contribute, IF relevant. (May be 'None assumed').")
    hexaco: HexacoScores
    evidence_breakdown: EvidenceBreakdown

class BehaviorAnalysisResult(BaseModel):
    behavioral_ambiguity: str = Field(description="How ambiguous the behavior is. Explicitly state that multiple different profiles can produce the same behavior.")
    specific_uncertainty: str = Field(description="Why uncertain: specific factors we don't know (e.g., financial circumstances, perceived ownership, urgency) that prevent a definitive conclusion.")
    cannot_be_inferred: list[str] = Field(description="Specific broad traits, capacities, or long-term characteristics that CANNOT be inferred from this single action.")
    missing_information: list[str] = Field(description="Specific pieces of missing context (e.g., 'Prior experiences with lost property', 'Cultural norms', 'Time pressure').")
    explanations: list[CompetingExplanation] = Field(min_length=3, max_length=3, description="Exactly 3 genuinely competing explanations.")


# ==========================================
# STREAMLIT APP CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="Behavioral Scenario Lab", layout="wide")

# PREMIUM CUSTOM CSS INJECTION
premium_css = """
<style>
    /* Base Theme & Typography */
    .stApp {
        background-color: #0d0d0e;
        color: #e5e5ea;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
        font-weight: 500;
        letter-spacing: -0.015em;
    }
    
    /* Inputs, Selectboxes, Textareas */
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
    
    /* Sliders */
    .stSlider [data-baseweb="slider"] div {
        background-color: #4F83F5 !important;
    }
    
    /* Buttons */
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
    
    /* Primary Button override */
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
    
    /* Tabs */
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

    /* Info / Warning / Success Alerts (Streamlit native overrides) */
    div[data-testid="stAlert"] {
        background-color: #151517 !important;
        border: 1px solid #2a2a2e !important;
        border-radius: 8px !important;
        color: #e5e5ea !important;
    }

    /* CUSTOM HTML COMPONENTS */
    
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
st.markdown("<h1 style='font-weight: 300; margin-bottom: 0;'>Behavioral Scenario Lab</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #8e8e93; font-size: 1.1rem; margin-bottom: 30px;'>Advanced predictive modeling and sociological counterfactual analysis.</p>", unsafe_allow_html=True)

# Initialize Session State early for stability
for key in ['parsed_predictions', 'reverse_parsed_predictions', 'last_sim', 'last_situation', 'last_chat_response', 'last_config', 'simulation_id']:
    if key not in st.session_state:
        st.session_state[key] = None

# Sidebar for API Key & Model Settings
with st.sidebar:
    st.markdown("<h3 style='margin-bottom: 20px;'>Lab Configurations</h3>", unsafe_allow_html=True)
    
    model_choice = st.selectbox(
        "Primary AI Engine",
        [
            "Gemini 3.6 Flash (Fast & Capable - Default)", 
            "Gemini 3.1 Pro (Heavy Reasoning)", 
            "Gemini 3.5 Flash-Lite (Ultra Fast)"
        ],
        index=0,
        help="Select the primary engine. If it hits a rate limit, the app automatically fails over to a backup model."
    )

    # Set the active model and backup engine based on selection
    if "3.1 Pro" in model_choice:
        primary_model = "gemini-3.1-pro"
        backup_model = "gemini-3.6-flash"
    elif "3.5 Flash-Lite" in model_choice:
        primary_model = "gemini-3.5-flash-lite"
        backup_model = "gemini-3.6-flash"
    else: 
        # Default: 3.6 Flash is selected
        primary_model = "gemini-3.6-flash"
        backup_model = "gemini-3.5-flash-lite" # Safety net if 3.6 goes down


    api_key = st.secrets.get("GEMINI_API_KEY", "")
    
    if not api_key:
        api_key = st.text_input("Enter Gemini API Key:", type="password")

if not api_key:
    st.warning("Please enter your Gemini API Key in the sidebar or configure your secrets to begin.")
    st.stop()

# Initialize Client
client = genai.Client(api_key=api_key)


# ==========================================
# HELPER FOR WEIGHT NORMALIZATION
# ==========================================
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


# Create Tabs
tab1, tab2, tab3 = st.tabs(["Forward Predictor", "Competing Explanations", "Science & Methodology"])


# ==========================================
# TAB 1: FORWARD PREDICTOR
# ==========================================
with tab1:
    
    # Input Configuration Sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='section-title'>Profile Configuration</div>", unsafe_allow_html=True)
        st.markdown("<div style='color: #8e8e93; font-size: 0.9rem; margin-bottom: 20px;'>Illustrative HEXACO Parameters (0–100). Hypothetical simulation inputs, not clinical measurements.</div>", unsafe_allow_html=True)
        
        h = st.slider("Honesty-Humility", 0, 100, 20, help="Sincerity, fairness, modesty, and reduced tendency toward status/material exploitation.")
        e = st.slider("Emotionality", 0, 100, 80, help="Fearfulness, sensitivity to threat, dependence, sentimentality, and emotional attachment.")
        x = st.slider("Extraversion", 0, 100, 50, help="Social self-esteem, social boldness, sociability, and liveliness.")
        a = st.slider("Agreeableness", 0, 100, 40, help="Forgiveness, gentleness, flexibility, and patience toward others.")
        c = st.slider("Conscientiousness", 0, 100, 90, help="Organization, diligence, perfectionism, prudence, and impulse control.")
        o = st.slider("Openness to Experience", 0, 100, 15, help="Aesthetic appreciation, inquisitiveness, creativity, and unconventionality.")

    with col2:
        st.markdown("<div class='section-title'>States & Modifiers</div>", unsafe_allow_html=True)
        
        sensory = st.selectbox(
            "Sensory Responsiveness", 
            ["Lower hypothetical sensory reactivity", "Moderate hypothetical sensory reactivity", "Higher hypothetical sensory reactivity"], 
            index=1,
            help="A simulation parameter representing sensory processing tendency."
        )
        
        sensory_domains = st.multiselect(
            "Relevant Sensory Domains", 
            ["Auditory", "Visual", "Tactile", "Olfactory", "Gustatory", "Vestibular"]
        )

        masking = st.selectbox(
           "Behavioral Masking Tendency", 
           ["None (Natural expression)", "Moderate", "High (Heavy camouflage)"]
        )

        stimming = st.multiselect(
           "Self-Regulation Tendency", 
           ["None", "Fidgeting", "Pacing", "Auditory stimming", "Tactile stimming", "Vocal scripting"],
           default=["None"]
        )
        
        reward_sensitivity = st.selectbox("Reward Sensitivity / Novelty Seeking", ["Low (Prefers predictable/stable options)", "Medium (Balanced)", "High (Sensitive to novelty, reward, stimulation)"], index=2)
        
        state_trait = st.selectbox("Current Internal State", ["High stress/arousal", "Relaxed/Calm", "Fatigued/Burnout", "Baseline"])
        
        cognitive_load = st.selectbox("Cognitive Load", ["Low (Clear headed)", "Medium (Busy)", "High (Distracted/Overwhelmed)"])
        
    st.markdown("<br><div class='section-title'>Scenario Context</div>", unsafe_allow_html=True)
    extra_details = st.text_input("Background Context", "Late for an important job interview.")
    situation = st.text_area("Immediate Situation", "Finds a wallet with $500 cash in a loud, crowded subway station.", height=100)
    
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Run Behavioral Simulation", type="primary", use_container_width=True):
        # Clear previous state
        st.session_state["parsed_predictions"] = None
        st.session_state["last_chat_response"] = None
        st.session_state["simulation_id"] = str(uuid.uuid4())[:8]
        
        # Store full configuration for accurate counterfactual analysis later
        st.session_state['last_config'] = {
            "HEXACO": {"H": h, "E": e, "X": x, "A": a, "C": c, "O": o},
            "Sensory Responsiveness": sensory,
            "Sensory Domains": sensory_domains,
            "Masking Tendency": masking,
            "Stimming Tendency": stimming,
            "Reward Sensitivity": reward_sensitivity,
            "Current State": state_trait,
            "Cognitive Load": cognitive_load,
            "Background Details": extra_details,
            "Situation": situation
        }
        
        prompt = f"""
        You are an AI generating plausible behavioral outcomes given hypothetical traits, states, and contexts, with uncertainty. 
        
        CRITICAL RULES:
        1. Treat all personality (HEXACO values), sensory, reward-sensitivity, and state variables strictly as hypothetical simulation parameters. They are not measurements of an actual person.
        2. Never infer autism, ADHD, anxiety disorders, personality disorders, trauma disorders, or any other clinical diagnosis from these parameters or behaviors.
        3. Do not treat any individual HEXACO dimension as deterministically producing an action. Personality traits should be treated as probabilistic tendencies whose relevance depends on context and interacting variables.
        4. A rationale is an explanatory hypothesis, not evidence that the specified trait caused the behavior.
        5. Separate concepts clearly: Traits (stable tendencies), States (temporary conditions), and Context (external circumstances).
        6. Do not force every modifier to matter. Irrelevant sensory, masking, reward, or state variables MUST be allowed to have negligible influence.
        7. Selected Sensory Domains are hypotheses: YOU must determine if the context actually triggers them.
        8. Masking tendency ≠ currently masking. It should ONLY affect the simulation when the context plausibly involves social suppression/monitoring.
        9. Stimming tendency ≠ guaranteed stimming. Do not automatically interpret it as anxiety or distress.
        10. High cognitive load reduces working-memory capacity and increases reliance on habitual/simplified strategies. DO NOT say it causes someone to "lose logic" or switch to "raw instinct."
        11. Include unknown factors: Explicitly acknowledge that learning history, goals, culture, and prior experiences may be missing but influence behavior.
        12. uncertainty_level must be exactly one of: Low, Moderate, High.
        13. These are scenario-consistent behavioral hypotheses, not predictions of what a real person will actually do.
        14. The 3 predicted actions MUST be meaningfully different from each other, representing different plausible pathways, not just three slight variations of the exact same action.
        
        TRAITS (Hypothetical):
        - HEXACO: H:{h}, E:{e}, X:{x}, A:{a}, C:{c}, O:{o}
        - Reward Sensitivity / Novelty Seeking: {reward_sensitivity}
        - Hypothetical Sensory Responsiveness: {sensory}
        
        STATES & MODIFIERS (Hypothetical):
        - Current State: {state_trait}
        - Cognitive Load: {cognitive_load}
        - Masking Tendency: {masking}
        - Stimming Tendency: {stimming}
        - Potentially Relevant Sensory Domains: {sensory_domains}
        
        CONTEXT:
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
                # Attempt primary engine (encompassing generation and schema validation)
                result_obj, raw_text = run_forward_generation(primary_model)
                st.session_state['parsed_predictions'] = result_obj.model_dump()
                st.session_state['last_sim'] = raw_text
                st.session_state['last_situation'] = situation
            except Exception as primary_error:
                st.warning("Primary engine or validation failed. Retrying with the backup engine...")
                try:
                    # Attempt backup engine
                    result_obj, raw_text = run_forward_generation(backup_model)
                    st.session_state['parsed_predictions'] = result_obj.model_dump()
                    st.session_state['last_sim'] = raw_text
                    st.session_state['last_situation'] = situation
                except Exception as backup_error:
                    st.error("System failure: Both AI engines are currently unavailable or failed structured validation.")
                    st.stop()

    # ==========================================
    # RENDER PREDICTIONS AND CHAT PERSISTENTLY
    # ==========================================
    if st.session_state['parsed_predictions']:
        result = st.session_state['parsed_predictions']
        
        st.markdown("<br><hr style='border-color: #1a1a1c;'><br>", unsafe_allow_html=True)
        st.markdown(f"<div style='color: #4F83F5; font-size: 0.9rem; letter-spacing: 0.05em; margin-bottom: 24px; text-transform: uppercase;'>Simulation ID: {st.session_state['simulation_id']}</div>", unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.info(f"**Modifier Relevance:**\n\n{result['modifier_relevance']}")
        with col_b:
            st.warning(f"**Uncertainty Level:** {result['uncertainty_level']}\n\n{result['uncertainty_reason']}")
    
        st.markdown("<br><div class='section-title'>Predicted Plausible Behaviors</div>", unsafe_allow_html=True)
        
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

        # === FOLLOW-UP CHAT FEATURE ===
        st.markdown("<br><div class='section-title'>Counterfactual Lab</div>", unsafe_allow_html=True)
        st.markdown("<div style='color: #8e8e93; font-size: 0.95rem; margin-bottom: 20px;'>Isolate and modify a single variable to observe cascading behavioral changes. <em>(e.g., 'What if the environment became quiet?' or 'What if the person wasn't late?')</em></div>", unsafe_allow_html=True)

        with st.form("chat_form"):
            query = st.text_input("Modify Context or State", placeholder="e.g., Reduce cognitive load to 'Low'")
            submit_q = st.form_submit_button("Run Counterfactual Simulation")
    
            if submit_q:
                st.session_state["last_chat_response"] = None 

            if submit_q and query:
                if not st.session_state.get('last_config'):
                    st.error("Please run a baseline simulation first.")
                else:
                    config_str = json.dumps(st.session_state['last_config'], indent=2)
                    
                    chat_prompt = f"""
                    You are continuing a behavioral simulation. The user wants to run a counterfactual analysis.
                    
                    ORIGINAL CONFIGURATION:
                    {config_str}
                    
                    PREVIOUS AI PREDICTIONS: {st.session_state['last_sim']}
            
                    USER HYPOTHESIS/REQUEST: {query}
                    
                    CRITICAL RULES:
                    1. Identify exactly what variable or context detail the user wants to change.
                    2. Treat all other original configuration parameters as constant.
                    3. Generate 3 meaningfully different new predicted actions based on this modified state.
                    4. Provide a comparison summary explaining how the new outcomes differ from the baseline.
                    5. Output strictly matching the requested JSON schema.
                    6. If the requested change is ambiguous or cannot be mapped to exactly one variable or context detail, do not guess. State that it is ambiguous and ask the user to specify one change. 
                    7. If the user changes the environment or situation, modify only the relevant contextual variable. Never alter HEXACO, sensory responsiveness, reward sensitivity, or other stable individual parameters unless the user explicitly requests that parameter to change. 
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
                        except Exception as e1:
                            try:
                                cf_obj = run_counterfactual_generation(backup_model)
                                st.session_state['last_chat_response'] = cf_obj.model_dump()
                            except Exception as e2:
                                st.error("System failed to construct a valid counterfactual branch.")

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
# TAB 2: BEHAVIOR -> COMPETING EXPLANATIONS
# ==========================================
with tab2:
    st.markdown("<div class='section-title' style='border: none;'>Behavioral Equifinality Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div style='color: #8e8e93; font-size: 1rem; margin-bottom: 30px; line-height: 1.5;'>This diagnostic tool isolates a single observed action and reverse-engineers multiple distinct psychological profiles and circumstantial architectures that could plausibly produce it.</div>", unsafe_allow_html=True)
    
    rev_situation = st.text_area("Environmental Context", "Finds a lost wallet containing $500 cash in a crowded subway station.", height=80)
    observed_action = st.text_area("Observed Behavior", "Grabbed the cash immediately, threw the wallet onto the tracks, and ran onto the train.", height=80)
    known_context = st.text_input("Verified Modifiers (Optional)", "Late for work")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔬 Generate Competing Hypotheses", type="primary", use_container_width=True):
        st.session_state["reverse_parsed_predictions"] = None
        
        prompt = f"""
        You are an AI performing a behavioral analysis to address the "One-to-Many" behavioral problem.
        
        Your goal is to answer: "What distinct mechanisms and psychological configurations could each plausibly produce this behavior, while recognizing that the behavior may be explained without stable personality traits?"
        
        CRITICAL RULES:
        1. Provide exactly 3 genuinely competing explanations (e.g., Hypothesis A: incentive-driven, Hypothesis B: situational-pressure-driven, Hypothesis C: stable-trait-compatible).
        2. A single behavior must never be treated as sufficient evidence for a broad personality trait. (e.g., Stealing once ≠ low Honesty-Humility as a person). It may be consistent with it, but it does not establish it.
        3. Actively try to explain the behavior without personality first. Consider alternative explanations like misunderstanding, lack of information, social pressure, time pressure, fatigue, immediate incentives, learned habits, cultural norms, or deliberate strategy.
        4. Use qualitative ranges (e.g., Low, Moderate, High) for HEXACO dimensions ONLY when the behavior provides clear evidence. If the observed behavior provides insufficient information about a dimension, explicitly output "Insufficient information" rather than inventing a range.
        5. Never infer autism, ADHD, anxiety disorders, personality disorders, trauma disorders, or any other clinical diagnosis from these parameters or behaviors.
        6. Treat all outputs as hypothesized explanations, not factual deductions.
        
        SITUATION (Context): {rev_situation}
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
            except Exception as primary_error:
                st.warning("Primary engine failed. Failing over to backup compute...")
                try:
                    rev_result_obj = run_reverse_generation(backup_model)
                    st.session_state['reverse_parsed_predictions'] = rev_result_obj.model_dump()
                except Exception as backup_error:
                    st.error("System failure: Unable to compute hypotheses.")
                    st.stop()

    # ==========================================
    # RENDER EXPLANATIONS PERSISTENTLY
    # ==========================================
    if st.session_state['reverse_parsed_predictions']:
        result = st.session_state['reverse_parsed_predictions']
        st.markdown("<br><hr style='border-color: #1a1a1c;'><br>", unsafe_allow_html=True)
        
        # Top-level ambiguity and uncertainty
        st.info(f"**Structural Ambiguity:** {result['behavioral_ambiguity']}")
        st.warning(f"**Specific Epistemic Gaps:** {result['specific_uncertainty']}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Epistemological Humility Block
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
        
        hypotheses = result.get('explanations', [])
        hypotheses = hypotheses[:3]

        cols = st.columns(min(len(hypotheses), 3)) if hypotheses else []
        for idx, (col, hyp) in enumerate(zip(cols, hypotheses)):
            with col:
                comp = hyp['compatibility']
                badge_class = "badge-strong" if comp == "Strong" else "badge-moderate" if comp == "Moderate" else "badge-weak"
                
                ed = hyp['evidence_breakdown']
                h_data = hyp['hexaco']
                
                # HTML Card injected into the column
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
                    
                    <div style="margin-top: 24px; font-size: 0.85rem; color: #8e8e93; border-top: 1px solid #2a2a2e; padding-top: 16px; line-height: 1.6;">
                        <div style="margin-bottom: 6px;"><strong style="color:#a1a1aa;">Context Factor:</strong> {hyp['situational_factors']}</div>
                        <div style="margin-bottom: 6px;"><strong style="color:#a1a1aa;">Internal State:</strong> {hyp['temporary_state']}</div>
                        <div><strong style="color:#a1a1aa;">Assumed Traits:</strong> {hyp['possible_trait_contribution']}</div>
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)


# ==========================================
# TAB 3: SCIENCE & METHODOLOGY (Gemini-Powered)
# ==========================================
with tab3:
    st.markdown("<div class='section-title' style='border: none;'>Theoretical Frameworks</div>", unsafe_allow_html=True)
    st.markdown("<div style='color: #8e8e93; font-size: 1rem; margin-bottom: 30px;'>Explore the sociological and psychological principles anchoring this predictive engine. Select a core concept or query your own to dynamically generate an accessible research brief.</div>", unsafe_allow_html=True)
    
    concept_choice = st.selectbox(
        "Select an investigative concept:",
        [
            "Equifinality (The One-to-Many Problem)", 
            "The HEXACO Personality Model", 
            "Cognitive Load & Working Memory", 
            "Sensory Processing & Overload", 
            "Behavioral Masking", 
            "Traits vs. States vs. Context", 
            "Custom (Ask your own question)"
        ]
    )
    
    custom_concept = ""
    if concept_choice == "Custom (Ask your own question)":
        custom_concept = st.text_input("Enter a custom behavioral science concept:")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Synthesize Knowledge Brief", type="primary"):
        target_concept = custom_concept if concept_choice == "Custom (Ask your own question)" else concept_choice
        
        if not target_concept:
            st.warning("Input required to generate synthesis.")
        else:
            explain_prompt = f"""
            You are an expert sociologist and author specializing in human resilience and behavioral science.
            Explain the concept of "{target_concept}" comprehensively but in a way that is highly accessible and easy for a layperson to understand.
            
            Structure your explanation logically:
            1. **The Core Definition:** A clear, simple explanation of what it is.
            2. **The Mechanics:** How this concept actually operates in real-world human behavior and structural environments.
            3. **Resilience & Reality:** A brief, relatable example of how this impacts everyday decision-making or coping.
            
            Use Markdown formatting (bolding, bullet points) for scannability. Do not use JSON formatting. Keep it engaging, scientific, and grounded in reality.
            """
            
            with st.spinner(f"Compiling research synthesis for '{target_concept}'..."):
                try:
                    # Try Primary Model
                    explanation = client.models.generate_content(
                        model=primary_model,
                        contents=explain_prompt
                    )
                    
                    st.markdown(f"""
                    <div class="premium-card" style="margin-top: 20px; padding: 40px; line-height: 1.7; font-size: 1.05rem;">
                        {explanation.text}
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e1:
                    try:
                        # Try Backup Model
                        explanation = client.models.generate_content(
                            model=backup_model,
                            contents=explain_prompt
                        )
                        st.markdown(f"""
                        <div class="premium-card" style="margin-top: 20px; padding: 40px; line-height: 1.7; font-size: 1.05rem;">
                            {explanation.text}
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e2:
                        st.error("Knowledge retrieval failed. AI engines unavailable.")



# --- PERMANENT FOOTER ---
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; color: #52525b; font-size: 0.85em; border-top: 1px solid #1a1a1c; padding-top: 30px; margin-top: 20px;'>
        <p style='margin-bottom: 8px; color: #71717a;'><b>Behavioral Scenario Lab</b> • Predictive Generative Architecture</p>
        <p style='max-width: 800px; margin: 0 auto; line-height: 1.5;'><em>Strictly for exploratory, theoretical, and educational simulation. Predictions are synthetic generative estimates based on heuristic parameters and do not establish causal, clinical, or psychological facts regarding actual individuals. Behavioral equifinality guarantees that multiple distinct psychological architectures can output identical behaviors.</em></p>
    </div>
    """,
    unsafe_allow_html=True
)
