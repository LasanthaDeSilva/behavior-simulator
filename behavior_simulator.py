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

# Inject Custom CSS for Premium Dark UI
st.markdown("""
<style>
    /* Global Styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
        background-color: #0b0c10;
        color: #e2e2e2;
    }
    
    /* Hide Streamlit default branding/menus */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Layout Adjustments */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
        font-weight: 500;
        letter-spacing: -0.02em;
    }
    
    /* Typography: Header */
    .lab-header {
        font-size: 2.5rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #ffffff;
        margin-bottom: 0.2rem;
        text-align: center;
    }
    .lab-subheader {
        font-size: 1.1rem;
        font-weight: 300;
        color: #8892b0;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    /* Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 1px solid #1e222a;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 0;
        color: #8892b0;
        font-weight: 500;
        font-size: 1rem;
    }
    .stTabs [aria-selected="true"] {
        color: #4A90E2 !important;
        border-bottom-color: #4A90E2 !important;
    }
    
    /* Custom Cards */
    .premium-card {
        background-color: #121419;
        border: 1px solid #1e222a;
        border-radius: 8px;
        padding: 24px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    .card-title {
        font-size: 0.85rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #8892b0;
        margin-bottom: 20px;
        border-bottom: 1px solid #1e222a;
        padding-bottom: 8px;
    }
    
    /* Inputs & Selects */
    .stTextInput > div > div > input, 
    .stTextArea > div > div > textarea, 
    .stSelectbox > div > div > div {
        background-color: #0b0c10 !important;
        border: 1px solid #2a2f3a !important;
        color: #e2e2e2 !important;
        border-radius: 4px;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #1a1e27;
        color: #ffffff;
        border: 1px solid #2a2f3a;
        border-radius: 6px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        letter-spacing: 0.05em;
        transition: all 0.2s ease-in-out;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #4A90E2;
        border-color: #4A90E2;
        color: #ffffff;
    }
    
    /* Results Progress Bar */
    .plausibility-track {
        background-color: #1e222a;
        height: 6px;
        border-radius: 3px;
        margin: 8px 0 16px 0;
        overflow: hidden;
    }
    .plausibility-fill {
        background-color: #4A90E2;
        height: 100%;
        border-radius: 3px;
    }
    
    /* Uncertainty Status */
    .status-indicator {
        border-left: 3px solid #e2a04a;
        padding-left: 16px;
        margin: 20px 0;
        background-color: #16181d;
        padding: 16px;
        border-radius: 0 6px 6px 0;
    }
    .status-title {
        font-size: 0.75rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #e2a04a;
        margin-bottom: 4px;
    }
    
    /* Evidence Grid (Tab 2) */
    .evidence-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-top: 16px;
    }
    .evidence-box {
        background-color: #0b0c10;
        border: 1px solid #1e222a;
        padding: 12px;
        border-radius: 6px;
        font-size: 0.85rem;
    }
    .evidence-box strong {
        display: block;
        font-size: 0.7rem;
        letter-spacing: 0.1em;
        margin-bottom: 6px;
        color: #8892b0;
    }
    .obs-tag { color: #4CAF50; }
    .int-tag { color: #4A90E2; }
    .spec-tag { color: #e2a04a; }
</style>
""", unsafe_allow_html=True)


# Initialize Session State early for stability
for key in ['parsed_predictions', 'reverse_parsed_predictions', 'last_sim', 'last_situation', 'last_chat_response', 'last_config', 'simulation_id']:
    if key not in st.session_state:
        st.session_state[key] = None

# Header Block
st.markdown('<div class="lab-header">BEHAVIORAL SCENARIO LAB</div>', unsafe_allow_html=True)
st.markdown('<div class="lab-subheader">Explore how traits, states, and context interact to produce different behavioral pathways.</div>', unsafe_allow_html=True)


# Sidebar for API Key & Model Settings
with st.sidebar:
    st.markdown("<h3 style='color:#fff; font-weight:400; font-size:1.2rem;'>System Settings</h3>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color: #1e222a;'>", unsafe_allow_html=True)
    
    model_choice = st.selectbox(
        "Primary AI Engine",
        [
            "Gemini 3.6 Flash (Fast & Capable - Default)", 
            "Gemini 3.1 Pro (Heavy Reasoning)", 
            "Gemini 3.5 Flash-Lite (Ultra Fast)"
        ],
        index=0
    )

    # Set the active model and backup engine based on selection
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
    st.warning("Please enter your Gemini API Key in the sidebar or configure your secrets.")
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
tab1, tab2, tab3 = st.tabs(["FORWARD PREDICTOR", "COMPETING EXPLANATIONS", "SCIENCE & METHODOLOGY"])


# ==========================================
# TAB 1: FORWARD PREDICTOR
# ==========================================
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")
    
    with col_left:
        with st.container():
            st.markdown('<div class="premium-card"><div class="card-title">PROFILE</div>', unsafe_allow_html=True)
            st.markdown("<span style='font-size:0.85rem; color:#8892b0;'>Illustrative HEXACO Parameters (Hypothetical simulation inputs)</span>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            h = st.slider("Honesty–Humility", 0, 100, 20, help="Sincerity, fairness, modesty")
            e = st.slider("Emotionality", 0, 100, 80, help="Fearfulness, sensitivity to threat, attachment")
            x = st.slider("Extraversion", 0, 100, 50, help="Social self-esteem, sociability")
            a = st.slider("Agreeableness", 0, 100, 40, help="Forgiveness, gentleness, flexibility")
            c = st.slider("Conscientiousness", 0, 100, 90, help="Organization, diligence, impulse control")
            o = st.slider("Openness to Experience", 0, 100, 15, help="Aesthetic appreciation, inquisitiveness")
            st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        with st.container():
            st.markdown('<div class="premium-card"><div class="card-title">STATES & MODIFIERS</div>', unsafe_allow_html=True)
            
            sensory = st.selectbox("Sensory Responsiveness", ["Lower hypothetical reactivity", "Moderate hypothetical reactivity", "Higher hypothetical reactivity"], index=1)
            sensory_domains = st.multiselect("Relevant Sensory Domains", ["Auditory", "Visual", "Tactile", "Olfactory", "Gustatory", "Vestibular"])
            masking = st.selectbox("Behavioral Masking Tendency", ["None (Natural expression)", "Moderate", "High (Heavy camouflage)"])
            stimming = st.multiselect("Stimming Tendency", ["None", "Fidgeting", "Pacing", "Auditory stimming", "Tactile stimming", "Vocal scripting"], default=["None"])
            reward_sensitivity = st.selectbox("Reward Sensitivity / Novelty Seeking", ["Low (Predictable options)", "Medium (Balanced)", "High (Reward sensitive)"], index=2)
            state_trait = st.selectbox("Current State", ["High stress/arousal", "Relaxed/Calm", "Fatigued/Burnout", "Baseline"])
            cognitive_load = st.selectbox("Cognitive Load", ["Low (Clear headed)", "Medium (Busy)", "High (Distracted/Overwhelmed)"])
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="premium-card"><div class="card-title">SCENARIO</div>', unsafe_allow_html=True)
    extra_details = st.text_input("Background Details", "Late for an important job interview.")
    situation = st.text_area("The Situation (External Circumstances)", "Finds a wallet with $500 cash in a loud, crowded subway station.", height=100)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("RUN BEHAVIORAL SIMULATION →", type="primary"):
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
            "Background Details": extra_details,
            "Situation": situation
        }
        
        prompt = f"""
        You are an AI generating plausible behavioral outcomes given hypothetical traits, states, and contexts, with uncertainty. 
        
        CRITICAL RULES:
        1. Treat all variables strictly as hypothetical simulation parameters. Not measurements of a real person.
        2. Never infer clinical diagnoses (autism, ADHD, personality disorders, etc.).
        3. Personality traits should be treated as probabilistic tendencies whose relevance depends on context.
        4. A rationale is an explanatory hypothesis, not evidence that the specified trait caused the behavior.
        5. Separate concepts clearly: Traits, States, and Context.
        6. Do not force every modifier to matter.
        7. Selected Sensory Domains are hypotheses: YOU must determine if the context actually triggers them.
        8. Masking tendency ONLY affects the simulation when context involves social monitoring.
        9. Stimming tendency DOES NOT automatically imply anxiety or distress.
        10. High cognitive load reduces working-memory capacity. DO NOT say it causes a loss of logic.
        11. Include unknown factors (learning history, culture) as limits on certainty.
        12. uncertainty_level must be exactly one of: Low, Moderate, High.
        13. 3 predicted actions MUST be meaningfully different pathways.
        
        TRAITS: HEXACO (H:{h}, E:{e}, X:{x}, A:{a}, C:{c}, O:{o}), Reward: {reward_sensitivity}, Sensory: {sensory}
        STATES & MODIFIERS: State: {state_trait}, Load: {cognitive_load}, Masking: {masking}, Stimming: {stimming}, Domains: {sensory_domains}
        CONTEXT: Background: {extra_details}, Situation: {situation}
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
            
        with st.spinner("Simulating behavioral pathways..."):
            try:
                result_obj, raw_text = run_forward_generation(primary_model)
                st.session_state['parsed_predictions'] = result_obj.model_dump()
                st.session_state['last_sim'] = raw_text
                st.session_state['last_situation'] = situation
            except Exception as primary_error:
                try:
                    result_obj, raw_text = run_forward_generation(backup_model)
                    st.session_state['parsed_predictions'] = result_obj.model_dump()
                    st.session_state['last_sim'] = raw_text
                    st.session_state['last_situation'] = situation
                except Exception as backup_error:
                    st.error("Simulation temporarily unavailable. Please try again.")
                    st.stop()

    # ==========================================
    # RENDER PREDICTIONS PERSISTENTLY
    # ==========================================
    if not st.session_state.get('parsed_predictions'):
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px; color: #555;">
            <p style="font-size: 0.9rem; letter-spacing: 0.1em; text-transform: uppercase;">Ready to Simulate</p>
            <p style="font-size: 0.85rem;">Configure a hypothetical profile and scenario to explore possible behavioral pathways.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        result = st.session_state['parsed_predictions']
        
        st.markdown('<br><div class="card-title" style="color:#ffffff;">SIMULATION RESULTS</div>', unsafe_allow_html=True)
        
        # Metadata / Uncertainty Block
        meta_html = f"""
        <div style="display: flex; gap: 20px; margin-bottom: 24px;">
            <div style="flex: 1; background: #121419; padding: 16px; border: 1px solid #1e222a; border-radius: 6px;">
                <div style="font-size: 0.75rem; color: #8892b0; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em;">Modifier Relevance</div>
                <div style="font-size: 0.85rem; color: #d0d0d0; line-height: 1.5;">{result['modifier_relevance']}</div>
            </div>
            <div style="flex: 1; background: #16181d; padding: 16px; border-left: 3px solid #e2a04a; border-radius: 0 6px 6px 0;">
                <div style="font-size: 0.75rem; color: #e2a04a; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em;">Uncertainty: {result['uncertainty_level']}</div>
                <div style="font-size: 0.85rem; color: #d0d0d0; line-height: 1.5;">{result['uncertainty_reason']}</div>
            </div>
        </div>
        """
        st.markdown(meta_html, unsafe_allow_html=True)
    
        # Plausible Actions
        predictions = result.get('predictions', [])
        percentages = calculate_normalized_percentages(predictions)
        
        for idx, action in enumerate(predictions):
            pct = percentages[idx]
            card_html = f"""
            <div class="premium-card" style="padding: 24px;">
                <div style="display: flex; align-items: baseline; gap: 12px; margin-bottom: 8px;">
                    <span style="font-size: 1.2rem; color: #4A90E2; font-weight: 600;">0{idx+1}</span>
                    <h3 style="margin: 0; font-size: 1.1rem;">{action['action']}</h3>
                </div>
                
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #8892b0; margin-top: 16px;">
                    <span>Relative Plausibility Weight — not an empirical probability</span>
                    <span>{pct}%</span>
                </div>
                <div class="plausibility-track">
                    <div class="plausibility-fill" style="width: {pct}%;"></div>
                </div>
                
                <div style="font-size: 0.9rem; color: #b0b0b0; line-height: 1.6;">
                    <strong style="color: #ffffff;">Rationale:</strong> {action['rationale']}
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

        # === COUNTERFACTUAL LAB ===
        st.markdown('<br><br><div class="card-title" style="color:#ffffff;">COUNTERFACTUAL LAB</div>', unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.9rem; color:#8892b0;'>Change one condition and explore how the behavioral landscape shifts.</p>", unsafe_allow_html=True)

        with st.form("chat_form"):
            query = st.text_input("Test a scenario change:", placeholder="What would happen if the environment became quiet?")
            submit_q = st.form_submit_button("EXPLORE CHANGE →")
    
            if submit_q:
                st.session_state["last_chat_response"] = None 

            if submit_q and query:
                config_str = json.dumps(st.session_state['last_config'], indent=2)
                chat_prompt = f"""
                You are continuing a behavioral simulation. 
                ORIGINAL CONFIGURATION: {config_str}
                PREVIOUS AI PREDICTIONS: {st.session_state['last_sim']}
                USER HYPOTHESIS/REQUEST: {query}
                
                CRITICAL RULES:
                1. Identify exactly what variable or context detail the user wants to change.
                2. Treat all other original configuration parameters as constant.
                3. Generate 3 meaningfully different new predicted actions based on this modified state.
                4. Provide a comparison summary explaining how the new outcomes differ from the baseline.
                5. If user changes situation, modify only relevant context variables.
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

                with st.spinner("Simulating counterfactual shift..."):
                    try:
                        cf_obj = run_counterfactual_generation(primary_model)
                        st.session_state['last_chat_response'] = cf_obj.model_dump()
                    except Exception:
                        try:
                            cf_obj = run_counterfactual_generation(backup_model)
                            st.session_state['last_chat_response'] = cf_obj.model_dump()
                        except Exception:
                            st.error("Failed to generate counterfactual response.")

        if st.session_state['last_chat_response']:
            cf_data = st.session_state['last_chat_response']
            
            comp_html = f"""
            <div style="display: flex; gap: 20px; margin: 24px 0;">
                <div style="flex: 1; background: #121419; padding: 20px; border: 1px solid #1e222a; border-radius: 6px;">
                    <div style="font-size: 0.75rem; color: #8892b0; margin-bottom: 8px; text-transform: uppercase;">IDENTIFIED CHANGE</div>
                    <div style="font-size: 0.95rem; color: #4A90E2;">{cf_data['identified_variable_changed']}</div>
                </div>
                <div style="flex: 2; background: #121419; padding: 20px; border: 1px solid #1e222a; border-radius: 6px;">
                    <div style="font-size: 0.75rem; color: #8892b0; margin-bottom: 8px; text-transform: uppercase;">COMPARISON TO BASELINE</div>
                    <div style="font-size: 0.9rem; color: #d0d0d0; line-height: 1.5;">{cf_data['comparison_summary']}</div>
                </div>
            </div>
            """
            st.markdown(comp_html, unsafe_allow_html=True)
            
            cf_predictions = cf_data.get('new_predictions', [])
            cf_percentages = calculate_normalized_percentages(cf_predictions)
            
            for idx, action in enumerate(cf_predictions):
                pct = cf_percentages[idx]
                card_html = f"""
                <div class="premium-card" style="padding: 20px; border-left: 3px solid #4A90E2;">
                    <div style="display: flex; align-items: baseline; gap: 12px; margin-bottom: 8px;">
                        <span style="font-size: 1rem; color: #4A90E2; font-weight: 600;">NEW 0{idx+1}</span>
                        <h4 style="margin: 0;">{action['action']}</h4>
                    </div>
                    <div class="plausibility-track" style="height: 4px; margin: 12px 0;">
                        <div class="plausibility-fill" style="width: {pct}%;"></div>
                    </div>
                    <div style="font-size: 0.85rem; color: #b0b0b0; line-height: 1.5;">
                        <strong style="color: #fff;">Rationale:</strong> {action['rationale']}
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)


# ==========================================
# TAB 2: BEHAVIOR -> COMPETING EXPLANATIONS
# ==========================================
with tab2:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">INVESTIGATION PARAMETERS</div>', unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.9rem; color:#8892b0; margin-bottom:20px;'>One behavior can emerge from multiple mechanisms. Explore competing interpretations without assuming a single cause.</p>", unsafe_allow_html=True)
    
    observed_action = st.text_area("OBSERVED BEHAVIOR", "Grabbed the cash immediately, threw the wallet onto the tracks, and ran onto the train.", height=80)
    rev_situation = st.text_area("SITUATION CONTEXT", "Finds a lost wallet containing $500 cash in a crowded subway station.", height=80)
    known_context = st.text_input("KNOWN CONTEXT (Optional)", "Late for work")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("ANALYZE BEHAVIOR →", type="primary"):
        st.session_state["reverse_parsed_predictions"] = None
        
        prompt = f"""
        You are an AI performing a behavioral analysis to address the "One-to-Many" behavioral problem.
        Your goal is to answer: "What distinct mechanisms could each plausibly produce this behavior, prioritizing non-personality factors first?"
        
        CRITICAL RULES:
        1. Provide exactly 3 genuinely competing explanations.
        2. A single behavior is never sufficient evidence for a broad personality trait.
        3. Actively try to explain without personality first (e.g., misunderstanding, time pressure, culture).
        4. Use qualitative HEXACO ranges ONLY when behavior provides clear evidence. Otherwise output "Insufficient information".
        5. Never infer clinical diagnoses.
        
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

        with st.spinner("Generating competing explanations..."):
            try:
                rev_result_obj = run_reverse_generation(primary_model)
                st.session_state['reverse_parsed_predictions'] = rev_result_obj.model_dump()
            except Exception:
                try:
                    rev_result_obj = run_reverse_generation(backup_model)
                    st.session_state['reverse_parsed_predictions'] = rev_result_obj.model_dump()
                except Exception:
                    st.error("Analysis temporarily unavailable.")

    # ==========================================
    # RENDER EXPLANATIONS PERSISTENTLY
    # ==========================================
    if st.session_state['reverse_parsed_predictions']:
        result = st.session_state['reverse_parsed_predictions']
        
        # Epistemological Block
        unc_html = f"""
        <div style="background: #121419; border: 1px solid #1e222a; border-radius: 8px; padding: 24px; margin: 30px 0;">
            <div style="font-size: 0.75rem; color: #8892b0; margin-bottom: 12px; text-transform: uppercase;">Epistemological Limits</div>
            <p style="font-size: 0.9rem; color: #d0d0d0;"><strong>Ambiguity:</strong> {result['behavioral_ambiguity']}</p>
            <p style="font-size: 0.9rem; color: #d0d0d0;"><strong>Uncertainty:</strong> {result['specific_uncertainty']}</p>
            <div style="display: flex; gap: 20px; margin-top: 20px;">
                <div style="flex: 1;">
                    <div style="font-size: 0.75rem; color: #e2a04a; margin-bottom: 8px;">MISSING INFORMATION</div>
                    <ul style="font-size: 0.85rem; color: #a0a0a0; padding-left: 15px;">
                        {"".join(f"<li>{item}</li>" for item in result.get('missing_information', []))}
                    </ul>
                </div>
                <div style="flex: 1;">
                    <div style="font-size: 0.75rem; color: #e45b5b; margin-bottom: 8px;">CANNOT BE INFERRED</div>
                    <ul style="font-size: 0.85rem; color: #a0a0a0; padding-left: 15px;">
                        {"".join(f"<li>{item}</li>" for item in result.get('cannot_be_inferred', []))}
                    </ul>
                </div>
            </div>
        </div>
        """
        st.markdown(unc_html, unsafe_allow_html=True)
        
        st.markdown('<div class="card-title" style="color:#ffffff; margin-top:40px;">COMPETING HYPOTHESES</div>', unsafe_allow_html=True)
        
        hypotheses = result.get('explanations', [])[:3]

        for idx, hyp in enumerate(hypotheses):
            comp = hyp['compatibility']
            color = "#4CAF50" if comp == "Strong" else "#e2a04a" if comp == "Moderate" else "#e45b5b"
            ed = hyp['evidence_breakdown']
            h_data = hyp['hexaco']
            
            hyp_html = f"""
            <div class="premium-card">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e222a; padding-bottom: 15px; margin-bottom: 15px;">
                    <div style="font-size: 1.1rem; color: #ffffff; font-weight: 500;">0{idx+1} — {hyp['explanation_name']}</div>
                    <div style="font-size: 0.8rem; background: #16181d; padding: 4px 10px; border-radius: 4px; border: 1px solid {color}; color: {color};">COMPATIBILITY: {comp.upper()}</div>
                </div>
                
                <div style="font-size: 0.9rem; color: #d0d0d0; margin-bottom: 20px;">
                    <strong style="color:#fff;">Mechanism:</strong> {hyp['primary_mechanism']}
                </div>
                
                <div style="display: flex; gap: 15px; margin-bottom: 20px; font-size: 0.85rem;">
                    <div style="flex:1; background: #0b0c10; padding: 12px; border-radius: 6px;">
                        <div style="color: #8892b0; font-size: 0.7rem; margin-bottom: 4px;">SITUATIONAL FACTORS</div>
                        <div style="color: #c0c0c0;">{hyp['situational_factors']}</div>
                    </div>
                    <div style="flex:1; background: #0b0c10; padding: 12px; border-radius: 6px;">
                        <div style="color: #8892b0; font-size: 0.7rem; margin-bottom: 4px;">TEMPORARY STATE</div>
                        <div style="color: #c0c0c0;">{hyp['temporary_state']}</div>
                    </div>
                </div>
                
                <div class="evidence-grid">
                    <div class="evidence-box">
                        <strong class="obs-tag">OBSERVED (DIRECT)</strong>
                        {ed['directly_supported']}
                    </div>
                    <div class="evidence-box">
                        <strong class="int-tag">INTERPRETED (PLAUSIBLE)</strong>
                        {ed['interpretation']}
                    </div>
                    <div class="evidence-box">
                        <strong class="spec-tag">SPECULATIVE (ASSUMED)</strong>
                        {ed['speculation']}
                    </div>
                </div>
            </div>
            """
            st.markdown(hyp_html, unsafe_allow_html=True)
            
            with st.expander("VIEW POSSIBLE HEXACO CONFIGURATION (IF APPLICABLE)"):
                st.markdown(f"""
                <div style="font-size: 0.85rem; color: #a0a0a0; padding: 10px;">
                <strong>H:</strong> {h_data['Honesty_Humility']} &nbsp;|&nbsp; 
                <strong>E:</strong> {h_data['Emotionality']} &nbsp;|&nbsp; 
                <strong>X:</strong> {h_data['Extraversion']}<br>
                <strong>A:</strong> {h_data['Agreeableness']} &nbsp;|&nbsp; 
                <strong>C:</strong> {h_data['Conscientiousness']} &nbsp;|&nbsp; 
                <strong>O:</strong> {h_data['Openness']}
                </div>
                """, unsafe_allow_html=True)


# ==========================================
# TAB 3: SCIENCE & METHODOLOGY 
# ==========================================
with tab3:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">KNOWLEDGE CENTER</div>', unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.9rem; color:#8892b0;'>Explore the sociological and psychological concepts powering this engine.</p><br>", unsafe_allow_html=True)
    
    concept_choice = st.selectbox(
        "Select Concept",
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
        custom_concept = st.text_input("Enter behavioral concept:")
        
    st.markdown('<br>', unsafe_allow_html=True)
    
    if st.button("GENERATE EXPLANATION →", type="primary"):
        target_concept = custom_concept if concept_choice == "Custom (Ask your own question)" else concept_choice
        
        if not target_concept:
            st.warning("Please enter or select a concept.")
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
            
            with st.spinner("Synthesizing research..."):
                try:
                    explanation = client.models.generate_content(model=primary_model, contents=explain_prompt)
                    st.markdown(f'<div style="background:#121419; padding:30px; border-radius:8px; border:1px solid #1e222a; margin-top:20px; line-height:1.6; color:#d0d0d0;">{explanation.text}</div>', unsafe_allow_html=True)
                except Exception:
                    try:
                        explanation = client.models.generate_content(model=backup_model, contents=explain_prompt)
                        st.markdown(f'<div style="background:#121419; padding:30px; border-radius:8px; border:1px solid #1e222a; margin-top:20px; line-height:1.6; color:#d0d0d0;">{explanation.text}</div>', unsafe_allow_html=True)
                    except Exception:
                        st.error("Knowledge synthesis temporarily unavailable.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- PERMANENT FOOTER ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<hr style='border-color: #1e222a;'>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; color: #8892b0; font-size: 0.85em;'>
        <p><b>BEHAVIORAL SCENARIO LAB</b> | Built with Streamlit & Google Gemini</p>
        <p>⚠️ <em>Disclaimer: This application is for educational, creative, and exploratory simulation purposes only. It generates plausible behavioral outcomes based on hypothetical trait and state parameters. It does not predict real-world actions, establish causal facts, or provide medical/psychological evaluations. Multiple distinct psychological profiles can produce identical behaviors. Numerical trait values shown by the simulator are illustrative configurations, not estimates of a person's actual traits.</em></p>
    </div>
    """,
    unsafe_allow_html=True
)
