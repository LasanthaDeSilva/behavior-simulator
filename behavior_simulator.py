import warnings
warnings.filterwarnings("ignore")

import json
import streamlit as st
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# ==========================================
# PYDANTIC SCHEMAS (FORCES JSON OUTPUTS)
# ==========================================

# Schemas for Forward Predictor
class PredictedAction(BaseModel):
    action: str = Field(description="A highly specific action the person might take.")
    raw_weight: int = Field(description="Relative weight of this outcome occurring (1-100). Do not worry about making them sum to 100, Python will normalize this.")
    rationale: str = Field(description="Explanation of why, detailing mechanisms and specific trait/state interactions.")

class ForwardPrediction(BaseModel):
    modifier_relevance: str = Field(description="Analysis of which specific modifiers (Sensory, Masking, etc.) actually mattered here, and which were irrelevant.")
    uncertainty_level: str = Field(description="Rate the uncertainty of this prediction: Low, Moderate, or High.")
    uncertainty_reason: str = Field(description="Explanation of why the prediction carries this level of uncertainty.")
    predictions: list[PredictedAction] = Field(description="Exactly 3 plausible actions, ranked by weight.")

# Schemas for Reverse Engineer
class HexacoScores(BaseModel):
    Honesty_Humility: int
    Emotionality: int
    Extraversion: int
    Agreeableness: int
    Conscientiousness: int
    Openness: int

class ProfileHypothesis(BaseModel):
    relative_plausibility_score: int = Field(description="Heuristic plausibility score (1-100) indicating how well this profile fits the action.")
    hexaco: HexacoScores
    sensory_threshold: str = Field(description="Deduced sensory threshold.")
    reward_sensitivity: str = Field(description="Deduced reward sensitivity / novelty seeking.")
    current_state: str = Field(description="Deduced temporary state (e.g., panicked, exhausted).")
    justification: str = Field(description="Explanation of why this specific mix of traits/state leads plausibly to the observed action. Can include 'Insufficient information' if highly ambiguous.")

class ReverseEngineeringResult(BaseModel):
    evidence_quality: str = Field(description="Assessment of the quality and specificity of the provided context and action.")
    behavioral_ambiguity: str = Field(description="How ambiguous the behavior is (e.g., 'Highly ambiguous, could be driven by greed OR panic').")
    hypotheses: list[ProfileHypothesis] = Field(description="Exactly 3 distinct profile hypotheses that could have caused the action.")


# ==========================================
# STREAMLIT APP CONFIGURATION
# ==========================================
st.set_page_config(page_title="Behavior Simulator", layout="wide")
st.title("🧠 AI-Assisted Behavioral Simulation")

# Sidebar for API Key & Model Settings
with st.sidebar:
    st.header("Settings")
    
    model_choice = st.selectbox(
        "Select AI Engine",
        ["Gemini 3.7 Flash (Primary)", "Gemini 3.5 Flash (Backup)"],
        help="Switch to backup if the primary model is unavailable."
    )

    if "3.7" in model_choice:
        primary_model = "gemini-3.7-flash"
        backup_model = "gemini-3.5-flash"
    else:
        primary_model = "gemini-3.5-flash"
        backup_model = "gemini-3.7-flash"

    api_key = st.secrets.get("GEMINI_API_KEY", "")
    
    if not api_key:
        api_key = st.text_input("Enter Gemini API Key:", type="password")

if not api_key:
    st.warning("Please enter your Gemini API Key in the sidebar or configure your secrets.")
    st.stop()

# Initialize Client
client = genai.Client(api_key=api_key)


# Create Tabs
tab1, tab2 = st.tabs(["🔮 Forward Predictor", "🔍 Reverse Engineer"])

# ==========================================
# TAB 1: FORWARD PREDICTOR
# ==========================================
with tab1:
    st.subheader("Configure Profile & Context")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**HEXACO Traits (0-100 Baseline)**")
        st.markdown("<br>", unsafe_allow_html=True)
        
        h = st.slider("Honesty-Humility", 0, 100, 20)
        st.caption("Measures sincerity, fairness, and modesty. High scorers are genuine; low scorers tend to be manipulative.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        e = st.slider("Emotionality", 0, 100, 80)
        st.caption("Fearfulness, anxiety sensitivity, dependence, and sentimentality.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        x = st.slider("Extraversion", 0, 100, 50)
        st.caption("Covers social boldness. High scorers thrive in crowds; low scorers prefer solitary, quiet settings.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        a = st.slider("Agreeableness", 0, 100, 40)
        st.caption("Forgiveness, patience, flexibility, and tolerance toward others.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        c = st.slider("Conscientiousness", 0, 100, 90)
        st.caption("Reflects organization and diligence. High scorers are disciplined; low scorers are impulsive.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        o = st.slider("Openness to Experience", 0, 100, 15)
        st.caption("Intellectual curiosity, creativity, unconventionality, and aesthetic/experiential engagement.")
        st.markdown("<br>", unsafe_allow_html=True)

    with col2:
        st.markdown("**Simulation Parameters (States & Modifiers)**")
        st.markdown("<br>", unsafe_allow_html=True)
        
        sensory = st.selectbox("Typical Sensory Sensitivity", ["Low (Needs intense input)", "Medium (Balanced)", "High (Easily overwhelmed)"])
        st.caption("General nervous system filtering threshold.")
        st.markdown("<br>", unsafe_allow_html=True)

        sensory_domains = st.multiselect(
            "Hypothesized Sensory Vectors", 
            ["Auditory", "Visual", "Tactile", "Olfactory", "Gustatory", "Vestibular"],
            help="Sensory domains hypothesized to be relevant. The AI will determine if the situation actually activates them."
        )
        st.markdown("<br>", unsafe_allow_html=True)

        masking = st.selectbox(
           "Behavioral Masking Tendency", 
           ["None (Natural expression)", "Moderate", "High (Heavy camouflage)"]
        )
        st.caption("Consider whether masking plausibly increases load in this particular context.")
        st.markdown("<br>", unsafe_allow_html=True)

        stimming = st.multiselect(
           "Stimming / Self-Regulation Tendency", 
           ["None", "Fidgeting", "Pacing", "Auditory stimming", "Tactile stimming", "Vocal scripting"],
           default=["None"]
        )
        st.caption("Potential self-regulatory, sensory-seeking, attention-regulating, emotional, or habitual behavior.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        reward_sensitivity = st.selectbox("Reward Sensitivity / Novelty Seeking", ["Low (Prefers predictable/stable options)", "Medium (Balanced)", "High (Sensitive to novelty, reward, stimulation)"], index=2)
        st.markdown("<br>", unsafe_allow_html=True)
        
        state_trait = st.selectbox("Current State", ["High stress/Panic", "Relaxed/Calm", "Fatigued/Burnout", "Baseline"])
        st.markdown("<br>", unsafe_allow_html=True)
        
        cognitive_load = st.selectbox("Cognitive Load", ["Low (Clear headed)", "Medium (Busy)", "High (Distracted/Overwhelmed)"])
        st.caption("High load can reduce working-memory capacity and increase reliance on habitual strategies.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("**Context**")
        extra_details = st.text_input("Extra Details", "Late for an important job interview.")
        situation = st.text_area("The Situation", "Finds a wallet with $500 cash in a loud, crowded subway station.")

    if st.button("🚀 Run Behavioral Simulation", type="primary"):
        prompt = f"""
        You are an AI performing a behavioral simulation. 
        CRITICAL RULE: Treat all personality, sensory, reward-sensitivity, and state variables as hypothetical simulation parameters. Do not interpret them as measurements of a real person's neurobiology, diagnosis, or psychological condition.
        
        PHILOSOPHY: Given these hypothetical parameters and this context, what behaviors are most plausible, what mechanisms support each possibility, and how uncertain is the prediction?
        
        Evaluate whether each modifier is relevant to the situation. Do not force irrelevant modifiers to influence the prediction. Determine whether the situation actually activates the hypothesized sensory domains. Consider whether masking plausibly increases cognitive/emotional load here; do not assume an effect when the situation doesn't involve social suppression.
        
        TRAITS:
        - HEXACO: H:{h}, E:{e}, X:{x}, A:{a}, C:{c}, O:{o}
        - Reward Sensitivity: {reward_sensitivity}
        - Typical Sensory Sensitivity: {sensory}
        
        STATES & MODIFIERS:
        - Current State: {state_trait}
        - Cognitive Load: {cognitive_load}
        - Masking Tendency: {masking}
        - Stimming Tendency: {stimming}
        - Hypothesized Sensory Domains: {sensory_domains}
        
        CONTEXT:
        - Background: {extra_details}
        - Situation: {situation}
        """
        
        with st.spinner("Simulating plausible behaviors..."):
            try:
                response = client.models.generate_content(
                    model=primary_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ForwardPrediction,
                        temperature=0.4
                    )
                )
            except Exception as e:
                st.warning("Primary engine failed. Retrying with the backup engine...")
                try:
                    response = client.models.generate_content(
                        model=backup_model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=ForwardPrediction,
                            temperature=0.4
                        )
                    )
                except Exception as backup_error:
                    st.error(f"Both AI engines are currently unavailable. Error: {str(backup_error)}")
                    st.stop()

            try:
                result = json.loads(response.text)
                st.session_state['parsed_predictions'] = result
                st.session_state['last_sim'] = response.text
                st.session_state['last_situation'] = situation
                
                if 'last_chat_response' in st.session_state:
                    del st.session_state['last_chat_response']
                    
            except Exception as parse_error:
                st.error(f"Error reading the AI output: {str(parse_error)}")

    # ==========================================
    # RENDER PREDICTIONS AND CHAT PERSISTENTLY
    # ==========================================
    if 'parsed_predictions' in st.session_state:
        result = st.session_state['parsed_predictions']
        
        st.markdown("---")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.info(f"**🔍 Modifier Relevance:** {result['modifier_relevance']}")
        with col_b:
            st.warning(f"**⚠️ Uncertainty Level:** {result['uncertainty_level']}\n\n{result['uncertainty_reason']}")
    
        st.markdown("### Plausible Actions")
        
        # Calculate percentages securely in Python
        raw_weights = [action['raw_weight'] for action in result['predictions']]
        total_weight = sum(raw_weights) if sum(raw_weights) > 0 else 1
        
        for idx, action in enumerate(result['predictions']):
            st.markdown(f"**{idx+1}. {action['action']}**")
            calculated_pct = int((action['raw_weight'] / total_weight) * 100)
            st.progress(calculated_pct / 100.0, text=f"Relative Plausibility: {calculated_pct}%")
            st.write(f"**Rationale:** {action['rationale']}")
            st.divider()

        # === FOLLOW-UP CHAT FEATURE ===
        st.markdown("### 💬 Counterfactual Analysis")
        st.caption("Try asking: *What if the environment became quiet?* or *What if the person wasn't late?*")

        with st.form("chat_form"):
            query = st.text_input("Test a scenario change:", placeholder="What if sensory load doubled?")
            submit_q = st.form_submit_button("Test Hypothesis")
    
            if submit_q and query:
                chat_prompt = f"""
                You are continuing a behavioral simulation. Treat all parameters as hypothetical.
                ORIGINAL SITUATION: {st.session_state['last_situation']}
                PREVIOUS AI PREDICTIONS: {st.session_state['last_sim']}
        
                USER HYPOTHESIS/QUESTION: {query}
        
                Provide a direct, concise, and scientifically grounded response addressing how this specific change alters the plausible behaviors. Do not use JSON formatting.
                """
        
                try:
                    chat_response = client.models.generate_content(
                        model=primary_model,
                        contents=chat_prompt
                    )
                    st.session_state['last_chat_response'] = chat_response.text
                except Exception:
                    chat_response = client.models.generate_content(
                        model=backup_model,
                        contents=chat_prompt
                    )
                    st.session_state['last_chat_response'] = chat_response.text

        if 'last_chat_response' in st.session_state and st.session_state['last_chat_response']:
            st.info(st.session_state['last_chat_response'])


# ==========================================
# TAB 2: REVERSE ENGINEER
# ==========================================
with tab2:
    st.subheader("Reverse Engineer Profile from Action")
    st.info("💡 Explores the 'One-to-Many' problem: A single action can be caused by completely different psychological profiles.")
    
    rev_situation = st.text_area("Situation Context", "Finds a lost wallet containing $500 cash in a crowded subway station.")
    observed_action = st.text_area("Observed Action", "Grabbed the cash immediately, threw the wallet onto the tracks, and ran onto the train.")
    known_context = st.text_input("Known Context (Optional)", "Late for work")
    
    if st.button("🔬 Analyze Plausible Profiles", type="primary"):
        prompt = f"""
        You are an AI performing a behavioral simulation. Address the "One-to-Many" behavioral problem.
        CRITICAL RULE: Treat all outputs as hypothetical simulation parameters, not medical diagnoses.
        
        SITUATION: {rev_situation}
        OBSERVED ACTION: {observed_action}
        KNOWN CONTEXT: {known_context}
        
        Provide the TOP 3 completely distinct profiles (combinations of HEXACO, Modifiers, and State) that plausibly explain this exact action. 
        If the behavior is highly ambiguous, state "Insufficient information to distinguish accurately" in your justifications and explain why.
        """
        
        with st.spinner("Analyzing behavioral heuristics..."):
            try:
                response = client.models.generate_content(
                    model=primary_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ReverseEngineeringResult,
                        temperature=0.6 
                    )
                )
            except Exception as e:
                st.warning("Primary engine failed. Retrying with the backup engine...")
                try:
                    response = client.models.generate_content(
                        model=backup_model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=ReverseEngineeringResult,
                            temperature=0.6 
                        )
                    )
                except Exception as backup_error:
                    st.error(f"Both AI engines are currently unavailable. Error: {str(backup_error)}")
                    st.stop()
                    
            try:
                result = json.loads(response.text)
                st.session_state['reverse_parsed_predictions'] = result
            except Exception as parse_error:
                st.error(f"Error reading the AI output: {str(parse_error)}")

    # ==========================================
    # RENDER REVERSE PREDICTIONS PERSISTENTLY
    # ==========================================
    if 'reverse_parsed_predictions' in st.session_state:
        result = st.session_state['reverse_parsed_predictions']
        st.markdown("---")
        
        col_ambig, col_ev = st.columns(2)
        col_ambig.info(f"**🧩 Behavioral Ambiguity:** {result['behavioral_ambiguity']}")
        col_ev.info(f"**🔬 Evidence Quality:** {result['evidence_quality']}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        cols = st.columns(3)
        for idx, (col, hyp) in enumerate(zip(cols, result['hypotheses'])):
            with col:
                st.markdown(f"### Profile Hypothesis {idx+1}")
                st.progress(hyp['relative_plausibility_score'] / 100.0, text=f"Relative Plausibility: {hyp['relative_plausibility_score']}/100")
                
                st.markdown("**🧠 Deduced State & Temperament**")
                st.write(f"- **Sensory:** {hyp['sensory_threshold']}")
                st.write(f"- **Reward Sens:** {hyp['reward_sensitivity']}")
                st.write(f"- **State:** {hyp['current_state']}")
                
                st.markdown("**📊 Estimated HEXACO**")
                h_data = hyp['hexaco']
                st.caption(f"H:{h_data['Honesty_Humility']} | E:{h_data['Emotionality']} | X:{h_data['Extraversion']} | A:{h_data['Agreeableness']} | C:{h_data['Conscientiousness']} | O:{h_data['Openness']}")
                
                with st.expander("Read Justification"):
                    st.write(hyp['justification'])

# --- PERMANENT FOOTER ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.85em;'>
        <p><b>AI-Assisted Behavioral Simulation</b> | Built with Streamlit & Google Gemini</p>
        <p>⚠️ <em>Disclaimer: This application is for educational, creative, and exploratory simulation purposes only. It does not provide medical diagnoses, psychological evaluations, or professional clinical advice. All outputs are generated heuristically by an AI model.</em></p>
    </div>
    """,
    unsafe_allow_html=True
)
