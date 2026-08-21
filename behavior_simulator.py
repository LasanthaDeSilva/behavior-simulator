import warnings
warnings.filterwarnings("ignore")

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

# Schemas for Forward Predictor
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

# Schemas for Reverse Engineer
class HexacoScores(BaseModel):
    Honesty_Humility: str = Field(description="Qualitative range (e.g., Low, Moderate, High) rather than an exact number to avoid false precision.")
    Emotionality: str = Field(description="Qualitative range (e.g., Low, Moderate, High)")
    Extraversion: str = Field(description="Qualitative range (e.g., Low, Moderate, High)")
    Agreeableness: str = Field(description="Qualitative range (e.g., Low, Moderate, High)")
    Conscientiousness: str = Field(description="Qualitative range (e.g., Low, Moderate, High)")
    Openness: str = Field(description="Qualitative range (e.g., Low, Moderate, High)")

class ProfileHypothesis(BaseModel):
    relative_plausibility_score: int = Field(ge=1, le=100, description="Heuristic compatibility score (1-100). This is NOT a probability, NOT a confidence metric, and NOT a personality estimate.")
    primary_mechanism: str = Field(description="Primary driver of the behavior (e.g., situational pressure, immediate incentive, learning history, stable trait tendency).")
    hexaco: HexacoScores
    sensory_responsiveness: str = Field(description="Hypothesized sensory responsiveness.")
    reward_sensitivity: str = Field(description="Hypothesized reward sensitivity / novelty seeking.")
    current_state: str = Field(description="Hypothesized temporary state (e.g., panicked, exhausted).")
    justification: str = Field(description="Explanation of why this specific mix of traits/state leads plausibly to the observed action. MUST include 'Insufficient information' if highly ambiguous.")

class ReverseEngineeringResult(BaseModel):
    evidence_quality: str = Field(description="Assessment of the quality and specificity of the provided context and action.")
    behavioral_ambiguity: str = Field(description="How ambiguous the behavior is. Explicitly state that multiple different profiles can produce the same behavior.")
    hypotheses: list[ProfileHypothesis] = Field(min_length=3, max_length=3, description="Exactly 3 distinct profile hypotheses that could plausibly explain the action.")


# ==========================================
# STREAMLIT APP CONFIGURATION
# ==========================================
st.set_page_config(page_title="Behavior Simulator", layout="wide")
st.title("Behavioral Scenario Lab")

# Initialize Session State early for stability
for key in ['parsed_predictions', 'reverse_parsed_predictions', 'last_sim', 'last_situation', 'last_chat_response', 'last_config', 'simulation_id']:
    if key not in st.session_state:
        st.session_state[key] = None

# Sidebar for API Key & Model Settings
with st.sidebar:
    st.header("Settings")
    
    model_choice = st.selectbox(
        "Primary AI Engine",
        ["Gemini 3.7 Flash", "Gemini 3.5 Flash"]
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


# ==========================================
# HELPER FOR WEIGHT NORMALIZATION
# ==========================================
def calculate_normalized_percentages(predictions_list):
    raw_weights = [action['raw_weight'] for action in predictions_list]
    total_weight = sum(raw_weights)
    
    exact = [(w / total_weight) * 100 for w in raw_weights]
    percentages = [math.floor(x) for x in exact]
    remainder = int(100 - sum(percentages))
    
    order = sorted(range(len(exact)), key=lambda i: exact[i] - percentages[i], reverse=True)
    for i in order[:remainder]:
        percentages[i] += 1
        
    return percentages


# Create Tabs
tab1, tab2 = st.tabs(["🔮 Forward Predictor", "🔍 Profile Hypotheses"])

# ==========================================
# TAB 1: FORWARD PREDICTOR
# ==========================================
with tab1:
    st.subheader("Configure Profile & Context")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Illustrative HEXACO Parameters (0–100)**")
        st.caption("These are hypothetical simulation inputs, not clinical measurements or estimates of a real person's personality.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        h = st.slider("Honesty-Humility", 0, 100, 20)
        st.caption("Sincerity, fairness, modesty, and reduced tendency toward status/material exploitation.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        e = st.slider("Emotionality", 0, 100, 80)
        st.caption("Fearfulness, sensitivity to threat, dependence, sentimentality, and emotional attachment.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        x = st.slider("Extraversion", 0, 100, 50)
        st.caption("Social self-esteem, social boldness, sociability, and liveliness.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        a = st.slider("Agreeableness", 0, 100, 40)
        st.caption("Forgiveness, gentleness, flexibility, and patience toward others.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        c = st.slider("Conscientiousness", 0, 100, 90)
        st.caption("Organization, diligence, perfectionism, prudence, and impulse control.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        o = st.slider("Openness to Experience", 0, 100, 15)
        st.caption("Aesthetic appreciation, inquisitiveness, creativity, and unconventionality.")
        st.markdown("<br>", unsafe_allow_html=True)

    with col2:
        st.markdown("**Simulation Parameters (States & Modifiers)**")
        st.markdown("<br>", unsafe_allow_html=True)
        
        sensory = st.selectbox(
            "Hypothetical Sensory Responsiveness", 
            ["Lower hypothetical sensory reactivity", "Moderate hypothetical sensory reactivity", "Higher hypothetical sensory reactivity"], 
            index=1
        )
        st.caption("A simulation parameter representing sensory processing tendency.")
        st.markdown("<br>", unsafe_allow_html=True)

        sensory_domains = st.multiselect(
            "Potentially Relevant Sensory Domains", 
            ["Auditory", "Visual", "Tactile", "Olfactory", "Gustatory", "Vestibular"],
            help="Select sensory domains that might be relevant. The model must determine whether they are actually relevant to the scenario."
        )
        st.markdown("<br>", unsafe_allow_html=True)

        masking = st.selectbox(
           "Behavioral Masking Tendency", 
           ["None (Natural expression)", "Moderate", "High (Heavy camouflage)"]
        )
        st.caption("Consider whether masking plausibly increases load in this particular context (only if social monitoring is required).")
        st.markdown("<br>", unsafe_allow_html=True)

        stimming = st.multiselect(
           "Stimming / Self-Regulation Tendency", 
           ["None", "Fidgeting", "Pacing", "Auditory stimming", "Tactile stimming", "Vocal scripting"],
           default=["None"]
        )
        st.caption("Potential self-regulatory or sensory-seeking behavior. (Does not automatically imply anxiety/distress).")
        st.markdown("<br>", unsafe_allow_html=True)
        
        reward_sensitivity = st.selectbox("Reward Sensitivity / Novelty Seeking", ["Low (Prefers predictable/stable options)", "Medium (Balanced)", "High (Sensitive to novelty, reward, stimulation)"], index=2)
        st.markdown("<br>", unsafe_allow_html=True)
        
        state_trait = st.selectbox("Current State", ["High stress/arousal", "Relaxed/Calm", "Fatigued/Burnout", "Baseline"])
        st.caption("Temporary conditions (distinct from stable traits).")
        st.markdown("<br>", unsafe_allow_html=True)
        
        cognitive_load = st.selectbox("Cognitive Load", ["Low (Clear headed)", "Medium (Busy)", "High (Distracted/Overwhelmed)"])
        st.caption("High load reduces working-memory capacity and increases reliance on habitual/simplified strategies.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("**Context**")
        extra_details = st.text_input("Extra Details", "Late for an important job interview.")
        situation = st.text_area("The Situation (External Circumstances)", "Finds a wallet with $500 cash in a loud, crowded subway station.")

    if st.button("🚀 Run Behavioral Simulation", type="primary"):
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
            
        with st.spinner("Simulating plausible behaviors..."):
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
                    st.error("Both AI engines are currently unavailable or failed structured validation.")
                    with st.expander("Technical error details"):
                        st.write(f"Primary error: {str(primary_error)}")
                        st.write(f"Backup error: {str(backup_error)}")
                    st.stop()

    # ==========================================
    # RENDER PREDICTIONS AND CHAT PERSISTENTLY
    # ==========================================
    if st.session_state['parsed_predictions']:
        result = st.session_state['parsed_predictions']
        
        st.markdown("---")
        st.caption(f"Simulation #{st.session_state['simulation_id']}")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.info(f"**🔍 Modifier Relevance:** {result['modifier_relevance']}")
        with col_b:
            st.warning(f"**⚠️ Uncertainty Level:** {result['uncertainty_level']}\n\n{result['uncertainty_reason']}")
    
        st.markdown("### Plausible Actions")
        
        predictions = result.get('predictions', [])
        percentages = calculate_normalized_percentages(predictions)
        
        for idx, action in enumerate(predictions):
            st.markdown(f"**{idx+1}. {action['action']}**")
            calculated_pct = percentages[idx]
            st.progress(calculated_pct / 100.0, text=f"Normalized Heuristic Weight: {calculated_pct}%")
            st.write(f"**Rationale:** {action['rationale']}")
            st.divider()

        # === FOLLOW-UP CHAT FEATURE ===
        st.markdown("### 💬 Counterfactual Analysis")
        st.caption("Try asking: *What if the environment became quiet?* or *What if the person wasn't late?*")

        with st.form("chat_form"):
            query = st.text_input("Test a scenario change:", placeholder="What if sensory load doubled?")
            submit_q = st.form_submit_button("Explore Counterfactual")
    
            if submit_q and query:
                if not st.session_state.get('last_config'):
                    st.error("⚠️ Please run a simulation first so there is a baseline configuration to modify.")
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

                    with st.spinner("Simulating counterfactual..."):
                        try:
                            cf_obj = run_counterfactual_generation(primary_model)
                            st.session_state['last_chat_response'] = cf_obj.model_dump()
                        except Exception as e1:
                            try:
                                cf_obj = run_counterfactual_generation(backup_model)
                                st.session_state['last_chat_response'] = cf_obj.model_dump()
                            except Exception as e2:
                                st.error("⚠️ Both engines failed to generate or validate a structured counterfactual response.")

        if st.session_state['last_chat_response']:
            cf_data = st.session_state['last_chat_response']
            st.success(f"**Identified Change:** {cf_data['identified_variable_changed']}")
            st.info(f"**Comparison to Baseline:** {cf_data['comparison_summary']}")
            
            cf_predictions = cf_data.get('new_predictions', [])
            cf_percentages = calculate_normalized_percentages(cf_predictions)
            
            st.markdown("#### Counterfactual Predictions")
            for idx, action in enumerate(cf_predictions):
                st.markdown(f"**{idx+1}. {action['action']}**")
                calculated_pct = cf_percentages[idx]
                st.progress(calculated_pct / 100.0, text=f"Normalized Heuristic Weight: {calculated_pct}%")
                st.write(f"**Rationale:** {action['rationale']}")
                st.markdown("<br>", unsafe_allow_html=True)


# ==========================================
# TAB 2: REVERSE ENGINEER (PROFILE HYPOTHESES)
# ==========================================
with tab2:
    st.subheader("Generate Plausible Profile Hypotheses")
    st.info("💡 This tool explores possible psychological configurations that could be consistent with an observed behavior. It does not infer or recover the person's actual personality.")
    st.warning("⚠️ **Important:** These hypotheses are not recovered personality profiles. The same observed behavior may be compatible with many different trait/state configurations.")
    
    rev_situation = st.text_area("Situation Context", "Finds a lost wallet containing $500 cash in a crowded subway station.")
    observed_action = st.text_area("Observed Action", "Grabbed the cash immediately, threw the wallet onto the tracks, and ran onto the train.")
    known_context = st.text_input("Known Context (Optional)", "Late for work")
    
    if st.button("🔬 Analyze Plausible Profiles", type="primary"):
        st.session_state["reverse_parsed_predictions"] = None
        
        prompt = f"""
        You are an AI performing a behavioral simulation to address the "One-to-Many" behavioral problem.
        
        CRITICAL RULES:
        1. Provide 3 distinct hypothetical profile configurations that are plausibly consistent with this exact action. 
        2. The three hypotheses must differ primarily in their explanatory mechanism (e.g., one driven by immediate incentive, another by situational pressure, another by trait tendencies).
        3. Use qualitative ranges (e.g., Low, Moderate, High) for HEXACO dimensions to avoid false precision.
        4. Explicitly state that multiple different profiles can produce the exact same behavior. Do not claim you can identify a real personality from one action.
        5. Never infer autism, ADHD, anxiety disorders, personality disorders, trauma disorders, or any other clinical diagnosis from these parameters or behaviors.
        6. Treat all outputs as hypothesized parameters, not actual detections or medical diagnoses.
        7. If the behavior is highly ambiguous, you MUST allow for high uncertainty and output "Insufficient information to distinguish accurately" in your justifications.
        8. Separate traits (stable), states (temporary), and context (external) conceptually.
        
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
                    response_schema=ReverseEngineeringResult,
                    temperature=0.6 
                )
            )
            return ReverseEngineeringResult.model_validate_json(response.text)

        with st.spinner("Analyzing behavioral hypotheses..."):
            try:
                rev_result_obj = run_reverse_generation(primary_model)
                st.session_state['reverse_parsed_predictions'] = rev_result_obj.model_dump()
            except Exception as primary_error:
                st.warning("Primary engine failed validation or API call. Retrying with backup engine...")
                try:
                    rev_result_obj = run_reverse_generation(backup_model)
                    st.session_state['reverse_parsed_predictions'] = rev_result_obj.model_dump()
                except Exception as backup_error:
                    st.error("Both AI engines are currently unavailable or failed structured validation.")
                    st.stop()

    # ==========================================
    # RENDER REVERSE PREDICTIONS PERSISTENTLY
    # ==========================================
    if st.session_state['reverse_parsed_predictions']:
        result = st.session_state['reverse_parsed_predictions']
        st.markdown("---")
        
        col_ambig, col_ev = st.columns(2)
        col_ambig.info(f"**🧩 Behavioral Ambiguity:** {result['behavioral_ambiguity']}")
        col_ev.info(f"**🔬 Evidence Quality:** {result['evidence_quality']}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        hypotheses = result.get('hypotheses', [])
        hypotheses = hypotheses[:3]

        cols = st.columns(min(len(hypotheses), 3)) if hypotheses else []
        for idx, (col, hyp) in enumerate(zip(cols, hypotheses)):
            with col:
                st.markdown(f"### Profile Hypothesis {idx+1}")
                st.progress(hyp['relative_plausibility_score'] / 100.0, text=f"Heuristic Compatibility: {hyp['relative_plausibility_score']}/100")
                
                st.markdown(f"**⚙️ Primary Mechanism:** {hyp['primary_mechanism']}")
                
                st.markdown("**🧠 Hypothesized State & Temperament**")
                st.write(f"- **Sensory Responsiveness:** {hyp['sensory_responsiveness']}")
                st.write(f"- **Reward Sens:** {hyp['reward_sensitivity']}")
                st.write(f"- **State:** {hyp['current_state']}")
                
                st.markdown("**📊 Illustrative HEXACO Configuration**")
                h_data = hyp['hexaco']
                st.caption(f"**H:** {h_data['Honesty_Humility']} | **E:** {h_data['Emotionality']} | **X:** {h_data['Extraversion']}")
                st.caption(f"**A:** {h_data['Agreeableness']} | **C:** {h_data['Conscientiousness']} | **O:** {h_data['Openness']}")
                
                with st.expander("Read Justification"):
                    st.write(hyp['justification'])

# --- PERMANENT FOOTER ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.85em;'>
        <p><b>AI-Assisted Behavioral Simulation</b> | Built with Streamlit & Google Gemini</p>
        <p>⚠️ <em>Disclaimer: This application is for educational, creative, and exploratory simulation purposes only. It generates plausible behavioral outcomes based on hypothetical trait and state parameters. It does not predict real-world actions, establish causal facts, or provide medical/psychological evaluations. Multiple distinct psychological profiles can produce identical behaviors. Numerical trait values shown by the simulator are illustrative configurations, not estimates of a person's actual traits.</em></p>
    </div>
    """,
    unsafe_allow_html=True
)
