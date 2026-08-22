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
tab1, tab2, tab3 = st.tabs(["🔮 Forward Predictor", "🔍 Behavior → Competing Explanations", "📖 Science & Methodology"])


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
    
            if submit_q:
                st.session_state["last_chat_response"] = None 

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
# TAB 2: BEHAVIOR -> COMPETING EXPLANATIONS
# ==========================================
with tab2:
    st.subheader("Generate 3 Competing Explanations for the Observed Behavior")
    st.info("💡 This tool attempts to explain *why* a behavior occurred by prioritizing situational pressure, temporary states, and immediate incentives before considering stable personality traits.")
    
    rev_situation = st.text_area("Situation Context", "Finds a lost wallet containing $500 cash in a crowded subway station.")
    observed_action = st.text_area("Observed Action", "Grabbed the cash immediately, threw the wallet onto the tracks, and ran onto the train.")
    known_context = st.text_input("Known Context (Optional)", "Late for work")
    
    if st.button("🔬 Analyze Behavior", type="primary"):
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

        with st.spinner("Generating competing explanations..."):
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
    # RENDER EXPLANATIONS PERSISTENTLY
    # ==========================================
    if st.session_state['reverse_parsed_predictions']:
        result = st.session_state['reverse_parsed_predictions']
        st.markdown("---")
        
        # Top-level ambiguity and uncertainty
        st.info(f"**🧩 Behavioral Ambiguity:** {result['behavioral_ambiguity']}")
        st.warning(f"**⚠️ Specific Uncertainty:** {result['specific_uncertainty']}")
        
        # Epistemological Humility Block
        col_missing, col_cannot = st.columns(2)
        with col_missing:
            st.markdown("#### ❓ Missing Information")
            st.caption("We cannot draw definitive conclusions without knowing:")
            for item in result.get('missing_information', []):
                st.markdown(f"- {item}")
        with col_cannot:
            st.markdown("#### 🛑 What Cannot Be Inferred")
            st.caption("This single behavior is insufficient evidence to determine:")
            for item in result.get('cannot_be_inferred', []):
                st.markdown(f"- {item}")
                
        st.markdown("---")
        st.markdown("### 3 Competing Explanations")
        
        hypotheses = result.get('explanations', [])
        hypotheses = hypotheses[:3]

        cols = st.columns(min(len(hypotheses), 3)) if hypotheses else []
        for idx, (col, hyp) in enumerate(zip(cols, hypotheses)):
            with col:
                st.markdown(f"#### {idx+1}. {hyp['explanation_name']}")
                
                # Compatibility Badge
                comp = hyp['compatibility']
                color = "green" if comp == "Strong" else "orange" if comp == "Moderate" else "red"
                st.markdown(f"**Compatibility:** :{color}[**{comp}**]")
                
                st.markdown(f"**⚙️ Mechanism:** {hyp['primary_mechanism']}")
                
                st.markdown("**Context & State**")
                st.write(f"- **Situation:** {hyp['situational_factors']}")
                st.write(f"- **Temporary State:** {hyp['temporary_state']}")
                st.write(f"- **Trait Contribution:** {hyp['possible_trait_contribution']}")
                
                # Expanders for detailed breakdown
                with st.expander("📊 Evidence vs. Speculation Breakdown"):
                    ed = hyp['evidence_breakdown']
                    st.markdown("**✅ Directly Supported (Observed):**")
                    st.write(ed['directly_supported'])
                    st.markdown("**🔍 Interpretation (Plausible Link):**")
                    st.write(ed['interpretation'])
                    st.markdown("**💭 Speculation (Missing Data Assumed):**")
                    st.write(ed['speculation'])
                
                with st.expander("🧬 Possible HEXACO Configuration (If Applicable)"):
                    h_data = hyp['hexaco']
                    st.markdown(f"**H:** {h_data['Honesty_Humility']}")
                    st.markdown(f"**E:** {h_data['Emotionality']}")
                    st.markdown(f"**X:** {h_data['Extraversion']}")
                    st.markdown(f"**A:** {h_data['Agreeableness']}")
                    st.markdown(f"**C:** {h_data['Conscientiousness']}")
                    st.markdown(f"**O:** {h_data['Openness']}")


# ==========================================
# TAB 3: SCIENCE & METHODOLOGY (Gemini-Powered)
# ==========================================
with tab3:
    st.header("📖 The Science Behind the Simulator")
    st.info("Explore the sociological and psychological concepts powering this engine. Select a core concept or ask your own!")
    
    st.markdown("---")
    
    concept_choice = st.selectbox(
        "Select a concept to learn about:",
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
        custom_concept = st.text_input("What behavioral concept would you like explained?")
        
    if st.button("🧠 Generate Explanation", type="primary"):
        target_concept = custom_concept if concept_choice == "Custom (Ask your own question)" else concept_choice
        
        if not target_concept:
            st.warning("Please enter or select a concept to explain.")
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
            
            with st.spinner(f"Analyzing '{target_concept}'..."):
                try:
                    # Try Primary Model
                    explanation = client.models.generate_content(
                        model=primary_model,
                        contents=explain_prompt
                    )
                    st.markdown(explanation.text)
                except Exception as e1:
                    try:
                        # Try Backup Model
                        explanation = client.models.generate_content(
                            model=backup_model,
                            contents=explain_prompt
                        )
                        st.markdown(explanation.text)
                    except Exception as e2:
                        st.error("Both AI engines are currently unavailable to generate this explanation.")



# --- PERMANENT FOOTER ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.85em;'>
        <p><b>Behavioral Scenario Lab</b> | Built with Streamlit & Google Gemini</p>
        <p>⚠️ <em>Disclaimer: This application is for educational, creative, and exploratory simulation purposes only. It generates plausible behavioral outcomes based on hypothetical trait and state parameters. It does not predict real-world actions, establish causal facts, or provide medical/psychological evaluations. Multiple distinct psychological profiles can produce identical behaviors. Numerical trait values shown by the simulator are illustrative configurations, not estimates of a person's actual traits.</em></p>
    </div>
    """,
    unsafe_allow_html=True
)
