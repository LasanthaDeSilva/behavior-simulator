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
    action: str = Field(description="A highly specific action the person will take.")
    probability_percentage: int = Field(description="Probability of this action occurring (0-100).")
    rationale: str = Field(description="Explanation of why, specifically detailing how temperament/state modified baseline traits.")

class ForwardPrediction(BaseModel):
    modifier_analysis: str = Field(description="Analysis of how the subject's Sensory/Dopamine/State actively overrode or modified their baseline HEXACO traits in this situation.")
    predictions: list[PredictedAction] = Field(description="Exactly 3 possible actions, ranked by probability.")

# Schemas for Reverse Engineer
class HexacoScores(BaseModel):
    Honesty_Humility: int
    Emotionality: int
    Extraversion: int
    Agreeableness: int
    Conscientiousness: int
    Openness: int

class ProfileHypothesis(BaseModel):
    probability_percentage: int = Field(description="Likelihood that this specific profile is the correct one (0-100).")
    hexaco: HexacoScores
    sensory_threshold: str = Field(description="Deduced sensory threshold.")
    dopaminergic_system: str = Field(description="Deduced dopamine baseline.")
    current_state: str = Field(description="Deduced temporary state (e.g., panicked, exhausted).")
    justification: str = Field(description="Explanation of why this specific mix of traits/state leads perfectly to the observed action.")

class ReverseEngineeringResult(BaseModel):
    hypotheses: list[ProfileHypothesis] = Field(description="Exactly 3 distinct profile hypotheses that could have caused the action.")


# ==========================================
# STREAMLIT APP CONFIGURATION
# ==========================================
st.set_page_config(page_title="Behavior Simulator", layout="wide")
st.title("🧠 Advanced Human Behavior Simulator")

# Initialize session state variables if they don't exist yet
if "forward_result" not in st.session_state:
    st.session_state.forward_result = None
if "reverse_result" not in st.session_state:
    st.session_state.reverse_result = None

# Sidebar for API Key & Model Settings
with st.sidebar:
    st.header("Settings")
    
    model_choice = st.selectbox(
        "Select AI Engine",
        ["Gemini 3.6 Flash (Primary)", "Gemini 2.5 Flash (Backup)"],
        help="Switch to 2.5 Flash if you hit rate limits on 3.6 Flash."
    )
    
    # Set the active model and backup engine
    if "3.6" in model_choice:
        primary_model = "gemini-3.6-flash"
        backup_model = "gemini-2.5-flash"
    else:
        primary_model = "gemini-2.5-flash"
        backup_model = "gemini-3.6-flash"

    api_key = st.secrets["GEMINI_API_KEY"]

if not api_key:
    st.warning("Please enter your Gemini API Key in the sidebar.")
    st.stop()

# Initialize Client
client = genai.Client(api_key=api_key)


# Create Tabs
tab1, tab2 = st.tabs(["🔮 Forward Predictor", "🔍 Reverse Engineer"])

# ==========================================
# TAB 1: FORWARD PREDICTOR
# ==========================================
with tab1:
    st.subheader("Configure Profile & Scenario")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**HEXACO Traits (0-100 Baseline)**")
        st.markdown("<br>", unsafe_allow_html=True)
        
        h = st.slider("Honesty-Humility", 0, 100, 20)
        st.caption("Measures sincerity, fairness, and modesty. High scorers are genuine; low scorers tend to be manipulative.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        e = st.slider("Emotionality", 0, 100, 80)
        st.caption("Focuses on anxiety and vulnerability. High scorers feel deeper anxiety; low scorers are highly independent.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        x = st.slider("Extraversion", 0, 100, 50)
        st.caption("Covers social boldness. High scorers thrive in crowds; low scorers prefer solitary, quiet settings.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        a = st.slider("Agreeableness", 0, 100, 40)
        st.caption("Involves forgiveness and patience. High scorers compromise easily; low scorers anger quickly.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        c = st.slider("Conscientiousness", 0, 100, 90)
        st.caption("Reflects organization and diligence. High scorers are disciplined; low scorers are impulsive.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        o = st.slider("Openness to Experience", 0, 100, 15)
        st.caption("Represents curiosity. High scorers are naturally drawn to novel ideas. Low scorers strongly prefer the familiar, practical solutions, and traditional ways of thinking.")
        st.markdown("<br>", unsafe_allow_html=True)

    with col2:
        st.markdown("**Temperament & State (The Modifiers)**")
        st.markdown("<br>", unsafe_allow_html=True)
        
        sensory = st.selectbox("Sensory Threshold", ["Low (Easily overwhelmed)", "Medium (Balanced)", "High (Requires intense input)"])
        st.caption("How the nervous system filters noise. Low = easily overwhelmed; High = needs intense stimulation.")
        st.markdown("<br>", unsafe_allow_html=True)

        sensory_domains = st.multiselect(
        "Primary Sensory Domains Affected", 
        [
            "General / All Senses", 
            "Auditory (Sound)", 
            "Visual (Sight)", 
            "Tactile (Touch)", 
            "Olfactory (Smell)", 
            "Gustatory (Taste)", 
            "Vestibular (Balance/Movement)"
        ])
        st.caption("The specific sensory system most intensely affected or triggered in this scenario.")
        st.markdown("<br>", unsafe_allow_html=True)

        masking = st.selectbox(
           "Behavioral Masking", 
           ["None (Unmasked / Natural expression)", "Moderate (Suppressing visible traits for social norms)", "High (Heavy camouflage, high exhaustion risk)"]
        )
        st.caption("The degree to which the subject is actively hiding their natural reactions, neurodivergent traits, or stress to conform.")
        st.markdown("<br>", unsafe_allow_html=True)

        stimming = st.multiselect(
           "Stimming / Self-Regulation Behaviors", 
           [
               "None",
               "Fidgeting / Hand movements", 
               "Pacing / Repetitive motion", 
               "Auditory stimming (Humming, repeating words)", 
               "Tactile stimming (Rubbing objects, textures)", 
               "Verbal masking / Scripting"
           ],
           default=["None"]
        )
        st.caption("Physical or vocal mechanisms used to manage anxiety, sensory overload, or emotional regulation.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        dopamine = st.selectbox("Dopaminergic System", ["Low (Cautious/Apathetic)", "Medium (Balanced)", "High (Thrill-seeking/Impulsive)"], index=2)
        st.caption("The brain's reward center. High = takes risks for thrills; Low = prefers stability and security.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        state_trait = st.selectbox("Current State", ["High stress/Panic state", "Relaxed/Calm state", "Fatigued/Burnout", "Baseline"])
        st.caption("The immediate physical/emotional baseline. Acts as a powerful multiplier to baseline traits.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        cognitive_load = st.selectbox("Cognitive Load", ["Low (Clear headed)", "Medium (Busy)", "High (Distracted/Overwhelmed)"])
        st.caption("Mental bandwidth in use. High load causes people to lose logic and default to raw instinct.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("**Context**")
        extra_details = st.text_input("Extra Details", "Late for an important job interview.")
        st.markdown("<br>", unsafe_allow_html=True)
        situation = st.text_area("The Situation", "Finds a wallet with $500 cash in a loud, crowded subway station.")
        st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Predict Action", type="primary"):
        hexaco_data = f"H:{h}, E:{e}, X:{x}, A:{a}, C:{c}, O:{o}"
        
        prompt = f"""
        You are an elite computational psychologist. 
        Analyze this subject and predict their behavioral response. 
        
        APPLY THE MODIFIER MULTIPLIER RULE: 
        The subject's Temperament, State, Sensory Domains, Masking levels, and Stimming behaviors must actively drive the outcome. Especially their specific Sensory Domains must actively amplify or trigger the outcome based on their Sensory Threshold ({sensory}). If specific sensory domains are selected below, they are the exact vectors causing overstimulation or craving—incorporate them directly into the physical and psychological reaction. High Masking should rapidly drain Cognitive Load and increase internal stress, while Stimming acts as a regulatory valve. Incorporate these mechanisms directly into the predicted actions.
        
        BASELINE HEXACO: {hexaco_data}
        
        MODIFIERS: 
        - Sensory: {sensory}
        - Affected Sensory Domains: {sensory_domains}
        - Dopamine: {dopamine}
        - State: {state_trait}
        - Cognitive Load: {cognitive_load}
        - Behavioral Masking: {masking}
        - Stimming Behaviors: {stimming}
        - Background: {extra_details}
        
        SITUATION: {situation}
        
        Provide the top 3 most statistically likely actions with probabilities summing to 100%.
        """
        
        with st.spinner("Simulating neurobiological response..."):
            try:
            # 1. Try the primary model
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
            # 2. If primary fails, warn the user and switch to backup
            st.warning(f"Primary engine hit a rate limit. Automatically retrying with backup...")
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
                # If both fail, show error and stop execution
                st.error(f"Both AI engines are currently unavailable. Error: {str(backup_error)}")
                st.stop()

        # 3. If either model succeeded, render the results to the UI
        try:
            result = json.loads(response.text)
            
            st.markdown("---")
            st.info(f"**🧠 Trait Modifier Analysis:** {result['modifier_analysis']}")
            
            for idx, action in enumerate(result['predictions']):
                st.markdown(f"### {idx+1}. {action['action']}")
                # Using your exact progress bar and markdown logic
                st.progress(action['probability_percentage'] / 100.0, text=f"{action['probability_percentage']}%")
                st.write(f"**Rationale:** {action['rationale']}")
                st.divider()
                
        except Exception as parse_error:
            st.error(f"Error reading the AI output: {str(parse_error)}")


# ==========================================
# TAB 2: REVERSE ENGINEER
# ==========================================
with tab2:
    st.subheader("Reverse Engineer Profile from Action")
    st.info("💡 Solves the 'One-to-Many' problem: Generates the top 3 distinct personality profiles that could result in the exact same behavior.")
    
    rev_situation = st.text_area("Situation Context", "Finds a lost wallet containing $500 cash in a crowded subway station.")
    observed_action = st.text_area("Observed Action", "Grabbed the cash immediately, threw the wallet onto the tracks, and ran onto the train.")
    known_context = st.text_input("Known Context (Optional)", "Late for work")
    
    if st.button("🔬 Reverse Engineer Profile", type="primary"):
        prompt = f"""
        You are an elite computational psychologist. Address the "One-to-Many" behavioral problem: A single action can be caused by completely different psychological profiles. 
        
        SITUATION: {rev_situation}
        OBSERVED ACTION: {observed_action}
        KNOWN CONTEXT: {known_context}
        
        Provide the TOP 3 completely distinct profiles (combinations of HEXACO, Temperament, and State) that would result in this exact action. Rank them by probability.
        """
        
        with st.spinner("Running reverse behavioral heuristics..."):
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ReverseEngineeringResult,
                        temperature=0.6 # Slightly higher temp allows for more creative reverse-engineering 
                    )
                )
                
                result = json.loads(response.text)
                st.markdown("---")
                
                cols = st.columns(3)
                for idx, (col, hyp) in enumerate(zip(cols, result['hypotheses'])):
                    with col:
                        st.markdown(f"### Hypothesis {idx+1}")
                        st.progress(hyp['probability_percentage'] / 100.0, text=f"Likelihood: {hyp['probability_percentage']}%")
                        
                        st.markdown("**🧠 Deduced State & Temperament**")
                        st.write(f"- **Sensory:** {hyp['sensory_threshold']}")
                        st.write(f"- **Dopamine:** {hyp['dopaminergic_system']}")
                        st.write(f"- **State:** {hyp['current_state']}")
                        
                        st.markdown("**📊 Estimated HEXACO**")
                        h_data = hyp['hexaco']
                        st.caption(f"H:{h_data['Honesty_Humility']} | E:{h_data['Emotionality']} | X:{h_data['Extraversion']} | A:{h_data['Agreeableness']} | C:{h_data['Conscientiousness']} | O:{h_data['Openness']}")
                        
                        with st.expander("Read Justification"):
                            st.write(hyp['justification'])

            except Exception as e:
                st.error(f"Error during reverse engineering: {str(e)}")


# --- PERMANENT FOOTER ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.85em;'>
        <p><b>Behavioral & Psychological Simulator</b> | Built with Streamlit & Google Gemini</p>
        <p>⚠️ <em>Disclaimer: This application is for educational, creative, and exploratory simulation purposes only. It does not provide medical diagnoses, psychological evaluations, or professional clinical advice.</em></p>
    </div>
    """,
    unsafe_allow_html=True
)

