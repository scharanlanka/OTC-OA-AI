import streamlit as st
import pandas as pd
import joblib
import requests
from io import BytesIO

st.markdown("""
    <style>
    /* Match multiselect tags with dark theme */
    [data-baseweb="tag"] {
        background-color: #2c2c2c !important;
        color: #fff !important;
        border: 1px solid #444 !important;
        border-radius: 5px;
        padding: 2px 6px;
    }
    [data-baseweb="tag"] svg {
        fill: #aaa !important;t
    }
    </style>
""", unsafe_allow_html=True)

# ─── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_URL = "https://otc-only-model.s3.amazonaws.com/otc_classifier_no_postpain.pkl"

@st.cache(allow_output_mutation=True, show_spinner=False)
def load_artifacts():
    otc_pre = joblib.load("otc_preprocessor_no_postpain.pkl")
    pain_model = joblib.load("pain_reduction_model.pkl")
    weeks_model = joblib.load("weeks_to_effect_model.pkl")

    r = requests.get(MODEL_URL)
    r.raise_for_status()
    otc_model = joblib.load(BytesIO(r.content))

    df = pd.read_csv("OTC-Data.csv", skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        'Best OTC': 'best_otc', 'OTCSleep': 'otc_sleep', 'OTC Cause': 'otc_cause',
        'OTC PainLocation': 'otc_pain_location', 'OTC PainTime': 'otc_pain_time',
        'OTC CocomtSymptom': 'otc_cocomt_symptom', 'Gender': 'gender',
        'Age': 'age', 'Height': 'height', 'Weight': 'weight',
        'Ethnicity': 'ethnicity', 'Race': 'race'
    })
    return otc_pre, otc_model, pain_model, weeks_model, df

# Load artifacts
otc_pre, otc_model, pain_model, weeks_model, df_full = load_artifacts()

st.markdown(
    """
    <div style='text-align: center; font-size: 32.2px; font-weight: bold; line-height: 1.4;'>
        AI (Artificial Intelligence) Decision-Making Tool<br>
        for<br>
        Over-the-Counter Medications for Knee Joint Pain
    </div>
    """,
    unsafe_allow_html=True
)


st.text("This AI decision-making model recommends the best OTC medications for indivduals with knee joint pain.")
# ─── PATIENT PROFILE ───────────────────────────────────────────────────────────
st.header("Patient Profile")
st.text("About You:")
age       = st.text_input("Age (50 or older)", value="")
gender = st.selectbox(
    "Gender",
    options=["", "Male", "Female", "Non-binary / third gender", "Prefer not to say"],
    format_func=lambda x: " " if x == "" else x,
    index=0
)

ethnicity = st.selectbox(
    "Spanish, Hispanic, or Latino origin?",
    options=["Yes", "No"],
    index=None,
    placeholder=""
)

race = st.selectbox(
    "Race",
    options=["White", "Black or African American","American Indian or Alaska Native", "Asian", "Native Hawaaiian or Pacific Islander", "Other/Unknown", "Prefer not to say"],
    index=None,
    placeholder=""
)

weight    = st.text_input("Weight (lbs)", value="", placeholder="e.g. 183")
height    = st.text_input("Height (inches)", value="", placeholder="e.g. 67")



# ─── PAIN LEVEL ────────────────────────────────────────────────────────────────
st.header("About your knee pain:")
pain_level = st.text_input("Rate the level of your knee joint pain. 0 = no pain, 10 = the worst pain", value="", placeholder="")


cause = st.selectbox("What caused your knee pain?", [
    "", "Overweight or obesity", "Injuries: Such as torn ligaments, torn cartilage, kneecap fracture, or bone fractures due to traumas like falls or car accidents.",
    "Medical conditions: Such as arthritis, gout, infections, tendonitis, bursitis, or instability.",
    "Aging Such as osteoarthritis", "Repeated stress: Such as overuse due to repetitive motions in physical activities and exercise/sports, like running, jumping, or working on your knees, prolonged standing or kneeling, or tight muscles.",
    "Other conditions: Such as patellofemoral pain syndrome, lupus, or rheumatoid arthritis.",
    "None of the above", "Don’t know"])

pain_location = st.selectbox("Where do you feel your knee pain?", [
    "", "In the front of your knee", "All over the knee", "Close to the surface above or behind your knee (usually an issue with muscles, tendons or ligaments)",
    "Deeper inside your knee (pain that comes from your bones or cartilage).", "In multiple parts of your knee or leg (pain on one side like coming from the back of your knee, or pain that spreads to areas around your knee like lower leg or thigh.)", "None of the above"])

pain_time = st.selectbox("When do you feel pain?", [
    "", "When you are moving or bending your knee, and getting better when you rest.", "Feel more pain first thing in the morning when you wake up.", "Feel more pain at night, especially if you were physically active earlier that day.",
    "Feel more pain during bad weather.", "Feel more pain when you are stressed/anxious/tired.", "Feel more pain when you are unwell.", "None of the above"])

symptoms = st.multiselect("Accompanying symptoms", [
    "", "Dull pain", "Throbbing pain", "Sharp pain", "Swelling", "Stiffness", "Redness of skin (Erythema) and warmth to the touch",
    "Instability or weakness (having trouble walking, limping) ", "Popping or crunching noises", "Limited range of motion (inability to fully straighten the knee)",
    "Locking of the knee joint", "Inability to bear weight", "Fever", "Disabling pain", "Others", "None"], placeholder="")


sleep = st.selectbox("Do you experience any of these?", ["", "Abnormal sleep pattern", "Pain at other joint(s) (spine, shoulder. elbow, wrist, fingers, hip, ankle, toes, etc.).", "None of the above"])


# ─── PREDICTION ────────────────────────────────────────────────────────────────
if st.button("Get OTC Recommendations"):
    required = [age, gender, race, ethnicity, weight, height,
                pain_level, pain_location, pain_time, sleep, cause]
    if not all(required):
        st.error("Please fill in every field.")
    else:
        try:
            age_v = int(age)
            w_v   = float(weight)
            h_v   = float(height)
            pl_v  = int(pain_level)
            if pl_v == 0:
                st.warning("A pain level of 0 indicates no pain. No OTC medication can be recommended in this case.")
                st.stop()
        except ValueError:
            st.error("Age, weight, height, and pain level must be numeric.")
            st.stop()
        
        if age_v < 50:
            st.error("This tool is designed for patients aged 50 and above.")
            st.stop()

        input_df = pd.DataFrame([{
            'otc_prepain': pl_v,
            'age':         age_v,
            'height':      h_v,
            'weight':      w_v,
            'gender':      gender,
            'race':        race,
            'ethnicity':   ethnicity,
            'otc_pain_location': pain_location,
            'otc_pain_time':     pain_time,
            'otc_cocomt_symptom': ",".join(symptoms) if symptoms else "",
            'otc_sleep':          sleep,
            'otc_cause':          cause
        }])

        # OTC prediction
        Xp = otc_pre.transform(input_df)
        probs = otc_model.predict_proba(Xp)[0]
        classes = otc_model.classes_
        top3 = probs.argsort()[-3:][::-1]

        st.subheader("Top 3 OTC Recommendations")
        for i in top3:
            st.write(f"- {classes[i]}: { 310*probs[i]:.1f}% confidence")

        # Pain reduction + weeks prediction (top-1 only)
        try:
            reg_input = input_df[['otc_prepain', 'age', 'height', 'weight', 'gender', 'ethnicity', 'race']].copy()
            reg_input['otc_usetime'] = 4  # simulate 4 weeks usage

            predicted_reduction = pain_model.predict(reg_input)[0]
            predicted_weeks = weeks_model.predict(reg_input)[0]

            st.success(f"\n\n✨ By following {classes[top3[0]]}, you may reduce your pain by **{predicted_reduction:.1f} points** in about **{predicted_weeks:.1f} weeks**.")
        except Exception as e:
            st.warning(f"Could not estimate pain reduction or weeks to effect: {e}")

