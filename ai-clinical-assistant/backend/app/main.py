# import os
# import tempfile
# import pandas as pd
# import spacy
# import whisper
# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from typing import Dict, List

# # -------------------------
# # Load models
# # -------------------------
# print("Loading spaCy model...")
# nlp = spacy.load("en_core_web_sm")

# print("Loading Whisper model...")
# whisper_model = whisper.load_model("medium.en")

# # -------------------------
# # Load datasets dynamically
# # -------------------------
# SYMPTOM_DATA_PATH = "datasets/disease_symptom.csv"
# MEDICINE_DATA_PATH = "datasets/medicines.csv"

# if os.path.exists(SYMPTOM_DATA_PATH):
#     symptom_df = pd.read_csv(SYMPTOM_DATA_PATH)
#     SYMPTOM_KEYWORDS = set(symptom_df.columns.tolist() + symptom_df.values.flatten().astype(str).tolist())
# else:
#     SYMPTOM_KEYWORDS = []

# if os.path.exists(MEDICINE_DATA_PATH):
#     medicine_df = pd.read_csv(MEDICINE_DATA_PATH)
#     MEDICINES = set(medicine_df.values.flatten().astype(str).tolist())
# else:
#     MEDICINES = []

# # -------------------------
# # FastAPI app setup
# # -------------------------
# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -------------------------
# # Entity extraction
# # -------------------------
# def extract_entities(text: str) -> Dict[str, List[str]]:
#     doc = nlp(text)
#     entities = {"PATIENT_NAME": [], "DATE": [], "SYMPTOMS": [], "MEDICINES": []}

#     # Named Entities (Person, Date)
#     for ent in doc.ents:
#         if ent.label_ == "PERSON":
#             entities["PATIENT_NAME"].append(ent.text)
#         elif ent.label_ == "DATE":
#             entities["DATE"].append(ent.text)

#     # Symptom Extraction (using dataset)
#     lower_text = text.lower()
#     for symptom in SYMPTOM_KEYWORDS:
#         if isinstance(symptom, str) and symptom.lower() in lower_text:
#             entities["SYMPTOMS"].append(symptom)

#     # Medicine Extraction (using dataset)
#     for med in MEDICINES:
#         if isinstance(med, str) and med.lower() in lower_text:
#             entities["MEDICINES"].append(med)

#     # Remove duplicates
#     for k in entities:
#         entities[k] = list(set(entities[k]))

#     return entities

# # -------------------------
# # API Endpoints
# # -------------------------
# @app.post("/transcribe")
# async def transcribe_audio(file: UploadFile = File(...)):
#     """Transcribe audio and extract entities"""
#     suffix = os.path.splitext(file.filename)[-1]
#     if suffix not in [".wav", ".webm", ".ogg", ".mp3"]:
#         suffix = ".wav"

#     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#         tmp.write(await file.read())
#         tmp_path = tmp.name

#     print(f" Received file: {file.filename}, saved to {tmp_path}")

#     # Transcribe
#     result = whisper_model.transcribe(tmp_path, task="translate")
#     english_text = result["text"]

#     # Extract entities
#     entities = extract_entities(english_text)

#     os.remove(tmp_path)

#     return {
#         "transcript": english_text,
#         "entities": entities,
#         "recommendations": entities.get("MEDICINES", [])[:3] or ["Consult Doctor"],
#     }

# @app.get("/")
# def root():
#     return {"message": "AI Clinical Assistant Backend Running"}



































# import os
# import tempfile
# import pandas as pd
# import spacy
# import whisper
# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from typing import Dict, List

# # -------------------------
# # Load models
# # -------------------------
# print("Loading spaCy model...")
# nlp = spacy.load("en_core_web_sm")

# print("Loading Whisper model...")
# whisper_model = whisper.load_model("medium.en")  # or "small.en" if system is slow

# # -------------------------
# # Load datasets dynamically
# # -------------------------
# SYMPTOM_DATA_PATH = "datasets/disease_symptom.csv"
# MEDICINE_DATA_PATH = "datasets/medicines.csv"

# SYMPTOM_KEYWORDS = set()
# MEDICINE_MAP = {}

# if os.path.exists(SYMPTOM_DATA_PATH):
#     print("Loading disease-symptom dataset...")
#     symptom_df = pd.read_csv(SYMPTOM_DATA_PATH)
#     # Flatten all symptom names and diseases into keywords
#     for col in symptom_df.columns:
#         SYMPTOM_KEYWORDS.update(symptom_df[col].dropna().astype(str).str.lower().tolist())

# if os.path.exists(MEDICINE_DATA_PATH):
#     print("Loading medicine dataset...")
#     medicine_df = pd.read_csv(MEDICINE_DATA_PATH)
#     # Build a mapping: indication -> list of medicine names
#     for _, row in medicine_df.iterrows():
#         indication = str(row.get("Indication", "")).strip().lower()
#         name = str(row.get("Name", "")).strip()
#         if indication and name:
#             MEDICINE_MAP.setdefault(indication, []).append(name)

# # -------------------------
# # FastAPI app setup
# # -------------------------
# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -------------------------
# # Entity extraction
# # -------------------------
# def extract_entities(text: str) -> Dict[str, List[str]]:
#     doc = nlp(text)
#     entities = {"PATIENT_NAME": [], "DATE": [], "SYMPTOMS": [], "MEDICINES": []}

#     # Named Entities (Person, Date)
#     for ent in doc.ents:
#         if ent.label_ == "PERSON":
#             entities["PATIENT_NAME"].append(ent.text)
#         elif ent.label_ == "DATE":
#             entities["DATE"].append(ent.text)

#     lower_text = text.lower()

#     # Symptom extraction
#     for symptom in SYMPTOM_KEYWORDS:
#         if symptom in lower_text:
#             entities["SYMPTOMS"].append(symptom)

#     # Medicine recommendation based on detected symptoms or diseases
#     recommended = []
#     for symptom in entities["SYMPTOMS"]:
#         for indication, meds in MEDICINE_MAP.items():
#             if symptom in indication:
#                 recommended.extend(meds)

#     entities["MEDICINES"] = list(set(recommended))  # remove duplicates

#     # Clean duplicates
#     for k in entities:
#         entities[k] = list(set(entities[k]))

#     return entities

# # -------------------------
# # API Endpoints
# # -------------------------
# @app.post("/transcribe")
# async def transcribe_audio(file: UploadFile = File(...)):
#     """Transcribe audio and extract entities"""
#     suffix = os.path.splitext(file.filename)[-1]
#     if suffix not in [".wav", ".webm", ".ogg", ".mp3"]:
#         suffix = ".wav"

#     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#         tmp.write(await file.read())
#         tmp_path = tmp.name

#     print(f"Received file: {file.filename}, saved to {tmp_path}")

#     # Transcribe
#     result = whisper_model.transcribe(tmp_path, task="translate")
#     english_text = result["text"]

#     # Extract entities
#     entities = extract_entities(english_text)

#     os.remove(tmp_path)

#     # Suggest top 3 medicines if available
#     recommended = entities.get("MEDICINES", [])[:3] or ["Consult Doctor"]

#     return {
#         "transcript": english_text,
#         "entities": entities,
#         "recommendations": recommended,
#     }

# @app.get("/")
# def root():
#     return {"message": "AI Clinical Assistant Backend Running"}




















# import os
# import tempfile
# import pandas as pd
# import spacy
# import whisper
# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from typing import Dict, List

# # -------------------------
# # Load models
# # -------------------------
# print("Loading spaCy model...")
# nlp = spacy.load("en_core_web_sm")

# print("Loading Whisper model...")
# whisper_model = whisper.load_model("medium.en")  # use "small.en" if system is slow

# # -------------------------
# # Load datasets dynamically
# # -------------------------
# SYMPTOM_DATA_PATH = "datasets/disease_symptom.csv"
# MEDICINE_DATA_PATH = "datasets/medicines.csv"

# SYMPTOM_KEYWORDS = set()
# DISEASE_SYMPTOM_MAP = {}
# MEDICINE_MAP = {}

# # -------------------------
# # Process disease-symptom dataset
# # -------------------------
# if os.path.exists(SYMPTOM_DATA_PATH):
#     print("✅ Loading disease-symptom dataset...")
#     symptom_df = pd.read_csv(SYMPTOM_DATA_PATH)

#     # assuming first column is 'diseases' and rest are symptom names with 1/0
#     SYMPTOM_KEYWORDS = set(symptom_df.columns[1:].str.lower())  # all symptom names

#     # Map: disease -> list of symptoms with value == 1
#     for _, row in symptom_df.iterrows():
#         disease = str(row["diseases"]).strip().lower()
#         active_symptoms = [col.lower() for col in symptom_df.columns[1:] if row[col] == 1]
#         DISEASE_SYMPTOM_MAP[disease] = active_symptoms

# # -------------------------
# # Process medicine dataset
# # -------------------------
# if os.path.exists(MEDICINE_DATA_PATH):
#     print("✅ Loading medicine dataset...")
#     medicine_df = pd.read_csv(MEDICINE_DATA_PATH)

#     # Build a mapping: indication (disease/condition) -> list of medicine names
#     for _, row in medicine_df.iterrows():
#         indication = str(row.get("Indication", "")).strip().lower()
#         name = str(row.get("Name", "")).strip()
#         if indication and name:
#             MEDICINE_MAP.setdefault(indication, []).append(name)

# # -------------------------
# # FastAPI setup
# # -------------------------
# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -------------------------
# # Entity extraction
# # -------------------------
# def extract_entities(text: str) -> Dict[str, List[str]]:
#     doc = nlp(text)
#     entities = {"PATIENT_NAME": [], "DATE": [], "SYMPTOMS": [], "DISEASES": [], "MEDICINES": []}

#     # Extract PERSON and DATE entities
#     for ent in doc.ents:
#         if ent.label_ == "PERSON":
#             entities["PATIENT_NAME"].append(ent.text)
#         elif ent.label_ == "DATE":
#             entities["DATE"].append(ent.text)

#     lower_text = text.lower()

#     # Extract symptoms from text
#     detected_symptoms = [symptom for symptom in SYMPTOM_KEYWORDS if symptom in lower_text]
#     entities["SYMPTOMS"].extend(detected_symptoms)

#     # Predict diseases based on symptoms
#     matched_diseases = []
#     for disease, symptoms in DISEASE_SYMPTOM_MAP.items():
#         if any(symptom in detected_symptoms for symptom in symptoms):
#             matched_diseases.append(disease)
#     entities["DISEASES"] = list(set(matched_diseases))

#     # Recommend medicines
#     recommended_medicines = []
#     for disease in matched_diseases:
#         for indication, meds in MEDICINE_MAP.items():
#             if disease in indication:
#                 recommended_medicines.extend(meds)

#     entities["MEDICINES"] = list(set(recommended_medicines))

#     # Clean up duplicates
#     for key in entities:
#         entities[key] = list(set(entities[key]))

#     return entities

# # -------------------------
# # API Endpoints
# # -------------------------
# @app.post("/transcribe")
# async def transcribe_audio(file: UploadFile = File(...)):
#     """Transcribe audio and extract entities"""
#     suffix = os.path.splitext(file.filename)[-1]
#     if suffix not in [".wav", ".webm", ".ogg", ".mp3"]:
#         suffix = ".wav"

#     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#         tmp.write(await file.read())
#         tmp_path = tmp.name

#     print(f"Received file: {file.filename}, saved to {tmp_path}")

#     # Transcribe using Whisper
#     result = whisper_model.transcribe(tmp_path, task="translate")
#     english_text = result["text"]

#     # Extract entities
#     entities = extract_entities(english_text)

#     os.remove(tmp_path)

#     # Select top recommendations
#     recommended = entities.get("MEDICINES", [])[:3] or ["Consult Doctor"]

#     return {
#         "transcript": english_text,
#         "entities": entities,
#         "recommendations": recommended,
#     }

# @app.get("/")
# def root():
#     return {"message": "AI Clinical Assistant Backend Running ✅"}


















































# import os
# import tempfile
# import pandas as pd
# import spacy
# import whisper
# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from typing import Dict, List

# # -------------------------
# # Load models
# # -------------------------
# print("🧠 Loading scispaCy medical model...")
# nlp = spacy.load("en_ner_bc5cdr_md")  # detects diseases + chemicals (medicines)

# print("🎧 Loading Whisper model...")
# whisper_model = whisper.load_model("medium.en")  # use "small.en" if system is slow

# # -------------------------
# # Load datasets dynamically
# # -------------------------
# SYMPTOM_DATA_PATH = "datasets/disease_symptom.csv"
# MEDICINE_DATA_PATH = "datasets/medicines.csv"

# DISEASE_SYMPTOM_MAP = {}
# MEDICINE_MAP = {}

# # -------------------------
# # Process disease-symptom dataset (optional)
# # -------------------------
# if os.path.exists(SYMPTOM_DATA_PATH):
#     print("✅ Loading disease-symptom dataset...")
#     symptom_df = pd.read_csv(SYMPTOM_DATA_PATH)
#     for _, row in symptom_df.iterrows():
#         disease = str(row["diseases"]).strip().lower()
#         active_symptoms = [col.lower() for col in symptom_df.columns[1:] if row[col] == 1]
#         DISEASE_SYMPTOM_MAP[disease] = active_symptoms

# # -------------------------
# # Process medicine dataset
# # -------------------------
# if os.path.exists(MEDICINE_DATA_PATH):
#     print("✅ Loading medicine dataset...")
#     medicine_df = pd.read_csv(MEDICINE_DATA_PATH)
#     for _, row in medicine_df.iterrows():
#         indication = str(row.get("Indication", "")).strip().lower()
#         name = str(row.get("Name", "")).strip()
#         if indication and name:
#             MEDICINE_MAP.setdefault(indication, []).append(name)

# # -------------------------
# # FastAPI setup
# # -------------------------
# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -------------------------
# # Entity extraction using scispaCy
# # -------------------------
# def extract_entities(text: str) -> Dict[str, List[str]]:
#     doc = nlp(text)
#     entities = {"PATIENT_NAME": [], "DATE": [], "SYMPTOMS": [], "DISEASES": [], "MEDICINES": []}

#     lower_text = text.lower()

#     # Detect diseases and medicines using scispaCy labels
#     for ent in doc.ents:
#         if ent.label_ == "DISEASE":
#             entities["DISEASES"].append(ent.text.lower())
#         elif ent.label_ == "CHEMICAL":
#             entities["MEDICINES"].append(ent.text.lower())

#     # Optional: detect PERSON and DATE using simple spaCy (fallback)
#     base_nlp = spacy.load("en_core_web_sm")
#     base_doc = base_nlp(text)
#     for ent in base_doc.ents:
#         if ent.label_ == "PERSON":
#             entities["PATIENT_NAME"].append(ent.text)
#         elif ent.label_ == "DATE":
#             entities["DATE"].append(ent.text)

#     # Add simple symptom inference if dataset is available
#     detected_symptoms = []
#     for disease, symptoms in DISEASE_SYMPTOM_MAP.items():
#         for symptom in symptoms:
#             if symptom in lower_text:
#                 detected_symptoms.append(symptom)
#     entities["SYMPTOMS"] = list(set(detected_symptoms))

#     # Recommend medicines based on detected diseases
#     recommended_medicines = []
#     for disease in entities["DISEASES"]:
#         for indication, meds in MEDICINE_MAP.items():
#             if disease in indication:
#                 recommended_medicines.extend(meds)

#     # If still empty, try matching symptoms
#     if not recommended_medicines:
#         for symptom in entities["SYMPTOMS"]:
#             for indication, meds in MEDICINE_MAP.items():
#                 if symptom in indication:
#                     recommended_medicines.extend(meds)

#     entities["MEDICINES"] = list(set(recommended_medicines))

#     # Clean duplicates
#     for key in entities:
#         entities[key] = list(set(entities[key]))

#     return entities

# # -------------------------
# # API Endpoints
# # -------------------------
# @app.post("/transcribe")
# async def transcribe_audio(file: UploadFile = File(...)):
#     """Transcribe audio and extract medical entities"""
#     suffix = os.path.splitext(file.filename)[-1]
#     if suffix not in [".wav", ".webm", ".ogg", ".mp3"]:
#         suffix = ".wav"

#     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#         tmp.write(await file.read())
#         tmp_path = tmp.name

#     print(f"🎙 Received file: {file.filename}, saved to {tmp_path}")

#     # Transcribe
#     result = whisper_model.transcribe(tmp_path, task="translate")
#     english_text = result["text"]

#     # Extract entities
#     entities = extract_entities(english_text)

#     os.remove(tmp_path)

#     # Recommend top 3 medicines or fallback
#     recommended = entities.get("MEDICINES", [])[:3] or ["Consult Doctor"]

#     return {
#         "transcript": english_text,
#         "entities": entities,
#         "recommendations": recommended,
#     }

# @app.get("/")
# def root():
#     return {"message": "AI Clinical Assistant Backend Running ✅ with scispaCy"}






import os
import tempfile
import pandas as pd
import spacy
import whisper
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Dict, List
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime

# -------------------------
# Load models
# -------------------------
print("🧠 Loading scispaCy medical model...")
nlp = spacy.load("en_ner_bc5cdr_md")  # detects diseases + chemicals

print("🎧 Loading Whisper model...")
whisper_model = whisper.load_model("medium.en")  # use "small.en" if system is slow

# -------------------------
# Load datasets
# -------------------------
SYMPTOM_DATA_PATH = "datasets/disease_symptom.csv"
MEDICINE_DATA_PATH = "datasets/medicines.csv"

DISEASE_SYMPTOM_MAP = {}
MEDICINE_MAP = {}

if os.path.exists(SYMPTOM_DATA_PATH):
    print("✅ Loading disease-symptom dataset...")
    symptom_df = pd.read_csv(SYMPTOM_DATA_PATH)
    for _, row in symptom_df.iterrows():
        disease = str(row["diseases"]).strip().lower()
        active_symptoms = [col.lower() for col in symptom_df.columns[1:] if row[col] == 1]
        DISEASE_SYMPTOM_MAP[disease] = active_symptoms

if os.path.exists(MEDICINE_DATA_PATH):
    print("✅ Loading medicine dataset...")
    medicine_df = pd.read_csv(MEDICINE_DATA_PATH)
    for _, row in medicine_df.iterrows():
        indication = str(row.get("Indication", "")).strip().lower()
        name = str(row.get("Name", "")).strip()
        if indication and name:
            MEDICINE_MAP.setdefault(indication, []).append(name)

# -------------------------
# FastAPI setup
# -------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Entity Extraction
# -------------------------
def extract_entities(text: str) -> Dict[str, List[str]]:
    doc = nlp(text)
    entities = {"PATIENT_NAME": [], "DATE": [], "SYMPTOMS": [], "DISEASES": [], "MEDICINES": []}
    lower_text = text.lower()

    # Detect diseases & medicines using scispaCy
    for ent in doc.ents:
        if ent.label_ == "DISEASE":
            entities["DISEASES"].append(ent.text.lower())
        elif ent.label_ == "CHEMICAL":
            entities["MEDICINES"].append(ent.text.lower())

    # Add basic PERSON & DATE recognition
    base_nlp = spacy.load("en_core_web_sm")
    base_doc = base_nlp(text)
    for ent in base_doc.ents:
        if ent.label_ == "PERSON":
            entities["PATIENT_NAME"].append(ent.text)
        elif ent.label_ == "DATE":
            entities["DATE"].append(ent.text)

    # Detect symptoms (from dataset)
    detected_symptoms = []
    for disease, symptoms in DISEASE_SYMPTOM_MAP.items():
        for symptom in symptoms:
            if symptom in lower_text:
                detected_symptoms.append(symptom)
    entities["SYMPTOMS"] = list(set(detected_symptoms))

    # Recommend medicines
    recommended_medicines = []
    for disease in entities["DISEASES"]:
        for indication, meds in MEDICINE_MAP.items():
            if disease in indication:
                recommended_medicines.extend(meds)
    if not recommended_medicines:
        for symptom in entities["SYMPTOMS"]:
            for indication, meds in MEDICINE_MAP.items():
                if symptom in indication:
                    recommended_medicines.extend(meds)

    entities["MEDICINES"] = list(set(recommended_medicines))

    # Remove duplicates
    for key in entities:
        entities[key] = list(set(entities[key]))
    return entities

# -------------------------
# PDF Prescription Generator
# -------------------------
def generate_prescription_pdf(entities: Dict[str, List[str]], transcript: str) -> str:
    file_path = tempfile.mktemp(suffix=".pdf")
    c = canvas.Canvas(file_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(200, height - 50, "AI Clinical Prescription")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    y = height - 140
    c.drawString(50, y, f"Patient Name: {', '.join(entities.get('PATIENT_NAME', ['Not Found']))}")
    y -= 20
    c.drawString(50, y, f"Detected Symptoms: {', '.join(entities.get('SYMPTOMS', ['Not Detected']))}")
    y -= 20
    c.drawString(50, y, f"Predicted Diseases: {', '.join(entities.get('DISEASES', ['Not Found']))}")
    y -= 20
    c.drawString(50, y, f"Recommended Medicines: {', '.join(entities.get('MEDICINES', ['Consult Doctor']))}")

    y -= 40
    c.setFont("Helvetica-Oblique", 11)
    c.drawString(50, y, "Doctor Advice:")
    y -= 20
    c.setFont("Helvetica", 10)
    advice = "Take medicines as prescribed. Drink plenty of water. Consult a doctor if symptoms persist."
    c.drawString(70, y, advice)

    y -= 40
    c.setFont("Helvetica-Oblique", 11)
    c.drawString(50, y, "Original Transcript:")
    y -= 20
    c.setFont("Helvetica", 9)
    for line in transcript.split("."):
        c.drawString(70, y, line.strip())
        y -= 15
        if y < 100:
            c.showPage()
            y = height - 100
            c.setFont("Helvetica", 9)

    c.showPage()
    c.save()
    return file_path

# -------------------------
# API Endpoints
# -------------------------
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[-1]
    if suffix not in [".wav", ".webm", ".ogg", ".mp3"]:
        suffix = ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    print(f"🎙 Received file: {file.filename}, saved to {tmp_path}")

    result = whisper_model.transcribe(tmp_path, task="translate")
    english_text = result["text"]
    entities = extract_entities(english_text)
    os.remove(tmp_path)

    pdf_path = generate_prescription_pdf(entities, english_text)
    recommended = entities.get("MEDICINES", [])[:3] or ["Consult Doctor"]

    return {
        "transcript": english_text,
        "entities": entities,
        "recommendations": recommended,
        "pdf_file": pdf_path
    }

@app.get("/download_prescription")
def download_prescription(file_path: str):
    """Download generated prescription"""
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/pdf", filename="Prescription.pdf")
    return {"error": "File not found"}

@app.get("/")
def root():
    return {"message": "AI Clinical Assistant Running ✅ with scispaCy + PDF Prescription"}
