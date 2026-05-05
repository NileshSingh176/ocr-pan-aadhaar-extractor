from flask import Flask, render_template, request, send_from_directory, jsonify
import os
import re
import json
from ultralytics import YOLO
import easyocr
from datetime import datetime
from PIL import Image, ImageOps

app = Flask(__name__)

# -------------------- CONFIG --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
JSON_FOLDER = os.path.join(BASE_DIR, "json")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(JSON_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load your specific best model
model = YOLO("model/best.pt")

# OCR - Initializing with GPU for performance
reader = easyocr.Reader(['en'], gpu=True)

# -------------------- SERVE IMAGE --------------------
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# -------------------- HELPERS --------------------
def contains_hindi(text):
    return any('\u0900' <= char <= '\u097F' for char in text)

def clean_lines(lines):
    cleaned = []
    for l in lines:
        l_no_hindi = re.sub(r'[\u0900-\u097F]+', '', l).upper().strip()
        l_clean = re.sub(r'[^A-Z0-9/ ]+', ' ', l_no_hindi)
        l_final = re.sub(r'\s+', ' ', l_clean).strip()
        if l_final:
            cleaned.append(l_final)
    return cleaned

# -------------------- EXTRACTION ENGINE --------------------
def extract_details(lines, doc_type):
    data = {
        "Document_Type": doc_type.upper(),
        "Extraction_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    full_text = " ".join(lines)
    doc_type = doc_type.lower()

    # 1. DOB Extraction
    dob = re.findall(r'\d{2}/\d{2}/\d{4}', full_text)
    if dob: data["DOB"] = dob[0]

    # ---------------- AADHAAR LOGIC ----------------
    if "aadhar" in doc_type:
        # Aadhaar Number (12 digits)
        aadhaar = re.search(r'\d{4}\s\d{4}\s\d{4}', full_text)
        if not aadhaar:
            aadhaar = re.search(r'\d{12}', full_text.replace(" ", ""))
        if aadhaar: 
            data["Aadhaar Number"] = aadhaar.group()

        # Name: Look for text immediately after "Govt of India"
        for i, line in enumerate(lines):
            if "GOVERNMENT" in line or "INDIA" in line:
                for j in range(1, 3):
                    if i + j < len(lines):
                        cand = lines[i+j].strip()
                        if len(cand) > 3 and not any(c.isdigit() for c in cand):
                            if not any(x in cand for x in ["UIDAI", "AADHAAR", "DOB"]):
                                data["Name"] = cand
                                break
                if "Name" in data: break

    # ---------------- PAN LOGIC ----------------
    elif "pan" in doc_type:
        # PAN Number (5L + 4D + 1L)
        pan = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', full_text.replace(" ", ""))
        if pan: data["PAN Number"] = pan.group()

        for i, line in enumerate(lines):
            # Father's Name logic
            if "FATHER" in line and i + 1 < len(lines):
                cand = lines[i+1].strip()
                if not any(c.isdigit() for c in cand):
                    data["Father Name"] = cand
            
            # Holder's Name logic (Name but not Father)
            if "NAME" in line and "FATHER" not in line and i + 1 < len(lines):
                cand = lines[i+1].strip()
                if not any(c.isdigit() for c in cand) and cand != data.get("Father Name"):
                    data["Name"] = cand

    return data

# -------------------- MAIN ROUTE --------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if file:
            filename = file.filename.replace(" ", "_")
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(path)

            # 1. Classification
            res = model(path)
            doc_type = res[0].names[res[0].probs.top1]

            # 2. OCR
            raw = reader.readtext(path, detail=0)
            cleaned = clean_lines(raw)

            # 3. Extraction
            details = extract_details(cleaned, doc_type)

            # 4. JSON SAVE TO FOLDER
            json_filename = f"{os.path.splitext(filename)[0]}.json"
            with open(os.path.join(JSON_FOLDER, json_filename), 'w') as f:
                json.dump(details, f, indent=4)

            return render_template("index.html", image_path=filename, 
                                   doc_type=doc_type, details=details)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000)