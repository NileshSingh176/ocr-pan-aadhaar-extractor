from flask import Flask, render_template, request, send_from_directory, jsonify
import os
import re
import json
import numpy as np
import cv2
from ultralytics import YOLO
import easyocr
from datetime import datetime
from PIL import Image
from crud import save_document

app = Flask(__name__)

# -------------------- CONFIG --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
JSON_FOLDER = os.path.join(BASE_DIR, "json")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(JSON_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load your specific best model
MODEL_PATH = os.path.join(BASE_DIR, "model", "best.pt")
model = YOLO(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# OCR - Initializing (using GPU=False as per uploaded sample, change to True if needed)
reader = easyocr.Reader(['en'], gpu=False, verbose=False)

# -------------------- SERVE IMAGE --------------------
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# -------------------- HELPERS FROM UPLOADED CODE --------------------
def clean_lines(lines):
    cleaned = []
    for l in lines:
        l = l.upper().strip()
        # Keep only letters, numbers, slashes and spaces
        l = re.sub(r'[^A-Z0-9/\s]+', '', l)
        # Remove extra spaces
        l = re.sub(r'\s+', ' ', l)
        if len(l) > 1:
            cleaned.append(l)
    return cleaned

def is_valid_name_line(line):
    """Returns True if the line looks like a human name."""
    words = line.split()
    if len(words) < 1 or len(words) > 4:
        return False
    good = 0
    bad = 0
    for w in words:
        if len(w) < 2:
            bad += 1
            continue
        if not re.search(r'[AEIOU]', w):
            bad += 1
            continue
        if re.search(r'(.)\1\1', w):
            bad += 1
            continue
        if len(w) > 15:
            bad += 1
            continue
        good += 1
    return True if (good > 0 and bad <= good) else False

# -------------------- EXTRACTION ENGINE (AS PER UPLOADED) --------------------
def extract_details(lines, doc_type):
    full_text = " ".join(lines)
    data = {
        "Document_Type": doc_type.upper(),
        "Extraction_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # =====================================================
    # ✅ PAN CARD LOGIC
    # =====================================================
    if "INCOME TAX" in full_text or re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', full_text):
        data["Document_Type"] = "PAN"
        
        pan_match = re.search(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', full_text)
        if pan_match:
            data["PAN Number"] = pan_match.group()

        dob_match = re.search(r'(\d{2}/\d{2}/(\d{4}))', full_text)
        if dob_match:
            data["DOB"] = dob_match.group(1)
            data["Year of Birth"] = dob_match.group(2)

        ignore_keywords = [
            "INCOME TAX", "DEPARTMENT", "GOVT", "GOVERNMENT",
            "PERMANENT", "ACCOUNT", "NUMBER", "CARD",
            "INDIA", "SIGNATURE", "NAME", "DATE", "BIRTH",
            "FATHER", "TAX", "SIGN"
        ]

        pan_anchor_idx = -1
        for idx, line in enumerate(lines):
            if re.search(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', line.strip().upper()):
                pan_anchor_idx = idx
                break

        start_idx = pan_anchor_idx + 1 if pan_anchor_idx != -1 else 0
        name_lines = []

        for idx in range(start_idx, len(lines)):
            line = lines[idx].strip().upper()
            if not line: continue
            if re.search(r'\d{2}/\d{2}/\d{4}', line): break
            if re.search(r'\d', line): continue
            if any(kw in line for kw in ignore_keywords): continue
            if not is_valid_name_line(line): continue
            
            name_lines.append(line)
            if len(name_lines) == 2: break

        data["Name"] = name_lines[0] if len(name_lines) >= 1 else "Not Detected"
        if len(name_lines) >= 2:
            data["Father Name"] = name_lines[1]
            
        return data

    # =====================================================
    # ✅ AADHAAR LOGIC
    # =====================================================
    else:
        data["Document_Type"] = "AADHAR"
        data["Name"] = "Not Detected"
        data["Year of Birth"] = "Not Detected"

        aadhaar = re.search(r'\d{4}\s\d{4}\s\d{4}', full_text)
        if aadhaar:
            data["Aadhaar Number"] = aadhaar.group()

        dob_match = re.search(r'(\d{2}/\d{2}/(\d{4}))', full_text)
        if dob_match:
            data["DOB"] = dob_match.group(1)
            data["Year of Birth"] = dob_match.group(2)
        else:
            yob_match = re.search(r'\b(19|20)\d{2}\b', full_text)
            if yob_match: data["Year of Birth"] = yob_match.group()

        if "FEMALE" in full_text: data["Gender"] = "Female"
        elif "MALE" in full_text: data["Gender"] = "Male"

        ignore_list = [
            "GOVERNMENT", "INDIA", "UIDAI", "AADHAAR", "FATHER",
            "ADDRESS", "MALE", "FEMALE", "HELP", "DOB", "YEAR",
            "BIRTH", "OF", "ENROLLMENT", "DOWNLOAD", "MOBILE",
            "VID", "DATE"
        ]

        valid_names = []
        for line in lines:
            line = line.strip()
            if any(word in line for word in ignore_list): continue
            if re.search(r'\d', line): continue
            words = line.split()
            if len(words) < 2 or len(words) > 4: continue
            
            bad_word = False
            for w in words:
                if len(w) <= 2 or not re.search(r'[AEIOU]', w) or re.search(r'(.)\1\1', w):
                    bad_word = True
            if bad_word: continue

            proper_word_count = sum(1 for w in words if len(w) >= 4 and re.search(r'[AEIOU]', w))
            if proper_word_count >= 2:
                valid_names.append(line)

        if valid_names:
            valid_names = sorted(valid_names, key=len, reverse=True) 
            data["Name"] = valid_names[0]

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
            doc_type = "UNKNOWN"
            if model:
                res = model(path)
                doc_type = res[0].names[res[0].probs.top1]

            # 2. Advanced OCR Processing (From uploaded code)
            img = np.array(Image.open(path).convert("RGB"))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            raw = reader.readtext(img, detail=1)

            if raw:
                # Sort top-to-bottom
                raw_sorted = sorted(raw, key=lambda r: r[0][0][1])
                
                # Merge horizontal fragments
                heights = [abs(b[0][2][1] - b[0][0][1]) for b in raw_sorted if abs(b[0][2][1] - b[0][0][1]) > 0]
                avg_height = (sum(heights) / len(heights)) if heights else 20
                row_threshold = avg_height * 0.6

                rows = []
                for det in raw_sorted:
                    top_y = det[0][0][1]
                    placed = False
                    for row in rows:
                        row_y = row[0][0][0][1]
                        if abs(top_y - row_y) <= row_threshold:
                            row.append(det)
                            placed = True
                            break
                    if not placed: rows.append([det])

                merged_lines = []
                for row in rows:
                    row.sort(key=lambda d: d[0][0][0])
                    merged_lines.append(" ".join(d[1] for d in row))

                cleaned = clean_lines(merged_lines)
                details = extract_details(cleaned, doc_type)
                save_document(details)

                # 3. JSON Save
                json_filename = f"{os.path.splitext(filename)[0]}.json"
                with open(os.path.join(JSON_FOLDER, json_filename), 'w') as f:
                    json.dump(details, f, indent=4)

                return render_template("index.html", image_path=filename, 
                                       doc_type=doc_type, details=details)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000)