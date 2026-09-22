# Identity Document Information Extractor

A professional AI-powered tool designed to classify and extract structured data from Indian Identity Documents (Aadhaar and PAN cards). This project uses YOLOv8 for document classification and EasyOCR for high-accuracy text extraction.

## 📂 Project Structure
```text
.
├── app.py              # Flask Backend & Extraction Logic
├── requirements.txt    # Dependency list for Windows & Mac
├── model/
│   └── best.pt         # Trained YOLOv8 Weights
├── templates/
│   └── index.html      # Web User Interface
├── uploads/            # Directory where uploaded images are stored
└── json/               # Directory where extracted data is saved as .json