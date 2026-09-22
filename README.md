PAN & Aadhaar OCR Extractor

This project is a Python-based OCR application that I built to extract useful information from PAN cards and Aadhaar cards.

The main idea is simple: upload a document image, detect what type of document it is, read the text from it, and then extract important details such as name, date of birth, PAN number, Aadhaar number, etc.

I used YOLOv8 for document detection/classification and EasyOCR for text extraction. Flask is used to run the web application and SQLite is used to store the extracted information.

What it does

* Detects PAN and Aadhaar documents
* Extracts text from the uploaded document using EasyOCR
* Extracts important fields from the OCR text
* Handles basic OCR text cleaning and processing
* Stores extracted information in SQLite
* Generates JSON output
* Provides a simple Flask web interface

PAN details

The application can extract:

* Name
* Father’s name
* Date of birth
* PAN number

Aadhaar details

The application can extract:

* Name
* Aadhaar number
* Date of birth
* Gender

How the project works

The basic flow of the application is:

Document Image
      ↓
YOLOv8
      ↓
Document Type Detection
      ↓
EasyOCR
      ↓
OCR Text
      ↓
Text Cleaning & Processing
      ↓
Field Extraction
      ↓
JSON / SQLite

For example, when a PAN card image is uploaded, the system first identifies the document and then uses OCR to read the text. After that, custom extraction logic is used to find the PAN number, name, father’s name and date of birth.

Technologies I used

* Python
* Flask
* YOLOv8
* EasyOCR
* OpenCV
* NumPy
* PIL
* SQLite
* HTML/CSS
* JSON

Project structure

ocr-pan-aadhaar-extractor/
│
├── Model/
│   └── best.pt
│
├── Output/
│
├── static/
│
├── templates/
│
├── app.py
├── database.py
├── crud.py
├── requirements.txt
├── .gitignore
└── README.md

Screenshots

Home Page

PAN Extraction

Aadhaar Extraction

The screenshots are from test/sample documents and do not contain real personal information.

Running the project

Clone the repository:

git clone https://github.com/NileshSingh176/ocr-pan-aadhaar-extractor.git
cd ocr-pan-aadhaar-extractor

Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate

Install the required packages:

pip install -r requirements.txt

Run the Flask application:

python app.py

Then open the local URL shown in the terminal.

Output

After processing a document, the extracted information is converted into structured data.

For example:

{
    "document_type": "PAN",
    "name": "Sample Name",
    "father_name": "Sample Father",
    "dob": "01/01/1998",
    "pan_number": "ABCDE1234F"
}

The project also stores the extracted information in the SQLite database.

Why I built this

I worked on this project to understand how OCR and computer vision can be combined with normal Python application development.

One of the interesting parts for me was that OCR does not always return perfectly structured text. Because of this, I had to work on text cleaning, ordering OCR results and writing extraction rules to identify the required fields.

Future improvements

Some things I would like to improve in the future:

* Improve OCR accuracy on low-quality images
* Support more types of documents
* Handle different document layouts
* Add better document validation
* Add confidence scores for extracted fields
* Expose the extraction functionality through a REST API
* Improve the UI and error handling

Author

Nilesh Singh

B.Tech CSE — AI & ML

GitHub: NileshSingh176
