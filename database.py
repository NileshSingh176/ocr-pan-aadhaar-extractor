import sqlite3

conn = sqlite3.connect(
    "documents.db",
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS aadhar_pan_01(

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    document_type TEXT,

    extraction_time TEXT,

    name TEXT,

    dob TEXT,

    aadhaar_number TEXT UNIQUE,

    pan_number TEXT UNIQUE,

    father_name TEXT

)
""")

conn.commit()