from database import conn, cursor

def save_document(details):

    cursor.execute("""
    INSERT INTO aadhar_pan_01 (
        document_type,
        extraction_time,
        name,
        dob,
        aadhaar_number,
        pan_number,
        father_name
    )
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (

        details.get("Document_Type"),
        details.get("Extraction_Time"),
        details.get("Name"),
        details.get("DOB"),
        details.get("Aadhaar Number"),
        details.get("PAN Number"),
        details.get("Father Name")

    ))
    conn.commit()

def get_employee():
    cursor.execute("""
                    SELECT *
                    FROM aadhar_pan_01
                   """)
    rows = cursor.fetchall()
    return rows     