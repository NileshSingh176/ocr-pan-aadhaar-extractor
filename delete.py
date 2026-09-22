from database import cursor , conn

# cursor.execute("""DROP TABLE IF EXISTS aadhar_pan_01""")

def get_employee():
    cursor.execute(""" SELECT * FROM aadhar_pan_01 """)
    rows = cursor.fetchall()
    return rows
conn.commit()