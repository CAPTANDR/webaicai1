import json

# กำหนดเส้นทางของไฟล์ data.json
data_file_path = '/app/backend/apps/rag/search/cai/data.json'

# อ่านไฟล์ data.json
with open(data_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# ทำการประมวลผลข้อมูลใน data.json ตามที่คุณต้องการ
# ตัวอย่างเช่น การแสดงข้อมูลในไฟล์
print("Data from data.json:")
print(json.dumps(data, indent=4))

# คุณสามารถเพิ่มโค้ดที่ใช้ประมวลผลข้อมูลที่อยู่ใน data.json ได้ที่นี่
# เช่น การวิเคราะห์ข้อมูล หรือการอัปเดตข้อมูลในฐานข้อมูล
