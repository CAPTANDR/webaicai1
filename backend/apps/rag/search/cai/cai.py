import fitz  # PyMuPDF

def extract_text_from_pdf(data):
    """Extracts text from a PDF file."""
    document = fitz.open(data)
    text = ""
    for page in document:
        text += page.get_text()
    return text

data = 'open-webui-main\backend\apps\rag\search\cai\data.json'
pdf_text = extract_text_from_pdf(data)

# การตัดแยกคำถามและคำตอบ
import re
pattern = r"(คำถาม .+?)\n\n(คำตอบ .+?\")"
qa_pairs = re.findall(pattern, pdf_text, re.DOTALL)

# จัดเก็บเป็น JSON
import json
qa_data = [{"question": qa[0], "answer": qa[1]} for qa in qa_pairs]
with open('/mnt/data/qa_data.json', 'w', encoding='utf-8') as f:
    json.dump(qa_data, f, ensure_ascii=False, indent=2)
