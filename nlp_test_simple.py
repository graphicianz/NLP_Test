# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r'N:\VersionControl\thirdparty\Windowspython3_packages_nlp')
from transformers import pipeline


def interpret_note(note_body):
    # ใช้ pipeline สำหรับ sentiment-analysis (หรือปรับโมเดล text classification)
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    # วิเคราะห์ข้อความ
    result = classifier(note_body)[0]
    sentiment = result['label']  # LABEL: POSITIVE / NEGATIVE
    score = result['score']  # ความมั่นใจใน label ที่ได้

    # ตีความ
    if sentiment == "NEGATIVE":
        interpretation = "Feedback from client"
    elif sentiment == "POSITIVE":
        interpretation = "Approved"
    else:
        interpretation = "Needs manual review"

    # คืนค่าผลลัพธ์พร้อมคะแนน
    return f"{interpretation} (Score: {score:.2f})"


# รายการข้อความ
notes = [
    "Please fix the textures and lighting. It's not consistent.",
    "Great job on this! Everything looks perfect.",
    "Looks good",
    "Not bad",
    "I need more detail",
    "Can I have some minor fix?",
    "Take a look a frame 100, I think we need to talk",
    "I fucking love it",
    "UNBELIEVABLE, WELL,  I HAVE A HUGE PROBLEM WITH THIS TASK. IT'S THAT YOU HAVEN'T SEND IT FOR ME SOONER",
]

# วนลูปส่งข้อความทีละข้อความไปยังฟังก์ชัน
for note in notes:
    print(f"Note: {note}")
    print(interpret_note(note))
    print("-" * 50)
