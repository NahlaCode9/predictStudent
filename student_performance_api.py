# -*- coding: utf-8 -*-
import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI, Query, HTTPException
from enum import IntEnum

class Ethnicity(IntEnum):
    Caucasian = 0
    AfricanAmerican = 1
    Asian = 2
    Other = 3

class ParentalEducation(IntEnum):
    None_ = 0
    HighSchool = 1
    SomeCollege = 2
    Bachelors = 3
    Higher = 4

class ParentalSupport(IntEnum):
    None_ = 0
    Low = 1
    Moderate = 2
    High = 3
    VeryHigh = 4

class GradeClass(IntEnum):
    A = 0
    B = 1
    C = 2
    D = 3
    F = 4

app = FastAPI(
    title="واجهة برمجة تطبيقات التنبؤ بأداء الطلاب",
    description="واجهة برمجة تطبيقات للتنبؤ بفئة درجة الطالب بناءً على التفاصيل الديموغرافية وعادات الدراسة ومشاركة الوالدين والأنشطة اللامنهجية.",
    version="1.4.0"
)

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],  # يسمح بالوصول من أي مصدر. قم بتقييد هذا في الإنتاج
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

try:
    model = load_model("student_performance_api")
except Exception as e:
    print(f"خطأ في تحميل النموذج: {e}")
    raise HTTPException(status_code=500, detail="لم يتمكن من تحميل النموذج")

@app.get("/predict", response_model=dict)
async def predict(
    Age: int = Query(..., ge=15, le=18, description="عمر الطالب (15-18 سنة)"),
    Gender: int = Query(..., description="جنس الطالب (0: ذكر، 1: أنثى)"),
    Ethnicity: int = Query(..., ge=0, le=3, description="عرق الطالب (0: قوقازي، 1: أفريقي أمريكي، 2: آسيوي، 3: آخر)"),
    ParentalEducation: int = Query(..., ge=0, le=4, description="مستوى تعليم الوالدين (0: لا يوجد، 1: ثانوية عامة، 2: بعض الدراسة الجامعية، 3: بكالوريوس، 4: أعلى)"),
    StudyTimeWeekly: float = Query(..., ge=0, le=20, description="وقت الدراسة الأسبوعي بالساعات (0-20)"),
    Absences: int = Query(..., ge=0, le=30, description="عدد الغيابات خلال العام الدراسي (0-30)"),
    Tutoring: int = Query(..., description="حالة التدريس الخصوصي (0: لا، 1: نعم)"),
    ParentalSupport: int = Query(..., ge=0, le=4, description="مستوى دعم الوالدين (0: لا يوجد، 1: منخفض، 2: متوسط، 3: عالي، 4: عالي جداً)"),
    Extracurricular: int = Query(..., description="المشاركة في الأنشطة اللامنهجية (0: لا، 1: نعم)"),
    Sports: int = Query(..., description="المشاركة في الرياضة (0: لا، 1: نعم)"),
    Music: int = Query(..., description="المشاركة في الأنشطة الموسيقية (0: لا، 1: نعم)"),
    Volunteering: int = Query(..., description="المشاركة في العمل التطوعي (0: لا، 1: نعم)")
):
    try:
        input_data = pd.DataFrame([{
            'Age': Age,
            'Gender': Gender,
            'Ethnicity': Ethnicity,
            'ParentalEducation': ParentalEducation,
            'StudyTimeWeekly': StudyTimeWeekly,
            'Absences': Absences,
            'Tutoring': Tutoring,
            'ParentalSupport': ParentalSupport,
            'Extracurricular': Extracurricular,
            'Sports': Sports,
            'Music': Music,
            'Volunteering': Volunteering
        }])

        predictions = predict_model(model, data=input_data)
        
        predicted_class = predictions["prediction_label"].iloc[0]
        prediction_score = predictions["prediction_score"].iloc[0]

        grade_class = GradeClass(predicted_class)
        gpa_range = {
            GradeClass.A: '(90-100%)',
            GradeClass.B: '(80-89%)',
            GradeClass.C: '(70-79%)',
            GradeClass.D: '(60-69%)',
            GradeClass.F: "(أقل من 60%)"
        }[grade_class]

        grade_class_arabic = {
            GradeClass.A: "ممتاز",
            GradeClass.B: "جيد جدًا",
            GradeClass.C: "جيد",
            GradeClass.D: "مقبول",
            GradeClass.F: "راسب"
        }[grade_class]

        return {
            "فئة_الدرجة_المتوقعة": grade_class_arabic,
            "نطاق_المعدل_التراكمي_المتوقع": gpa_range,
            "درجة_التنبؤ": float(prediction_score),
            "رسالة": "تم التنبؤ بنجاح",
            "تفاصيل": f"يتنبأ النموذج بفئة درجة '{grade_class_arabic}'، والتي تتوافق مع نطاق معدل تراكمي {gpa_range}. درجة التنبؤ {prediction_score:.2f} تشير إلى ثقة النموذج في هذا التنبؤ."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"حدث خطأ أثناء التنبؤ: {str(e)}")
