import pandas as pd
import joblib

def main():
    # 1) قراءة ملف البيانات الجديد 
    df_new = pd.read_csv(r"C:\Users\Dell\Desktop\الذكاء الاصطناعي\Pearson-Raghad\GitHub\new_data.csv")

    # 2) تجهيز الأعمدة 
    df_new["subject"] = df_new["subject"].fillna("")
    df_new["body"] = df_new["body"].fillna("")
    df_new["text"] = (df_new["subject"] + " " + df_new["body"]).str.lower()

    # 3) تحميل النموذج وال vectorizer 
    model = joblib.load(r"C:\Users\Dell\Desktop\الذكاء الاصطناعي\Pearson-Raghad\GitHub\v.2.0.pkl")
    vectorizer = joblib.load(r"C:\Users\Dell\Desktop\الذكاء الاصطناعي\Pearson-Raghad\GitHub\vectorizer.pkl")

    # 4) تحويل النص إلى أرقام بنفس طريقة التدريب (Transform فقط)
    X_new = vectorizer.transform(df_new["text"])

    # 5) عمل تنبؤ بالفئات
    predictions = model.predict(X_new)

    # 6) إضافة النتائج وحفظ ملف جديد
    df_new["Predicted_Queue"] = predictions
    df_new.to_csv(r"C:\Users\Dell\Desktop\الذكاء الاصطناعي\Pearson-Raghad\GitHub\tickets_with_predictions.csv", index=False)


if __name__ == "__main__":
    main()