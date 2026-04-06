# model.py
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

def train_motion_model():
    """
    تدريب نموذج انحدار خطي للتنبؤ بالسرعة النهائية لجسم في مائع.
    المدخلات: [عمق الجسم (m), كثافة المائع (kg/m³), حجم الجسم (m³)]
    المخرجات: السرعة النهائية (m/s) (محاكاة باستخدام قانون أرخميدس + مقاومة لزجة)
    """
    np.random.seed(42)
    n_samples = 500
    depth = np.random.uniform(0.1, 10, n_samples)
    density = np.random.uniform(500, 1500, n_samples)
    volume = np.random.uniform(0.001, 0.1, n_samples)
    # محاكاة سرعة نهائية (علاقة خطية مع إضافة ضوضاء)
    velocity = 0.5 * depth + 0.02 * density + 5 * volume + np.random.normal(0, 0.5, n_samples)
    X = np.column_stack((depth, density, volume))
    y = velocity
    model = LinearRegression()
    model.fit(X, y)
    return model

# تحميل النموذج (سيتم تدريبه مرة واحدة عند بدء التشغيل)
try:
    model = train_motion_model()
except Exception as e:
    model = None
    print(f"⚠️ Error training model: {e}")