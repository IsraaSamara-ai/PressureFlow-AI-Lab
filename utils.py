# utils.py
import numpy as np

def fluid_pressure(density_fluid, gravity, depth):
    """ضغط المائع الساكن: P = ρgh"""
    return density_fluid * gravity * depth

def gas_pressure(moles, temperature, volume, R=8.314):
    """قانون الغاز المثالي: PV = nRT"""
    return (moles * R * temperature) / volume if volume > 0 else 0

def buoyant_force(density_fluid, volume_displaced, gravity):
    """قوة الطفو = ρ * V * g"""
    return density_fluid * volume_displaced * gravity

def apparent_weight(real_weight, buoyant_force):
    """الوزن الظاهري = الوزن الحقيقي - قوة الطفو"""
    return max(0, real_weight - buoyant_force)

def predict_motion(model, depth, density_fluid, volume_body):
    """استخدام نموذج AI للتنبؤ بالسرعة النهائية"""
    if model is None:
        return 0.0
    X = np.array([[depth, density_fluid, volume_body]])
    return model.predict(X)[0]

def generate_motion_trajectory(depth, velocity_final, steps=20):
    """محاكاة مسار الجسم (الزمن مقابل العمق) لرسم متحرك"""
    times = np.linspace(0, 5, steps)
    positions = depth - velocity_final * times  # حركة تصاعدية مبسطة
    positions = np.clip(positions, 0, depth)
    return times, positions