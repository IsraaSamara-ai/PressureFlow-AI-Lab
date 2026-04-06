# app.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import *
from model import model as ai_model  # النموذج المدرب مسبقاً

# ========== إعداد الصفحة ==========
st.set_page_config(
    page_title="PressureFlow AI - مختبر ضغط الموائع",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CSS مخصص لتحسين المظهر ==========
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1E88E5; text-align: center; }
    .sub-header { font-size: 1.2rem; color: #0D47A1; margin-bottom: 20px; }
    .info-box { background-color: #f0f2f6; padding: 15px; border-radius: 10px; }
    .footer { text-align: center; margin-top: 50px; font-size: 0.9rem; color: gray; }
</style>
""", unsafe_allow_html=True)

# ========== العنوان والمطور ==========
st.markdown("<h1 class='main-header'>🌊 PressureFlow AI – مختبر ضغط الموائع والغازات</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>🔄 مع تنبؤ حركي بالذكاء الاصطناعي | محاكاة تفاعلية | أجهزة قياس افتراضية</p>", unsafe_allow_html=True)
st.caption("إعداد وتطوير: **Israa Samara** |  🧪 AI Motion Prediction | Fluid Mechanics")

# ========== الشريط الجانبي (المتغيرات) ==========
with st.sidebar:
    st.header("⚙️ إعدادات التجربة")
    experiment_type = st.selectbox("نوع المائع", ("سائل (مائع ساكن)", "غاز مثالي"))
    
    gravity = st.number_input("تسارع الجاذبية (m/s²)", value=9.81, step=0.1)
    
    if experiment_type == "سائل (مائع ساكن)":
        fluid_density = st.number_input("كثافة المائع (kg/m³)", value=1000.0, step=50.0, help="مثلاً: الماء 1000، الزيت 800")
        depth = st.slider("العمق (m)", 0.0, 20.0, 5.0, 0.1)
        # إظهار ضغط المائع
        pressure = fluid_pressure(fluid_density, gravity, depth)
        st.metric("💧 ضغط المائع الساكن (Pascal)", f"{pressure:.2f} Pa")
        st.caption(f"المعادلة: P = ρ·g·h = {fluid_density} × {gravity} × {depth}")
        
    else:  # غاز مثالي
        gas_type = st.selectbox("نوع الغاز", ("هيليوم (He)", "نيتروجين (N₂)", "ثاني أكسيد الكربون (CO₂)"))
        gas_density_map = {"هيليوم (He)": 0.1785, "نيتروجين (N₂)": 1.2506, "ثاني أكسيد الكربون (CO₂)": 1.977}
        gas_density = gas_density_map[gas_type]
        st.info(f"كثافة {gas_type} ≈ {gas_density} kg/m³ عند STP")
        volume_gas = st.number_input("حجم الوعاء (m³)", value=1.0, min_value=0.1, step=0.1)
        temperature = st.number_input("درجة الحرارة (K)", value=293.0, step=5.0)
        moles = st.number_input("عدد المولات (mol)", value=1.0, step=0.5)
        pressure = gas_pressure(moles, temperature, volume_gas)
        st.metric("🎈 ضغط الغاز (Pascal)", f"{pressure:.2f} Pa")
        st.caption(f"المعادلة: P = nRT/V  (R=8.314)")
        depth = 0.0  # الغاز لا يعتمد على العمق بهذا الشكل
    
    st.divider()
    st.header("🧪 الجسم والحركة")
    body_volume = st.number_input("حجم الجسم (m³)", value=0.01, step=0.005, format="%.4f")
    body_weight = st.number_input("وزن الجسم الحقيقي (N)", value=10.0, step=1.0)
    body_depth = st.slider("النقطة التي يقع فيها الجسم (عمق m)", 0.0, 20.0, 3.0, 0.1 if experiment_type=="سائل (مائع ساكن)" else 0.0, disabled=(experiment_type=="غاز مثالي"))
    
    # حساب قوة الطفو والوزن الظاهري
    if experiment_type == "سائل (مائع ساكن)":
        buoyant = buoyant_force(fluid_density, body_volume, gravity)
        app_weight = apparent_weight(body_weight, buoyant)
        st.metric("⬆️ قوة الطفو (N)", f"{buoyant:.2f}")
        st.metric("⚖️ الوزن الظاهري (N)", f"{app_weight:.2f}")
        st.caption("الوزن الظاهري = الوزن الحقيقي - قوة الطفو")
    else:
        buoyant = 0.0
        app_weight = body_weight
    
    # تنبؤ الذكاء الاصطناعي بالحركة
    if st.button("🚀 توقع سرعة الجسم (AI Prediction)"):
        with st.spinner("AI يتنبأ بحركة الجسم..."):
            if experiment_type == "سائل (مائع ساكن)":
                pred_velocity = predict_motion(ai_model, body_depth, fluid_density, body_volume)
            else:
                pred_velocity = np.random.uniform(0.5, 2.0)  # مثال للغاز
            st.session_state["pred_vel"] = pred_velocity
            st.success(f"🧠 السرعة النهائية المتوقعة: {pred_velocity:.2f} m/s")
    if "pred_vel" in st.session_state:
        st.info(f"📈 آخر توقع للسرعة: {st.session_state['pred_vel']:.2f} m/s")

# ========== الأعمدة الرئيسية ==========
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 توزيع الضغط مع العمق (رسم تفاعلي)")
    if experiment_type == "سائل (مائع ساكن)":
        depths_plot = np.linspace(0, max(1, depth*1.2), 50)
        pressures_plot = fluid_pressure(fluid_density, gravity, depths_plot)
        fig = px.line(x=depths_plot, y=pressures_plot, labels={"x": "العمق (m)", "y": "الضغط (Pa)"}, title="ضغط المائع الساكن")
        fig.add_vline(x=depth, line_dash="dash", line_color="red", annotation_text=f"عمق الجسم = {depth}m")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ضغط الغاز لا يعتمد على العمق، اعرض منحنى الضغط مقابل الحجم أو درجة الحرارة.")
        volumes = np.linspace(0.5, 2.0, 50)
        pressures_gas = [gas_pressure(moles, temperature, v) for v in volumes]
        fig = px.line(x=volumes, y=pressures_gas, labels={"x": "الحجم (m³)", "y": "الضغط (Pa)"}, title="علاقة الضغط بالحجم (قانون بويل)")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🎬 محاكاة حركة الجسم (رسوم متحركة سلسة)")
    if experiment_type == "سائل (مائع ساكن)" and body_depth > 0:
        # إنشاء إطار متحرك باستخدام Plotly
        vel = st.session_state.get("pred_vel", predict_motion(ai_model, body_depth, fluid_density, body_volume))
        times, positions = generate_motion_trajectory(body_depth, vel, steps=30)
        fig_anim = go.Figure(
            data=[go.Scatter(x=[0], y=[positions[0]], mode="markers+text", marker=dict(size=15, color="red"), text=["⚪"], textposition="middle center")],
            layout=go.Layout(
                xaxis=dict(range=[0, 6], title="الزمن (s)"),
                yaxis=dict(range=[0, body_depth+1], title="العمق (m)"),
                title="حركة الجسم نحو السطح (الطفو)",
                updatemenus=[dict(type="buttons", showactive=False, buttons=[dict(label="▶️ تشغيل", method="animate", args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}])])]
            ),
            frames=[go.Frame(data=[go.Scatter(x=[times[i]], y=[positions[i]], mode="markers+text", marker=dict(size=15, color="red"), text=["⚪"])]) for i in range(len(times))]
        )
        st.plotly_chart(fig_anim, use_container_width=True)
        st.caption(f"🕹️ السرعة المتوقعة: {vel:.2f} m/s | حركة تصاعدية بفعل الطفو")
    else:
        st.warning("اختر مائعاً ساكناً وعمقاً أكبر من صفر لتشغيل المحاكاة المتحركة.")

with col2:
    st.subheader("📟 أجهزة قياس الضغط")
    # جهاز 1: مقياس بوردون
    st.markdown("**🔧 مقياس بوردون (Bourdon gauge)**")
    st.progress(min(100, int(pressure/1000)), text=f"القراءة: {pressure:.0f} Pa")
    # جهاز 2: مانومتر رقمي
    st.markdown("**📱 مانومتر رقمي (Digital manometer)**")
    st.metric("الضغط المُقاس", f"{pressure:.2f} Pa", delta=None)
    # جهاز 3: مستشعر ضغط إلكتروني
    st.markdown("**⚡ مستشعر ضغط إلكتروني (Pressure transducer)**")
    st.code(f"Signal: {pressure/10000:.2f} V (4-20mA scale)", language="text")
    st.caption("الأجهزة تعرض نفس قيمة الضغط المحسوبة من المعادلات أعلاه.")
    
    st.divider()
    st.subheader("📘 شرح سريع (مصطلحات إنجليزية)")
    st.markdown("""
    - **Static Fluid Pressure**: `P = ρgh` (الضغط الساكن)  
    - **Ideal Gas Law**: `PV = nRT`  
    - **Buoyancy (Archimedes)**: قوة الطفو = وزن السائل المزاح  
    - **AI Motion Prediction**: نموذج انحدار خطي يتنبأ بالسرعة النهائية  
    - **Bourdon tube**, **Manometer**, **Transducer**: أجهزة قياس الضغط  
    """)

# ========== تذييل ==========
st.markdown("<div class='footer'>© 2025 PressureFlow AI Lab | Developed by Israa Samara | Streamlit + AI Motion Forecasting</div>", unsafe_allow_html=True)

# لتجنب أخطاء النموذج إذا لم يتم تحميله
if ai_model is None:
    st.error("⚠️ لم يتم تحميل نموذج الذكاء الاصطناعي، سيتم استخدام توقع افتراضي. تأكد من تشغيل train_motion_model()")