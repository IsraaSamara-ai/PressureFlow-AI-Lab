"""
PressureLab - Interactive Fluid & Gas Pressure Laboratory
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="PressureLab | معمل الضغط التفاعلي",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');
*{font-family:'Inter','Segoe UI',Tahoma,sans-serif;direction:rtl;}
.stApp{background:#070b14;color:#e2e8f0;}
.main{padding-top:2rem;}
.block-container{padding-top:2rem;padding-bottom:2rem;}

@keyframes fadeUp{from{opacity:0;transform:translateY(30px);}to{opacity:1;transform:translateY(0);}}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(0,212,255,0.3);}50%{box-shadow:0 0 20px 5px rgba(0,212,255,0.15);}}
@keyframes gradientMove{0%{background-position:0% 50%;}50%{background-position:100% 50%;}100%{background-position:0% 50%;}}
@keyframes shimmer{0%{background-position:-200% 0;}100%{background-position:200% 0;}}

.fade-up{animation:fadeUp .6s ease-out both;}
.delay-1{animation-delay:.1s;}.delay-2{animation-delay:.2s;}.delay-3{animation-delay:.3s;}

.hero-title{
    font-size:2.8rem;font-weight:900;line-height:1.2;
    background:linear-gradient(135deg,#00d4ff,#7c3aed,#f59e0b,#00d4ff);
    background-size:300% 300%;animation:gradientMove 6s ease infinite;
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    text-align:center;margin-bottom:.5rem;
}
.hero-sub{text-align:center;color:#94a3b8;font-size:1.1rem;margin-bottom:2rem;}

.card{
    background:linear-gradient(145deg,#0f1729,#131c31);
    border:1px solid rgba(255,255,255,0.06);border-radius:16px;
    padding:1.5rem;transition:all .35s ease;position:relative;overflow:hidden;
}
.card::before{
    content:'';position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,transparent,#00d4ff,transparent);
    opacity:0;transition:opacity .35s ease;
}
.card:hover{border-color:rgba(0,212,255,0.25);transform:translateY(-3px);box-shadow:0 8px 30px rgba(0,0,0,0.3);}
.card:hover::before{opacity:1;}

.metric-box{
    background:linear-gradient(145deg,#0d1321,#111b2e);
    border:1px solid rgba(255,255,255,0.05);border-radius:14px;
    padding:1.2rem;text-align:center;transition:all .3s ease;
}
.metric-box:hover{border-color:rgba(0,212,255,0.3);animation:pulse 2s infinite;}
.metric-val{
    font-size:2rem;font-weight:800;
    background:linear-gradient(135deg,#00d4ff,#38bdf8);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.metric-val.warm{
    background:linear-gradient(135deg,#f59e0b,#fbbf24);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.metric-val.purple{
    background:linear-gradient(135deg,#7c3aed,#a78bfa);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.metric-label{color:#64748b;font-size:.85rem;margin-bottom:.3rem;}
.metric-unit{color:#475569;font-size:.75rem;margin-top:.2rem;}

.formula-box{
    background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.15);
    border-radius:12px;padding:1rem 1.5rem;text-align:center;
    font-size:1.3rem;font-weight:700;color:#00d4ff;letter-spacing:1px;
    font-family:'Courier New',monospace;direction:ltr;
}

.section-title{
    font-size:1.6rem;font-weight:800;color:#f1f5f9;
    margin-bottom:.3rem;display:flex;align-items:center;gap:.5rem;
}
.section-desc{color:#64748b;font-size:.95rem;margin-bottom:1.5rem;}

.stTabs [data-baseweb="tab-list"]{gap:.5rem;background:transparent;border:none;}
.stTabs [data-baseweb="tab"]{
    background:#111827;border:1px solid rgba(255,255,255,0.06);
    border-radius:10px;padding:.5rem 1.2rem;color:#94a3b8;
    font-weight:600;font-size:.9rem;transition:all .3s ease;
}
.stTabs [data-baseweb="tab"]:hover{color:#e2e8f0;border-color:rgba(0,212,255,0.3);}
.stTabs [aria-selected="true"]{
    background:linear-gradient(135deg,rgba(0,212,255,0.15),rgba(124,58,237,0.15));
    border-color:rgba(0,212,255,0.4);color:#00d4ff;
}
.stTabs [data-baseweb="tab-highlight"]{background:transparent;}
.stTabs [data-baseweb="tab-border"]{display:none;}

.comparison-table{width:100%;border-collapse:separate;border-spacing:0;}
.comparison-table th{
    background:rgba(0,212,255,0.1);color:#00d4ff;padding:.7rem 1rem;
    font-weight:600;font-size:.85rem;text-align:center;border-bottom:2px solid rgba(0,212,255,0.2);
}
.comparison-table td{
    padding:.6rem 1rem;text-align:center;border-bottom:1px solid rgba(255,255,255,0.04);
    font-size:.85rem;color:#cbd5e1;
}
.comparison-table tr:hover td{background:rgba(255,255,255,0.02);}

.device-card{
    background:linear-gradient(145deg,#0f1729,#131c31);
    border:1px solid rgba(255,255,255,0.06);border-radius:14px;
    padding:1.5rem;transition:all .35s ease;
}
.device-card:hover{border-color:rgba(245,158,11,0.3);transform:scale(1.01);}

.shimmer-line{
    height:3px;border-radius:2px;
    background:linear-gradient(90deg,transparent,rgba(0,212,255,0.3),transparent);
    background-size:200% 100%;animation:shimmer 2s infinite;
    margin:1rem 0;
}

div[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0a0f1c,#070b14);
    border-left:1px solid rgba(255,255,255,0.05);
}
div[data-testid="stSidebar"] *{color:#cbd5e1;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# DATA CONSTANTS
# ═══════════════════════════════════════════════════════════════
FLUIDS = {
    "Water":       {"density": 1000,  "color": "#2196F3", "ar": "الماء",            "icon": "💧"},
    "Sea Water":   {"density": 1025,  "color": "#1565C0", "ar": "ماء البحر",         "icon": "🌊"},
    "Mercury":     {"density": 13546, "color": "#B0BEC5", "ar": "الزئبق",           "icon": "🪩"},
    "Oil":         {"density": 850,   "color": "#FF9800", "ar": "الزيت",            "icon": "🛢️"},
    "Glycerin":    {"density": 1261,  "color": "#EC407A", "ar": "الغليسرين",         "icon": "🧪"},
    "Ethanol":     {"density": 789,   "color": "#AB47BC", "ar": "الإيثانول",         "icon": "⚗️"},
}

GASES = {
    "Air":      {"M": 0.029, "rho": 1.225, "color": "#64B5F6", "ar": "الهواء",              "icon": "🌬️"},
    "Helium":   {"M": 0.004, "rho": 0.164, "color": "#FFD54F", "ar": "الهيليوم",            "icon": "🎈"},
    "CO2":      {"M": 0.044, "rho": 1.842, "color": "#81C784", "ar": "ثاني أكسيد الكربون",   "icon": "☁️"},
    "Nitrogen": {"M": 0.028, "rho": 1.165, "color": "#90CAF9", "ar": "النيتروجين",          "icon": "🟦"},
    "Oxygen":   {"M": 0.032, "rho": 1.331, "color": "#EF9A9A", "ar": "الأكسجين",            "icon": "🔴"},
    "Methane":  {"M": 0.016, "rho": 0.657, "color": "#CE93D8", "ar": "الميثان",             "icon": "🔥"},
}

PRESSURE_UNITS = {
    "Pa":  1,
    "kPa": 1000,
    "MPa": 1e6,
    "atm": 101325,
    "bar": 1e5,
    "mmHg": 133.322,
}

DEVICES_INFO = [
    {
        "name": "Mercury Barometer", "ar": "بارومتر الزئبق", "icon": "🌡️",
        "principle": "يعتمد على وزن عمود الزئبق الذي يتوازن مع الضغط الجوي. اخترعه إيفانجليستا توريشيلي عام 1643.",
        "range": "700 - 800 mmHg", "accuracy": "±0.5 mmHg",
        "uses": "قياس الضغط الجوي، التنبؤ بالطقس", "formula": "P = ρ × g × h"
    },
    {
        "name": "U-tube Manometer", "ar": "مانومتر الأنبوب على شكل U", "icon": "📏",
        "principle": "يقيس فرق الضغط بين نقطتين عن طريق فرق ارتفاع السائل في فرعي الأنبوب. يمكن استخدامه مع الزئبق أو الماء.",
        "range": "0 - 200 kPa", "accuracy": "±0.5 mm",
        "uses": "قياس ضغط الغازات والسوائل في المختبرات", "formula": "ΔP = ρ × g × Δh"
    },
    {
        "name": "Bourdon Tube Gauge", "ar": "مقياس أنبوب بوردون", "icon": "⚙️",
        "principle": "أنبوب معدني مسطح منحنٍ يتفتح قليلاً عند تعرضه للضغط، تحول هذه الحركة إلى مؤشر على dial.",
        "range": "0 - 1000 bar", "accuracy": "±1% من المدى الكامل",
        "uses": "الصناعة، أنظمة الهيدروليك، غلايات البخار", "formula": "Deformation ∝ Applied Pressure"
    },
    {
        "name": "Piezoelectric Sensor", "ar": "مستشعر الضغط الكهربائي الضغطي", "icon": "📡",
        "principle": "مواد بلورية تنتج شحنة كهربائية عند تعرضها لإجهاد ميكانيكي (ضغط). الشحنة تتناسب مع الضغط المؤثر.",
        "range": "0 - 700 MPa", "accuracy": "±0.5%",
        "uses": "محركات الاحتراق، المراقبة الصناعية، أبحاث الصدمات", "formula": "Q = d × F (d = piezoelectric coefficient)"
    },
    {
        "name": "Aneroid Barometer", "ar": "بارومتر الـ Aneroid (بدون سائل)", "icon": "🔄",
        "principle": "صندوق معدني مرن مفرغ جزئياً يتغير شكله مع تغير الضغط الجوي. الحركة تُنقل عبر نظام ميكانيكي إلى مؤشر.",
        "range": "870 - 1085 hPa", "accuracy": "±0.5 hPa",
        "uses": "التنبؤ بالطقس، الارتفاع عن سطح البحر، الملاحة", "formula": "ΔV ∝ ΔP (elastic deformation)"
    },
    {
        "name": "Digital Pressure Transducer", "ar": "محول الضغط الرقمي", "icon": "💻",
        "principle": "يجمع بين حساس ضغط ومعالج دقيق لتحويل الضغط إلى إشارة رقمية. يدعم بروتوكالات الاتصال المختلفة.",
        "range": "0 - 600 bar", "accuracy": "±0.1%",
        "uses": "الأنظمة الآلية، المراقبة عن بعد، IoT", "formula": "Digital Output = f(Pressure)"
    },
]

# ═══════════════════════════════════════════════════════════════
# PHYSICS FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def fluid_pressure(density, height, g=9.81):
    if height < 0:
        return 0.0
    return density * g * height

def total_pressure_at_point(density, depth, surface_pressure=101325.0, g=9.81):
    return surface_pressure + density * g * max(depth, 0)

def barometric_pressure(P0, M, h, T, g=9.81, R=8.314):
    exponent = -M * g * h / (R * T)
    if exponent < -500:
        return 0.0
    return P0 * math.exp(exponent)

def ideal_gas_pressure(n, T, V, R=8.314):
    if V <= 0:
        return float('inf')
    return n * R * T / V

def gravity_at_altitude(alt_m):
    g0 = 9.80665
    Re = 6_371_000
    return g0 * (Re / (Re + alt_m)) ** 2

def density_at_altitude(rho0, M, h, T, g=9.81, R=8.314):
    exponent = -M * g * h / (R * T)
    if exponent < -500:
        return 0.0
    return rho0 * math.exp(exponent)

def convert_pressure(p_pa, unit):
    if unit not in PRESSURE_UNITS:
        return p_pa
    return p_pa / PRESSURE_UNITS[unit]

# ═══════════════════════════════════════════════════════════════
# AI MODEL (Cached)
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def train_pressure_ai_model():
    np.random.seed(42)
    n_samples = 8000
    altitudes = np.random.uniform(0, 15000, n_samples)
    temps = np.random.uniform(220, 350, n_samples)
    gas_idx = np.random.randint(0, len(GASES), n_samples)
    molar_masses = np.array([list(GASES.values())[i]["M"] for i in gas_idx])
    base_pressures = np.random.uniform(95000, 105000, n_samples)

    pressures = np.array([
        barometric_pressure(base_pressures[i], molar_masses[i],
                          altitudes[i], temps[i])
        for i in range(n_samples)
    ])

    X = np.column_stack([altitudes, temps, molar_masses, base_pressures])
    feature_names = ["Altitude (m)", "Temperature (K)", "Molar Mass (kg/mol)", "Base Pressure (Pa)"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, pressures, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        min_samples_split=5, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "rmse": math.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
        "mape": np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100,
    }

    return model, feature_names, metrics, X_test, y_test, y_pred

# ═══════════════════════════════════════════════════════════════
# VISUALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════
def draw_fluid_container(fluid_key, fluid_depth, point_depth, container_h=10):
    fig, ax = plt.subplots(figsize=(5, 9))
    fig.patch.set_facecolor('#0a0e1a')
    ax.set_facecolor('#0a0e1a')

    fl = FLUIDS[fluid_key]
    w = 4

    container = Rectangle((0, 0), w, container_h, linewidth=2,
                           edgecolor='#475569', facecolor='none', zorder=3)
    ax.add_patch(container)

    fd = min(fluid_depth, container_h)
    fluid_rect = Rectangle((0, 0), w, fd, facecolor=fl['color'], alpha=0.35, zorder=2)
    ax.add_patch(fluid_rect)

    fluid_top = Rectangle((0, fd - 0.15), w, 0.3, facecolor=fl['color'], alpha=0.6, zorder=2)
    ax.add_patch(fluid_top)

    n_arrows = 6
    arrow_depths = np.linspace(fd * 0.1, fd * 0.9, n_arrows) if fd > 0.5 else []
    for ad in arrow_depths:
        p_norm = fluid_pressure(fl['density'], ad) / max(fluid_pressure(fl['density'], fd), 1)
        alen = 0.3 + 1.2 * p_norm
        alph = 0.3 + 0.5 * p_norm
        ax.annotate('', xy=(0.05, ad), xytext=(-alen, ad),
                    arrowprops=dict(arrowstyle='->', color='#fbbf24', lw=1.2 + p_norm, alpha=alph))
        ax.annotate('', xy=(w - 0.05, ad), xytext=(w + alen, ad),
                    arrowprops=dict(arrowstyle='->', color='#fbbf24', lw=1.2 + p_norm, alpha=alph))

    if fd > 0.5:
        ax.annotate('', xy=(w / 2, 0.05), xytext=(w / 2, -0.8),
                    arrowprops=dict(arrowstyle='->', color='#fbbf24', lw=2, alpha=0.9))

    pd_clamped = min(max(point_depth, 0), fd)
    ax.plot(w / 2, pd_clamped, 'o', color='#ef4444', markersize=14, zorder=5,
            markeredgecolor='white', markeredgewidth=2)
    p_at_point = fluid_pressure(fl['density'], pd_clamped)
    ax.annotate(f'P = {p_at_point / 1000:.2f} kPa',
                xy=(w / 2 + 0.3, pd_clamped), xytext=(w + 1.8, pd_clamped),
                fontsize=10, fontweight='bold', color='#ef4444',
                arrowprops=dict(arrowstyle='->', color='#ef4444', lw=1.5))

    for i in range(1, int(fd) + 1):
        if i <= fd:
            ax.plot([0, 0.3], [i, i], '-', color='white', alpha=0.3, lw=0.8)
            ax.text(-0.3, i, f'{i}m', color='#64748b', fontsize=7, ha='right', va='center')

    ax.text(w / 2, container_h + 0.4, 'P₀ (Atmospheric)', ha='center',
            color='#94a3b8', fontsize=9, style='italic')
    ax.text(w / 2, fd / 2, fl['ar'], ha='center', va='center',
            color=fl['color'], fontsize=16, fontweight='bold', alpha=0.7, zorder=4)

    ax.set_xlim(-2, w + 5)
    ax.set_ylim(-1.5, container_h + 1)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    return fig


def draw_building_section(num_floors, supply_kpa, floor_h=3.0):
    fig, ax = plt.subplots(figsize=(7, max(num_floors * 0.7 + 2, 4
