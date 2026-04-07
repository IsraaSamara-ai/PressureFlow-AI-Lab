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
    fig, ax = plt.subplots(figsize=(7, max(num_floors * 0.7 + 2, 4)))
    fig.patch.set_facecolor('#0a0e1a')
    ax.set_facecolor('#0a0e1a')

    bw = 5
    fh_px = 1.2
    supply_pa = supply_kpa * 1000

    for i in range(num_floors + 1):
        y = i * fh_px
        slab = Rectangle((0, y), bw, 0.12, facecolor='#334155', edgecolor='#475569', lw=0.5)
        ax.add_patch(slab)

        p_pa = supply_pa - 1000 * 9.81 * i * floor_h
        p_kpa = p_pa / 1000

        if p_kpa >= 200:
            clr = '#10b981'
            status = 'Good'
        elif p_kpa >= 100:
            clr = '#f59e0b'
            status = 'Low'
        else:
            clr = '#ef4444'
            status = 'Critical'

        room = Rectangle((0.15, y + 0.15), bw - 0.3, fh_px - 0.18,
                         facecolor=clr, alpha=0.08, edgecolor=clr, lw=0.5)
        ax.add_patch(room)

        fname = "Ground" if i == 0 else f"F{i}"
        ax.text(-0.3, y + fh_px / 2, fname, color='#e2e8f0', fontsize=8,
                ha='right', va='center', fontweight='600')

        ax.text(bw + 0.3, y + fh_px / 2, f'{p_kpa:.1f} kPa',
                color=clr, fontsize=9, va='center', fontweight='bold')

        ax.text(bw + 2.5, y + fh_px / 2, status, color=clr, fontsize=7,
                va='center', style='italic', alpha=0.8)

    ax.annotate('Water Supply', xy=(bw / 2, -0.4), ha='center', va='top',
                color='#00d4ff', fontsize=9, fontweight='600')

    ax.set_xlim(-2, bw + 4)
    ax.set_ylim(-1, (num_floors + 1) * fh_px + 0.5)
    ax.set_aspect('auto')
    ax.axis('off')
    plt.tight_layout()
    return fig


def draw_barometer():
    fig, ax = plt.subplots(figsize=(3, 7))
    fig.patch.set_facecolor('#0a0e1a')
    ax.set_facecolor('#0a0e1a')

    ax.plot([1, 1], [1, 7], color='#475569', lw=2.5)
    ax.plot([2, 2], [1, 7], color='#475569', lw=2.5)
    ax.plot([1, 2], [7, 7], color='#475569', lw=2.5)

    ax.fill_between([1, 2], [1, 1], [4.5, 4.5], color='#B0BEC5', alpha=0.8)
    ax.fill_between([1, 2], [1, 1], [1.5, 1.5], color='#B0BEC5', alpha=0.4)

    ax.text(1.5, 5.8, 'Vacuum', ha='center', color='#64748b', fontsize=8, style='italic')
    ax.annotate('', xy=(2.3, 2.8), xytext=(2.3, 4.5),
                arrowprops=dict(arrowstyle='<->', color='#fbbf24', lw=1.5))
    ax.text(2.6, 3.65, 'h', color='#fbbf24', fontsize=12, fontweight='bold')
    ax.text(1.5, 0.5, 'P_atm', ha='center', color='#00d4ff', fontsize=9)

    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 7.5)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    return fig


def draw_manometer():
    fig, ax = plt.subplots(figsize=(5, 6))
    fig.patch.set_facecolor('#0a0e1a')
    ax.set_facecolor('#0a0e1a')

    ax.plot([1, 1], [2, 6], color='#475569', lw=2.5)
    ax.plot([2, 2], [2, 6], color='#475569', lw=2.5)
    ax.plot([4, 4], [2, 6], color='#475569', lw=2.5)
    ax.plot([5, 5], [2, 6], color='#475569', lw=2.5)
    ax.plot([1, 2], [6, 6], color='#475569', lw=2.5)
    ax.plot([4, 5], [6, 6], color='#475569', lw=2.5)
    ax.plot([2, 4], [2, 2], color='#475569', lw=2.5)

    ax.fill_between([1, 2], [2, 2], [3.5, 3.5], color='#2196F3', alpha=0.5)
    ax.fill_between([4, 5], [2, 2], [4.5, 4.5], color='#2196F3', alpha=0.5)
    ax.fill_between([2, 4], [2, 2], [2.3, 2.3], color='#2196F3', alpha=0.5)

    ax.text(1.5, 5.3, 'P1', ha='center', color='#ef4444', fontsize=11, fontweight='bold')
    ax.text(4.5, 5.3, 'P2', ha='center', color='#10b981', fontsize=11, fontweight='bold')
    ax.annotate('', xy=(3, 3.5), xytext=(3, 4.5),
                arrowprops=dict(arrowstyle='<->', color='#fbbf24', lw=1.5))
    ax.text(3.4, 4, 'Δh', color='#fbbf24', fontsize=10, fontweight='bold')

    ax.set_xlim(0, 6)
    ax.set_ylim(1, 6.5)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    return fig


def draw_bourdon():
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor('#0a0e1a')
    ax.set_facecolor('#0a0e1a')

    theta = np.linspace(np.pi * 0.2, np.pi * 1.6, 100)
    r = 2
    x_inner = (r - 0.15) * np.cos(theta)
    y_inner = (r - 0.15) * np.sin(theta)
    x_outer = (r + 0.15) * np.cos(theta)
    y_outer = (r + 0.15) * np.sin(theta)

    ax.fill(np.concatenate([x_inner, x_outer[::-1]]),
            np.concatenate([y_inner, y_outer[::-1]]),
            color='#475569', alpha=0.6)
    ax.plot(x_inner, y_inner, color='#94a3b8', lw=1.5)
    ax.plot(x_outer, y_outer, color='#94a3b8', lw=1.5)

    ax.plot([r * np.cos(np.pi * 0.2)], [r * np.sin(np.pi * 0.2)],
            'o', color='#fbbf24', markersize=8)
    ax.text(r * np.cos(np.pi * 0.2) - 0.5, r * np.sin(np.pi * 0.2) - 0.4,
            'Input', color='#fbbf24', fontsize=8)

    end_x = r * np.cos(np.pi * 1.6)
    end_y = r * np.sin(np.pi * 1.6)
    ax.plot([end_x, end_x + 1.2], [end_y, end_y + 0.8],
            color='#ef4444', lw=2)
    ax.plot(end_x + 1.2, end_y + 0.8, 'v', color='#ef4444', markersize=8)
    ax.text(end_x + 1.4, end_y + 0.9, 'Pointer', color='#ef4444', fontsize=8)

    arc_t = np.linspace(np.pi * 1.2, np.pi * 1.8, 50)
    ax.plot(3.5 * np.cos(arc_t), 3.5 * np.sin(arc_t), color='#64748b', lw=1, alpha=0.5)
    for i, t in enumerate(np.linspace(np.pi * 1.2, np.pi * 1.8, 6)):
        ax.plot([3.3 * np.cos(t), 3.7 * np.cos(t)],
                [3.3 * np.sin(t), 3.7 * np.sin(t)], color='#64748b', lw=0.5)

    ax.set_xlim(-3, 5)
    ax.set_ylim(-1, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
def render_sidebar():
    st.sidebar.markdown("""
    <div style="text-align:center;padding:1rem 0;">
        <div style="font-size:2rem;">🔬</div>
        <div style="font-size:1.1rem;font-weight:800;color:#00d4ff;">PressureLab</div>
        <div style="font-size:.75rem;color:#64748b;">Fluid & Gas Pressure Lab</div>
    </div>
    <div class="shimmer-line"></div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("#### 📐 محول الوحدات")
    with st.sidebar.form("unit_converter"):
        val = st.number_input("القيمة", value=101.325, format="%.4f")
        from_u = st.selectbox("من وحدة", list(PRESSURE_UNITS.keys()), index=2)
        to_u = st.selectbox("إلى وحدة", list(PRESSURE_UNITS.keys()), index=0)
        submitted = st.form_submit_button("تحويل", use_container_width=True)
        if submitted:
            pa = val * PRESSURE_UNITS[from_u]
            result = pa / PRESSURE_UNITS[to_u]
            st.success(f"**{val:.4f} {from_u} = {result:.6f} {to_u}**")

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📋 ثوابت فيزيائية")
    st.sidebar.markdown("""
    <div style="font-size:.8rem;color:#94a3b8;line-height:2;">
    <b>g₀</b> = 9.80665 m/s²<br>
    <b>P₀</b> = 101,325 Pa (1 atm)<br>
    <b>R</b> = 8.314 J/(mol·K)<br>
    <b>T₀</b> = 288.15 K (15°C)<br>
    <b>R_earth</b> = 6,371 km
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align:center;font-size:.7rem;color:#475569;padding:.5rem;">
    PressureLab v1.0<br>
    Educational Purpose Only
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SECTION: INTRODUCTION
# ═══════════════════════════════════════════════════════════════
def show_introduction():
    st.markdown('<div class="hero-title">PressureLab</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">معمل تفاعلي لاستكشاف ضغط المائع والغاز — من المبادئ الأساسية إلى التنبؤ بالذكاء الاصطناعي</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="shimmer-line"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="card fade-up">
            <div style="font-size:2rem;margin-bottom:.5rem;">💧</div>
            <div style="font-weight:700;font-size:1.05rem;color:#e2e8f0;margin-bottom:.5rem;">ضغط المائع</div>
            <div style="font-size:.85rem;color:#94a3b8;line-height:1.7;">
            دراسة الضغط الهيدروستاتيكي (Hydrostatic Pressure) الناتج عن وزن عمود المائع، والعوامل المؤثرة فيه: الكثافة (Density)، العمق (Depth)، وتسارع الجاذبية (Gravity).
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card fade-up delay-1">
            <div style="font-size:2rem;margin-bottom:.5rem;">🌬️</div>
            <div style="font-weight:700;font-size:1.05rem;color:#e2e8f0;margin-bottom:.5rem;">ضغط الغاز</div>
            <div style="font-size:.85rem;color:#94a3b8;line-height:1.7;">
            استكشاف سلوك ضغط الغاز مع الارتفاع باستخدام المعادلة البارومترية (Barometric Formula) وقانون الغاز المثالي (Ideal Gas Law)، ومقارنة غازات مختلفة.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="card fade-up delay-2">
            <div style="font-size:2rem;margin-bottom:.5rem;">🤖</div>
            <div style="font-weight:700;font-size:1.05rem;color:#e2e8f0;margin-bottom:.5rem;">التنبؤ بالذكاء الاصطناعي</div>
            <div style="font-size:.85rem;color:#94a3b8;line-height:1.7;">
            نموذج تعلم آلي (Machine Learning) مدرب على بيانات فيزيائية حقيقية للتنبؤ بضغط الغاز عند أي ارتفاع ودرجة حرارة ونوع غاز.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div class="formula-box">P = ρ × g × h</div>
        <div style="text-align:center;color:#64748b;font-size:.8rem;margin-top:.5rem;">
        معادلة ضغط المائع الساكن<br>
        <span style="color:#94a3b8;">Hydrostatic Pressure Equation</span>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="formula-box">P = P₀ × e<sup>-Mgh/RT</sup></div>
        <div style="text-align:center;color:#64748b;font-size:.8rem;margin-top:.5rem;">
        المعادلة البارومترية<br>
        <span style="color:#94a3b8;">Barometric Formula</span>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="formula-box">PV = nRT</div>
        <div style="text-align:center;color:#64748b;font-size:.8rem;margin-top:.5rem;">
        قانون الغاز المثالي<br>
        <span style="color:#94a3b8;">Ideal Gas Law</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <div style="font-weight:700;font-size:1.05rem;color:#e2e8f0;margin-bottom:.8rem;">📐 العوامل المؤثرة على الضغط — ملخص مقارن</div>
        <table class="comparison-table">
            <thead>
                <tr>
                    <th>العامل / Factor</th>
                    <th>تأثيره على ضغط المائع</th>
                    <th>تأثيره على ضغط الغاز</th>
                </tr>
            </thead>
            <tbody>
                <tr><td>الكثافة (Density ρ)</td><td>علاقة طردية مباشرة</td><td>يؤثر عبر الكتلة المولية M</td></tr>
                <tr><td>العمق/الارتفاع (h)</td><td>علاقة طردية (يزداد مع العمق)</td><td>علاقة عكسية أُسّية (ينقص مع الارتفاع)</td></tr>
                <tr><td>الجاذبية (g)</td><td>علاقة طردية مباشرة</td><td>علاقة عكسية (في المعادلة البارومترية)</td></tr>
                <tr><td>درجة الحرارة (T)</td><td>تأثير ضعيف (عبر تغير الكثافة)</td><td>علاقة طردية (في المعادلة البارومترية)</td></tr>
                <tr><td>الحجم (V)</td><td>لا يؤثر (مائع غير قابل للانضغاط)</td><td>علاقة عكسية (PV=nRT)</td></tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SECTION: FLUID PRESSURE
# ═══════════════════════════════════════════════════════════════
def show_fluid_pressure():
    st.markdown('<div class="section-title">💧 ضغط المائع الساكن</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Hydrostatic Pressure — تفاعل مباشر مع معادلة P = ρgh وتطبيقاتها في الأبنية السكنية</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="margin-bottom:1.5rem;">
        <div style="color:#94a3b8;font-size:.9rem;line-height:1.8;">
        <b style="color:#00d4ff;">المبدأ الأساسي:</b> الضغط الهيدروستاتيكي هو الضغط الذي يمارسه عمود المائع الساكن due to its weight.
        يزداد الضغط خطياً مع العمق ويكون متساوياً في جميع الاتجاهات عند نقطة معينة
        (Pascal's Law — قانون باسكال).
        <br><br>
        <b style="color:#f59e0b;">المعادلة:</b> &nbsp; <span style="font-family:monospace;color:#00d4ff;">P = ρ × g × h</span>
        &nbsp; حيث: ρ = الكثافة (kg/m³) ، g = الجاذبية (m/s²) ، h = العمق (m)
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_vis, col_ctrl = st.columns([1, 1])

    with col_ctrl:
        st.markdown('<div class="card"><div style="font-weight:700;color:#e2e8f0;margin-bottom:1rem;">⚙️ معلمات التجربة</div>',
                    unsafe_allow_html=True)
        fluid_key = st.selectbox(
            "نوع المائع / Fluid Type",
            list(FLUIDS.keys()),
            format_func=lambda x: f"{FLUIDS[x]['icon']} {FLUIDS[x]['ar']} ({x}) — ρ = {FLUIDS[x]['density']} kg/m³"
        )
        fl = FLUIDS[fluid_key]

        container_h = st.slider("ارتفاع الحاوية / Container Height (m)", 1.0, 50.0, 10.0, 0.5)
        fluid_depth = st.slider("عمق المائع / Fluid Depth (m)", 0.0, container_h, min(8.0, container_h), 0.1)
        point_depth = st.slider("نقطة القياس / Measurement Point Depth (m)", 0.0, fluid_depth, fluid_depth * 0.5, 0.1)
        alt_site = st.slider("ارتفاع الموقع / Site Altitude (m)", 0, 5000, 0, 50)

        g_local = gravity_at_altitude(alt_site)
        p_gauge = fluid_pressure(fl['density'], point_depth, g_local)
        p_total = total_pressure_at_point(fl['density'], point_depth, 101325.0, g_local)

        st.markdown('</div>', unsafe_allow_html=True)

    with col_vis:
        fig_c = draw_fluid_container(fluid_key, fluid_depth, point_depth, container_h)
        st.pyplot(fig_c, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-box"><div class="metric-label">ضغط القياس (Gauge)</div>
        <div class="metric-val">{p_gauge/1000:.2f}</div><div class="metric-unit">kPa</div></div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-box"><div class="metric-label">الضغط الكلي (Absolute)</div>
        <div class="metric-val warm">{p_total/1000:.2f}</div><div class="metric-unit">kPa</div></div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-box"><div class="metric-label">الجاذبية المحلية</div>
        <div class="metric-val purple">{g_local:.4f}</div><div class="metric-unit">m/s²</div></div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-box"><div class="metric-label">الضغط الجوي المعدّل</div>
        <div class="metric-val">{101325/1000:.2f}</div><div class="metric-unit">kPa (sea level)</div></div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    depths_arr = np.linspace(0, fluid_depth, 200)
    pressures_arr = fl['density'] * g_local * depths_arr / 1000

    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=pressures_arr, y=depths_arr, mode='lines',
                               line=dict(color=fl['color'], width=3), name='P vs Depth'))
    fig_p.add_trace(go.Scatter(x=[p_gauge / 1000], y=[point_depth], mode='markers',
                               marker=dict(color='#ef4444', size=14, symbol='circle',
                                           line=dict(color='white', width=2)),
                               name=f'Point ({p_gauge/1000:.1f} kPa)'))
    fig_p.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', family='Inter'),
        xaxis_title="Pressure (kPa)", yaxis_title="Depth (m)",
        xaxis=dict(gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.1)',
                   autorange='reversed'),
        margin=dict(l=60, r=30, t=30, b=60), height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig_p, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <div style="font-weight:700;font-size:1.05rem;color:#e2e8f0;margin-bottom:1rem;">
        📊 مقارنة ضغط جميع الموائع عند نفس العمق
        </div>
    """, unsafe_allow_html=True)
    comp_depth = st.slider("العمق المشترك للمقارنة / Common Depth (m)", 0.5, 30.0, 5.0, 0.5,
                           key="comp_depth")

    rows = []
    for name, data in FLUIDS.items():
        p = data['density'] * g_local * comp_depth
        rows.append({
            "المائع": f"{data['icon']} {data['ar']}",
            "Fluid": name,
            "ρ (kg/m³)": data['density'],
            f"P at {comp_depth}m (kPa)": round(p / 1000, 2),
            f"P (atm)": round(p / 101325, 4),
            f"P (bar)": round(p / 1e5, 4),
        })
    df_comp = pd.DataFrame(rows)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Residential Building
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="font-size:1.3rem;">🏢 ضغط الماء في المبنى السكني</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Residential Water Pressure — حساب الضغط عند كل طابق وتحديد الحاجة لمضخة رافعة (Booster Pump)</div>',
                unsafe_allow_html=True)

    bc1, bc2 = st.columns([1, 1])
    with bc1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        supply_kpa = st.number_input("ضغط التغذية / Supply Pressure (kPa)", 100.0, 800.0, 350.0, 5.0, key="bldg_supply")
        n_floors = st.slider("عدد الطوابق / Number of Floors", 1, 30, 10, 1, key="bldg_floors")
        floor_h = st.slider("ارتفاع الطابق / Floor Height (m)", 2.5, 5.0, 3.0, 0.1, key="bldg_fh")
        st.markdown('</div>', unsafe_allow_html=True)

        supply_pa = supply_kpa * 1000
        min_floor_pressure = supply_pa - 1000 * 9.81 * n_floors * floor_h
        min_kpa = min_floor_pressure / 1000

        if min_kpa >= 200:
            st.success(f"✅ الضغط في أعلى طابق: **{min_kpa:.1f} kPa** — جيد (لا حاجة لمضخة)")
        elif min_kpa >= 100:
            st.warning(f"⚠️ الضغط في أعلى طابق: **{min_kpa:.1f} kPa** — ضعيف (يُنصح بمضخة رافعة)")
        else:
            st.error(f"❌ الضغط في أعلى طابق: **{min_kpa:.1f} kPa** — حرج (مضخة رافعة ضرورية)")

    with bc2:
        fig_b = draw_building_section(n_floors, supply_kpa, floor_h)
        st.pyplot(fig_b, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# SECTION: GAS PRESSURE
# ═══════════════════════════════════════════════════════════════
def show_gas_pressure():
    st.markdown('<div class="section-title">🌬️ ضغط الغاز والمعادلة البارومترية</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Gas Pressure & Barometric Formula — تأثير الارتفاع ودرجة الحرارة ونوع الغاز على الضغط</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="margin-bottom:1.5rem;">
        <div style="color:#94a3b8;font-size:.9rem;line-height:1.8;">
        <b style="color:#00d4ff;">المعادلة البارومترية (Barometric Formula):</b>
        &nbsp;<span style="font-family:monospace;color:#00d4ff;">P = P₀ × e<sup>-Mgh / RT</sup></span>
        <br>حيث: P₀ = الضغط عند سطح البحر (101325 Pa) ، M = الكتلة المولية (kg/mol) ،
        g = الجاذبية (m/s²) ، h = الارتفاع (m) ، R = ثابت الغاز (8.314 J/(mol·K)) ،
        T = درجة الحرارة المطلقة (K)
        <br><br>
        <b style="color:#f59e0b;">قانون الغاز المثالي (Ideal Gas Law):</b>
        &nbsp;<span style="font-family:monospace;color:#f59e0b;">PV = nRT → P = nRT / V</span>
        <br>حيث: n = عدد المولات ، V = الحجم (m³)
        </div>
    </div>
    """, unsafe_allow_html=True)

    g1, g2 = st.columns([1, 1])
    with g1:
        st.markdown('<div class="card"><div style="font-weight:700;color:#e2e8f0;margin-bottom:1rem;">⚙️ معلمات الغاز</div>',
                    unsafe_allow_html=True)
        gas_key = st.selectbox(
            "نوع الغاز / Gas Type", list(GASES.keys()),
            format_func=lambda x: f"{GASES[x]['icon']} {GASES[x]['ar']} ({x}) — M = {GASES[x]['M']} kg/mol",
            key="gas_sel"
        )
        gas = GASES[gas_key]

        altitude = st.slider("الارتفاع / Altitude (m)", 0, 40000, 5000, 100, key="alt_gas")
        temp_c = st.slider("درجة الحرارة / Temperature (°C)", -50, 50, 15, 1, key="temp_gas")
        temp_k = temp_c + 273.15

        g_alt = gravity_at_altitude(altitude)
        p_calc = barometric_pressure(101325, gas['M'], altitude, temp_k, g_alt)
        rho_alt = density_at_altitude(gas['rho'], gas['M'], altitude, temp_k, g_alt)

        st.markdown('</div>', unsafe_allow_html=True)

    with g2:
        gm1, gm2 = st.columns(2)
        with gm1:
            st.markdown(f"""
            <div class="metric-box"><div class="metric-label">الضغط / Pressure</div>
            <div class="metric-val">{p_calc/1000:.2f}</div><div class="metric-unit">kPa</div></div>
            """, unsafe_allow_html=True)
        with gm2:
            st.markdown(f"""
            <div class="metric-box"><div class="metric-label">الكثافة / Density</div>
            <div class="metric-val warm">{rho_alt:.4f}</div><div class="metric-unit">kg/m³</div></div>
            """, unsafe_allow_html=True)

        gm3, gm4 = st.columns(2)
        with gm3:
            st.markdown(f"""
            <div class="metric-box"><div class="metric-label">الضغط / Pressure</div>
            <div class="metric-val purple">{p_calc/101325:.4f}</div><div class="metric-unit">atm</div></div>
            """, unsafe_allow_html=True)
        with gm4:
            pct = (p_calc / 101325) * 100
            st.markdown(f"""
            <div class="metric-box"><div class="metric-label">نسبة الضغط الجوي</div>
            <div class="metric-val">{pct:.1f}</div><div class="metric-unit">% of sea level</div></div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-weight:700;color:#e2e8f0;font-size:1.05rem;margin-bottom:.5rem;">📈 مقارنة ضغط الغازات مع الارتفاع</div>',
                unsafe_allow_html=True)

    alt_range = np.linspace(0, 40000, 500)
    fig_gas = go.Figure()
    for gname, gdata in GASES.items():
        p_arr = np.array([barometric_pressure(101325, gdata['M'], a, temp_k, g_alt) for a in alt_range])
        fig_gas.add_trace(go.Scatter(
            x=alt_range / 1000, y=p_arr / 1000,
            mode='lines', name=f"{gdata['ar']} ({gname})",
            line=dict(color=gdata['color'], width=2.5),
        ))

    fig_gas.add_trace(go.Scatter(
        x=[altitude / 1000], y=[p_calc / 1000],
        mode='markers', marker=dict(color='#ef4444', size=12, symbol='star',
                                     line=dict(color='white', width=2)),
        name='Selected Point', showlegend=True
    ))

    fig_gas.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', family='Inter'),
        xaxis_title="Altitude (km)", yaxis_title="Pressure (kPa)",
        xaxis=dict(gridcolor='rgba(255,255,255,0.06)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.06)'),
        margin=dict(l=60, r=30, t=30, b=60), height=450,
        hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.08)
    )
    st.plotly_chart(fig_gas, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-weight:700;color:#e2e8f0;font-size:1.05rem;margin-bottom:.5rem;">🌡️ تأثير درجة الحرارة على الضغط</div>',
                unsafe_allow_html=True)
    temp_range = np.linspace(-50, 50, 200)
    temp_k_range = temp_range + 273.15
    p_temp_arr = np.array([barometric_pressure(101325, gas['M'], altitude, tk, g_alt) for tk in temp_k_range])

    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=temp_range, y=p_temp_arr / 1000,
        mode='lines', line=dict(color=gas['color'], width=3),
        name=f"{gas['ar']} ({gas_key}) at {altitude}m"
    ))
    fig_temp.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', family='Inter'),
        xaxis_title="Temperature (°C)", yaxis_title="Pressure (kPa)",
        xaxis=dict(gridcolor='rgba(255,255,255,0.06)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.06)'),
        margin=dict(l=60, r=30, t=30, b=60), height=350,
    )
    st.plotly_chart(fig_temp, use_container_width=True)

    # Ideal Gas Calculator
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <div style="font-weight:700;font-size:1.05rem;color:#e2e8f0;margin-bottom:1rem;">
        ⚗️ حاسبة قانون الغاز المثالي / Ideal Gas Law Calculator
        </div>
    """, unsafe_allow_html=True)

    ig1, ig2, ig3, ig4 = st.columns(4)
    with ig1:
        n_mol = st.number_input("عدد المولات / n (mol)", 0.01, 100.0, 1.0, 0.01, key="ig_n")
    with ig2:
        t_ig = st.number_input("درجة الحرارة / T (K)", 100, 1000, 300, 1, key="ig_t")
    with ig3:
        v_ig = st.number_input("الحجم / V (m³)", 0.001, 100.0, 0.0224, 0.001, key="ig_v", format="%.4f")
    with ig4:
        p_ig = ideal_gas_pressure(n_mol, t_ig, v_ig)
        st.markdown(f"""
        <div class="metric-box" style="margin-top:1.5rem;">
        <div class="metric-label">الضغط / P</div>
        <div class="metric-val">{p_ig/1000:.2f}</div><div class="metric-unit">kPa ({p_ig/101325:.3f} atm)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Gas comparison table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <div style="font-weight:700;font-size:1.05rem;color:#e2e8f0;margin-bottom:1rem;">
        📋 جدول مقارنة خصائص الغازات عند الارتفاع المحدد
        </div>
    """, unsafe_allow_html=True)
    gas_rows = []
    for gn, gd in GASES.items():
        p_g = barometric_pressure(101325, gd['M'], altitude, temp_k, g_alt)
        rho_g = density_at_altitude(gd['rho'], gd['M'], altitude, temp_k, g_alt)
        gas_rows.append({
            "الغاز": f"{gd['icon']} {gd['ar']}",
            "Gas": gn,
            "M (kg/mol)": gd['M'],
            "ρ₀ (kg/m³)": gd['rho'],
            f"ρ at {altitude}m": round(rho_g, 4),
            f"P at {altitude}m (kPa)": round(p_g / 1000, 2),
            f"P (atm)": round(p_g / 101325, 4),
            "% of P₀": round((p_g / 101325) * 100, 1),
        })
    df_gas = pd.DataFrame(gas_rows)
    st.dataframe(df_gas, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SECTION: MEASUREMENT DEVICES
# ═══════════════════════════════════════════════════════════════
def show_measurement_devices():
    st.markdown('<div class="section-title">🔧 أجهزة قياس الضغط</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Pressure Measurement Devices — المبادئ، المدى، الدقة، والاستخدامات</div>',
                unsafe_allow_html=True)

    sd1, sd2, sd3 = st.columns(3)
    with sd1:
        st.markdown('<div style="text-align:center;font-weight:700;color:#e2e8f0;margin-bottom:.5rem;">Mercury Barometer</div>',
                    unsafe_allow_html=True)
        fig_bar = draw_barometer()
        st.pyplot(fig_bar, use_container_width=True)
    with sd2:
        st.markdown('<div style="text-align:center;font-weight:700;color:#e2e8f0;margin-bottom:.5rem;">U-tube Manometer</div>',
                    unsafe_allow_html=True)
        fig_man = draw_manometer()
        st.pyplot(fig_man, use_container_width=True)
    with sd3:
        st.markdown('<div style="text-align:center;font-weight:700;color:#e2e8f0;margin-bottom:.5rem;">Bourdon Tube Gauge</div>',
                    unsafe_allow_html=True)
        fig_bou = draw_bourdon()
        st.pyplot(fig_bou, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    for i, dev in enumerate(DEVICES_INFO):
        col_d1, col_d2 = st.columns([2, 1])
        with col_d1:
            st.markdown(f"""
            <div class="device-card">
                <div style="display:flex;align-items:center;gap:.8rem;margin-bottom:.8rem;">
                    <div style="font-size:1.8rem;">{dev['icon']}</div>
                    <div>
                        <div style="font-weight:700;color:#e2e8f0;font-size:1.05rem;">{dev['ar']}</div>
                        <div style="color:#64748b;font-size:.8rem;">{dev['name']}</div>
                    </div>
                </div>
                <div style="color:#94a3b8;font-size:.9rem;line-height:1.8;margin-bottom:.8rem;">
                {dev['principle']}
                </div>
                <div style="font-family:monospace;color:#00d4ff;font-size:.85rem;background:rgba(0,212,255,0.05);
                padding:.4rem .8rem;border-radius:8px;display:inline-block;">
                {dev['formula']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_d2:
            st.markdown(f"""
            <div class="metric-box" style="margin-bottom:.8rem;">
                <div class="metric-label">المدى / Range</div>
                <div style="font-size:.95rem;font-weight:700;color:#e2e8f0;">{dev['range']}</div>
            </div>
            <div class="metric-box" style="margin-bottom:.8rem;">
                <div class="metric-label">الدقة / Accuracy</div>
                <div style="font-size:.95rem;font-weight:700;color:#f59e0b;">{dev['accuracy']}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">الاستخدام / Uses</div>
                <div style="font-size:.8rem;color:#94a3b8;line-height:1.6;">{dev['uses']}</div>
            </div>
            """, unsafe_allow_html=True)
        if i < len(DEVICES_INFO) - 1:
            st.markdown('<div class="shimmer-line"></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <div style="font-weight:700;font-size:1.05rem;color:#e2e8f0;margin-bottom:1rem;">
        📏 محاكاة مانومتر الأنبوب U — حساب فرق الضغط
        </div>
        <div style="color:#94a3b8;font-size:.85rem;margin-bottom:1rem;">
        ΔP = ρ × g × Δh — حيث Δh هو فرق ارتفاع السائل في فرعي الأنبوب
        </div>
    """, unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        man_fluid = st.selectbox("سائل المانومتر", list(FLUIDS.keys()), index=0, key="man_fluid")
    with mc2:
        delta_h = st.number_input("Δh (m)", 0.001, 2.0, 0.15, 0.001, key="man_dh", format="%.3f")
    with mc3:
        delta_p = fluid_pressure(FLUIDS[man_fluid]['density'], delta_h)
        st.markdown(f"""
        <div class="metric-box" style="margin-top:1.5rem;">
        <div class="metric-label">فرق الضغط ΔP</div>
        <div class="metric-val">{delta_p/1000:.2f}</div><div class="metric-unit">kPa</div></div>
        """, unsafe_allow_html=True)
    with mc4:
        st.markdown(f"""
        <div class="metric-box" style="margin-top:1.5rem;">
        <div class="metric-label">فرق الضغط ΔP</div>
        <div class="metric-val warm">{delta_p/101325:.5f}</div><div class="metric-unit">atm</div></div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SECTION: AI PREDICTION
# ═══════════════════════════════════════════════════════════════
def show_ai_prediction():
    st.markdown('<div class="section-title">🤖 التنبؤ بالذكاء الاصطناعي</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">AI Pressure Prediction — نموذج Gradient Boosting مدرب على المعادلة البارومترية للتنبؤ بضغط الغاز أثناء الحركة</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="margin-bottom:1.5rem;">
        <div style="color:#94a3b8;font-size:.9rem;line-height:1.8;">
        <b style="color:#00d4ff;">كيف يعمل النموذج؟</b> تم توليد 8,000 نقطة بيانات باستخدام المعادلة البارومترية الفيزيائية
        Barometric Formula مع إضافة ضوضاء عشوائية (Noise) بسيطة.
        ثم تم تدريب نموذج <span style="color:#f59e0b;">Gradient Boosting Regressor</span> (200 شجرة قرار)
        على هذه البيانات ليتعلم العلاقة بين:
        الارتفاع (Altitude) ، درجة الحرارة (Temperature) ، الكتلة المولية (Molar Mass) ،
        والضغط الأساسي (Base Pressure) ← الضغط الناتج (Pressure).
        <br><br>
        <b style="color:#a78bfa;">الهدف:</b> إظهار قدرة الذكاء الاصطناعي على تعلم القوانين الفيزيائية
        والتنبؤ بالضغط عند أي نقطة خلال مسار الحركة.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("🔄 تحميل النموذج المدرب..."):
        model, feature_names, metrics, X_test, y_test, y_pred = train_pressure_ai_model()

    st.markdown('<div style="font-weight:700;color:#e2e8f0;font-size:1.05rem;margin-bottom:.8rem;">📊 أداء النموذج / Model Performance</div>',
                unsafe_allow_html=True)
    mm1, mm2, mm3, mm4 = st.columns(4)
    with mm1:
        st.markdown(f"""
        <div class="metric-box"><div class="metric-label">R² Score</div>
        <div class="metric-val">{metrics['r2']:.6f}</div>
        <div class="metric-unit">{'ممتاز' if metrics['r2']>0.999 else 'جيد جداً'}</div></div>
        """, unsafe_allow_html=True)
    with mm2:
        st.markdown(f"""
        <div class="metric-box"><div class="metric-label">RMSE</div>
        <div class="metric-val warm">{metrics['rmse']:.2f}</div><div class="metric-unit">Pa</div></div>
        """, unsafe_allow_html=True)
    with mm3:
        st.markdown(f"""
        <div class="metric-box"><div class="metric-label">MAE</div>
        <div class="metric-val purple">{metrics['mae']:.2f}</div><div class="metric-unit">Pa</div></div>
        """, unsafe_allow_html=True)
    with mm4:
        st.markdown(f"""
        <div class="metric-box"><div class="metric-label">MAPE</div>
        <div class="metric-val">{metrics['mape']:.4f}</div><div class="metric-unit">%</div></div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-weight:700;color:#e2e8f0;font-size:1.05rem;margin-bottom:.5rem;">🎯 التنبؤ مقابل القياس الفعلي</div>',
                unsafe_allow_html=True)

    sample_idx = np.random.choice(len(y_test), min(2000, len(y_test)), replace=False)
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=y_test[sample_idx] / 1000, y=y_pred[sample_idx] / 1000,
        mode='markers', marker=dict(color='#00d4ff', size=4, opacity=0.5),
        name='Predictions'
    ))
    max_p = max(y_test.max(), y_pred.max()) / 1000
    fig_scatter.add_trace(go.Scatter(
        x=[0, max_p], y=[0, max_p],
        mode='lines', line=dict(color='#ef4444', width=2, dash='dash'),
        name='Perfect Prediction'
    ))
    fig_scatter.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', family='Inter'),
        xaxis_title="Actual Pressure (kPa)", yaxis_title="AI Predicted Pressure (kPa)",
        xaxis=dict(gridcolor='rgba(255,255,255,0.06)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.06)'),
        margin=dict(l=60, r=30, t=30, b=60), height=420,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <div style="font-weight:700;font-size:1.05rem;color:#e2e8f0;margin-bottom:1rem;">
        🎮 تنبؤ تفاعلي — أدخل المعلمات وشاهد نتيجة الذكاء الاصطناعي
        </div>
    """, unsafe_allow_html=True)

    pi1, pi2, pi3, pi4 = st.columns(4)
    with pi1:
        ai_alt = st.number_input("الارتفاع / Altitude (m)", 0, 40000, 5000, 100, key="ai_alt")
    with pi2:
        ai_temp = st.number_input("الحرارة / Temperature (K)", 200, 400, 288, 1, key="ai_temp")
    with pi3:
        ai_gas = st.selectbox("الغاز / Gas", list(GASES.keys()), key="ai_gas",
                              format_func=lambda x: f"{GASES[x]['ar']} ({x})")
    with pi4:
        ai_p0 = st.number_input("الضغط الأساسي / P₀ (Pa)", 90000, 110000, 101325, 100, key="ai_p0")

    ai_M = GASES[ai_gas]['M']
    ai_g = gravity_at_altitude(ai_alt)

    X_input = np.array([[ai_alt, ai_temp, ai_M, ai_p0]])
    ai_prediction = model.predict(X_input)[0]
    theoretical = barometric_pressure(ai_p0, ai_M, ai_alt, ai_temp, ai_g)
    error_pct = abs(ai_prediction - theoretical) / (theoretical + 1e-10) * 100

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"""
        <div class="metric-box"><div class="metric-label">🤖 تنبؤ الذكاء الاصطناعي</div>
        <div class="metric-val">{ai_prediction/1000:.2f}</div><div class="metric-unit">kPa</div></div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="metric-box"><div class="metric-label">📐 القيمة الفيزيائية الدقيقة</div>
        <div class="metric-val warm">{theoretical/1000:.2f}</div><div class="metric-unit">kPa</div></div>
        """, unsafe_allow_html=True)
    with r3:
        err_color = "#10b981" if error_pct < 1 else "#f59e0b" if error_pct < 5 else "#ef4444"
        st.markdown(f"""
        <div class="metric-box"><div class="metric-label">📊 نسبة الخطأ</div>
        <div class="metric-val" style="background:linear-gradient(135deg,{err_color},{err_color}cc);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">{error_pct:.3f}</div>
        <div class="metric-unit">%</div></div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <div style="font-weight:700;font-size:1.05rem;color:#e2e8f0;margin-bottom:.5rem;">
        🛤️ التنبؤ بالحركة — مسار الضغط من نقطة بداية إلى نقطة نهاية
        </div>
        <div style="color:#94a3b8;font-size:.85rem;margin-bottom:1rem;">
        حدد مسار الحركة (الارتفاع) وسيقوم الذكاء الاصطناعي بالتنبؤ بالضغط عند كل نقطة على المسار
        ومقارنته بالقيم الفيزيائية الدقيقة.
        </div>
    """, unsafe_allow_html=True)

    pp1, pp2, pp3 = st.columns(3)
    with pp1:
        path_start = st.number_input("ارتفاع البداية / Start Altitude (m)", 0, 39000, 0, 100, key="path_s")
    with pp2:
        path_end = st.number_input("ارتفاع النهاية / End Altitude (m)", 100, 40000, 20000, 100, key="path_e")
    with pp3:
        path_n = st.slider("عدد النقاط / Points", 10, 200, 80, key="path_n")

    if path_end <= path_start:
        path_end = path_start + 100

    path_alts = np.linspace(path_start, path_end, int(path_n))
    path_temps = np.full_like(path_alts, ai_temp)
    path_M = np.full_like(path_alts, ai_M)
    path_p0 = np.full_like(path_alts, ai_p0)

    X_path = np.column_stack([path_alts, path_temps, path_M, path_p0])
    ai_path_pred = model.predict(X_path)
    theo_path = np.array([barometric_pressure(ai_p0, ai_M, a, ai_temp, ai_g) for a in path_alts])
    path_errors = np.abs(ai_path_pred - theo_path) / (theo_path + 1e-10) * 100

    fig_path = make_subplots(rows=1, cols=2, subplot_titles=("Pressure Along Path", "Error Along Path"))

    fig_path.add_trace(go.Scatter(
        x=path_alts / 1000, y=theo_path / 1000,
        mode='lines', line=dict(color='#00d4ff', width=3), name='Theoretical'
    ), row=1, col=1)
    fig_path.add_trace(go.Scatter(
        x=path_alts / 1000, y=ai_path_pred / 1000,
        mode='lines+markers', line=dict(color='#f59e0b', width=2, dash='dot'),
        marker=dict(size=4), name='AI Prediction'
    ), row=1, col=1)

    fig_path.add_trace(go.Scatter(
        x=path_alts / 1000, y=path_errors,
        mode='lines', line=dict(color='#ef4444', width=2), name='Error %',
        fill='tozeroy', fillcolor='rgba(239,68,68,0.1)'
    ), row=1, col=2)

    fig_path.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', family='Inter'),
        height=400, margin=dict(l=60, r=30, t=50, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    fig_path.update_xaxes(title_text="Altitude (km)", gridcolor='rgba(255,255,255,0.06)')
    fig_path.update_yaxes(title_text="Pressure (kPa)", gridcolor='rgba(255,255,255,0.06)')
    fig_path.update_yaxes(title_text="Error (%)", gridcolor='rgba(255,255,255,0.06)', row=1, col=2)

    st.plotly_chart(fig_path, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════
def main():
    render_sidebar()

    tab_intro, tab_fluid, tab_gas, tab_devices, tab_ai = st.tabs([
        "🏠 Introduction",
        "💧 Fluid Pressure",
        "🌬️ Gas Pressure",
        "🔧 Measurement Devices",
        "🤖 AI Prediction"
    ])

    with tab_intro:
        show_introduction()
    with tab_fluid:
        show_fluid_pressure()
    with tab_gas:
        show_gas_pressure()
    with tab_devices:
        show_measurement_devices()
    with tab_ai:
        show_ai_prediction()


if __name__ == "__main__":
    main()
