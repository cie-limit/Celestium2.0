"""
Celestium 2.0 - Main Application
물리 기반 궤도 시뮬레이터
"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, time, date

from simulation.spacecraft import VEHICLES
from simulation.optimization import generate_trajectories, get_moon_state
from simulation.physics import R_EARTH, R_MOON


# --- [1] CONFIG & DESIGN ---
st.set_page_config(
    page_title="CELESTIUM 2.0",
    page_icon="🚀",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
* { font-family: 'Orbitron', sans-serif !important; }
.stApp { background-color: #000000; color: #aaddff; }

.sc-frame {
    border: 1px solid #004488;
    background: rgba(5, 10, 20, 0.95);
    padding: 15px;
    margin-bottom: 15px;
    box-shadow: 0 0 15px rgba(0, 100, 255, 0.2);
    border-radius: 6px;
}
.sc-header {
    color: #00ffff;
    font-size: 1.1em;
    border-bottom: 1px solid #004488;
    margin-bottom: 15px;
    padding-bottom: 8px;
    letter-spacing: 3px;
}
.version-tag {
    color: #ffcc00;
    font-size: 0.7em;
    margin-left: 10px;
}
.physics-badge {
    background: linear-gradient(90deg, #004488, #006699);
    color: #00ffff;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 0.7em;
    margin-left: 5px;
}
.hl-val { color: #ffcc00; }
.hl-good { color: #33ff33; }
.hl-bad { color: #ff3333; }
.hl-info { color: #88ccff; }

.stButton>button {
    border: 1px solid #00ffff;
    color: #00ffff;
    background: rgba(0, 50, 100, 0.5);
    width: 100%;
    height: 50px;
    font-weight: bold;
    transition: all 0.3s;
    letter-spacing: 2px;
}
.stButton>button:hover {
    background: #00ffff;
    color: #000;
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
}

.rocket-img img {
    border: 1px solid #333;
    border-radius: 5px;
    max-height: 180px;
    object-fit: cover;
}

.metric-box {
    background: rgba(0, 50, 100, 0.3);
    border: 1px solid #003366;
    padding: 10px;
    border-radius: 4px;
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)

# Session State 초기화
if 'page' not in st.session_state:
    st.session_state.page = 'start'
if 'sim_data' not in st.session_state:
    st.session_state.sim_data = {}
if 'vehicle_name' not in st.session_state:
    st.session_state.vehicle_name = list(VEHICLES.keys())[0]
if 'computing' not in st.session_state:
    st.session_state.computing = False


# --- [2] START SCREEN ---
if st.session_state.page == 'start':
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown("""
        <h1 style='text-align: center; color: #00ffff;'>
            CELESTIUM
            <span class='version-tag'>2.0</span>
        </h1>
        """, unsafe_allow_html=True)
        st.markdown("""
        <p style='text-align: center; letter-spacing: 3px; color: #88ccff;'>
            PHYSICS-BASED ORBITAL SIMULATOR
        </p>
        <p style='text-align: center; color: #666; font-size: 0.8em;'>
            Powered by RK4 Numerical Integration
        </p>
        """, unsafe_allow_html=True)
        st.markdown("---")
        
        # 새 기능 표시
        st.markdown("""
        <div style='text-align: center; color: #aaddff; font-size: 0.85em; margin-bottom: 20px;'>
            ✨ NEW IN 2.0 ✨<br>
            • Real gravity simulation (Earth + Moon)<br>
            • Runge-Kutta 4th order integration<br>
            • Accurate ΔV calculations<br>
            • Adaptive time-step propagation
        </div>
        """, unsafe_allow_html=True)
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("🚀 INITIALIZE SYSTEMS"):
                st.session_state.page = 'lobby'
                st.rerun()


# --- [3] LOBBY ---
elif st.session_state.page == 'lobby':
    st.markdown("""
    <h2>MISSION CONFIGURATION 
        <span class='physics-badge'>PHYSICS ENGINE v2.0</span>
    </h2>
    """, unsafe_allow_html=True)
    
    c_left, c_right = st.columns([1, 1.5])
    
    with c_left:
        # 시간 설정
        st.markdown("<div class='sc-frame'><div class='sc-header'>📅 TEMPORAL TARGETING</div>", unsafe_allow_html=True)
        
        d_in = st.date_input("Launch Date", date.today())
        t_in = st.time_input("Launch Time (UTC)", time(12, 0))
        launch_datetime = datetime.combine(d_in, t_in)
        
        # 달 상태 조회
        moon_state = get_moon_state(launch_datetime)
        dist = moon_state["dist"]
        dec = moon_state["dec"]
        
        # 발사 창 상태 평가
        if abs(dec) < 5:
            status, color, status_desc = "OPTIMAL", "hl-good", "Minimal plane change required"
        elif abs(dec) < 15:
            status, color, status_desc = "NOMINAL", "hl-val", "Moderate plane change penalty"
        else:
            status, color, status_desc = "CRITICAL", "hl-bad", "High fuel penalty for plane change"
        
        st.markdown(f"""
        <div class='metric-box'>
            <b>LUNAR DISTANCE:</b> <span class='hl-val'>{dist:,.0f}</span> km<br>
            <b>LUNAR DECLINATION:</b> <span class='hl-val'>{dec:+.2f}°</span><br>
        </div>
        <div style='display: flex; align-items: center; justify-content: space-between; margin-top: 10px;'>
            <span>LAUNCH WINDOW:</span>
            <span class='{color}' style='font-size: 1.1em; border: 1px solid; padding: 3px 12px;'>{status}</span>
        </div>
        <p style='color: #666; font-size: 0.8em; margin-top: 5px;'>{status_desc}</p>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 시뮬레이션 설정
        st.markdown("<div class='sc-frame'><div class='sc-header'>⚙️ SIMULATION SETTINGS</div>", unsafe_allow_html=True)
        
        integration_method = st.selectbox(
            "Integration Method",
            ["RK4 (Recommended)", "Euler (Fast)"],
            help="RK4 is more accurate, Euler is faster but less precise"
        )
        
        adaptive_dt = st.checkbox("Adaptive Time Step", value=True, 
                                  help="Automatically adjust time step based on gravity strength")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with c_right:
        # 기체 선택
        st.markdown("<div class='sc-frame'><div class='sc-header'>🛸 VEHICLE SELECTION</div>", unsafe_allow_html=True)
        
        v_name = st.selectbox("Select Launch Vehicle", list(VEHICLES.keys()))
        st.session_state.vehicle_name = v_name
        vehicle = VEHICLES[v_name]
        
        rc1, rc2 = st.columns([1, 2])
        with rc1:
            st.markdown("<div class='rocket-img'>", unsafe_allow_html=True)
            st.image(vehicle.img_url, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with rc2:
            st.markdown(vehicle.get_description(), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 계산 버튼
        if st.button("⚡ COMPUTE TRAJECTORIES", disabled=st.session_state.computing):
            st.session_state.computing = True
            
            with st.spinner("🔄 Running physics simulation..."):
                try:
                    st.session_state.sim_data = generate_trajectories(launch_datetime, vehicle)
                    st.session_state.page = 'simulation'
                except Exception as e:
                    st.error(f"Simulation error: {e}")
                finally:
                    st.session_state.computing = False
            
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)


# --- [4] SIMULATION RESULTS ---
elif st.session_state.page == 'simulation':
    st.markdown("""
    <h2>TRAJECTORY ANALYSIS 
        <span class='physics-badge'>REAL PHYSICS</span>
    </h2>
    """, unsafe_allow_html=True)
    
    if st.button("⬅️ RETURN TO CONFIGURATION"):
        st.session_state.page = 'lobby'
        st.rerun()
    
    data = st.session_state.sim_data
    vehicle = VEHICLES[st.session_state.vehicle_name]
    
    c1, c2 = st.columns([1, 2.5])
    
    with c1:
        # 텔레메트리 데이터
        st.markdown("<div class='sc-frame'><div class='sc-header'>📊 TELEMETRY DATA</div>", unsafe_allow_html=True)
        
        tabs = st.tabs(["FAST", "BAL", "OPT", "STD", "RET"])
        keys = ["fast", "bal", "opt", "ho", "fr"]
        
        for i, key in enumerate(keys):
            with tabs[i]:
                d = data[key]
                st.markdown(f"<h4 style='color:{d['color']}'>{d['name']}</h4>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("ΔV", f"{d['delta_v']} m/s")
                with col2:
                    st.metric("Time", d['time'])
                
                st.caption(d['desc'])
                
                # 연료 게이지
                fuel = float(d['fuel_mass'])
                cap = float(vehicle.fuel_capacity)
                pct = (fuel / cap) * 100
                
                if pct > 100:
                    st.error(f"⚠️ FUEL EXCEEDED: {pct:.1f}%")
                else:
                    st.progress(min(pct / 100, 1.0), text=f"Fuel: {fuel:,.0f} kg ({pct:.1f}%)")
                
                # 페널티 표시
                if d.get('penalty', 0) > 0:
                    st.warning(f"Plane change penalty: +{d['penalty']:.0f} m/s")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 레이어 컨트롤
        st.markdown("<div class='sc-frame'><div class='sc-header'>🎛️ LAYER CONTROL</div>", unsafe_allow_html=True)
        
        v_fast = st.checkbox("High Speed", True)
        v_bal = st.checkbox("Balanced", True)
        v_opt = st.checkbox("Fuel Optimized", False)
        v_ho = st.checkbox("Standard Hohmann", False)
        v_fr = st.checkbox("Free Return", False)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with c2:
        # 3D 시각화
        st.markdown("<div class='sc-frame'><div class='sc-header'>🌍 TACTICAL MAP (1:1 SCALE)</div>", unsafe_allow_html=True)
        
        fig = go.Figure()
        
        # 지구 생성
        def create_sphere(radius, center, color, name, opacity=0.8):
            u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
            x = center[0] + radius * np.cos(u) * np.sin(v)
            y = center[1] + radius * np.sin(u) * np.sin(v)
            z = center[2] + radius * np.cos(v)
            return go.Surface(
                x=x, y=y, z=z,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                opacity=opacity,
                name=name,
                hoverinfo='name'
            )
        
        # 지구
        fig.add_trace(create_sphere(R_EARTH, [0, 0, 0], '#1E90FF', 'Earth'))
        
        # 달
        moon_pos = data['fast']['target_pos']
        fig.add_trace(create_sphere(R_MOON, moon_pos, '#808080', 'Moon'))
        
        # 궤도 표시
        def add_trajectory(key, visible):
            if visible and key in data:
                d = data[key]
                fig.add_trace(go.Scatter3d(
                    x=d['x'], y=d['y'], z=d['z'],
                    mode='lines',
                    line=dict(color=d['color'], width=3),
                    name=d['name'],
                    hovertemplate=f"{d['name']}<br>ΔV: {d['delta_v']} m/s<br>Time: {d['time']}<extra></extra>"
                ))
        
        add_trajectory('fast', v_fast)
        add_trajectory('bal', v_bal)
        add_trajectory('opt', v_opt)
        add_trajectory('ho', v_ho)
        add_trajectory('fr', v_fr)
        
        # 레이아웃
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False, backgroundcolor="#050510"),
                yaxis=dict(visible=False, backgroundcolor="#050510"),
                zaxis=dict(visible=False, backgroundcolor="#050510"),
                bgcolor="#050510",
                aspectmode='data',
                camera=dict(eye=dict(x=0.8, y=0.8, z=0.5))
            ),
            paper_bgcolor="#050510",
            margin=dict(l=0, r=0, t=0, b=0),
            height=700,
            showlegend=True,
            legend=dict(
                x=0.02, y=0.98,
                font=dict(color='white', size=10),
                bgcolor="rgba(0,20,40,0.8)",
                bordercolor="#004488",
                borderwidth=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 추가 정보
        st.markdown("""
        <div class='sc-frame'>
            <div class='sc-header'>ℹ️ SIMULATION INFO</div>
            <p style='color: #88ccff; font-size: 0.85em;'>
                • Trajectories computed using <b>RK4 numerical integration</b><br>
                • Gravity model: <b>Restricted 3-body</b> (Earth + Moon)<br>
                • All ΔV values include plane change penalties if applicable<br>
                • Fuel mass calculated via <b>Tsiolkovsky equation</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
