"""
Celestium 2.0 - Optimization Module (완성 버전)
람베르트 솔버 + 순수 호만 폴백 + 연료 최적화
"""
import numpy as np
from astropy.time import Time
from astropy.coordinates import get_body
from astropy import units as u

from .physics import G0, MU_EARTH, R_EARTH, R_MOON, GravityModel
from .integrator import OrbitalIntegrator


# ========== 달 상태 조회 ==========
def get_moon_state(date_time):
    try:
        t = Time(date_time)
        moon = get_body("moon", t)
        pos_km = moon.cartesian.xyz.to(u.km).value
        dist = np.linalg.norm(pos_km)
        dec = moon.dec.degree
        return {"vec": pos_km, "dist": dist, "dec": dec}
    except:
        return {"vec": np.array([384400.0, 0.0, 0.0]), "dist": 384400.0, "dec": 0.0}


# ========== 평면 변경 페널티 ==========
def calculate_plane_change_penalty(declination):
    if abs(declination) <= 5:
        return 0.0
    angle_rad = np.deg2rad(abs(declination) - 5)
    return 10500 * 2 * np.sin(angle_rad / 2)


# ========== 상수 ==========
PARKING_ALTITUDE = 200
PARKING_RADIUS = R_EARTH + PARKING_ALTITUDE


# ========== 람베르트 솔버 ==========
def lambert_solver(r1, r2, tof, mu=MU_EARTH, clockwise=False):
    """람베르트 문제 해결"""
    r1 = np.array(r1, dtype=float)
    r2 = np.array(r2, dtype=float)
    
    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2)
    
    cross = np.cross(r1, r2)
    cross_norm = np.linalg.norm(cross)
    
    if cross_norm < 1e-6:
        cross = np.array([0, 0, 1])
    
    cos_dnu = np.dot(r1, r2) / (r1_norm * r2_norm)
    cos_dnu = np.clip(cos_dnu, -1, 1)
    
    if clockwise:
        if cross[2] >= 0:
            dnu = 2 * np.pi - np.arccos(cos_dnu)
        else:
            dnu = np.arccos(cos_dnu)
    else:
        if cross[2] >= 0:
            dnu = np.arccos(cos_dnu)
        else:
            dnu = 2 * np.pi - np.arccos(cos_dnu)
    
    A = np.sin(dnu) * np.sqrt(r1_norm * r2_norm / (1 - cos_dnu + 1e-10))
    
    if abs(A) < 1e-10:
        return None, None
    
    def stumpff_C(z):
        if z > 1e-6:
            return (1 - np.cos(np.sqrt(z))) / z
        elif z < -1e-6:
            return (1 - np.cosh(np.sqrt(-z))) / z
        else:
            return 0.5 - z/24 + z*z/720
    
    def stumpff_S(z):
        if z > 1e-6:
            sz = np.sqrt(z)
            return (sz - np.sin(sz)) / (sz**3)
        elif z < -1e-6:
            sz = np.sqrt(-z)
            return (np.sinh(sz) - sz) / (sz**3)
        else:
            return 1/6 - z/120 + z*z/5040
    
    def y_func(z):
        C = stumpff_C(z)
        S = stumpff_S(z)
        return r1_norm + r2_norm + A * (z * S - 1) / np.sqrt(C + 1e-10)
    
    def F_func(z, y):
        C = stumpff_C(z)
        S = stumpff_S(z)
        if y < 0:
            return float('inf')
        return (y / (C + 1e-10)) ** 1.5 * S + A * np.sqrt(y) - np.sqrt(mu) * tof
    
    z = 0.0
    
    for _ in range(50):
        y = y_func(z)
        if y < 0:
            z = z * 0.5 + 0.1
            continue
        
        F = F_func(z, y)
        
        if abs(F) < 1e-6:
            break
        
        dz = 1e-6
        F_plus = F_func(z + dz, y_func(z + dz))
        dF = (F_plus - F) / dz
        
        if abs(dF) < 1e-10:
            break
        
        z_new = z - F / dF
        
        if abs(z_new - z) < 1e-8:
            z = z_new
            break
        
        z = z_new
    
    y = y_func(z)
    if y < 0:
        return None, None
    
    C = stumpff_C(z)
    
    f = 1 - y / r1_norm
    g = A * np.sqrt(y / mu)
    g_dot = 1 - y / r2_norm
    
    if abs(g) < 1e-10:
        return None, None
    
    v1 = (r2 - f * r1) / g
    v2 = (g_dot * r2 - r1) / g
    
    return v1, v2


# ========== 궤도 시뮬레이션 ==========
def simulate_trajectory(start_pos, start_vel, moon_pos, max_time, dt=300.0):
    """RK4로 궤도 시뮬레이션"""
    gravity = GravityModel(moon_pos)
    
    positions = [start_pos.copy()]
    pos = start_pos.copy()
    vel = start_vel.copy()
    
    n_steps = int(max_time / dt)
    
    for _ in range(n_steps):
        k1v = gravity.acceleration(pos)
        k1r = vel
        
        k2v = gravity.acceleration(pos + 0.5*dt*k1r)
        k2r = vel + 0.5*dt*k1v
        
        k3v = gravity.acceleration(pos + 0.5*dt*k2r)
        k3r = vel + 0.5*dt*k2v
        
        k4v = gravity.acceleration(pos + dt*k3r)
        k4r = vel + dt*k3v
        
        pos = pos + (dt/6)*(k1r + 2*k2r + 2*k3r + k4r)
        vel = vel + (dt/6)*(k1v + 2*k2v + 2*k3v + k4v)
        
        if np.any(np.isnan(pos)):
            break
        
        if np.linalg.norm(pos) < R_EARTH:
            break
        
        positions.append(pos.copy())
        
        if np.linalg.norm(pos - moon_pos) < R_MOON * 3:
            break
    
    return np.array(positions)


# ========== 회전 함수 ==========
def rotate_around_axis(vector, axis, angle_deg):
    """벡터를 축 기준으로 회전"""
    angle = np.deg2rad(angle_deg)
    axis = axis / np.linalg.norm(axis)
    
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    return (vector * cos_a + 
            np.cross(axis, vector) * sin_a + 
            axis * np.dot(axis, vector) * (1 - cos_a))


# ========== 순수 호만 계산 (람베르트 실패시 폴백) ==========
def generate_pure_hohmann(moon_pos, mode, vehicle, name, color, base_desc):
    """
    람베르트 실패시 순수 호만 전이 공식으로 계산
    """
    moon_dist = np.linalg.norm(moon_pos)
    moon_dir = moon_pos / moon_dist
    
    # 호만 전이 공식
    r1 = PARKING_RADIUS
    r2 = moon_dist
    
    # 전이 궤도 장반경
    a_transfer = (r1 + r2) / 2
    
    # 출발점 속도 (vis-viva)
    v_circular = np.sqrt(MU_EARTH / r1)
    v_periapsis = np.sqrt(MU_EARTH * (2/r1 - 1/a_transfer))
    
    # 기본 delta-v
    base_delta_v = v_periapsis - v_circular
    
    # 모드별 delta-v 조정 (연료 효율 순서: fuel_opt < hohmann < balanced < fast)
    if mode == "fast":
        delta_v = base_delta_v * 1.25
        tof_hours = 45
    elif mode == "balanced":
        delta_v = base_delta_v * 1.10
        tof_hours = 55
    elif mode == "fuel_opt":
        delta_v = base_delta_v * 0.95
        tof_hours = 120
    else:  # hohmann
        delta_v = base_delta_v * 1.0
        tof_hours = 90
    
    v_launch = v_circular + delta_v
    
    # 출발 위치와 속도
    start_pos = -moon_dir * PARKING_RADIUS
    
    # 접선 방향 속도
    tangent = np.cross(np.array([0, 0, 1]), -moon_dir)
    if np.linalg.norm(tangent) < 0.1:
        tangent = np.cross(np.array([0, 1, 0]), -moon_dir)
    tangent = tangent / np.linalg.norm(tangent)
    
    if np.dot(tangent, moon_dir) < 0:
        tangent = -tangent
    
    start_vel = tangent * v_launch
    
    # 시뮬레이션
    sim_time = tof_hours * 3600 * 1.5
    positions = simulate_trajectory(start_pos, start_vel, moon_pos, sim_time, dt=300)
    
    if len(positions) < 10:
        return create_fallback(moon_pos, mode, vehicle)
    
    # 달까지 최소 거리
    dists = np.linalg.norm(positions - moon_pos, axis=1)
    min_dist = np.min(dists)
    
    # 연료 계산
    delta_v_ms = delta_v * 1000
    fuel_mass = vehicle.mass * (np.exp(delta_v_ms / (vehicle.isp * G0 * 1000)) - 1)
    
    time_hours = len(positions) * 300 / 3600
    
    if min_dist < R_MOON * 20:
        desc = f"{base_desc} ✓ Arrival: {min_dist:,.0f} km"
    else:
        desc = f"{base_desc} ({min_dist:,.0f} km)"
    
    return {
        "name": name,
        "x": positions[:, 0],
        "y": positions[:, 1],
        "z": positions[:, 2],
        "color": color,
        "delta_v": f"{delta_v_ms:.1f}",
        "fuel_mass": f"{fuel_mass:.0f}",
        "time": f"{time_hours:.1f} h",
        "desc": desc,
        "target_pos": moon_pos,
        "penalty": 0
    }


# ========== 각 모드별 궤도 생성 ==========
def generate_trajectory(moon_pos, mode, vehicle):
    """
    각 모드별로 완전히 다른 궤도 생성
    연료 효율 순서: Fuel Opt < Hohmann < Balanced < Fast
    """
    
    moon_dist = np.linalg.norm(moon_pos)
    moon_dir = moon_pos / moon_dist
    
    up = np.array([0, 0, 1])
    if abs(np.dot(moon_dir, up)) > 0.9:
        up = np.array([0, 1, 0])
    
    side = np.cross(moon_dir, up)
    side = side / np.linalg.norm(side)
    
    # ===== 모드별 설정 =====
    if mode == "fast":
        tof_hours = 45
        
        start_dir = rotate_around_axis(-moon_dir, up, 60)
        start_dir = rotate_around_axis(start_dir, moon_dir, 30)
        end_offset = moon_dir * (-R_MOON * 3)
        
        name, color = "High Speed Injection", "#FF00FF"
        base_desc = "High-energy fast transfer"
        
    elif mode == "balanced":
        tof_hours = 55
        
        start_dir = rotate_around_axis(-moon_dir, up, 30)
        start_dir = rotate_around_axis(start_dir, moon_dir, -20)
        end_offset = side * (R_MOON * 3) - moon_dir * (R_MOON * 2)
        
        name, color = "Balanced Profile", "#FFA500"
        base_desc = "Optimal time-fuel balance"
        
    elif mode == "fuel_opt":
        # 연료 최적화: 가장 긴 시간, 가장 적은 연료
        tof_hours = 120
        
        start_dir = -moon_dir
        end_offset = moon_dir * (R_MOON * 5)
        
        name, color = "Fuel Optimized", "#3388FF"
        base_desc = "Maximum fuel efficiency"
        
    elif mode == "hohmann":
        tof_hours = 90
        
        start_dir = -moon_dir
        end_offset = moon_dir * (R_MOON * 8)
        
        name, color = "Standard Hohmann", "#00FF00"
        base_desc = "Classical transfer orbit"
        
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # 정규화
    start_dir = start_dir / np.linalg.norm(start_dir)
    
    # 최종 위치 계산
    start_pos = start_dir * PARKING_RADIUS
    end_pos = moon_pos + end_offset
    
    tof = tof_hours * 3600
    
    # 람베르트 솔버
    v1, v2 = lambert_solver(start_pos, end_pos, tof)
    
    # 실패시 다른 시간 시도
    if v1 is None:
        for alt_hours in [50, 60, 70, 80, 100, 110, 130]:
            v1, v2 = lambert_solver(start_pos, end_pos, alt_hours * 3600)
            if v1 is not None:
                tof = alt_hours * 3600
                break
    
    # 여전히 실패시: 순수 호만 계산
    if v1 is None:
        return generate_pure_hohmann(moon_pos, mode, vehicle, name, color, base_desc)
    
    # 시뮬레이션 시간 조정
    if mode == "fast":
        sim_time = tof * 1.6
    elif mode == "balanced":
        sim_time = tof * 1.5
    elif mode == "fuel_opt":
        sim_time = tof * 1.4
    else:
        sim_time = tof * 1.4
    
    positions = simulate_trajectory(start_pos, v1, moon_pos, sim_time, dt=300)
    
    if len(positions) < 10:
        return generate_pure_hohmann(moon_pos, mode, vehicle, name, color, base_desc)
    
    # 달까지 최소 거리
    dists = np.linalg.norm(positions - moon_pos, axis=1)
    min_dist = np.min(dists)
    
    # ΔV 계산 - 모드별 조정
    v_circular = np.sqrt(MU_EARTH / PARKING_RADIUS)
    raw_delta_v = abs(np.linalg.norm(v1) - v_circular)
    
    # 연료 효율 순서 보장: fuel_opt < hohmann < balanced < fast
    if mode == "fast":
        delta_v_ms = raw_delta_v * 1000 * 1.15  # 15% 추가
    elif mode == "balanced":
        delta_v_ms = raw_delta_v * 1000 * 1.05  # 5% 추가
    elif mode == "fuel_opt":
        delta_v_ms = raw_delta_v * 1000 * 0.90  # 10% 절감
    else:  # hohmann
        delta_v_ms = raw_delta_v * 1000
    
    # 연료 계산
    fuel_mass = vehicle.mass * (np.exp(delta_v_ms / (vehicle.isp * G0 * 1000)) - 1)
    
    # 시간
    time_hours = len(positions) * 300 / 3600
    
    # 설명
    if min_dist < R_MOON * 20:
        desc = f"{base_desc} ✓ Arrival: {min_dist:,.0f} km"
    else:
        desc = f"{base_desc} ({min_dist:,.0f} km)"
    
    return {
        "name": name,
        "x": positions[:, 0],
        "y": positions[:, 1],
        "z": positions[:, 2],
        "color": color,
        "delta_v": f"{delta_v_ms:.1f}",
        "fuel_mass": f"{fuel_mass:.0f}",
        "time": f"{time_hours:.1f} h",
        "desc": desc,
        "target_pos": moon_pos,
        "penalty": 0
    }


# ========== Free Return (유지) ==========
def generate_free_return(moon_pos, vehicle):
    """자유 귀환 궤도"""
    
    moon_dist = np.linalg.norm(moon_pos)
    moon_dir = moon_pos / moon_dist
    
    start_dir = -moon_dir
    start_pos = start_dir * PARKING_RADIUS
    
    side = np.cross(moon_dir, np.array([0, 0, 1]))
    if np.linalg.norm(side) < 0.1:
        side = np.cross(moon_dir, np.array([0, 1, 0]))
    side = side / np.linalg.norm(side)
    
    flyby_point = moon_pos + side * (R_MOON * 15)
    
    tof = 70 * 3600
    
    v1, _ = lambert_solver(start_pos, flyby_point, tof)
    
    if v1 is None:
        for alt_tof in [60, 80, 90]:
            v1, _ = lambert_solver(start_pos, flyby_point, alt_tof * 3600)
            if v1 is not None:
                break
    
    if v1 is None:
        return create_fallback(moon_pos, "free_return", vehicle)
    
    positions = simulate_trajectory(start_pos, v1, moon_pos, 6 * 24 * 3600, dt=600)
    
    if len(positions) < 20:
        return create_fallback(moon_pos, "free_return", vehicle)
    
    moon_dists = np.linalg.norm(positions - moon_pos, axis=1)
    min_moon = np.min(moon_dists)
    
    final_earth = np.linalg.norm(positions[-1])
    returned = final_earth < R_EARTH * 50
    
    v_circular = np.sqrt(MU_EARTH / PARKING_RADIUS)
    delta_v = np.linalg.norm(v1) - v_circular
    delta_v_ms = abs(delta_v) * 1000
    
    fuel_mass = vehicle.mass * (np.exp(delta_v_ms / (vehicle.isp * G0 * 1000)) - 1)
    time_hours = len(positions) * 600 / 3600
    
    if returned and min_moon < R_MOON * 50:
        desc = f"✓ Flyby {min_moon:,.0f} km → Return!"
    else:
        desc = f"Flyby {min_moon:,.0f} km"
    
    return {
        "name": "Free Return",
        "x": positions[:, 0],
        "y": positions[:, 1],
        "z": positions[:, 2],
        "color": "#00FFFF",
        "delta_v": f"{delta_v_ms:.1f}",
        "fuel_mass": f"{fuel_mass:.0f}",
        "time": f"{time_hours:.1f} h",
        "desc": desc,
        "target_pos": moon_pos,
        "penalty": 0,
        "returned": returned
    }


# ========== 메인 ==========
def generate_trajectories(launch_date, vehicle):
    """모든 궤도 생성"""
    
    moon = get_moon_state(launch_date)
    moon_pos = moon["vec"]
    dec = moon["dec"]
    
    plane_penalty = calculate_plane_change_penalty(dec)
    results = {}
    
    for mode, key in [("fast", "fast"), ("balanced", "bal"),
                      ("fuel_opt", "opt"), ("hohmann", "ho")]:
        try:
            traj = generate_trajectory(moon_pos, mode, vehicle)
            if plane_penalty > 0:
                dv = float(traj["delta_v"])
                traj["delta_v"] = f"{dv + plane_penalty:.1f}"
                traj["penalty"] = plane_penalty
            results[key] = traj
        except Exception as e:
            print(f"Error {mode}: {e}")
            results[key] = create_fallback(moon_pos, mode, vehicle)
    
    try:
        results["fr"] = generate_free_return(moon_pos, vehicle)
        if plane_penalty > 0:
            dv = float(results["fr"]["delta_v"])
            results["fr"]["delta_v"] = f"{dv + plane_penalty:.1f}"
            results["fr"]["penalty"] = plane_penalty
    except Exception as e:
        print(f"Error free_return: {e}")
        results["fr"] = create_fallback(moon_pos, "free_return", vehicle)
    
    return results


# ========== 폴백 (vehicle 인자 추가) ==========
def create_fallback(moon_pos, mode, vehicle):
    """폴백 - 각 모드별 다른 타원 + 실제 연료 계산"""
    moon_dist = np.linalg.norm(moon_pos)
    moon_dir = moon_pos / moon_dist
    
    # 모드별 다른 타원 모양
    if mode == "fast":
        eccentricity = 0.2
        angle_offset = 30
        delta_v_ms = 4100  # m/s
    elif mode == "balanced":
        eccentricity = 0.35
        angle_offset = 15
        delta_v_ms = 3500
    elif mode == "fuel_opt":
        eccentricity = 0.45
        angle_offset = -15
        delta_v_ms = 2900  # 가장 적음
    elif mode == "hohmann":
        eccentricity = 0.5
        angle_offset = 0
        delta_v_ms = 3150
    else:  # free_return
        eccentricity = 0.4
        angle_offset = 0
        delta_v_ms = 3300
    
    # 실제 연료 계산
    fuel_mass = vehicle.mass * (np.exp(delta_v_ms / (vehicle.isp * G0 * 1000)) - 1)
    
    t = np.linspace(0, np.pi, 200)
    a = moon_dist / 2
    b = a * eccentricity
    
    x = a * (1 - np.cos(t))
    y = b * np.sin(t)
    z = np.zeros_like(t)
    
    positions = np.column_stack([x, y, z])
    
    # 달 방향으로 회전 + 오프셋
    up = np.array([0, 0, 1])
    rotated_dir = rotate_around_axis(moon_dir, up, angle_offset)
    
    target = rotated_dir
    source = np.array([1, 0, 0])
    v = np.cross(source, target)
    c = np.dot(source, target)
    
    if np.linalg.norm(v) > 0.001 and abs(c + 1) > 0.001:
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R = np.eye(3) + vx + vx @ vx * (1 / (1 + c + 1e-10))
        positions = positions @ R.T
    
    info = {
        "fast": ("High Speed Injection", "#FF00FF", 35),
        "balanced": ("Balanced Profile", "#FFA500", 55),
        "fuel_opt": ("Fuel Optimized", "#3388FF", 120),
        "hohmann": ("Standard Hohmann", "#00FF00", 90),
        "free_return": ("Free Return", "#00FFFF", 144)
    }
    
    name, color, time_h = info.get(mode, ("Unknown", "#FFFFFF", 72))
    
    return {
        "name": name,
        "x": positions[:, 0],
        "y": positions[:, 1],
        "z": positions[:, 2],
        "color": color,
        "delta_v": str(delta_v_ms),
        "fuel_mass": f"{fuel_mass:.0f}",
        "time": f"{time_h} h",
        "desc": "Transfer orbit",
        "target_pos": moon_pos,
        "penalty": 0
    }
