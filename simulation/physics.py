"""
Celestium 2.0 - Physics Engine
실제 중력 물리 계산을 담당하는 모듈
"""
import numpy as np

# 물리 상수
G = 6.67430e-20  # km^3 / (kg * s^2)
MU_EARTH = 398600.4418  # km^3/s^2 (지구 중력 상수)
MU_MOON = 4902.8000  # km^3/s^2 (달 중력 상수)
R_EARTH = 6371.0  # km
R_MOON = 1737.4  # km
G0 = 9.80665e-3  # km/s^2


class GravityModel:
    """중력 모델 클래스"""
    
    def __init__(self, moon_position):
        """
        Args:
            moon_position: 달의 위치 벡터 [x, y, z] (km)
        """
        self.earth_pos = np.array([0.0, 0.0, 0.0])
        self.moon_pos = np.array(moon_position)
    
    def acceleration(self, pos):
        """
        주어진 위치에서의 총 가속도 계산 (지구 + 달 중력)
        
        Args:
            pos: 우주선 위치 [x, y, z] (km)
        
        Returns:
            acceleration: 가속도 벡터 [ax, ay, az] (km/s^2)
        """
        # 지구로부터의 거리 및 가속도
        r_earth = pos - self.earth_pos
        dist_earth = np.linalg.norm(r_earth)
        if dist_earth < R_EARTH:
            dist_earth = R_EARTH  # 지표면 아래로 가지 않도록
        a_earth = -MU_EARTH * r_earth / (dist_earth ** 3)
        
        # 달로부터의 거리 및 가속도
        r_moon = pos - self.moon_pos
        dist_moon = np.linalg.norm(r_moon)
        if dist_moon < R_MOON:
            dist_moon = R_MOON
        a_moon = -MU_MOON * r_moon / (dist_moon ** 3)
        
        return a_earth + a_moon
    
    def acceleration_earth_only(self, pos):
        """지구 중력만 계산 (2체 문제용)"""
        r_earth = pos - self.earth_pos
        dist_earth = np.linalg.norm(r_earth)
        if dist_earth < R_EARTH:
            dist_earth = R_EARTH
        return -MU_EARTH * r_earth / (dist_earth ** 3)


def vis_viva_velocity(r, a, mu=MU_EARTH):
    """
    비스-비바 방정식으로 궤도 속도 계산
    v^2 = μ(2/r - 1/a)
    
    Args:
        r: 현재 위치의 거리 (km)
        a: 궤도 장반경 (km)
        mu: 중력 상수
    
    Returns:
        velocity: 속도 크기 (km/s)
    """
    return np.sqrt(mu * (2/r - 1/a))


def hohmann_delta_v(r1, r2, mu=MU_EARTH):
    """
    호만 전이에 필요한 ΔV 계산
    
    Args:
        r1: 출발 궤도 반경 (km)
        r2: 도착 궤도 반경 (km)
    
    Returns:
        dv1: 첫 번째 분사 ΔV (km/s)
        dv2: 두 번째 분사 ΔV (km/s)
        total_dv: 총 ΔV (km/s)
    """
    # 전이 궤도 장반경
    a_transfer = (r1 + r2) / 2
    
    # 출발 궤도에서의 원 궤도 속도
    v1_circular = np.sqrt(mu / r1)
    
    # 전이 궤도 근지점 속도
    v1_transfer = vis_viva_velocity(r1, a_transfer, mu)
    
    # 첫 번째 ΔV
    dv1 = v1_transfer - v1_circular
    
    # 도착 궤도에서의 원 궤도 속도
    v2_circular = np.sqrt(mu / r2)
    
    # 전이 궤도 원지점 속도
    v2_transfer = vis_viva_velocity(r2, a_transfer, mu)
    
    # 두 번째 ΔV
    dv2 = v2_circular - v2_transfer
    
    return dv1, dv2, abs(dv1) + abs(dv2)


def calculate_tli_velocity(r_parking, target_distance):
    """
    Trans-Lunar Injection 속도 계산
    
    Args:
        r_parking: 지구 주차 궤도 반경 (km)
        target_distance: 목표 거리 (달 궤도 반경, km)
    
    Returns:
        v_tli: TLI 속도 (km/s)
        dv: 필요한 ΔV (km/s)
    """
    # 전이 궤도의 장반경
    a_transfer = (r_parking + target_distance) / 2
    
    # 주차 궤도 속도
    v_parking = np.sqrt(MU_EARTH / r_parking)
    
    # TLI 속도 (비스-비바)
    v_tli = vis_viva_velocity(r_parking, a_transfer)
    
    # 필요 ΔV
    dv = v_tli - v_parking
    
    return v_tli, dv


def calculate_transfer_time(r1, r2, mu=MU_EARTH):
    """
    호만 전이 시간 계산
    
    Args:
        r1: 출발 궤도 반경 (km)
        r2: 도착 궤도 반경 (km)
    
    Returns:
        time: 전이 시간 (초)
    """
    a_transfer = (r1 + r2) / 2
    # 케플러 제3법칙: T = 2π√(a³/μ)
    period = 2 * np.pi * np.sqrt(a_transfer**3 / mu)
    # 호만 전이는 반 주기
    return period / 2


def escape_velocity(r, mu=MU_EARTH):
    """탈출 속도 계산"""
    return np.sqrt(2 * mu / r)


def sphere_of_influence(a, m_small, m_large):
    """
    중력권 반경 계산 (Hill sphere 근사)
    
    Args:
        a: 두 천체 사이 거리
        m_small: 작은 천체 질량
        m_large: 큰 천체 질량
    """
    return a * (m_small / m_large) ** (2/5)


# 달의 중력권 반경 (약 66,000 km)
MOON_SOI = sphere_of_influence(384400, 7.342e22, 5.972e24)
