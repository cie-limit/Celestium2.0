"""
Celestium 2.0 - Numerical Integrator
오일러/RK4 수치 적분기 (안정화 버전)
"""
import numpy as np
from .physics import GravityModel, R_EARTH, R_MOON


class OrbitalIntegrator:
    """궤도 수치 적분기"""
    
    def __init__(self, gravity_model, method='rk4'):
        self.gravity = gravity_model
        self.method = method
    
    def step_euler(self, pos, vel, dt):
        """오일러 방법"""
        acc = self.gravity.acceleration(pos)
        new_pos = pos + vel * dt
        new_vel = vel + acc * dt
        return new_pos, new_vel
    
    def step_rk4(self, pos, vel, dt):
        """4차 Runge-Kutta"""
        try:
            k1_v = self.gravity.acceleration(pos)
            k1_r = vel
            
            k2_v = self.gravity.acceleration(pos + 0.5*dt*k1_r)
            k2_r = vel + 0.5*dt*k1_v
            
            k3_v = self.gravity.acceleration(pos + 0.5*dt*k2_r)
            k3_r = vel + 0.5*dt*k2_v
            
            k4_v = self.gravity.acceleration(pos + dt*k3_r)
            k4_r = vel + dt*k3_v
            
            new_pos = pos + (dt/6) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
            new_vel = vel + (dt/6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
            
            return new_pos, new_vel
        except:
            return self.step_euler(pos, vel, dt)
    
    def step(self, pos, vel, dt):
        if self.method == 'euler':
            return self.step_euler(pos, vel, dt)
        else:
            return self.step_rk4(pos, vel, dt)
    
    def propagate(self, initial_pos, initial_vel, total_time, dt=60.0, max_points=2000):
        """궤도 전파"""
        
        n_steps = int(total_time / dt)
        if n_steps < 1:
            n_steps = 1
        
        save_interval = max(1, n_steps // max_points)
        
        positions = [initial_pos.copy()]
        velocities = [initial_vel.copy()]
        times = [0.0]
        
        pos = initial_pos.copy()
        vel = initial_vel.copy()
        
        moon_pos = self.gravity.moon_pos
        
        for i in range(1, n_steps + 1):
            try:
                pos, vel = self.step(pos, vel, dt)
            except:
                break
            
            # NaN 체크
            if np.any(np.isnan(pos)) or np.any(np.isnan(vel)):
                break
            
            # 지구 충돌 체크
            earth_dist = np.linalg.norm(pos)
            if earth_dist < R_EARTH * 0.9:
                break
            
            # 너무 멀리 가면 중단 (달 거리의 3배)
            if earth_dist > np.linalg.norm(moon_pos) * 3:
                break
            
            # 달 근처 도달
            moon_dist = np.linalg.norm(pos - moon_pos)
            if moon_dist < R_MOON * 1.5:
                positions.append(pos.copy())
                velocities.append(vel.copy())
                times.append(i * dt)
                break
            
            # 저장
            if i % save_interval == 0:
                positions.append(pos.copy())
                velocities.append(vel.copy())
                times.append(i * dt)
        
        return np.array(positions), np.array(velocities), np.array(times)


class AdaptiveIntegrator(OrbitalIntegrator):
    """적응형 시간 간격 적분기"""
    
    def propagate_adaptive(self, initial_pos, initial_vel, total_time,
                          dt_min=10.0, dt_max=3600.0, tolerance=1e-6, max_points=2000):
        """적응형 시간 간격 전파"""
        
        positions = [initial_pos.copy()]
        velocities = [initial_vel.copy()]
        times = [0.0]
        
        pos = initial_pos.copy()
        vel = initial_vel.copy()
        t = 0.0
        
        moon_pos = self.gravity.moon_pos
        moon_orbit_dist = np.linalg.norm(moon_pos)
        
        while t < total_time and len(positions) < max_points:
            try:
                # 현재 위치 기반 dt 조정
                earth_dist = np.linalg.norm(pos)
                moon_dist = np.linalg.norm(pos - moon_pos)
                
                # 천체에 가까우면 작은 dt
                min_dist = min(earth_dist - R_EARTH, moon_dist - R_MOON)
                if min_dist < 50000:
                    dt = dt_min
                elif min_dist < 200000:
                    dt = dt_min * 5
                else:
                    dt = dt_max
                
                # 남은 시간 체크
                if t + dt > total_time:
                    dt = total_time - t
                
                pos, vel = self.step_rk4(pos, vel, dt)
                t += dt
                
                # NaN 체크
                if np.any(np.isnan(pos)) or np.any(np.isnan(vel)):
                    break
                
                # 지구 충돌
                if np.linalg.norm(pos) < R_EARTH * 0.9:
                    break
                
                # 너무 멀리
                if np.linalg.norm(pos) > moon_orbit_dist * 3:
                    break
                
                # 달 도달
                if np.linalg.norm(pos - moon_pos) < R_MOON * 1.5:
                    positions.append(pos.copy())
                    velocities.append(vel.copy())
                    times.append(t)
                    break
                
                positions.append(pos.copy())
                velocities.append(vel.copy())
                times.append(t)
                
            except:
                break
        
        return np.array(positions), np.array(velocities), np.array(times)
