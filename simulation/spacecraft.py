"""
Celestium 2.0 - Spacecraft Definitions
"""
import numpy as np

class Vehicle:
    def __init__(self, name, dry_mass, isp, fuel_capacity, img_url, description):
        """
        Args:
            name: 기체 이름
            dry_mass: 건조 질량 (kg) - 연료 제외
            isp: 비추력 (초)
            fuel_capacity: 연료 탑재량 (kg)
            img_url: 이미지 URL
            description: 설명
        """
        self.name = name
        self.mass = dry_mass
        self.isp = isp
        self.fuel_capacity = fuel_capacity
        self.img_url = img_url
        self.description = description
    
    @property
    def total_mass(self):
        """총 질량 (건조 + 연료)"""
        return self.mass + self.fuel_capacity
    
    def get_description(self):
        return f"""
        <div style="color: #aaddff;">
            <b>Dry Mass:</b> {self.mass:,} kg<br>
            <b>ISP:</b> {self.isp} s<br>
            <b>Fuel Cap:</b> {self.fuel_capacity:,} kg<br>
            <b>Total Mass:</b> {self.total_mass:,} kg
            <br><br>{self.description}
        </div>
        """
    
    def calculate_delta_v_budget(self):
        """이 기체로 가능한 최대 ΔV 계산"""
        from .physics import G0
        mass_ratio = self.total_mass / self.mass
        return self.isp * G0 * 1000 * np.log(mass_ratio)  # m/s


# 차량 데이터베이스
VEHICLES = {
    "SpaceX Starship": Vehicle(
        "SpaceX Starship",
        dry_mass=120000,
        isp=380,
        fuel_capacity=1200000,
        img_url="https://cdn.digitaltoday.co.kr/news/photo/202411/540369_504944_5721.jpg",
        description="Fully reusable super-heavy lift vehicle. Assumes orbital refueling complete for lunar missions."
    ),
    "Apollo Saturn V (S-IVB)": Vehicle(
        "Apollo Saturn V (S-IVB)",
        dry_mass=13500,  # S-IVB 3단계 건조 질량
        isp=421,
        fuel_capacity=110000,
        img_url="https://images-assets.nasa.gov/image/KSC-71PC-0571/KSC-71PC-0571~large.jpg",
        description="Third stage of Saturn V used for Trans-Lunar Injection. Historic Apollo program vehicle."
    ),
    "SLS Block 1B (EUS)": Vehicle(
        "SLS Block 1B (EUS)",
        dry_mass=12000,  # Exploration Upper Stage
        isp=462,
        fuel_capacity=130000,
        img_url="https://cdn.jetphotos.com/full/5/1491940_1712030325.jpg",
        description="NASA's Exploration Upper Stage for Artemis lunar missions. RL10 engines."
    ),
    "Custom Vehicle": Vehicle(
        "Custom Vehicle",
        dry_mass=50000,
        isp=350,
        fuel_capacity=200000,
        img_url="https://via.placeholder.com/300x200?text=Custom+Vehicle",
        description="User-defined vehicle parameters. Adjust in settings."
    )
}
