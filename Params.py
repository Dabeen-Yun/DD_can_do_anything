# constellation
POLAR_LATITUDE = 70
CONSTELLATION_PARAM_F = 0
ORBIT_ALTITUDE = 780000    # m
R_EARTH_RADIUS = 6371000   # m
SATELLITE_ORBITAL_PERIOD = 5400 #sec (현재 사용 안 함 ms 단위로 변환 필요?)
NUM_ORBITS = 6
NUM_SATELLITES = 66
NUM_SATELLITES_PER_ORBIT = NUM_SATELLITES // NUM_ORBITS
SATELLITE_SPEED = 7.4 #km/s 27000 km/h
DEGREE_TO_KM = 111 # 6378 지구 반지름에 따른 위도 1도 거리
LAT_RANGE = [-90, 90]
LON_RANGE = [0, 360]
LAT_STEP = 60
LON_STEP = 72
PARAM_C = 299792458 # [m/s]

# SFC
# VNF_PER_SFC = (1, 2) # 현재 5개 고정
SFC_TYPE_LIST = {
    0: ['0', '1', '2', '3'], #eMBB
    1: ['0'], #uRLLC
    2: ['4', '5', '6'], # 'mMTC'
}
VNF_SIZE = 1e6 # [bit]  ≈125 KB
#100 * 8 # [bit] (100 B * 8)
SFC_SIZE = 5e6 # [bit]  ≈625 KB
#512 * 8 # [bit]

# simulation
NUM_ITERATIONS = 100 #400
NUM_GSFC = 3 #33 # int(2*1024*1024/SFC_SIZE*8) #[per ms]
TAU = 1000 # 1ms 단위로 맞추기

# VSG
# 0: FW
# 1: DPI
# 2: UE-UE local forwarder
# 3: IP-over-IP
# 4: compress
# 5: security wrapper
# 6: egress
VNF_TYPES_PER_VSG = (0, 7) #3

NUM_VNFS = 5 #10

# satellite capacity
SAT_QUEUE_SIZE = (SFC_SIZE // 8) * 50 * 8 # [bit] (SFC_SIZE[B] * 50) * 8
SAT_LINK_CAPACITY = 250 * 1e6 # [Mbps] -> [bps]
# SAT_PROCESSING_RATE: 50 MB/S -> 50 * 8 Mbps = 400 Mbps
SAT_PROCESSING_RATE = 400 * 1e6 # [Mbps] -> [bps]

# Gserver capacity
GSERVER_QUEUE_SIZE = (SFC_SIZE // 8) * 100 * 8 # [bit] (SFC_SIZE[B] * 100) * 8
GSERVER_LINK_CAPACITY = 500 * 1e6 # [bps]
# GSERVER_PROCESSING_RATE: 200 MB/S -> 200 * 8 Mbps = 1600 Mbps
GSERVER_PROCESSING_RATE = 1600 * 1e6 # [bps]
