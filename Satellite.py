from Params import *

import random
import numpy as np
import math
from collections import deque
d2r=np.deg2rad
r2d=np.rad2deg

class Satellite:
    def __init__(self, id, orb, spo, alt, phasing_inter_plane, inc_deg, sat_list, method=None, active_satellites=None):
        self.id = id
        self.sat_list = sat_list

        # 위치
        self.orb = orb                  # P
        self.spo = spo                  # S
        self.sat = orb * spo            # t = P * S
        self.x = self.id // self.spo    # plane index
        self.y = self.id % self.spo     # slot index
        self.lat = 0               # grid latitude [0~360]
        self.lon = 0
        self.alt = alt
        self.inc_deg = inc_deg          # inclination (deg)

        self.phasing_offset_deg = lambda y: (360 * CONSTELLATION_PARAM_F * self.y) / self.sat
        self.orbit_spacing_deg = 360 / self.orb
        self.phasing_inter_plane = phasing_inter_plane

        self.phasing_intra_plane = 360 / self.spo
        self.phasing_adjacent_plane = CONSTELLATION_PARAM_F * 360 / self.sat
        self.phase_list = [self.phasing_intra_plane, self.phasing_inter_plane, self.phasing_adjacent_plane]

        # 시간 [ms]
        self.time = 0

        # routing table
        self.routing_table = []

        self.vnf_list = []      #list [int]
        self.current_vsg_id = -1   #int
        self.vsg_enter_time = -1 #int
        self.drop_rate = random.uniform(0, 1) #-1

        # Inter Satellite Link (ISL)
        # adjacent satellite : [intra1, intra2, inter1, inter2]
        self.adj_sat_index_list = [-1, -1, -1, -1]
        self.adj_sat_p_d_list = [-1, -1, -1, -1]
        self.intra_ISL_list = []
        self.inter_ISL_list = []
        self.intra_ISL_p_d = []
        self.inter_ISL_p_d = []

        # processing queue
        self.process_queue = deque()

        self.num_process_vnf = 2 # 한 번에 처리할 수 있는 vnf 개수

        # ISL queue # transmitting queue
        self.queue_ISL_intra_1 = []
        self.queue_ISL_intra_2 = []
        self.queue_ISL_inter_1 = []
        self.queue_ISL_inter_2 = []
        self.queue_ISL = [self.queue_ISL_intra_1, self.queue_ISL_intra_2,
                          self.queue_ISL_inter_1, self.queue_ISL_inter_2]

        self.queue_TSL = []

        self.set_lla()
        # self.set_adjacent_node()
        # self.get_propagation_delay()

        # 자원
        self.moving_probability = 0 # PPCC 파라미터
        self.ISL_queue = [[],[],[],[]]

        self.current_load = 0  # 현재 처리 중인 traffic (in Mbps)

        # if "PPCC" in method: #PPCC-k, PPCC-f
        #     self.set_ppcc_info(active_satellites)

    def get_ecef_coords(self, lat, lon):
        """LLA (Lat, Lon, Alt)를 ECEF (x, y, z)로 변환합니다."""

        # Convert to radians
        lat_rad = d2r(lat)
        lon_rad = d2r(lon)

        R_obj = R_EARTH_RADIUS + ORBIT_ALTITUDE

        # ECEF conversion
        x = R_obj * math.cos(lat_rad) * math.cos(lon_rad)
        y = R_obj * math.cos(lat_rad) * math.sin(lon_rad)
        z = R_obj * math.sin(lat_rad)

        return x, y, z

    def calculate_delay_to_sat(self, other_sat):
        """현재 위성과 다른 위성 간의 전파 지연 시간(ms)을 ECEF 기반으로 계산합니다."""

        # 1. Get ECEF coords for self
        x1, y1, z1 = self.get_ecef_coords(self.lat, self.lon)

        # 2. Get ECEF coords for other_sat
        x2, y2, z2 = self.get_ecef_coords(other_sat.lat, other_sat.lon)

        # 3. Calculate 3D Euclidean distance (m)
        distance_m = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

        # 4. Calculate propagation delay (ms)
        delay_s = distance_m / PARAM_C
        delay_ms = delay_s * 1000

        # Original code used 'round(length / PARAM_C)', so we round the result to the nearest integer (ms).
        return delay_ms

    def get_propagation_delay(self):
        """
        인접 위성 간의 전파 지연 시간(Propagation Delay, ms)을 ECEF 기반으로 계산하여 업데이트
        """
        # 리스트 초기화
        self.adj_sat_p_d_list = []
        self.intra_ISL_p_d = []
        self.inter_ISL_p_d = []

        # adj_sat_index_list: [intra1, intra2, inter1, inter2] 순서
        for i, adj_sat_id in enumerate(self.adj_sat_index_list):
            if adj_sat_id == -1:
                delay = -1
            else:
                try:
                    adj_sat = self.sat_list[adj_sat_id]
                    # ECEF 기반 지연 시간 계산
                    delay = self.calculate_delay_to_sat(adj_sat)
                except IndexError:
                    delay = -1

            # Update the main adjacent delay list
            self.adj_sat_p_d_list.append(delay)

            # Update the intra/inter lists for compatibility
            if i < 2:  # Indices 0 and 1 are intra-plane (vertical)
                self.intra_ISL_p_d.append(delay)
            else:  # Indices 2 and 3 are inter-plane (horizontal)
                self.inter_ISL_p_d.append(delay)

    def set_adjacent_node(self):
        # horizontal adjacent node
        h_adj_1 = self.id + self.spo
        h_adj_2 = self.id - self.spo
        if h_adj_1 >= self.sat:
            h_adj_1 = h_adj_1 % self.sat
        if h_adj_2 < 0:
            h_adj_2 = self.sat + h_adj_2
        self.adj_sat_index_list[2] = h_adj_1
        self.adj_sat_index_list[3] = h_adj_2
        self.inter_ISL_list.append(h_adj_1)
        self.inter_ISL_list.append(h_adj_2)

        # vertical adjacent node
        v_adj_1 = self.id + 1
        v_adj_2 = self.id - 1
        if self.id // self.spo != v_adj_1 // self.spo:
            v_adj_1 -= self.spo
        if self.id // self.spo != v_adj_2 // self.spo:
            v_adj_2 += self.spo
        self.adj_sat_index_list[0] = v_adj_1
        self.adj_sat_index_list[1] = v_adj_2
        self.intra_ISL_list.append(v_adj_1)
        self.intra_ISL_list.append(v_adj_2)

    # walker-star constellation
    def set_lla(self):
        self.lon = (self.x * self.orbit_spacing_deg + self.phasing_offset_deg(self.y)) % 360

        phase = (2 * np.pi * self.y) / self.spo
        self.lat = 90 * np.sin(phase)  # POLAR_LATITUDE * np.sin(phase)

    def time_tic(self, delta_time=1): # 1ms 마다
        self.time += delta_time

        orbital_period_ms = SATELLITE_ORBITAL_PERIOD * 1000 # sec->ms 변환

        # 궤도 주기 계산 (초당 360도 회전)
        mean_motion_deg_per_ms = 360 / orbital_period_ms

        # 위성의 초기 위상 차이 (경도 기준)
        init_lon = (self.x * self.orbit_spacing_deg + self.phasing_offset_deg(self.y)) % 360

        # 시간에 따라 경도 업데이트
        self.lon = (init_lon + mean_motion_deg_per_ms * self.time) % 360

        # 위도는 경사 궤도를 따라 sin 형태로 주기적 움직임
        # 전체 궤도 주기 90분 기준으로 위도 변화 (위상은 y에 따라 달라짐)
        phase = 2 * np.pi * self.y / self.spo
        self.lat = 90 * np.sin(2 * np.pi * self.time / orbital_period_ms + phase)

    def get_vnf_type(self, vnf_type):
        vnf_string_to_process = None

        # 1. 입력 형태를 분석하여 'vnf'가 포함된 실제 문자열을 찾습니다.
        if isinstance(vnf_type, (list, tuple)):
            # 리스트/튜플에서 'vnf'가 포함된 문자열 찾기
            for item in vnf_type:
                if isinstance(item, str) and 'vnf' in item.lower():
                    vnf_string_to_process = item
                    break
        elif isinstance(vnf_type, str) and 'vnf' in vnf_type.lower():
            # 단일 문자열인 경우
            vnf_string_to_process = vnf_type
        else:
            # 'src', 'dst' 등 VNF가 아닌 경우이거나 VNF 문자열을 찾지 못한 경우
            vnf_string_to_process = vnf_type

            # 2. VNF 문자열에서 숫자만 추출
        if isinstance(vnf_string_to_process, str) and 'vnf' in vnf_string_to_process.lower():
            # 소문자 변환 후 'vnf' 접두사를 빈 문자열로 대체하여 숫자만 남깁니다.
            # 예: 'vnf0' -> '0', 'VNF10' -> '10'
            next_vnf_type = vnf_string_to_process.lower().replace('vnf', '')

            # 🚨 참고: VNF 타입이 'vnf0'처럼 숫자만 남는 형태라면,
            # 아래처럼 숫자가 아닌 모든 문자를 제거하는 정규 표현식이 가장 안전합니다.
            # next_vnf_type = re.sub(r'\D+', '', vnf_string_to_process)
        else:
            # VNF가 아닌 경우 (예: 'src', 'dst')는 원본 문자열을 그대로 사용합니다.
            next_vnf_type = vnf_string_to_process

        return next_vnf_type

    def add_to_process_queue(self, gsfc, vnf_type=None): #vnf_id: vnf 종류가 아닌, 해당 gsfc에서 실행되는 순서
        if vnf_type is not None: #mMTC (현 위성에 vnf 탑재 안되어있는데 process queue에 들어온 경우)
            next_vnf_type = self.get_vnf_type(vnf_type)

            if next_vnf_type in self.vnf_list:
                self.process_queue.append([gsfc.id, gsfc.num_completed_vnf, gsfc.vnf_sizes[gsfc.num_completed_vnf]])
                gsfc.is_keeping = False
            else:
                gsfc.is_keeping = True
        else:
            self.process_queue.append([gsfc.id, gsfc.num_completed_vnf, gsfc.vnf_sizes[gsfc.num_completed_vnf]])

        # print(f"[QUEUE LOG] Sat/ {self.id} | Added GSFC {gsfc_id} (VNF {vnf_id}) to PROC Queue. Size: {vnf_size}")

    def pop_from_process_queue(self, idx):
        if 0 <= idx < len(self.process_queue):
            del self.process_queue[idx]
            return True
        else:
            print(f"[WARNING] Satellite {self.id}: Index {idx} out of range in Process queue.")
            return None

    def process_vnfs(self, all_gsfc_list, mode='dd', processing_rate=None): # all_gsfc_list: 메인 루프에서 self.gsfc_list
        if processing_rate is None:
            processing_rate = SAT_PROCESSING_RATE

        # GSFC ID로 객체를 빠르게 조회하기 위한 맵 생성
        gsfc_map = {gsfc.id: gsfc for gsfc in all_gsfc_list}

        current_tasks_info = [] # [(queue_index, gsfc_id, vnf_id)]
        completed_gsfc_id = []  # 제거할 인덱스를 저장

        # vnf_size를 통해 현재 처리해야하는 총 용량 파악
        for i, item in enumerate(self.process_queue):
            gsfc_id, vnf_id, _ = item

            if gsfc_id not in gsfc_map:
                continue
            gsfc = gsfc_map[gsfc_id]
            # 큐의 VNF 순서와 GSFC의 다음 VNF 순서가 일치하고, 남은 사이즈가 0보다 커야 유효
            if gsfc.num_completed_vnf != vnf_id:
                print(f"[ERROR] synch xxxxx: gsfc_id {gsfc.id}, num_completed_vnf: {gsfc.num_completed_vnf}, vnf_id: {vnf_id}")
                continue

            # 현재 그 위성에서 처리할 수 있는 gsfc id들과 vnf size 합 구하기
            if i < self.num_process_vnf:  # 최대 처리 개수 제한 내 (처리 지연 발생)
                current_tasks_info.append((i, gsfc_id, vnf_id))
            else:
                # 처리 대상 개수 초과 -> 큐 지연 발생 1ms씩 증가
                gsfc.accumulate_queue_delay(mode=mode)

        num_tasks = len(current_tasks_info)
        if num_tasks == 0:
            print(f"[WARNING] NO left tasks in satellite {self.id}, process_queue: {self.process_queue}")
            return

        power_per_vnf = (processing_rate / TAU) / num_tasks # [bits/ms]
        # print(f"process vnfs / satellite processing rate {processing_rate}, power per vnf {power_per_vnf}")
        for queue_index, gsfc_id, vnf_id in current_tasks_info:
            gsfc = gsfc_map[gsfc_id]

            # GSFC의 processing_vnf를 호출하여 사이즈 업데이트 (완료 여부 확인)
            is_completed = gsfc.processing_vnf(power_per_vnf, mode=mode)

            if is_completed: # 완료된 gsfc 확인 - transmitting queue에 등록
                # print(f"[PROCESSING COMPLETE] Sat {self.id}: GSFC {gsfc.id} VNF 처리 완료. 다음 홉으로 포워딩.")
                # TODO 다음 경로도 동일한 위성인지 확인 (동일한 위성인데, 또 vnf 처리한다면 completed_gsfc_id지웠다가 맨 뒤에 다시 깔아야함)
                completed_gsfc_id.append(queue_index)
                self.add_to_transmit_queue(gsfc, mode=mode)
            else:
                # [사용자 요청 반영] 미완료: 큐에 남은 사이즈를 직접 업데이트 (인덱스 2가 남은 사이즈)
                # 현재는 vnf 개수로 power를 등분하기 때문에 불필요
                self.process_queue[queue_index][2] -= power_per_vnf

        # 큐 인덱스가 틀어지지 않도록 역순으로 정렬 후 제거a
        completed_gsfc_id.sort(reverse=True)
        for idx in completed_gsfc_id:
            self.pop_from_process_queue(idx)

    def transmit_gsfcs(self, all_gsfc_list, mode='dd'):
        """Transmission Delay [ms] = Packet Length [bit] / Data Rate [bit/ms]"""

        gsfc_map = {gsfc.id:gsfc for gsfc in all_gsfc_list}
        completed_gsfc_ids = []

        for link_index in range(len(self.queue_ISL)):
            isl_queue = self.queue_ISL[link_index]
            if not isl_queue:
                continue

            # 첫 번째 gsfc 이외는 transmit 대기 -> queueing 지연 추가
            for q_idx in range(1, len(isl_queue)):
                gsfc_id, _ = isl_queue[q_idx]
                gsfc = gsfc_map[gsfc_id]
                if gsfc is not None:
                    gsfc.accumulate_queue_delay(mode=mode)
                else:
                    print(f"[ERROR] Sat {self.id} (Link {link_index}) has unknown GSFC {gsfc_id}; skipping queue delay")
                    return []

            # 첫 번째 gsfc부터 내보내기 - transmitting delay 추가 / 나머지 gsfc는 transmit queueing delay 추가
            gsfc_id, remaining_size = isl_queue[0] #나머지는 queueing delay 추가하도록

            if gsfc_id not in gsfc_map:
                # GSFC 객체를 찾을 수 없으면 제거 (오류 상황 가정)
                print(f"[ERROR] Cannot find gsfc id in ISL queue")
                isl_queue.pop(0)
                continue
            gsfc = gsfc_map[gsfc_id]
            # print(f"[TRANS LOG] Sat {self.id} (Link {link_index}): Processing GSFC {gsfc_id}. Remaining: {remaining_size:.2f}. Capacity: {SAT_LINK_CAPACITY/TAU:.2f}")

            # 이번에 처리할 수 있는 양
            transmitted = min(SAT_LINK_CAPACITY/TAU, remaining_size)
            # 전송 지연 업데이트 TODO 지금은 1ms 단위로 업데이트
            gsfc.accumulate_trans_delay(mode=mode)

            # 큐 업데이트 (GSFC size 업데이트)
            isl_queue[0][1] -= transmitted

            if isl_queue[0][1] <= 0: # 첫 번째 gsfc 내보내기 완료
                # propagation delay 추가, transmitting 변수 False, 다음 vnf 여부 확인
                isl_queue.pop(0)
                propagation_delay = self.adj_sat_p_d_list[link_index]

                delay_attr = f"{mode}_prop_delay_ms"
                current_delay = getattr(gsfc, delay_attr)
                setattr(gsfc, delay_attr, current_delay + propagation_delay)
                setattr(gsfc, f"{mode}_is_transmitting", False)
                completed_gsfc_ids.append(gsfc.id)

                # 다음 위성에게 전달하는 로직은 Main 루프에서 별도로 처리해야 함 (경로 업데이트 등)
                # print(f"[TRANS LOG] Sat {self.id} (Link {link_index}): GSFC {gsfc_id} COMPLETED Transmission. Handing off.")
            # else:
                # 전송 미완료
                # print(f"[TRANS LOG] Sat {self.id} (Link {link_index}): GSFC {gsfc_id} Transmitted {transmitted:.2f}. Remaining: {isl_queue[0][1]:.2f}")

        return completed_gsfc_ids

    def get_isl_queue_idx(self, next_sat_id):
        for i, adj_id in enumerate(self.adj_sat_index_list):
            if adj_id == next_sat_id:
                return i
        return -1

    # 현재 위성에서 vnf 처리하는지 확인
    def has_dst_tag(self, x):
        if x is None:
            return False
        if isinstance(x, (list, tuple)):
            return any(isinstance(e, str) and 'dst' in e.lower() for e in x)
        if isinstance(x, str):
            return 'dst' in x.lower()
        return False

    def add_to_transmit_queue(self, gsfc, mode='dd'):
        remain_sat_path = gsfc.get_remain_path(mode=mode)

        if len(remain_sat_path) < 1:
            satellite_path_attr = f"{mode}_satellite_path"
            satellite_path = getattr(gsfc, satellite_path_attr, [])

            was_dst = self.has_dst_tag(satellite_path[-1][1])
            if was_dst:
                setattr(gsfc, f"{mode}_succeed", True)
            return

        next_sat_id = remain_sat_path[0][0]
        if next_sat_id >= NUM_SATELLITES:
            self.add_to_TSL_queue(gsfc, mode=mode)
            return

        link_index = self.get_isl_queue_idx(next_sat_id)

        if link_index != -1:
            # ISL 큐에 추가 및 부하 업데이트
            gsfc_id = gsfc.id
            gsfc_size = gsfc.sfc_size # [bit]

            # self.queue_ISL은 list of lists ([queue_ISL_intra_1, ...])
            self.queue_ISL[link_index].append([gsfc_id, gsfc_size])

            setattr(gsfc, f"{mode}_is_transmitting", True)

            # print(f"[ISL LOG] MODE {mode} Sat {self.id}: Added GSFC {gsfc_id} to ISL Queue (Link {link_index} to Sat {next_sat_id}). Size: {gsfc_size}.")
        else:
            next_vnf = remain_sat_path[0][1]
            next_is_vnf = False
            if isinstance(next_vnf, (list, tuple)):
                next_is_vnf = True if (any(isinstance(e, str) and 'vnf' in e.lower() for e in next_vnf)) else False
            elif isinstance(next_vnf, str):
                next_is_vnf = True if 'vnf' in next_vnf.lower() else False

            if next_sat_id == self.id:  # 위치 이동 안함
                if next_is_vnf:  # vnf 다시 수행
                    if gsfc.num_completed_vnf >= 5:
                        print("shshshsh")
                    self.add_to_process_queue(gsfc, next_vnf)
                    return
                else:
                    getattr(gsfc, f"{mode}_processed_satellite_path").append(remain_sat_path[0])
                    self.add_to_transmit_queue(gsfc, mode=mode)
            else:
                print(f"[ERROR] Satellite {self.id}: Next hop {next_sat_id} not adjacent. GSFC {gsfc.id} Dropped.")
                setattr(gsfc, f"{mode}_dropped", True)

    def add_to_TSL_queue(self, gsfc, mode='sd'):
        self.queue_TSL.append([gsfc.id, gsfc.sfc_size])
        setattr(gsfc, f"{mode}_is_transmitting", True)

    def transmit_TSL_gsfcs(self, all_gsfc_list, mode='dd'):
        """Transmission Delay [ms] = Packet Length [bit] / Data Rate [bit/ms]"""
        gsfc_map = {gsfc.id:gsfc for gsfc in all_gsfc_list}
        completed_gsfc_ids = []

        for q_idx in range(1, len(self.queue_TSL)):
            gsfc_id, _ = self.queue_TSL[q_idx]
            gsfc = gsfc_map[gsfc_id]
            if gsfc is not None:
                gsfc.accumulate_queue_delay(mode=mode)
            else:
                print(f"[ERROR] Sat {self.id} (TSL) has unknown GSFC {gsfc_id}; skipping queue delay")
                return []

        # 첫 번째 gsfc부터 내보내기 - transmitting delay 추가
        gsfc_id, remaining_size = self.queue_TSL[0]
        if gsfc_id not in gsfc_map:
            print(f"[ERROR] Cannot find gsfc id in TSL queue")
            self.queue_TSL.pop(0)
            return []
        gsfc = gsfc_map[gsfc_id]
        # print(f"[TRANS LOG] Sat {self.id} (TSL): Processing GSFC {gsfc_id}. Remaining: {remaining_size:.2f}. Capacity: {SAT_LINK_CAPACITY/TAU:.2f}")

        # 이번에 처리할 수 있는 양
        transmitted = min(SAT_LINK_CAPACITY/TAU, remaining_size)
        # 전송 지연 업데이트 TODO 지금은 1ms 단위로 업데이트
        gsfc.accumulate_trans_delay(mode=mode)

        # 큐 업데이트 (GSFC size 업데이트)
        self.queue_TSL[0][1] -= transmitted

        if self.queue_TSL[0][1] <= 0:  # 첫 번째 gsfc 내보내기 완료
            # propagation delay 추가, transmitting 변수 False, 다음 vnf 여부 확인
            self.queue_TSL.pop(0)
            propagation_delay = self.calculate_TSL_propagation_delay(gsfc.gserver)
            gsfc.sd_prop_delay_ms += propagation_delay
            gsfc.sd_is_transmitting = False
            completed_gsfc_ids.append(gsfc.id)

            # 다음 위성에게 전달하는 로직은 Main 루프에서 별도로 처리해야 함 (경로 업데이트 등)
            # print(f"[TRANS LOG] Sat {self.id} (TSL): GSFC {gsfc_id} COMPLETED Transmission. Handing off.")
        # else:
            # 전송 미완료
            # print(f"[TRANS LOG] Sat {self.id} (TSL): GSFC {gsfc_id} Transmitted {transmitted:.2f}. Remaining: {self.queue_TSL[0][1]:.2f}")

        return completed_gsfc_ids

    def calculate_TSL_propagation_delay(self, gserver):
        sat_lat_rad = d2r(self.lat)
        sat_lon_rad = d2r(self.lon)
        sat_alt_m = self.alt
        sat_R_obj = R_EARTH_RADIUS + sat_alt_m

        sat_x = sat_R_obj * math.cos(sat_lat_rad) * math.cos(sat_lon_rad)
        sat_y = sat_R_obj * math.cos(sat_lat_rad) * math.sin(sat_lon_rad)
        sat_z = sat_R_obj * math.sin(sat_lat_rad)

        gserver_lat_rad = d2r(gserver.lat)
        gserver_lon_rad = d2r(gserver.lon)
        gserver_alt_m = 0   # Gserver의 alt는 0 m
        gserver_R_obj = R_EARTH_RADIUS + gserver_alt_m

        gserver_x = gserver_R_obj * math.cos(gserver_lat_rad) * math.cos(gserver_lon_rad)
        gserver_y = gserver_R_obj * math.cos(gserver_lat_rad) * math.sin(gserver_lon_rad)
        gserver_z = gserver_R_obj * math.sin(gserver_lat_rad)

        # 3D 유클리드 거리 계산 (미터)
        distance_m = math.sqrt((sat_x - gserver_x) ** 2 + (sat_y - gserver_y) ** 2 + (sat_z - gserver_z) ** 2)
        # 전파 지연 시간 계산 (초)
        delay_s = distance_m / PARAM_C
        # 결과 반환 (밀리초, 소수점 둘째 자리까지 반올림)
        delay_ms = delay_s * 1000

        return round(delay_ms, 2)

    def assign_vnf(self, vnf_type):
        # if self.vnf_list is not None:
        #     print(f"[ERROR] Satellite {self.id} already has VNF {self.vnf_list[0]}")
        #     return
        self.vnf_list.append(vnf_type)
