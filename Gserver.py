from collections import deque
from Params import *
import math
import numpy as np

d2r=np.deg2rad

class Gserver:
    def __init__(self, id, lon, lat, vsg_id):
        self.id = id
        self.lon = lon
        self.lat = lat
        self.vsg_id = vsg_id
        self.vnf_list = []  # 클래스 속성 초기화

        self.GSERVER_CAPACITY = 300
        self.num_process_vnf = 4

        self.queue_TSL = []
        # processing queue
        self.process_queue = deque()

    def assign_vnf(self, vnf_type):
        self.vnf_list.append(vnf_type)

    # 현재 위성에서 vnf 처리하는지 확인
    def has_dst_tag(self, x):
        if x is None:
            return False
        if isinstance(x, (list, tuple)):
            return any(isinstance(e, str) and 'dst' in e.lower() for e in x)
        if isinstance(x, str):
            return 'dst' in x.lower()
        return False

    def calculate_TSL_propagation_delay(self, sat):
        sat_lat_rad = d2r(sat.lat)
        sat_lon_rad = d2r(sat.lon)
        sat_alt_m = sat.alt
        sat_R_obj = R_EARTH_RADIUS + sat_alt_m

        sat_x = sat_R_obj * math.cos(sat_lat_rad) * math.cos(sat_lon_rad)
        sat_y = sat_R_obj * math.cos(sat_lat_rad) * math.sin(sat_lon_rad)
        sat_z = sat_R_obj * math.sin(sat_lat_rad)


        gserver_lat_rad = d2r(self.lat)
        gserver_lon_rad = d2r(self.lon)
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

    def add_to_TSL_queue(self, gsfc, mode='sd'):
        self.queue_TSL.append([gsfc.id, SFC_SIZE])
        setattr(gsfc, f"{mode}_is_transmitting", True)

    def transmit_TSL_gsfcs(self, all_gsfc_list, all_sat_list, mode='sd'):
        """Transmission Delay [ms] = Packet Length [bit] / Data Rate [bit/ms]"""
        gsfc_map = {gsfc.id: gsfc for gsfc in all_gsfc_list}
        completed_gsfc_ids = []

        for q_idx in range(1, len(self.queue_TSL)):
            gsfc_id, _ = self.queue_TSL[q_idx]
            gsfc = gsfc_map[gsfc_id]
            if gsfc is not None:
                gsfc.accumulate_queue_delay(mode=mode)
            else:
                print(f"[ERROR] Gserver {self.id} (TSL) has unknown GSFC {gsfc_id}; skipping queue delay")
                pass

        # 첫 번째 gsfc부터 내보내기 - transmitting delay 추가
        gsfc_id, remaining_size = self.queue_TSL[0]
        if gsfc_id not in gsfc_map:
            print(f"[ERROR] Cannot find gsfc id in TSL queue")
            self.queue_TSL.pop(0)
            return []

        gsfc = gsfc_map[gsfc_id]
        # print(f"[TRANS LOG] Gserver {self.id} (TSL): Processing GSFC {gsfc_id}. Remaining: {remaining_size:.2f}. Capacity: {MAX_CAPACITY_PER_ISL:.2f}")

        # 이번에 처리할 수 있는 양
        transmitted = min(GSERVER_LINK_CAPACITY/TAU, remaining_size)
        # 전송 지연 업데이트 TODO 지금은 1ms 단위로 업데이트
        gsfc.accumulate_trans_delay(mode=mode)

        # 큐 업데이트 (GSFC size 업데이트)
        self.queue_TSL[0][1] -= transmitted

        if self.queue_TSL[0][1] <= 0:  # 첫 번째 gsfc 내보내기 완료
            # propagation delay 추가, transmitting 변수 False, 다음 vnf 여부 확인
            self.queue_TSL.pop(0)

            remain_sat_path = gsfc.get_remain_path(mode=mode)
            # TODO NEW! processed_satellite_path의 마지막에 'dst'가 있는지
            if len(remain_sat_path) < 1:
                satellite_path_attr = f"{mode}_satellite_path"
                satellite_path = getattr(gsfc, satellite_path_attr, [])

                was_dst = self.has_dst_tag(satellite_path[-1][1])
                if was_dst:
                    setattr(gsfc, f"{mode}_succeed", True)
                    # print(f"[PATH LOG] GSFC {gsfc.id} on Sat {self.id}: Destination reached. Success.")
                return

            next_sat_id = remain_sat_path[0][0]
            if next_sat_id >= NUM_SATELLITES:
                self.add_to_TSL_queue(gsfc, mode=mode)
                return
            next_sat = all_sat_list[next_sat_id]
            propagation_delay = self.calculate_TSL_propagation_delay(next_sat)

            gsfc.sd_prop_delay_ms += propagation_delay
            gsfc.sd_is_transmitting = False
            completed_gsfc_ids.append(gsfc.id)

            # 다음 위성에게 전달하는 로직은 Main 루프에서 별도로 처리해야 함 (경로 업데이트 등)
            # print(f"[TRANS LOG] Gserver {self.id} (TSL): GSFC {gsfc_id} COMPLETED Transmission. Handing off.")
        # else:
            # 전송 미완료
            # print(f"[TRANS LOG] Gserver {self.id} (TSL): GSFC {gsfc_id} Transmitted {transmitted:.2f}. Remaining: {self.queue_TSL[0][1]:.2f}")

        return completed_gsfc_ids

    def add_to_process_queue(self, gsfc_id, vnf_id, vnf_size):
        self.process_queue.append([gsfc_id, vnf_id, vnf_size])

        print(f"[QUEUE L/OG] Gserver {self.id} | Added GSFC {gsfc_id} (VNF {vnf_id}) to PROC Queue. Size: {vnf_size}")

    def pop_from_process_queue(self, idx):
        if 0 <= idx < len(self.process_queue):
            del self.process_queue[idx]
            return True
        else:
            print(f"[WARNING] Gserver {self.id}: Index {idx} out of range in Process queue.")
            return None

    def process_vnfs(self, all_gsfc_list, mode='sd', processing_rate=None):
        if processing_rate is None:
            processing_rate = GSERVER_PROCESSING_RATE

        gsfc_map = {gsfc.id: gsfc for gsfc in all_gsfc_list}

        current_tasks_info = []  # [(queue_index, gsfc_id, vnf_id)]
        completed_gsfc_id = []  # 제거할 인덱스를 저장

        for i, item in enumerate(self.process_queue):
            gsfc_id, vnf_id, _ = item

            if gsfc_id not in gsfc_map:
                continue
            gsfc = gsfc_map[gsfc_id]
            if gsfc.num_completed_vnf != vnf_id:
                print(f"[ERROR] synch xxxxx")
                continue

            if i < self.num_process_vnf:
                current_tasks_info.append((i, gsfc_id, vnf_id))
            else:
                gsfc.accumulate_queue_delay(mode=mode)

        num_tasks = len(current_tasks_info)
        if num_tasks == 0:
            print(f"[WARNING] NO left tasks in gserver {self.id}, process_queue: {self.process_queue}")
            return

        power_per_vnf = (processing_rate / TAU) / num_tasks  # [bits/ms]
        # print(f"process vnfs / gserver processing rate {processing_rate}, power per vnf {power_per_vnf}")
        for queue_index, gsfc_id, vnf_id in current_tasks_info:
            gsfc = gsfc_map[gsfc_id]

            is_completed = gsfc.processing_vnf(power_per_vnf, mode=mode)

            if is_completed:
                print(f"[P/ROCESSING COMPLETE] Gserver {self.id}: GSFC {gsfc.id} SFC 처리 완료. 다음 홉으로 포워딩.")
                completed_gsfc_id.append(queue_index)
                self.add_to_TSL_queue(gsfc, mode=mode)
            else:
                # 현재는 vnf 개수로 power를 등분하기 때문에 불필요
                self.process_queue[queue_index][2] -= power_per_vnf

        # 큐 인덱스가 틀어지지 않도록 역순으로 정렬 후 제거a
        completed_gsfc_id.sort(reverse=True)
        for idx in completed_gsfc_id:
            self.pop_from_process_queue(idx)
