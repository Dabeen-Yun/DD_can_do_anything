#PPT 삽입 용 시뮬레이션 네트워크
from astropy.coordinates.earth_orientation import eccentricity
from jinja2.nodes import Continue

from Satellite import *
from VSG import *
from GSFC import *
from Params import *
from Gserver import *
from Plot import *

import numpy as np
import random
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
from matplotlib.animation import ArtistAnimation
from itertools import product
from time import time
import os, json, csv
import math
from tqdm import tqdm
from statistics import mean

d2r=np.deg2rad


class Simulation:
    def __init__(self):
        np.random.seed(921)
        random.seed(921)

        self.G = None
        self.TG = None # include terrestrial nodes
        self.vsg_G = None # vsg graph for basic algorithm (VSG 논문 구현 용)
        self.results_dir = "results"

        # ==== basic =====
        self.GSFC_flow_rules = {}   #key: GSFC id, value: (function, vsg_idx)

        self.sat_list = [] # constellation 내 총 위성
        self.vsgs_list = []
        self.gsfc_list = []
        self.gserver_list = []
        self.vsg_path = {}

        self.hop_table = {}
        self.TG_hop_table = {} # include terrestrial nodes

        self.gsfc_id = 0

        self.congested_sat_ids = []
        self.data_drop_threshold = None

        self.csv_gsfc_set = []
        self.figure_list = []  # 각 time step의 Figure 객체를 저장할 리스트
        self.fig, self.ax = plt.subplots(figsize=(12, 8))


    # === 결과 CSV 생성 함수 ===
    def ensure_results_dir(self):
        os.makedirs(self.results_dir, exist_ok=True)

    def append_csv_row(self, filepath, fieldnames, row):
        file_exists = os.path.exists(filepath)
        with open(filepath, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                w.writeheader()
            w.writerow(row)

    def write_satellite_status_csv(self, t, filename="satellite_status.csv"):
        """
        매 시간 틱(t)마다 모든 위성의 큐 상태 (process, ISL, TSL)를 기록합니다.
        t=0일 때만 파일을 초기화하고, 이후에는 이어쓰기를 수행합니다.
        """
        self.ensure_results_dir()
        save_path = self.results_dir + filename

        # t=0일 때만 기존 파일 삭제 (초기화)
        if t == 0:
            if os.path.exists(save_path):
                try:
                    os.remove(save_path)
                    print(f"[INFO] Initializing satellite status file: {save_path}")
                except Exception as e:
                    print(f"[WARN] remove {save_path} failed: {e}")

        fieldnames = ["t", "sat_id", "process_queue_size",
                      "isl_queue_0", "isl_queue_1", "isl_queue_2", "isl_queue_3",
                      "isl_queue_count", "tsl_queue_count"]

        # 모든 위성을 순회하며 큐 상태를 기록
        for sat in self.sat_list:
            # process_queue: VNF 처리를 기다리는 GSFC/VNF 항목 수
            proc_size = len(sat.process_queue)

            # queue_ISL: ISL 전송을 기다리는 항목 수 (queue_ISL은 큐들의 리스트이므로 전체 합계)
            isl_0_count = sum(packet[1] for packet in sat.queue_ISL[0])
            isl_1_count = sum(packet[1] for packet in sat.queue_ISL[1])
            isl_2_count = sum(packet[1] for packet in sat.queue_ISL[2])
            isl_3_count = sum(packet[1] for packet in sat.queue_ISL[3])

            # 전체 ISL 큐에 쌓인 총 비트(bit) 크기 계산
            isl_count = sum(sum(packet[1] for packet in q) for q in sat.queue_ISL)

            # queue_TSL: TSL 전송을 기다리는 항목 수
            tsl_count = sum(packet[1] for packet in sat.queue_TSL)

            row = {
                "t": t,
                "sat_id": sat.id,
                "process_queue_size": proc_size,
                "isl_queue_0": isl_0_count,
                "isl_queue_1": isl_1_count,
                "isl_queue_2": isl_2_count,
                "isl_queue_3": isl_3_count,
                "isl_queue_count": isl_count,
                "tsl_queue_count": tsl_count,
            }

            self.append_csv_row(
                filepath=save_path,
                fieldnames=fieldnames,
                row=row
            )
        # print(f"[LOG] Time {t}: Satellite status saved.") # 너무 많은 로그 방지를 위해 주석 처리

    def write_results_csv(self, t, filename, mode='dd', IS_PROPOSED=False):
        self.ensure_results_dir()
        save_path = self.results_dir + filename

        if t == 0:
            if os.path.exists(save_path):
                try: os.remove(save_path)
                except Exception as e: print(f"[WARN] remove {save_path} failed: {e}")

        fieldnames = ["t", "gsfc_id", "is_succeed", "is_dropped", "src_vsg", "dst_vsg", "vnf_sequence", "vsg_path", "satellite_path", "processed_satellite_path", "actual_dst_vsg", "successfully_arrived",
                      "hop_count", "propagation_delay_ms", "processing_delay_ms", "queueing_delay_ms","transmission_delay_ms", "e2e_delay_ms"]

        # 모드별 속성 이름 설정
        succeed_attr = f"{mode}_succeed"
        dropped_attr = f"{mode}_dropped"
        satellite_path = f"{mode}_satellite_path"
        processed_satellite_attr = f"{mode}_processed_satellite_path"
        hop_attr = f"{mode}_hop_count"
        prop_attr = f"{mode}_prop_delay_ms"
        proc_attr = f"{mode}_proc_delay_ms"
        queue_attr = f"{mode}_queue_delay_ms"
        trans_attr = f"{mode}_trans_delay_ms"
        # e2e_attr = f"{mode}_e2e_delay_ms"

        for gsfc in self.gsfc_list:
            if gsfc in self.csv_gsfc_set:
                continue
            is_succeed = getattr(gsfc, succeed_attr)
            is_dropped = getattr(gsfc, dropped_attr)

            processed_satellite_path = getattr(gsfc, processed_satellite_attr, "[]")
            dst_vsg = -1

            if is_succeed:
                dst_sat_id = processed_satellite_path[-1][0]
                dst_sat = self.sat_list[dst_sat_id]
                dst_vsg = dst_sat.current_vsg_id

            e2e_delay = (getattr(gsfc, prop_attr, float('inf')) + getattr(gsfc, proc_attr, float('inf'))
                         + getattr(gsfc, queue_attr, float('inf')) + getattr(gsfc, trans_attr, float('inf')))

            row = {
                "t": t,
                "gsfc_id": gsfc.id,
                "is_succeed": True if is_succeed else False,
                "is_dropped": True if is_dropped else False,
                "src_vsg": gsfc.src_vsg_id,
                "dst_vsg": gsfc.dst_vsg_id,
                "vnf_sequence": gsfc.vnf_sequence,
                "vsg_path": self.vsg_path[gsfc.id],
                "satellite_path": getattr(gsfc, satellite_path, "[]"),
                "processed_satellite_path": getattr(gsfc, processed_satellite_attr, "[]"),
                "actual_dst_vsg": dst_vsg,
                "successfully_arrived": True if gsfc.dst_vsg_id == dst_vsg else False,
                "hop_count": getattr(gsfc, hop_attr, float('inf')),
                "propagation_delay_ms": getattr(gsfc, prop_attr, float('inf')),
                "processing_delay_ms": getattr(gsfc, proc_attr, float('inf')),
                "queueing_delay_ms": getattr(gsfc, queue_attr, float('inf')),
                "transmission_delay_ms": getattr(gsfc, trans_attr, float('inf')),
                "e2e_delay_ms": e2e_delay,
            }

            self.append_csv_row(
                filepath=save_path,
                fieldnames=fieldnames,
                row=row
            )

            if is_succeed:
                self.csv_gsfc_set.append(gsfc)

    def write_success_results_csv(self, filename, mode='dd', IS_PROPOSED=False):
        self.ensure_results_dir()
        save_path = self.results_dir + filename

        if os.path.exists(save_path):
            try: os.remove(save_path)
            except Exception as e: print(f"[WARN] remove {save_path} failed: {e}")

        fieldnames = ["gsfc_id", "is_succeed", "is_dropped", "src_vsg", "dst_vsg", "vnf_sequence", "vsg_path", "satellite_path", "processed_satellite_path", "actual_dst_vsg", "successfully_arrived",
                      "hop_count", "propagation_delay_ms", "processing_delay_ms", "queueing_delay_ms","transmission_delay_ms", "e2e_delay_ms"]

        # 모드별 속성 이름 설정
        succeed_attr = f"{mode}_succeed"
        dropped_attr = f"{mode}_dropped"
        satellite_attr = f"{mode}_satellite_path"
        processed_satellite_attr = f"{mode}_processed_satellite_path"
        hop_attr = f"{mode}_hop_count"
        prop_attr = f"{mode}_prop_delay_ms"
        proc_attr = f"{mode}_proc_delay_ms"
        queue_attr = f"{mode}_queue_delay_ms"
        trans_attr = f"{mode}_trans_delay_ms"
        # e2e_attr = f"{mode}_e2e_delay_ms"

        drop_count = 0

        for gsfc in self.gsfc_list:
            is_succeed = getattr(gsfc, succeed_attr)
            processed_satellite_path = getattr(gsfc, processed_satellite_attr, "[]")
            dst_sat_id = processed_satellite_path[-1][0]
            dst_sat = self.sat_list[dst_sat_id]
            dst_vsg = dst_sat.current_vsg_id

            if is_succeed:
                e2e_delay = (getattr(gsfc, prop_attr, float('inf')) + getattr(gsfc, proc_attr, float('inf'))
                             + getattr(gsfc, queue_attr, float('inf')) + getattr(gsfc, trans_attr, float('inf')))
                is_dropped = getattr(gsfc, dropped_attr)

                row = {
                    "gsfc_id": gsfc.id,
                    "is_succeed": True if is_succeed else False,
                    "is_dropped": True if is_dropped else False,
                    "src_vsg": gsfc.src_vsg_id,
                    "dst_vsg": gsfc.dst_vsg_id,
                    "vnf_sequence": gsfc.vnf_sequence,
                    "vsg_path": self.vsg_path[gsfc.id],
                    "satellite_path": getattr(gsfc, satellite_attr, "[]"),
                    "processed_satellite_path": getattr(gsfc, processed_satellite_attr, "[]"),
                    "actual_dst_vsg": dst_vsg,
                    "successfully_arrived": True if gsfc.dst_vsg_id == dst_vsg else False,
                    "hop_count": getattr(gsfc, hop_attr, float('inf')),
                    "propagation_delay_ms": getattr(gsfc, prop_attr, float('inf')),
                    "processing_delay_ms": getattr(gsfc, proc_attr, float('inf')),
                    "queueing_delay_ms": getattr(gsfc, queue_attr, float('inf')),
                    "transmission_delay_ms": getattr(gsfc, trans_attr, float('inf')),
                    "e2e_delay_ms": e2e_delay,
                }

                self.append_csv_row(
                    filepath=save_path,
                    fieldnames=fieldnames,
                    row=row
                )

    # === walker-star constellation ===
    def set_constellation(self, n_sat, n_orb):
        phasing_inter_plane = 180 / n_orb # walker-star

        for sat_id in range(n_sat):
            sat = Satellite(sat_id, NUM_ORBITS, NUM_SATELLITES_PER_ORBIT, ORBIT_ALTITUDE, phasing_inter_plane, POLAR_LATITUDE, self.sat_list)
            self.sat_list.append(sat)

        for sat_id in range(n_sat):
            sat = self.sat_list[sat_id]
            sat.set_adjacent_node()
            sat.get_propagation_delay()

        # print("constellation (ID, lat, lon):")
        # for sat in self.sat_list:
        #     print(f"  [{sat.id:2d}]  lat: {sat.lat:6.2f}°,  lon: {sat.lon:6.2f}°")

    def compute_all_pair_hop_counts(self, congested_sat_ids=None, mode='dd'):
        if congested_sat_ids is None:
            congested_sat_ids = set()
        else:
            congested_sat_ids = set(congested_sat_ids)

        self.G = nx.Graph()
        # self.TG = nx.Graph()

        # congestion 아닌 위성들만으로 그래프 구성
        for sat in self.sat_list:
            if sat.id in congested_sat_ids:
                continue
            for neighbor_id in sat.adj_sat_index_list:
                if neighbor_id != -1 and neighbor_id not in congested_sat_ids:

                    neighbor_sat = self.sat_list[neighbor_id]
                    prop_delay_ms = sat.calculate_delay_to_sat(neighbor_sat)

                    self.G.add_edge(sat.id, neighbor_id, weight=prop_delay_ms, link_type='isl')
                    # self.TG.add_edge(sat.id, neighbor_id, weight=prop_delay_ms, link_type='tsl')

                    # self.G.add_edge(sat.id, neighbor_id, weight=1, link_type='isl')
                    # self.TG.add_edge(sat.id, neighbor_id, weight=1, link_type='tsl')

            print("\n--- ISL 엣지 및 Propagation Delay 확인 (샘플) ---")
            sample_sat_id = sat.id
            if sample_sat_id in self.G:
                print(f"위성 {sample_sat_id}의 인접 엣지:")
                for neighbor, data in self.G[sample_sat_id].items():
                    # 'weight'는 전파 지연 (ms)
                    print(f"  -> 위성 {neighbor}: weight (Delay) = {data.get('weight', 'N/A'):.2f} ms")
            else:
                print(f"위성 {sample_sat_id}가 그래프에 존재하지 않습니다. (혼잡 상태 등으로 제외되었을 수 있음)")
            print("------------------------------------------")

            # 모든 쌍의 shortest path length 계산
            path_lengths = dict(nx.all_pairs_shortest_path_length(self.G))

            # hop_table 저장
            for src in range(len(self.sat_list)):
                for dst in range(len(self.sat_list)):
                    if src in congested_sat_ids or dst in congested_sat_ids:
                        self.hop_table[(src, dst)] = float('inf')
                    else:
                        self.hop_table[(src, dst)] = path_lengths.get(src, {}).get(dst, float('inf'))

    def get_full_path(self, src_id, dst_id, graph=None):
        if graph is None:
            graph = self.G

        try:
            return nx.shortest_path(graph, src_id, dst_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def set_congested_satellites(self, drop_rate_threshold=0.5):
        self.congested_sat_ids = {
            sat.id for sat in self.sat_list if sat.drop_rate >= drop_rate_threshold
        }
        # print(f"[INFO] Set {len(self.congested_sat_ids)} congested satellites (threshold={drop_rate_threshold}): {sorted(self.congested_sat_ids)}")

    def get_distance_between_VSGs(self, vid1, vid2):
        vsg1 = next((x for x in self.vsgs_list if x.id == vid1), None)
        vsg2 = next((x for x in self.vsgs_list if x.id == vid2), None)

        vsg1_lon = vsg1.center_coords[0]
        vsg1_lat = vsg1.center_coords[1]
        vsg2_lon = vsg2.center_coords[0]
        vsg2_lat = vsg2.center_coords[1]

        vsg1_lon_rad = d2r(vsg1_lon)
        vsg1_lat_rad = d2r(vsg1_lat)
        vsg1_alt_m = ORBIT_ALTITUDE
        vsg1_R_obj = R_EARTH_RADIUS + vsg1_alt_m

        vsg2_lon_rad = d2r(vsg2_lon)
        vsg2_lat_rad = d2r(vsg2_lat)
        vsg2_alt_m = ORBIT_ALTITUDE
        vsg2_R_obj = R_EARTH_RADIUS + vsg2_alt_m

        vsg1_x = vsg1_R_obj * math.cos(vsg1_lat_rad) * math.cos(vsg1_lon_rad)
        vsg1_y = vsg1_R_obj * math.cos(vsg1_lat_rad) * math.sin(vsg1_lon_rad)
        vsg1_z = vsg1_R_obj * math.sin(vsg1_lat_rad)

        vsg2_x = vsg2_R_obj * math.cos(vsg2_lat_rad) * math.cos(vsg2_lon_rad)
        vsg2_y = vsg2_R_obj * math.cos(vsg2_lat_rad) * math.sin(vsg2_lon_rad)
        vsg2_z = vsg2_R_obj * math.sin(vsg2_lat_rad)

        # 3D 유클리드 거리 계산 (미터)
        distance_m = math.sqrt((vsg1_x - vsg2_x) ** 2 + (vsg1_y - vsg2_y) ** 2 + (vsg1_z - vsg2_z) ** 2)

        return distance_m


    def initial_vsg_regions(self):
        self.vsgs_list = []
        self.gserver_list = []
        self.vsg_G = nx.Graph()

        vid = 0
        gid = 0

        lat_bins = np.arange(LAT_RANGE[0], LAT_RANGE[1]+1, LAT_STEP)
        lon_bins = np.arange(LON_RANGE[0], LON_RANGE[1]+1, LON_STEP)

        num_row = math.ceil((LAT_RANGE[1] - LAT_RANGE[0]) / LAT_STEP)
        num_col = math.ceil((LON_RANGE[1] - LON_RANGE[0]) / LON_STEP)

        for lat_min in lat_bins:
            lat_max = lat_min + LAT_STEP
            for lon_min in lon_bins:
                lon_max = lon_min + LON_STEP

                # 현재 그리드 셀 안에 속하는 위성 추출
                cell_sats = [
                    sat for sat in self.sat_list
                    if lat_min <= sat.lat < lat_max and lon_min <= sat.lon < lon_max
                ]

                if not cell_sats:
                    continue

                center_lat = (lat_min + LAT_STEP) / 2
                center_lon = (lon_min + LON_STEP) / 2

                ground_server = Gserver(gid, center_lon, center_lat, vid)

                vsg = VSG(vid, (center_lon, center_lat), lon_min, lat_min, cell_sats, ground_server)
                for sat in cell_sats:
                    sat.current_vsg_id = vid
                    # TODO. 그럼 다 똑같은 시간 아닌가?
                    sat.vsg_enter_time = time()

                self.vsgs_list.append(vsg)
                self.gserver_list.append(ground_server)

                vid += 1
                gid += 1

        DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        existing = {v.id for v in self.vsgs_list}

        for vid in range(num_row * num_col):
            if vid not in existing:
                continue

            row, col = divmod(vid, num_col)
            for dr, dc in DIRS:
                nrow = (row + dr) % num_row
                ncol = (col + dc) % num_col
                nvid = nrow * num_col + ncol

                if nvid not in existing:
                    continue  # 이웃이 비어 있으면 스킵

                if self.vsg_G.has_edge(vid, nvid):
                    continue

                vsg_distance = self.get_distance_between_VSGs(vid, nvid)
                self.vsg_G.add_edge(vid, nvid, weight=vsg_distance)

    def initial_vnfs_to_vsgs(self, mode='basic', alpha=0.5):
        # NUM_SATELLITES는 self.sat_list의 길이 또는 전역 상수를 사용합니다.
        vnf_set_sat_ids = sorted(random.sample(range(0,NUM_SATELLITES), int(NUM_SATELLITES*0.8)))

        for sat in self.sat_list:
            if sat.id in vnf_set_sat_ids:
                # 3개 이상 탑재 (최대 개수는 넘기지 않도록)
                vnf_per_sat = random.randint(3,NUM_VNFS_PER_SAT) # 흠...
                vnfs = sorted(random.sample(range(*VNF_TYPES_PER_VSG), vnf_per_sat))
                assigned_vnfs = [str(v) for v in vnfs]
                sat.vnf_list = assigned_vnfs

        for vsg in self.vsgs_list:
            sampled_vnf_types = random.sample(range(VNF_TYPES_PER_VSG[0], VNF_TYPES_PER_VSG[1] + 1), k=NUM_VNFS_PER_VSG)
            assigned_vnfs = [str(v) for v in sampled_vnf_types]
            vsg.assigned_vnfs = assigned_vnfs

            for vnf_type in assigned_vnfs:
                # 2-1. 현재 VSG 내에 해당 VNF를 호스팅하는 위성이 있는지 확인
                is_covered = any(vnf_type in sat.vnf_list for sat in vsg.satellites)
                if is_covered:
                    continue  # 이미 커버됨

                target_sat = random.choice(vsg.satellites)

                if len(target_sat.vnf_list) < NUM_VNFS_PER_SAT:
                    # 공간이 남은 경우: 추가 (Addition)
                    target_sat.vnf_list.append(vnf_type)
                else:
                    # 공간이 없는 경우: 교체 (Replacement)

                    # ✨ [수정된 핵심 로직]: VSG에 할당된 VNF가 아닌 것을 제거 대상으로 선택
                    non_assigned_vnfs = [v for v in target_sat.vnf_list if v not in assigned_vnfs]
                    victim_vnf = random.choice(non_assigned_vnfs)
                    target_sat.vnf_list.remove(victim_vnf)
                    target_sat.vnf_list.append(vnf_type)
                # # alpha가 클수록 진입 시간이 느린 위성을 우선, alpha가 작을 수록 드롭율이 낮은(신뢰성 높은) 위성을 우선
                # if mode == 'proposed':
                #     best_sat = None
                #     best_efficiency = -1
                #     max_time = max([sat.vsg_enter_time for sat in unassigned_sats], default=1e-6)
                #     max_drop = max([sat.drop_rate for sat in unassigned_sats], default=1e-6)
                #     for sat in unassigned_sats:
                #         norm_time = sat.vsg_enter_time / max_time
                #         norm_drop = sat.drop_rate / max_drop if max_drop > 0 else 0
                #         efficiency = alpha * norm_time + (1 - alpha) * (1 - norm_drop)
                #         if efficiency > best_efficiency:
                #             best_efficiency = efficiency
                #             best_sat = sat
                #     if best_sat:
                #         best_sat.assign_vnf(vnf_type)
                # else:
                #     target_sat = random.choice(unassigned_sats)
                #     target_sat.assign_vnf(vnf_type)

    def check_vsg_regions(self):
        vid = 0
        is_changed = False

        lat_bins = np.arange(LAT_RANGE[0], LAT_RANGE[1], LAT_STEP)
        lon_bins = np.arange(LON_RANGE[0], LON_RANGE[1], LON_STEP)
        print(f"DEBUG: lat_bins size: {len(lat_bins)}, lon_bins size: {len(lon_bins)}")

        for lat_min in lat_bins:
            lat_max = lat_min + LAT_STEP
            for lon_min in lon_bins:
                lon_max = lon_min + LON_STEP

                # 현재 그리드 셀 안에 속하는 위성 추출
                cell_sats = [
                    sat for sat in self.sat_list
                    if lat_min <= sat.lat < lat_max and lon_min <= sat.lon < lon_max
                ]

                if not cell_sats:
                    print("[WARNING] NO SATELLITES")
                    continue

                for sat in cell_sats:
                    sat.current_vsg_id = vid

                if not is_changed:
                    is_changed = self.vsgs_list[vid].satellites != cell_sats
                self.vsgs_list[vid].satellites = cell_sats
                vid += 1

    # Simulation.py (Simulation 클래스 내부)

    def get_satellite_load(self, sat):
        """
        위성(sat)의 process_queue에 쌓여있는 VNF 종류별 패킷 사이즈(로드)를 계산합니다.

        :param sat: Satellite 객체
        :return: VNF 종류별 로드 딕셔너리(dict)
        """
        # VNF 종류별 로드를 저장할 딕셔너리: {'vnf1': load, 'vnf2': load, ...}
        vnf_load = {}

        # 큐의 각 항목은 [gsfc_id, vnf_idx, vnf_size] 형식
        for item in sat.process_queue:
            if len(item) < 3: continue

            gsfc_id = item[0]
            vnf_idx = item[1]
            vnf_size = item[2]

            try:
                # 1. GSFC 객체와 VNF Sequence를 사용하여 VNF 종류 확인
                gsfc = self.gsfc_list[gsfc_id]
                # vnf_sequence가 SFC의 VNF 종류를 저장하는 리스트라고 가정
                vnf_kind = gsfc.vnf_sequence[vnf_idx]

            except IndexError:
                # gsfc_id나 vnf_idx가 유효하지 않은 경우 (데이터 불일치)
                continue

            if isinstance(vnf_size, (int, float)):
                if vnf_kind not in vnf_load:
                    vnf_load[vnf_kind] = 0

                vnf_load[vnf_kind] += vnf_size

        # VNF 종류별 전체 로드 딕셔너리 반환
        return vnf_load

    def reassign_vnfs_to_satellite(self, vsg):
        # TODO 2. 해당 VSG 내 위성들에 대해서, 가장 적은 load를 가지고 있는 VNF 종류 및 해당 위성 id 찾기, 더 이상 처리 못하게 된 sfc는 다른 위성으로 포워딩

        for vnf in vsg.assigned_vnfs:
            has_vnf = any(vnf in sat.vnf_list for sat in vsg.satellites)
            if has_vnf:
                continue

            # ----------------------------------------------------------------------
            # 🎯 [우선 순위 1]: VNF 최대 개수 미만인 위성에게 할당
            # ----------------------------------------------------------------------
            capacity_candidate = []
            for sat in vsg.satellites:
                # 1. VNF 최대 개수(3) 미만인 위성
                if len(sat.vnf_list) < NUM_VNFS_PER_SAT:
                    capacity_candidate.append(sat)

            if capacity_candidate:
                # VNF 슬롯이 비어있는 위성에게 할당하고 다음 VNF로 넘어갑니다.
                selected_sat = random.choice(capacity_candidate)
                selected_sat.vnf_list.append(vnf)
                print(f"[REASSIGN] VNF **{vnf}** assigned to Sat **{selected_sat.id}** (Simple Capacity Check: {len(selected_sat.vnf_list)}/{NUM_VNFS_PER_SAT}).")
                continue  # 다음 vnf로 넘어감

            # ----------------------------------------------------------------------
            # 🎯 [우선 순위 2]: VNF 슬롯이 가득 찼거나 없으므로, 로드 밸런싱을 통해 할당
            # ----------------------------------------------------------------------
            # VSG 내 모든 위성 중 가장 로드가 적은 위성을 찾습니다.
            min_overall_load = float('inf')
            best_sat = None
            best_vnf_kind_in_sat = None

            for sat in vsg.satellites:
                # 2-1. 위성(sat)의 VNF 종류별 로드 딕셔너리를 가져옵니다.
                vnf_loads_dict = self.get_satellite_load(sat)
                if vnf_loads_dict is None:
                    print(f"vsg_id {vsg.id}, sat_id {sat.id} sat_process_queue {sat.process_queue}")
                    input()

                # 2-2. 해당 위성 내에서 VSG에 할당되지 않은 VNF 중 최소 로드를 찾기
                min_vnf_load_in_sat = float('inf')
                min_vnf_kind_in_sat = None

                is_processed = False
                for vnf_kind, load in vnf_loads_dict.items():

                    # VSG에 할당된 VNF는 무시하고 (nothing), 할당되지 않은 VNF만 검사
                    if vnf_kind not in vsg.assigned_vnfs:
                        is_processed = True
                        if load < min_vnf_load_in_sat:
                            min_vnf_load_in_sat = load
                            min_vnf_kind_in_sat = vnf_kind

                # # 2-3. (VSG에 할당되지 않은 VNF가 전혀 없는 경우도 처리)
                # if not is_processed:
                #     # 모든 VNF가 할당된 VNF이거나 큐가 비어있는 경우
                #     min_vnf_load_in_sat = 0.0
                #     min_vnf_kind_in_sat = "N/A (Empty/Assigned Only)"
                # elif min_vnf_load_in_sat == float('inf'):
                #     # 큐는 있는데, 모든 VNF가 assigned_vnfs에 포함된 경우
                #     min_vnf_load_in_sat = 0.0  # 필터링된 로드는 0
                #     min_vnf_kind_in_sat = "N/A (Filtered Out)"

                # 2-4. 전 VSG를 통틀어 가장 적은 큐 로드를 가진 쌍을 갱신
                # (min_vnf_load_in_sat는 VSG에 할당되지 않은 VNF 중 최소 로드임)
                if min_vnf_load_in_sat < min_overall_load:
                    min_overall_load = min_vnf_load_in_sat
                    best_sat = sat
                    best_vnf_kind_in_sat = min_vnf_kind_in_sat

            # 3. 할당 가능한 위성이 있는지 확인
            if best_sat is None:
                # Capacity Check도 실패했고, Load Balancing으로도 후보를 찾지 못한 경우
                print(f"[ERROR] Cannot assign VNF {vnf} in VSG {vsg.id}. All satellites are full or unavailable.")
                continue

                # 4. 가장 로드가 적은 위성(best_sat)에 VNF 할당 (재할당 로직 실행)
            selected_sat = best_sat
            selected_sat.vnf_list.remove(best_vnf_kind_in_sat)
            selected_sat.vnf_list.append(vnf)

            # 5. 재할당 정보 출력
            print(f"[REASSIGN] VNF **{vnf}** assigned to Sat **{selected_sat.id}** in VSG **{vsg.id}**.")
            print(
                f"           Selection Criterion: Found minimum queue process (Load: **{min_overall_load:.2f}** bytes) across the VSG.")
            print(
                f"           The least loaded VNF Queue was **{best_vnf_kind_in_sat}** on Sat **{selected_sat.id}** (Filtering out VSG {vsg.id}'s assigned VNFs).")

    def supposed_reassign_vnfs_to_satellite(self, vsgs_to_reassign, alpha=0.5):
        for vsg in vsgs_to_reassign:
            for vnf in vsg.assigned_vnfs:
                has_vnf = any(sat.vnf_list == vnf for sat in vsg.satellites)
                if has_vnf:
                    continue

                # 필터링: 이미 다른 VNF 갖고 있는 위성은 제외
                candidate_sats = [
                    sat for sat in vsg.satellites
                    if sat.vnf_list not in vsg.assigned_vnfs
                ]

                max_time = max([sat.vsg_enter_time for sat in candidate_sats], default=1e-6)
                max_drop_rate = max([sat.drop_rate for sat in candidate_sats], default=1e-6)

                best_sat = None
                best_efficiency = -1

                for sat in candidate_sats:
                    norm_time = sat.vsg_enter_time / max_time
                    norm_drop_rate = sat.drop_rate / max_drop_rate
                    efficiency = alpha * norm_time + (1-alpha) * (1 - norm_drop_rate)

                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_sat = sat

                if best_sat:
                    best_sat.vnf_list = vnf

    def select_another_satellite_in_the_same_vsg(self, vsg, vnf_type, alpha=0.5):
        best_sat = None
        best_efficiency = -1
        max_time = max([sat.vsg_enter_time for sat in vsg.satellites], default=1e-6)
        max_drop = max([sat.drop_rate for sat in vsg.satellites], default=1e-6)

        for sat in vsg.satellites:
            if sat.vnf_list == None:
                continue
            if sat.vnf_list in vsg.assigned_vnfs:
                continue
            if sat.id in self.congested_sat_ids:
                continue

            norm_time = sat.vsg_enter_time / max_time
            norm_drop = sat.drop_rate / max_drop if max_drop > 0 else 0
            efficiency = alpha * norm_time + (1-alpha)*(1-norm_drop)
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_sat = sat

        if best_sat:
            best_sat.assign_vnf(vnf_type)

        return best_sat

    def generate_gsfc(self, num_gsfcs=None, vnf_size_mode="VSG"):
        if num_gsfcs is None:
            num_gsfcs = NUM_GSFC

        for i in range(num_gsfcs):
            # 1. SFC 내 vnf 시퀀스 생성 (min 1 ~ max 2)
            # sfc_length = np.random.randint(VNF_PER_SFC[0], VNF_PER_SFC[1]+1)
            # sfc_length = 5

            # # 방법 1. VSG에 할당되어 있는 VNF 중에서 GSFC 설정
            # all_vnfs = sorted({vnf for sat in self.sat_list for vnf in sat.vnf_list if sat.vnf_list})
            # if not all_vnfs:
            #     print(f"[WARNING] No VNFs found in current network.")
            #     vnf_sequence = []
            # else:
            #     vnf_sequence = random.choices(all_vnfs, k=sfc_length)

            sfc_type_idx = random.randint(0, 2)
            vnf_sequence = SFC_TYPE_LIST.get(sfc_type_idx)

            # TODO 0. SFC 종류 별 경로 설정 (일단 수정 완. 디버깅 필요)
            if sfc_type_idx == 1: # uRLLC:
                # SRC VSG와 DST VSG가 동일하도록
                src_vsg = random.choice(self.vsgs_list)
                src_vsg_id = src_vsg.id
                dst_vsg_id = src_vsg_id
            else:
                src_vsg = random.choice(self.vsgs_list)
                src_vsg_id = src_vsg.id
                src_lon = src_vsg.center_coords[0]  # (lon, lat)이므로 [0]이 lon

                # 2. dst_vsg 후보: src_vsg_id를 제외하고, 경도가 src_vsg의 경도보다 큰 VSG들
                # 이렇게 하면 'src가 왼쪽, dst가 오른쪽' 조건이 만족됩니다.
                dst_candidates = [
                    v for v in self.vsgs_list
                    if v.id != src_vsg_id and v.center_coords[0] > src_lon
                ]

                if dst_candidates:
                    # 조건(src_lon < dst_lon)을 만족하는 후보가 있으면 그 중에서 무작위 선택
                    dst_vsg_id = random.choice(dst_candidates).id
                else:
                    # 조건(src_lon < dst_lon)을 만족하는 후보가 없거나, src_vsg 밖에 없는 경우
                    # (예: src가 가장 동쪽에 있는 VSG인 경우)
                    # 3. 차선책: src_vsg_id를 제외한 나머지 모든 VSG 중에서 무작위 선택
                    other_vsgs = [v for v in self.vsgs_list if v.id != src_vsg_id]
                    if other_vsgs:
                        dst_vsg_id = random.choice(other_vsgs).id
                    else:
                        # VSG가 하나뿐인 경우. 생성을 건너뜁니다.
                        print(f"[WARNING] Only one VSG found (ID: {src_vsg_id}). Skipping GSFC creation.")
                        continue  # 다음 루프로 이동

            gsfc = GSFC(self.gsfc_id, src_vsg_id, dst_vsg_id, vnf_sequence, sfc_type_idx, vnf_size_mode)
            # print("gsfc list : ", self.gsfc_id, src_vsg_id, vnf_sequence, dst_vsg_id)
            self.gsfc_list.append(gsfc)
            self.gsfc_id += 1

    # ========= VSG 논문 구현 ========= #
    # 생성된 gsfc를 기반으로 vsg 구성 -> satellite path 구성
    def set_gsfc_flow_rule(self, gsfc):
        ## gsfc 내 vnf를 기반으로 필수 vsg 설정
        # 1. VNF를 수행할 VSG 리스트
        self.GSFC_flow_rules[gsfc.id] = {}
        self.GSFC_flow_rules[gsfc.id][0] = ("src", gsfc.src_vsg_id)

        prev_vsg_id = gsfc.src_vsg_id
        index = 1

        for idx, vnf in enumerate(gsfc.vnf_sequence):
            candidate_vsgs = []
            for vsg in self.vsgs_list:
                if vnf in vsg.assigned_vnfs:
                    available_sats = [
                        sat for sat in vsg.satellites
                        if vnf in sat.vnf_list
                    ]
                    if available_sats:
                        candidate_vsgs.append(vsg)

            if not candidate_vsgs: # 해당 VNF를 수행할 VSG가 없음
                print(f"[ERROR] 1-1 No non-congested VSG found for VNF {vnf}")
                self.GSFC_flow_rules[gsfc.id] = {}
                gsfc.basic_dropped = True
                return []

            try:
                hop_sorted = sorted(
                    candidate_vsgs,
                    key=lambda vsg: nx.shortest_path_length(self.vsg_G, source=prev_vsg_id, target=vsg.id)
                )
            except nx.NetworkXNoPath:
                print(f"[ERROR] 1-2 No path from src VSG to any candidate VSG for VNF {vnf}")
                self.GSFC_flow_rules[gsfc.id] = {}
                gsfc.basic_dropped = True
                return []

            for vsg in hop_sorted:
                self.GSFC_flow_rules[gsfc.id][index] = (vnf, vsg.id)
                prev_vsg_id = vsg.id
                index += 1
                break

            if not any(entry[0] == vnf for entry in self.GSFC_flow_rules[gsfc.id].values()):
                print(f"[ERROR] 1-3 No valid VSG selected for VNF {vnf}")
                self.GSFC_flow_rules[gsfc.id] = {}
                gsfc.basic_dropped = True
                return []

        self.GSFC_flow_rules[gsfc.id][index] = ("dst", gsfc.dst_vsg_id)

    def set_vsg_path(self, gsfc):
        self.vsg_path[gsfc.id] = []
        essential_vsgs = self.GSFC_flow_rules[gsfc.id]

        for i, (key, value) in enumerate(essential_vsgs.items()):
            if i < len(essential_vsgs) - 1:
                src_vnf = value[0]  # 첫 번째 요소
                src_vsg = value[1]  # 두 번째 요소보호되는 속성
                dst_vnf = essential_vsgs[key + 1][0]  # 다음 항목의 첫 번째 요소
                dst_vsg = essential_vsgs[key + 1][1]  # 다음 항목의 두 번째 요소

                if i == 0:
                    if src_vsg == dst_vsg:
                        self.vsg_path[gsfc.id].append((src_vsg, ("src", f"vnf{dst_vnf}")))
                        continue
                elif i == (len(essential_vsgs) - 2):
                    if src_vsg == dst_vsg:
                        self.vsg_path[gsfc.id][-1] = (src_vsg, ("dst", f"vnf{src_vnf}"))
                        continue

                # src, dst vsg가 첫 번째 vnf, 두 번째 vnf의 vsg와 같다면 합치기
                try:
                    sub_path = nx.shortest_path(self.vsg_G, source=src_vsg, target=dst_vsg)

                    if i == 0:
                        self.vsg_path[gsfc.id].append((src_vsg, (src_vnf)))

                    for vid in sub_path[1:-1]:
                        self.vsg_path[gsfc.id].append((vid, (None)))

                    if i == (len(essential_vsgs) - 2):
                        self.vsg_path[gsfc.id].append((sub_path[-1], (dst_vnf)))
                    else:
                        self.vsg_path[gsfc.id].append((sub_path[-1], (f"vnf{dst_vnf}")))

                except nx.NetworkXNoPath:
                    print(f"[ERROR] 2-1 No path between VSG {src_vsg} and {dst_vsg}")
                    gsfc.basic_dropped = True
                    return []

    def get_vnf_id_for_list(self, vnf_tag):
        """
        경로 태그(예: 'vnf1', ('src', 'vnf1'))에서 VNF 번호(예: '1')를 추출합니다.

        :param vnf_tag: VNF 정보가 담긴 문자열 또는 튜플.
        :return: VNF 번호('1', '2' 등)를 담은 문자열, 또는 False (VNF가 없는 경우).
        """
        # 1. vnf_tag가 튜플일 경우 (예: ('src', 'vnf1'))
        if isinstance(vnf_tag, tuple):
            for item in vnf_tag:
                if isinstance(item, str) and item.startswith('vnf'):
                    # 'vnf1'에서 '1'을 추출하여 반환
                    return item[3:]
            return False

        # 2. vnf_tag가 단일 문자열일 경우 (예: 'vnf1' 또는 'src')
        elif isinstance(vnf_tag, str) and vnf_tag.startswith('vnf'):
            return vnf_tag[3:]
        return False

    def _to_ecef_m(self, lat_deg, lon_deg, alt_m=ORBIT_ALTITUDE):
        """(deg, deg, m) -> ECEF (x,y,z) in meters (구형 지구 근사)"""
        lat = np.radians(lat_deg)
        lon = np.radians(lon_deg)
        r = R_EARTH_RADIUS + (alt_m if alt_m is not None else 0.0)
        x = r * np.cos(lat) * np.cos(lon)
        y = r * np.cos(lat) * np.sin(lon)
        z = r * np.sin(lat)
        return x, y, z

    def _filter_sats_with_xyz_m(self, vsg_sats, candidate_ids):
        """
        VSG 위성 리스트에서 후보 ID만 추출하고 ECEF(m) 좌표까지 준비.
        sat.alt가 km라면 alt_m = s.alt * 1000.0 로 바꿔주세요.
        """
        cid = set(candidate_ids or [])
        rows = []
        for s in vsg_sats:
            if s.id in cid:
                alt_m = getattr(s, "alt", 0.0)  # meters 가정
                x, y, z = self._to_ecef_m(float(s.lat), float(s.lon), float(alt_m))
                rows.append((int(s.id), x, y, z))
        return rows

    def _best_pair_euclid_broadcast_m(self, src_arr, dst_arr):
        """
        src_arr: (n,4)[id,x,y,z] in meters, dst_arr: (m,4)
        브로드캐스팅으로 제곱거리 행렬 계산 후 최소 쌍.
        """
        sx = src_arr[:, 1][:, None]
        sy = src_arr[:, 2][:, None]
        sz = src_arr[:, 3][:, None]
        dx = dst_arr[:, 1][None, :]
        dy = dst_arr[:, 2][None, :]
        dz = dst_arr[:, 3][None, :]

        D2 = (sx - dx) ** 2 + (sy - dy) ** 2 + (sz - dz) ** 2
        k = int(np.argmin(D2))
        i, j = divmod(k, D2.shape[1])
        return int(src_arr[i, 0]), int(dst_arr[j, 0]), float(np.sqrt(D2[i, j]))  # 거리(m)

    def _best_pair_euclid_ckdtree_m(self, src_arr, dst_arr):
        """
        큰 스케일에서는 KD-트리로 최근접 탐색 (meters).
        """
        from scipy.spatial import cKDTree
        tree = cKDTree(dst_arr[:, 1:4])  # xyz (meters)
        dists, idxs = tree.query(src_arr[:, 1:4], k=1)
        k = int(np.argmin(dists))
        return int(src_arr[k, 0]), int(dst_arr[int(idxs[k]), 0]), float(dists[k])  # 거리(m)

    def get_src_dst_sat(self, src_vsg, dst_vsg, candidate_src_sats, candidate_dst_sats,
                        brute_threshold_pairs=200_000, prefer_ckdtree=True):
        """
        src_vsg/dst_vsg: VSG 인덱스
        candidate_*_sats: 고려할 위성 id 모음
        반환: (best_src_id, best_dst_id) 또는 return_distance=True면 (best_src_id, best_dst_id, best_dist_m)
        전부 미터(m) 기준.
        """
        src_rows = self._filter_sats_with_xyz_m(self.vsgs_list[src_vsg].satellites, candidate_src_sats)
        dst_rows = self._filter_sats_with_xyz_m(self.vsgs_list[dst_vsg].satellites, candidate_dst_sats)

        src_arr = np.array(src_rows, dtype=float)
        dst_arr = np.array(dst_rows, dtype=float)

        n, m = len(src_arr), len(dst_arr)
        pairs = n * m

        if pairs <= brute_threshold_pairs:
            sid, did, dist_m = self._best_pair_euclid_broadcast_m(src_arr, dst_arr)
        else:
            if prefer_ckdtree:
                try:
                    sid, did, dist_m = self._best_pair_euclid_ckdtree_m(src_arr, dst_arr)
                except Exception:
                    sid, did, dist_m = self._best_pair_euclid_broadcast_m(src_arr, dst_arr)
            else:
                sid, did, dist_m = self._best_pair_euclid_broadcast_m(src_arr, dst_arr)

        return (sid, did, dist_m)

    def get_longgest_vnf_dst_sat_id(self, candidate_dst_sats):
        # 1. candidate_dst_sats ID를 실제 위성 객체로 매핑 (vsg_enter_time을 읽기 위함)
        dst_sat_objects = [self.sat_list[sat_id] for sat_id in candidate_dst_sats]

        # 2. vsg_enter_time 기준으로 내림차순 정렬 (가장 큰 시간이 맨 앞으로)
        #    sorted_sats 리스트의 첫 번째 요소가 가장 나중에 VSG에 들어온 위성입니다.
        sorted_sats = sorted(
            dst_sat_objects,
            key=lambda sat: sat.vsg_enter_time,
            reverse=True
        )

        # 3. dst_sat을 가장 최근에 들어온 위성으로 설정
        dst_sat_id = sorted_sats[0].id

        return dst_sat_id


    # TODO 0. sfc 종류에 맞게 경로 생성 잘 되는지 확인 (mMTC: 중간 경로 없음만 처리하면 될 듯)
    def set_satellite_path_noname(self, gsfc, mode):
        if gsfc.id not in self.vsg_path or not self.vsg_path[gsfc.id]:
            print(f"[ERROR] 2-1 ~AGAIN~ No VSG between VSG")
            gsfc.noname_dropped = True
            return []

        if gsfc.sfc_type_idx == 1: #mMTC:
            if gsfc.noname_satellite_path == []:
                cur_vsg_path_id = gsfc.noname_cur_vsg_path_id
                src_vsg_id, src_vnf = self.vsg_path[gsfc.id][cur_vsg_path_id]
                src_vsg = self.vsgs_list[src_vsg_id]

                selected_sat = []

                sat_all_vnfs = [
                    sat.id for sat in src_vsg.satellites
                    if all(vnf in sat.vnf_list for vnf in gsfc.vnf_sequence)
                ]

                if sat_all_vnfs:
                    # Priority 1 만족: 해당 위성들 중 무작위 선택
                    selected_sat = random.choice(sat_all_vnfs)
                else:
                    first_vnf = gsfc.vnf_sequence[0]

                    sat_first_vnf = [
                        sat.id for sat in src_vsg.satellites
                        if first_vnf in sat.vnf_list
                    ]

                    if sat_first_vnf:
                        # Priority 2 만족: 해당 위성들 중 무작위 선택
                        selected_sat = random.choice(sat_first_vnf)

                if selected_sat == []:
                    selected_sat = random.choice([sat.id for sat in src_vsg.satellites])

                gsfc.noname_satellite_path.append([selected_sat, 'src'])
                for vnf in gsfc.vnf_sequence:
                    gsfc.noname_satellite_path.append([selected_sat, f"vnf{vnf}"])
                gsfc.noname_satellite_path.append([selected_sat, "dst"])
            else:
                print(f"[NO GENERATION] already generation is done in mMTC scenario")
        else:
            if gsfc.noname_satellite_path == []:
                cur_vsg_path_id = gsfc.noname_cur_vsg_path_id
                src_vsg, src_vnf = self.vsg_path[gsfc.id][cur_vsg_path_id]

                is_vnf_src = self.has_vnf_tag(src_vnf)
                if is_vnf_src:
                    current_vnf_id = self.get_vnf_id_for_list(src_vnf)
                    candidate_src_sats = [
                        sat.id for sat in self.vsgs_list[src_vsg].satellites
                        if current_vnf_id in sat.vnf_list
                    ]
                else:
                    candidate_src_sats = [
                        sat.id for sat in self.vsgs_list[src_vsg].satellites
                    ]

                if not candidate_src_sats:
                    print(f"[ERROR] 3-1 No SATELLITE TO SRC")
                    gsfc.noname_dropped = True
                    return []

                # TODO. random choice?
                src_sat = random.choice(candidate_src_sats)
                gsfc.noname_satellite_path.append([src_sat, src_vnf])

            else:
                #  TODO src_vsg랑 dst_vsg가 같으면 prev_sat이랑 prev_vnf 추가하고 break
                prev_sat = gsfc.noname_satellite_path[-1][0]

                cur_vsg_path_id = gsfc.noname_cur_vsg_path_id - 1
                next_vsg_path_id = gsfc.noname_cur_vsg_path_id

                prev_vsg, prev_vnf = self.vsg_path[gsfc.id][cur_vsg_path_id]
                dst_vsg, dst_vnf = self.vsg_path[gsfc.id][next_vsg_path_id]

                is_vnf_src = self.has_vnf_tag(prev_vnf)
                if is_vnf_src:
                    current_vnf_id = self.get_vnf_id_for_list(prev_vnf)
                    candidate_src_sats = [
                        sat.id for sat in self.vsgs_list[prev_vsg].satellites
                        if current_vnf_id in sat.vnf_list
                    ]
                else:
                    candidate_src_sats = [
                        sat.id for sat in self.vsgs_list[prev_vsg].satellites
                    ]

                if not candidate_src_sats:
                    print(f"[ERROR] 3-1 No SATELLITE TO SRC")
                    gsfc.noname_dropped = True
                    return []

                is_vnf_dst = self.has_vnf_tag(dst_vnf)
                if is_vnf_dst:
                    current_vnf_id = self.get_vnf_id_for_list(dst_vnf)
                    candidate_dst_sats = [
                        sat.id for sat in self.vsgs_list[dst_vsg].satellites
                        if current_vnf_id in sat.vnf_list
                    ]
                else:
                    candidate_dst_sats = [
                        sat.id for sat in self.vsgs_list[dst_vsg].satellites
                    ]

                if not candidate_dst_sats:
                    print(f"[ERROR] 3-1 No SATELLITE TO DST")
                    gsfc.noname_dropped = True
                    return []

                if prev_vsg != dst_vsg: # vsg 이동이 필요할 때만 진행
                    src_vsg_sats = [sat.id for sat in self.vsgs_list[prev_vsg].satellites]
                    dst_vsg_sats = [sat.id for sat in self.vsgs_list[dst_vsg].satellites]
                    src_sat, dst_sat, src_dst_distance_m = self.get_src_dst_sat(prev_vsg, dst_vsg, src_vsg_sats, dst_vsg_sats)

                    # prev_sat -> src_sat
                    if prev_sat != src_sat:
                        try:
                            sub_path = nx.shortest_path(self.G, source=prev_sat, target=src_sat)

                            for sid in sub_path[1:]:
                                gsfc.noname_satellite_path.append([sid, None])
                            gsfc.noname_cur_sat_id = src_sat
                        except nx.NetworkXNoPath:
                            print(f"[ERROR] 3-2 No TRAJECTORY SAT TO SAT")
                            gsfc.noname_dropped = True
                            return []
                    else:
                        prev_sat = src_sat

                    # src_sat -> dst_sat
                    if src_sat == dst_sat:  # 이동 X
                        # SRC랑 DST는 다른 VSG 내에 있음.
                        print(f"[WARN] src_sat is not in dst_vsg")
                        return
                    else:
                        try:
                            sub_path = nx.shortest_path(self.G, source=src_sat, target=dst_sat)

                            for sid in sub_path[1:]:
                                gsfc.noname_satellite_path.append([sid, None])
                            gsfc.noname_cur_sat_id = dst_sat
                        except nx.NetworkXNoPath:
                            print(f"[ERROR] 3-2 No TRAJECTORY SAT TO SAT")
                            gsfc.noname_dropped = True
                            return []

                if prev_vsg == dst_vsg:
                    dst_sat = prev_sat

                # dst_sat -> vnf_sat
                if "longgest" in mode:
                    vnf_sat = self.get_longgest_vnf_dst_sat_id(candidate_dst_sats)
                else:
                    _, vnf_sat, dst_vnf_distance_m = self.get_src_dst_sat(dst_vsg, dst_vsg, [dst_sat], candidate_dst_sats)


                if dst_sat != vnf_sat:
                    dst_sat_avg_queue = mean([len(isl_k) for isl_k in self.sat_list[dst_sat].queue_ISL])

                    if dst_sat_avg_queue > 200:
                        # gserver까지 graph에 추가
                        current_G = self.G
                        selected_gserver_id = dst_vsg.gserver
                        if selected_gserver_id is not None:
                            current_G = self.create_temp_gserver_graph(selected_gserver_id)
                            # gsfc에 처리 gserver 추가
                            selected_gserver = self.gserver_list[selected_gserver_id]
                            gsfc.gserver = selected_gserver

                        # dst_sat -> vnf_g
                        if prev_sat == vnf_sat:  # 이동 X
                            gsfc.noname_satellite_path.append([vnf_sat, dst_vnf])
                            gsfc.noname_cur_sat_id = vnf_sat
                        else:
                            try:
                                sub_path = nx.shortest_path(current_G, source=prev_sat, target=vnf_sat)

                                if len(sub_path) > 2:
                                    for sid in sub_path[1:-1]:
                                        gsfc.noname_satellite_path.append([sid, None])
                                gsfc.noname_satellite_path.append([vnf_sat, dst_vnf])
                                gsfc.noname_cur_sat_id = vnf_sat
                            except nx.NetworkXNoPath:
                                print(f"[ERROR] 3-2 No TRAJECTORY SAT TO SAT")
                                gsfc.noname_dropped = True
                                return []

                    else:
                        try:
                            sub_path = nx.shortest_path(self.G, source=dst_sat, target=vnf_sat)

                            if len(sub_path) > 2:
                                for sid in sub_path[1:-1]:
                                    gsfc.noname_satellite_path.append([sid, None])
                            gsfc.noname_satellite_path.append([vnf_sat, dst_vnf])
                            gsfc.noname_cur_sat_id = vnf_sat
                        except nx.NetworkXNoPath:
                            print(f"[ERROR] 3-2 No TRAJECTORY SAT TO SAT")
                            gsfc.noname_dropped = True
                            return []

                else:
                    gsfc.noname_satellite_path[-1] = [dst_sat, dst_vnf]

            gsfc.noname_cur_vsg_path_id += 1

    def set_satellite_path(self, gsfc):
        if gsfc.id not in self.vsg_path or not self.vsg_path[gsfc.id]:
            print(f"[ERROR] 2-1 ~AGAIN~ No VSG between VSG")
            gsfc.basic_dropped = True
            return []

        prev_sat = -1

        for i in range(len(self.vsg_path[gsfc.id]) - 1):
            src_vsg, src_vnf = self.vsg_path[gsfc.id][i]
            dst_vsg, dst_vnf = self.vsg_path[gsfc.id][i + 1]

            if prev_sat == -1:
                is_vnf = self.has_vnf_tag(src_vnf)
                if is_vnf:
                    current_vnf_id = self.get_vnf_id_for_list(src_vnf)
                    candidate_src_sats = [
                        sat.id for sat in self.vsgs_list[src_vsg].satellites
                        if current_vnf_id in sat.vnf_list
                    ]
                else:
                    candidate_src_sats = [
                        sat.id for sat in self.vsgs_list[src_vsg].satellites
                    ]
                if not candidate_src_sats:
                    print(f"[ERROR] 3-1 No SATELLITE TO SRC")
                    gsfc.basic_dropped = True
                    return []
                src_sat = random.choice(candidate_src_sats)
                prev_sat = src_sat

            is_vnf = self.has_vnf_tag(dst_vnf)
            if is_vnf:
                current_vnf_id = self.get_vnf_id_for_list(dst_vnf)
                candidate_dst_sats = [
                    sat.id for sat in self.vsgs_list[dst_vsg].satellites
                    if current_vnf_id in sat.vnf_list
                ]
            else:
                candidate_dst_sats = [
                    sat.id for sat in self.vsgs_list[dst_vsg].satellites
                ]

            if not candidate_dst_sats:
                print(f"[ERROR] 3-1 No SATELLITE TO DST")
                gsfc.basic_dropped = True
                return []
            dst_sat = random.choice(candidate_dst_sats)

            if prev_sat == dst_sat: # 이동 X
                if i == 0:
                    gsfc.basic_satellite_path.append([src_sat, src_vnf])
                gsfc.basic_satellite_path.append([dst_sat, dst_vnf])
                prev_sat = dst_sat
            else:
                try:
                    sub_path = nx.shortest_path(self.G, source=prev_sat, target=dst_sat)

                    if i == 0:
                        gsfc.basic_satellite_path.append([src_sat, src_vnf])
                    if len(sub_path) > 2:
                        for sid in sub_path[1:-1]:
                            gsfc.basic_satellite_path.append([sid, None])
                    gsfc.basic_satellite_path.append([dst_sat, dst_vnf])
                    prev_sat = dst_sat
                except nx.NetworkXNoPath:
                    print(f"[ERROR] 3-2 No TRAJECTORY SAT TO SAT")
                    gsfc.basic_dropped = True
                    return []

    # ========= DD ========= #
    # vnf sequence를 처리할 수 있는 위성 리스트로 총 조합을 구한 뒤, 전역 최적 경로 선택
    def find_shortest_satellite_vnf_path(self, gsfc):
        vnf_sequence = gsfc.vnf_sequence
        src_vsg_id = gsfc.src_vsg_id
        dst_vsg_id = gsfc.dst_vsg_id

        # vnf 별 가능한 위성 후보군 추출
        unique_vnf_types = sorted(list(set(vnf_sequence)))
        vnf_to_sat_ids = {}
        for vnf in unique_vnf_types:
            candidate_sats = [
                sat.id for sat in self.sat_list
                if vnf in sat.vnf_list and sat.id not in self.congested_sat_ids
            ]
            if not candidate_sats:
                print(f"[ERROR] No satellites available for VNF {vnf}")
            vnf_to_sat_ids[vnf] = candidate_sats

        # src/dst VSG 내 위성 목록
        src_vsg_sat_ids = [sat.id for sat in self.vsgs_list[src_vsg_id].satellites if
                           sat.id not in self.congested_sat_ids]
        dst_vsg_sat_ids = [sat.id for sat in self.vsgs_list[dst_vsg_id].satellites if
                           sat.id not in self.congested_sat_ids]

        best_full_path = None
        min_total_hops = float('inf')

        # 모든 VNF 조합
        list_of_candidate_lists = [vnf_to_sat_ids[vnf_type] for vnf_type in vnf_sequence]
        vnf_combinations = list(product(*list_of_candidate_lists))
        # print(f"combinations in dd: {vnf_combinations}")

        for vnf_combo in vnf_combinations:
            path = list(vnf_combo)
            total_hops = 0
            valid = True
            full_path = []  # 전체 경로 누적

            # 시작점 처리
            if not src_vsg_sat_ids or not dst_vsg_sat_ids:
                print("[ERROR] There is no available satellites in SRC or DST VSG")
                break

            # src_vsg에서 첫 번째 vnf 실행 불가 -> src_vsg에서 가장 체류 시간 긴 위성 선택
            is_src_sat_added_vnf = True
            if path[0] not in src_vsg_sat_ids:
                # src_vsg 내 가장 작은 인덱스의 위성 붙이기
                start_sat = min(src_vsg_sat_ids)
                path.insert(0, start_sat)
                is_src_sat_added_vnf = False

            # dst_vsg에서 마지막 vnf 실행 불가 -> dst_vsg에서 가장 체류 시간 긴 위성 선택
            is_dst_sat_added_vnf = True
            if path[-1] not in dst_vsg_sat_ids:
                end_sat = min(dst_vsg_sat_ids)
                path.append(end_sat)
                is_dst_sat_added_vnf = False

            # 전체 경로 유효성 및 홉 수 계산
            current_vnf_id = 0
            for i in range(len(path) - 1):
                if (i == 0) and is_src_sat_added_vnf:
                    full_path.append([path[i], ("src", f"vnf{vnf_sequence[current_vnf_id]}")])
                    current_vnf_id += 1
                elif (i == 0) and not is_src_sat_added_vnf:
                    full_path.append([path[i], ("src")])
                else:
                    if path[i - 1] == path[i]:
                        full_path.append([path[i], (f"vnf{vnf_sequence[current_vnf_id]}")])
                        current_vnf_id += 1
                    else:
                        segment = self.get_full_path(path[i-1], path[i])
                        if not segment:
                            valid = False
                            break

                        total_hops += len(segment) - 1
                        if full_path:
                            if len(segment) > 2:
                                for seg in segment[1:-1]:
                                    full_path.append([seg, (None)])
                            full_path.append([segment[-1], (f"vnf{vnf_sequence[current_vnf_id]}")])
                            current_vnf_id += 1
                        else:
                            full_path.append([segment[0], (f"vnf{vnf_sequence[current_vnf_id]}")])
                            current_vnf_id += 1
                            if len(segment) > 2:
                                for seg in segment[:-1]:
                                    full_path.append([seg, (None)])
                            full_path.append([segment[-1], (f"vnf{vnf_sequence[current_vnf_id]}")])
                            current_vnf_id += 1
            # 마지막 도착지 처리
            if path[-2] == path[-1]:
                if is_dst_sat_added_vnf:
                    full_path.append([path[-1], ("dst", f"vnf{vnf_sequence[current_vnf_id]}")])
                else:
                    full_path.append([path[-1], ("dst")])
            else:
                segment = self.get_full_path(path[-2], path[-1])
                if not segment:
                    valid = False
                    break

                total_hops += len(segment) - 1
                if full_path:
                    if len(segment) > 2:
                        for seg in segment[1:-1]:
                            full_path.append([seg, (None)])
                    if is_dst_sat_added_vnf:
                        full_path.append([segment[-1], ("dst", f"vnf{vnf_sequence[current_vnf_id]}")])
                    else:
                        full_path.append([segment[-1], ("dst")])
                    current_vnf_id += 1
                else:
                    full_path.append([segment[0], (f"vnf{vnf_sequence[current_vnf_id]}")])
                    current_vnf_id += 1
                    if len(segment) > 2:
                        for seg in segment[:-1]:
                            full_path.append([seg, (None)])
                    if is_dst_sat_added_vnf:
                        full_path.append([segment[-1], ("dst", f"vnf{vnf_sequence[current_vnf_id]}")])
                    else:
                        full_path.append([segment[-1], ("dst")])
                    current_vnf_id += 1

            if valid and total_hops < min_total_hops:
                # print(f"NEWNEW {total_hops}, vnf_combo: {vnf_combo}")
                min_total_hops = total_hops
                best_full_path = full_path
                gsfc.dd_hop_count = min_total_hops

        gsfc.dd_satellite_path = best_full_path


    # ========= SD ========= #
    # SRC-DST 간 최적 경로 설정 후, src vsg의 지상국에서 전체 vnf 처리
    def find_shortest_satellite_path_between_src_dst(self, gsfc):
        src_vsg = self.vsgs_list[gsfc.src_vsg_id]
        dst_vsg = self.vsgs_list[gsfc.dst_vsg_id]

        src_vsg_sat_ids = [sat.id for sat in src_vsg.satellites if sat.id not in self.congested_sat_ids]
        dst_vsg_sat_ids = [sat.id for sat in dst_vsg.satellites if sat.id not in self.congested_sat_ids]

        if not src_vsg_sat_ids or not dst_vsg_sat_ids:
            return None

        # 대표 위성 선택 (가장 작은 ID)
        start_sat_id = min(src_vsg_sat_ids)
        end_sat_id = min(dst_vsg_sat_ids)

        # 최단 경로 찾기
        shortest_path = self.get_full_path(start_sat_id, end_sat_id)

        if not shortest_path:
            return None

        # 경로 포맷: [(sat_id, "src"), (sat_id, None), ..., (sat_id, "dst")]
        formatted_path = []
        if shortest_path:
            formatted_path.append([shortest_path[0], 'src'])
            for sat_id in shortest_path[1:-1]:
                formatted_path.append([sat_id, None])
            if len(shortest_path) > 1:
                formatted_path.append([shortest_path[-1], 'dst'])

        gsfc.sd_satellite_path = formatted_path
        gsfc.sd_hop_count = len(gsfc.sd_satellite_path) + 1
        gsfc.gserver = src_vsg.gserver
        gsfc.sd_total_vnf_size = sum(gsfc.vnf_sizes)

    # ========= SD+DBPR ========= #
    # 지상국 layer 포함. queue 상태를 고려한 delay 최적 경로 생성
    def advanced_proposed_find_satellite_path(self, gsfc):
        gserver_node_ids = [NUM_SATELLITES + gserver.id for gserver in self.gserver_list]

        # option 1. 한 가지의 VNF만 지상국에서 처리
        # option 2. 모든 VNF를 지상국에서 처리
        vnf_sequence = gsfc.vnf_sequence
        num_vnf = len(vnf_sequence)
        src_vsg_id = gsfc.src_vsg_id
        dst_vsg_id = gsfc.dst_vsg_id

        vnf_combinations = []

        # vnf 별 가능한 위성 후보군 추출
        unique_vnf_types = sorted(list(set(vnf_sequence)))
        vnf_to_sat_ids = {}
        for vnf in unique_vnf_types:
            candidate_sats = [
                sat.id for sat in self.sat_list
                if vnf in sat.vnf_list and sat.id not in self.congested_sat_ids
            ]
            if not candidate_sats:
                print(f"[ERROR] No satellites available for VNF {vnf}")
            vnf_to_sat_ids[vnf] = candidate_sats

        # gserver mask
        gserver_positions_masks = []
        # gserver 안 거침
        gserver_positions_masks.append([False] * num_vnf)
        for i in range(num_vnf):
            mask = [False] * num_vnf
            mask[i] = True
            gserver_positions_masks.append(mask)

        for gserver_mask in gserver_positions_masks:
            # gserver를 사용하지 않는 경우
            if not any(gserver_mask):
                list_of_candidate_lists = [vnf_to_sat_ids[vnf_sequence[i]] for i in range(num_vnf)]
                new_combinations = list(product(*list_of_candidate_lists))
                vnf_combinations.extend(new_combinations)

            else:  # gserver를 사용하는 경우
                for gserver_node_id in gserver_node_ids:
                    list_of_candidate_lists_with_gserver = []

                    for i in range(num_vnf):
                        current_vnf_type = vnf_sequence[i]

                        if gserver_mask[i]:
                            # Gserver가 처리하는 VNF 위치 -> Gserver 노드 ID만 포함
                            list_of_candidate_lists_with_gserver.append([gserver_node_id])
                        else:
                            # 위성이 처리하는 VNF 위치 -> 해당 VNF를 처리 가능한 위성 ID 목록 포함
                            list_of_candidate_lists_with_gserver.append(vnf_to_sat_ids[current_vnf_type])

                    # 최종 조합 생성
                    new_combinations = product(*list_of_candidate_lists_with_gserver)
                    vnf_combinations.extend(new_combinations)

        # src/dst VSG 내 위성 목록
        src_vsg_sat_ids = [sat.id for sat in self.vsgs_list[src_vsg_id].satellites if
                           sat.id not in self.congested_sat_ids]
        dst_vsg_sat_ids = [sat.id for sat in self.vsgs_list[dst_vsg_id].satellites if
                           sat.id not in self.congested_sat_ids]

        best_delay = float('inf')
        best_full_path = None

        for vnf_combo in vnf_combinations:
            path = list(vnf_combo)
            total_delay = 0.0  # hop count가 아닌 propagation ms로 변환
            total_hops = 0
            current_queue_delay = 0
            valid = True
            full_path = []

            # gserver까지 graph에 추가
            current_G = self.G
            selected_gserver_id = self.get_selected_gserver_id(vnf_combo)

            if selected_gserver_id is not None:
                current_G = self.create_temp_gserver_graph(selected_gserver_id)

                # gsfc에 처리 gserver 추가
                selected_gserver = self.gserver_list[selected_gserver_id]
                gsfc.gserver = selected_gserver

            # 시작점 처리
            if not src_vsg_sat_ids or not dst_vsg_sat_ids:
                print("[ERROR] There is no available satellites in SRC or DST VSG")
                break

            # src_vsg에서 첫 번째 vnf 실행 불가 -> src_vsg에서 가장 체류 시간 긴 위성 선택
            is_src_sat_added_vnf = True
            start_sat_id = path[0]
            if path[0] not in src_vsg_sat_ids:
                start_node_id = min(src_vsg_sat_ids)
                path.insert(0, start_node_id)
                is_src_sat_added_vnf = False
            elif self.is_gserver(path[0]):
                start_node_id = min(src_vsg_sat_ids)
                path.insert(0, start_node_id)
                is_src_sat_added_vnf = False

            # dst_vsg에서 마지막 vnf 실행 불가 -> dst_vsg에서 가장 체류 시간 긴 위성 선택
            is_dst_sat_added_vnf = True
            end_sat_id = path[-1]
            if path[-1] not in dst_vsg_sat_ids:
                end_node_id = min(dst_vsg_sat_ids)
                path.append(end_node_id)
                is_dst_sat_added_vnf = False
            elif self.is_gserver(path[-1]):
                end_node_id = min(dst_vsg_sat_ids)
                path.append(end_node_id)
                is_dst_sat_added_vnf = False

            current_vnf_id = 0
            for i in range(len(path) - 1):
                prev_node_id = path[i - 1] if (i - 1) >= 0 else None
                curr_node_id = path[i]

                if (i == 0) and is_src_sat_added_vnf:
                    node_type = 'sat' if not self.is_gserver(curr_node_id) else 'gserver'
                    full_path.append([curr_node_id, ("src", f"vnf{vnf_sequence[current_vnf_id]}")])

                    transmit_queue = self.get_node_link_queue(curr_node_id, node_type=node_type)
                    proc_queue = self.get_node_process_queue(curr_node_id, node_type=node_type)
                    current_queue_delay += proc_queue + transmit_queue  # process_queue의 길이 + queue_ISL 길이

                    current_vnf_id += 1
                elif (i == 0) and not is_src_sat_added_vnf:
                    node_type = 'sat' if not self.is_gserver(curr_node_id) else 'gserver'
                    full_path.append([curr_node_id, ("src")])

                    transmit_queue = self.get_node_link_queue(curr_node_id, node_type=node_type)
                    current_queue_delay += transmit_queue  # queue_ISL 길이만
                elif prev_node_id == curr_node_id:
                    # Gserver도 VNF 연쇄 처리 가능
                    node_type = "sat" if not self.is_gserver(curr_node_id) else "gserver"
                    full_path.append([curr_node_id, (f"vnf{vnf_sequence[current_vnf_id]}")])

                    proc_queue = self.get_node_process_queue(curr_node_id, node_type=node_type)
                    transmit_queue = self.get_node_link_queue(curr_node_id, node_type=node_type)
                    hop_distance = self.hop_table.get((start_sat_id, curr_node_id), 1.0)
                    hop_alpha = (1 / hop_distance) if hop_distance > 0 else 1.0
                    current_queue_delay += hop_alpha * (proc_queue + transmit_queue)

                    current_vnf_id += 1
                else:
                    segment = self.get_full_path(prev_node_id, curr_node_id,
                                                 current_G)  # Terrestrial included graph가 아닌 satellite grpah로 탐색 (단순 경유로는 gserver 경유 X)

                    if not segment:
                        valid = False
                        break

                    # total_delay += self.hop_table.get((prev_node_id, curr_node_id), float('inf'))
                    # 이건 prev_node와 curre_node가 연결되어야지만 나오는 거 아닌가?

                    total_hops += len(segment) - 1
                    if len(segment) > 2:
                        # 시작 노드는 이미 처리되었으므로 segment[1:-1]만 순회
                        for seg_node_id in segment[1:-1]:
                            # 중간 노드는 위성만 가능하다고 가정 (Gserver는 VNF 처리 노드로만 등장)
                            node_type = "sat" if not self.is_gserver(seg_node_id) else "gserver"
                            full_path.append([seg_node_id, (None)])

                            # Queue Delay 반영 (송신 큐만)
                            transmit_queue = self.get_node_link_queue(seg_node_id, node_type=node_type)  # 중간 노드는 위성
                            hop_distance = self.hop_table.get((start_sat_id, seg_node_id), 1.0)
                            hop_alpha = (1 / hop_distance) if hop_distance > 0 else 1.0

                            current_queue_delay += hop_alpha * transmit_queue
                    # 현재 VNF를 처리할 노드 (segment[-1] == curr_node)
                    node_type = "sat" if not self.is_gserver(curr_node_id) else "gserver"
                    full_path.append([curr_node_id, (f"vnf{vnf_sequence[current_vnf_id]}")])

                    # Queue Delay 반영 (처리 큐 + 송신 큐)
                    proc_queue = self.get_node_process_queue(curr_node_id, node_type=node_type)
                    transmit_queue = self.get_node_link_queue(curr_node_id, node_type=node_type)
                    hop_distance = self.hop_table.get((start_sat_id, curr_node_id), 1.0)
                    hop_alpha = (1 / hop_distance) if hop_distance > 0 else 1.0
                    current_queue_delay += hop_alpha * (proc_queue + transmit_queue)

                    current_vnf_id += 1
            if path[-2] == path[-1]:
                if is_dst_sat_added_vnf:
                    node_type = "sat" if not self.is_gserver(path[-1]) else "gserver"
                    full_path.append([path[-1], ("dst", f"vnf{vnf_sequence[current_vnf_id]}")])

                    hop_distance = self.hop_table.get((start_sat_id, path[-1]), 1.0)
                    hop_alpha = (1 / hop_distance) if hop_distance > 0 else 1.0
                    proc_queue = self.get_node_process_queue(path[-1], node_type=node_type)

                    current_queue_delay += hop_alpha * proc_queue
                else:
                    full_path.append([path[-1], ("dst")])
            else:
                segment = self.get_full_path(path[-2], path[-1],
                                             current_G)  # Terrestrial included graph가 아닌 satellite grpah로 탐색 (단순 경유로는 gserver 경유 X)
                if not segment:
                    valid = False
                    break

                total_hops += len(segment) - 1
                if len(segment) > 2:
                    # 시작 노드는 이미 처리되었으므로 segment[1:-1]만 순회
                    for seg_node_id in segment[1:-1]:
                        # 중간 노드는 위성만 가능하다고 가정 (Gserver는 VNF 처리 노드로만 등장)
                        node_type = "sat" if not self.is_gserver(seg_node_id) else "gserver"
                        full_path.append([seg_node_id, (None)])

                        # Queue Delay 반영 (송신 큐만)
                        transmit_queue = self.get_node_link_queue(seg_node_id, node_type=node_type)  # 중간 노드는 위성
                        hop_distance = self.hop_table.get((start_sat_id, seg_node_id), 1.0)
                        hop_alpha = (1 / hop_distance) if hop_distance > 0 else 1.0
                        current_queue_delay += hop_alpha * transmit_queue

                if is_dst_sat_added_vnf:
                    node_type = "sat" if not self.is_gserver(segment[-1]) else "gserver"
                    full_path.append([segment[-1], ("dst", f"vnf{vnf_sequence[current_vnf_id]}")])

                    proc_queue = self.get_node_process_queue(segment[-1], node_type=node_type)
                    hop_distance = self.hop_table.get((start_sat_id, segment[-1]), 1.0)
                    hop_alpha = (1 / hop_distance) if hop_distance > 0 else 1
                    current_queue_delay += hop_alpha * proc_queue
                else:
                    full_path.append([segment[-1], ("dst")])
                current_vnf_id += 1

            if valid and (total_hops + current_queue_delay) < best_delay:
                best_delay = (total_hops + current_queue_delay)
                best_full_path = full_path

        gsfc.sd_satellite_path = best_full_path

    # ========= DD+DBPR ========= #
    # 위성 layer 한정. queue 상태를 고려한 delay 최적 경로 생성
    def proposed_find_satellite_path(self, gsfc):
        vnf_sequence = gsfc.vnf_sequence
        src_vsg_id = gsfc.src_vsg_id
        dst_vsg_id = gsfc.dst_vsg_id

        # vnf 별 가능한 위성 후보군 추출
        unique_vnf_types = sorted(list(set(vnf_sequence)))
        vnf_to_sat_ids = {}
        for vnf in unique_vnf_types:
            candidate_sats = [
                sat.id for sat in self.sat_list
                if vnf in sat.vnf_list and sat.id not in self.congested_sat_ids
            ]
            if not candidate_sats:
                print(f"[ERROR] No satellites available for VNF {vnf}")
            vnf_to_sat_ids[vnf] = candidate_sats

        # src/dst VSG 내 위성 목록
        src_vsg_sat_ids = [sat.id for sat in self.vsgs_list[src_vsg_id].satellites if
                           sat.id not in self.congested_sat_ids]
        dst_vsg_sat_ids = [sat.id for sat in self.vsgs_list[dst_vsg_id].satellites if
                           sat.id not in self.congested_sat_ids]

        best_full_path = None
        min_total_delay = float('inf')

        # 모든 VNF 조합
        list_of_candidate_lists = [vnf_to_sat_ids[vnf_type] for vnf_type in vnf_sequence]
        vnf_combinations = list(product(*list_of_candidate_lists))
        # print(f"combinations in proposed: {vnf_combinations}")

        # 고려할 사항 1. 다음 노드와의 거리
        # 고려할 사항 2. 해당 노드의 큐 상태
        for vnf_combo in vnf_combinations:
            path = list(vnf_combo)
            total_hops = 0
            current_queue_delay = 0
            valid = True
            full_path = []

            # 시작점 처리
            if not src_vsg_sat_ids or not dst_vsg_sat_ids:
                print("[ERROR] There is no available satellites in SRC or DST VSG")
                break

            # src_vsg에서 첫 번째 vnf 실행 불가 -> src_vsg에서 가장 체류 시간 긴 위성 선택
            is_src_sat_added_vnf = True
            start_sat_id = path[0]
            if path[0] not in src_vsg_sat_ids:
                # src_vsg 내 가장 작은 인덱스의 위성 붙이기
                start_sat_id = min(src_vsg_sat_ids)
                path.insert(0, start_sat_id)
                is_src_sat_added_vnf = False

            # dst_vsg에서 마지막 vnf 실행 불가 -> dst_vsg에서 가장 체류 시간 긴 위성 선택
            is_dst_sat_added_vnf = True
            end_sat_id = path[-1]
            if path[-1] not in dst_vsg_sat_ids:
                end_sat_id = min(dst_vsg_sat_ids)
                path.append(end_sat_id)
                is_dst_sat_added_vnf = False

            current_vnf_id = 0
            for i in range(len(path) - 1):
                if (i == 0) and is_src_sat_added_vnf:
                    full_path.append([path[i], ("src", f"vnf{vnf_sequence[current_vnf_id]}")])
                    next_sat = self.sat_list[path[i]]
                    transmit_queue = self.get_node_link_queue(next_sat.id, node_type='sat')
                    proc_queue = self.get_node_process_queue(next_sat.id, node_type='sat')
                    current_queue_delay += proc_queue + transmit_queue  # process_queue의 길이 + queue_ISL 길이
                    current_vnf_id += 1
                elif (i == 0) and not is_src_sat_added_vnf:
                    full_path.append([path[i], ("src")])
                    next_sat = self.sat_list[path[i]]
                    transmit_queue = self.get_node_link_queue(next_sat.id, node_type='sat')
                    current_queue_delay += transmit_queue  # queue_ISL 길이만
                else:
                    if path[i - 1] == path[i]:
                        full_path.append([path[i], (f"vnf{vnf_sequence[current_vnf_id]}")])

                        next_sat = self.sat_list[path[i]]
                        hop_distance = self.hop_table[(start_sat_id, next_sat.id)]
                        transmit_queue = self.get_node_link_queue(next_sat.id, node_type='sat')
                        proc_queue = self.get_node_process_queue(next_sat.id, node_type='sat')
                        hop_alpha = (1 / hop_distance) if hop_distance > 0 else 1
                        current_queue_delay += hop_alpha * (
                                    proc_queue + transmit_queue)  # process_queue의 길이 + queue_ISL 길이
                        current_vnf_id += 1
                    else:
                        segment = self.get_full_path(path[i - 1], path[i])
                        if not segment:
                            valid = False
                            break

                        total_hops += len(segment) - 1
                        if full_path:
                            if len(segment) > 2:
                                for seg in segment[1:-1]:
                                    full_path.append([seg, (None)])
                                    next_sat = self.sat_list[seg]
                                    transmit_queue = self.get_node_link_queue(next_sat.id, node_type='sat')
                                    hop_distance = self.hop_table[(start_sat_id, next_sat.id)]
                                    hop_alpha = (1 / hop_distance) if hop_distance > 0 else 1
                                    current_queue_delay += hop_alpha * transmit_queue  # queue_ISL 길이만
                            full_path.append([segment[-1], (f"vnf{vnf_sequence[current_vnf_id]}")])
                            next_sat = self.sat_list[segment[-1]]
                            transmit_queue = self.get_node_link_queue(next_sat.id, node_type='sat')
                            proc_queue = self.get_node_process_queue(next_sat.id, node_type='sat')
                            hop_distance = self.hop_table[(start_sat_id, next_sat.id)]
                            hop_alpha = (1 / hop_distance) if hop_distance > 0 else 1
                            current_queue_delay += hop_alpha * (
                                        proc_queue + transmit_queue)  # process_queue의 길이 + queue_ISL 길이
                            current_vnf_id += 1
                        else:
                            full_path.append([segment[0], (f"vnf{vnf_sequence[current_vnf_id]}")])
                            next_sat = self.sat_list[segment[0]]
                            transmit_queue = self.get_node_link_queue(next_sat.id, node_type='sat')
                            proc_queue = self.get_node_process_queue(next_sat.id, node_type='sat')
                            hop_distance = self.hop_table[(start_sat_id, next_sat.id)]
                            hop_alpha = (1 / hop_distance) if hop_distance > 0 else 1
                            current_queue_delay += hop_alpha * (
                                        proc_queue + transmit_queue)  # process_queue의 길이 + queue_ISL 길이
                            current_vnf_id += 1
                            if len(segment) > 2:
                                for seg in segment[:-1]:
                                    full_path.append([seg, (None)])
                                    next_sat = self.sat_list[seg]
                                    transmit_queue = self.get_node_link_queue(next_sat.id, node_type='sat')
                                    hop_distance = self.hop_table[(start_sat_id, next_sat.id)]
                                    hop_alpha = (1 / hop_distance) if hop_distance > 0 else 1
                                    current_queue_delay += hop_alpha * transmit_queue  # queue_ISL 길이만
                            full_path.append([segment[-1], (f"vnf{vnf_sequence[current_vnf_id]}")])
                            next_sat = segment[-1]
                            transmit_queue = self.get_node_link_queue(next_sat.id, node_type='sat')
                            proc_queue = self.get_node_process_queue(next_sat.id, node_type='sat')
                            hop_distance = self.hop_table[(start_sat_id, next_sat.id)]
                            hop_alpha = (1 / hop_distance) if hop_distance > 0 else 1
                            current_queue_delay += hop_alpha * (
                                        proc_queue + transmit_queue)  # process_queue의 길이 + queue_ISL 길이
                            current_vnf_id += 1

            # 마지막 도착지 처리
            if path[-2] == path[-1]:
                if is_dst_sat_added_vnf:
                    full_path.append([path[-1], ("dst", f"vnf{vnf_sequence[current_vnf_id]}")])

                    next_sat = self.sat_list[path[-1]]
                    hop_distance = self.hop_table[(start_sat_id, next_sat.id)]
                    hop_alpha = (1 / hop_distance) if hop_distance > 0 else 1
                    proc_queue = self.get_node_process_queue(next_sat.id, node_type='sat')
                    current_queue_delay += hop_alpha * proc_queue  # process_queue의 길이
                else:
                    full_path.append([path[-1], ("dst")])
            else:
                segment = self.get_full_path(path[-2], path[-1])
                if not segment:
                    valid = False
                    break

                total_hops += len(segment) - 1
                if full_path:
                    if len(segment) > 2:
                        for seg in segment[1:-1]:
                            full_path.append([seg, (None)])
                            next_sat = self.sat_list[seg]
                            transmit_queue = self.get_node_link_queue(next_sat.id, node_type='sat')
                            hop_distance = self.hop_table[(start_sat_id, next_sat.id)]
                            hop_alpha = (1 / hop_distance) if hop_distance > 0 else 1
                            current_queue_delay += hop_alpha * transmit_queue  # queue_ISL 길이만
                    if is_dst_sat_added_vnf:
                        full_path.append([segment[-1], ("dst", f"vnf{vnf_sequence[current_vnf_id]}")])
                        next_sat = self.sat_list[segment[-1]]
                        proc_queue = self.get_node_process_queue(next_sat.id, node_type='sat')
                        hop_distance = self.hop_table[(start_sat_id, next_sat.id)]
                        hop_alpha = (1 / hop_distance) if hop_distance > 0 else 1
                        current_queue_delay += hop_alpha * proc_queue  # process_queue의 길이
                    else:
                        full_path.append([segment[-1], ("dst")])
                    current_vnf_id += 1
                else:
                    full_path.append([segment[0], (f"vnf{vnf_sequence[current_vnf_id]}")])
                    next_sat = self.sat_list[segment[0]]
                    transmit_queue = self.get_node_link_queue(next_sat.id, node_type='sat')
                    proc_queue = self.get_node_process_queue(next_sat.id, node_type='sat')
                    hop_distance = self.hop_table[(start_sat_id, next_sat.id)]
                    hop_alpha = (1 / hop_distance) if hop_distance > 0 else 1
                    current_queue_delay += hop_alpha * (
                                proc_queue + transmit_queue)  # process_queue의 길이 + queue_ISL 길이
                    current_vnf_id += 1

                    if len(segment) > 2:
                        for seg in segment[:-1]:
                            full_path.append([seg, (None)])
                            next_sat = self.sat_list[seg]
                            transmit_queue = self.get_node_link_queue(next_sat.id, node_type='sat')
                            hop_distance = self.hop_table[(start_sat_id, next_sat.id)]
                            hop_alpha = (1 / hop_distance) if hop_distance > 0 else 1
                            current_queue_delay += hop_alpha * transmit_queue  # queue_ISL 길이만
                    if is_dst_sat_added_vnf:
                        full_path.append([segment[-1], ("dst", f"vnf{vnf_sequence[current_vnf_id]}")])
                        next_sat = self.sat_list[segment[-1]]
                        proc_queue = self.get_node_process_queue(next_sat.id, node_type='sat')
                        hop_distance = self.hop_table[(start_sat_id, next_sat.id)]
                        hop_alpha = (1 / hop_distance) if hop_distance > 0 else 1
                        current_queue_delay += hop_alpha * proc_queue  # process_queue의 길이
                    else:
                        full_path.append([segment[-1], ("dst")])
                    current_vnf_id += 1

            if valid and (total_hops + current_queue_delay) < min_total_delay:  # total_delay말고, 거리 반비례 가중치 * 그에 대한 홉수의 합
                # print(f"NEWNEW {total_hops} + {current_queue_delay}, vnf_combo: {vnf_combo}")
                min_total_delay = (total_hops + current_queue_delay)
                best_full_path = full_path

            gsfc.dd_satellite_path = best_full_path

    # node의 transmit queue 크기 반환
    def get_node_link_queue(self, node_id, node_type='sat'):
        link_total_size = 0

        if node_type == 'sat':
            sat = self.sat_list[node_id]
            link_total_size = 0

            for link_queue in sat.queue_ISL:
                link_total_size += sum(packet[1] for packet in link_queue)
            link_total_size += sum(packet[1] for packet in sat.queue_TSL)

        elif node_type == 'gserver':
            if self.is_gserver(node_id):
                gserver_obj = self.get_gserver_obj(node_id)
                return len(gserver_obj.process_queue) if gserver_obj else 0

        return link_total_size

    # node의 process queue 크기 반환
    def get_node_process_queue(self, node_id, node_type='sat'):
        if node_type == 'sat':
            return len(self.sat_list[node_id].process_queue)
        elif node_type == 'gserver':
            if self.is_gserver(node_id):
                gserver_obj = self.get_gserver_obj(node_id)
                return len(gserver_obj.process_queue) if gserver_obj else 0

    # 거리 기반 가중치 반환 (거리가 멀 수록 작은 가중치 생성)
    def calculate_inverse_hop_cost(self, start_sat_id, current_sat_id):
        """홉 거리를 조회하고 0일 경우 1.0을, 무한대일 경우 0.0을 반환합니다."""
        hop_distance = self.hop_table.get((start_sat_id, current_sat_id), float('inf'))

        if hop_distance == 0:
            return 1.0  # 홉 거리가 0인 경우 (출발 위성), ZeroDivisionError 방지 및 비용 최대값(1배) 적용
        elif hop_distance == float('inf'):
            return 0.0  # 경로가 없는 경우, 비용 기여 없음 (valid=False로 처리됨)
        else:
            return 1.0 / hop_distance  # 일반적인 역수 가중치

    def is_gserver(self, node_id):
        """주어진 노드 ID가 Gserver 노드 ID인지 확인합니다."""
        return node_id >= NUM_SATELLITES

    # SD+DBPR에서 지상국을 포함한 경로 생성하기 위한 그래프
    # 단순 경유 케이스에서 지상국 경유를 막기 위해, vnf 처리하기 위해 선택한 gserver의 vsg만 활성화
    def create_temp_gserver_graph(self, gserver_id):
        TG_temp = self.G.copy()

        gserver = self.gserver_list[gserver_id]
        gserver_node_id = NUM_SATELLITES + gserver_id
        TG_temp.add_node(gserver_node_id, type="gserver", vsg_id=gserver.vsg_id)

        target_vsg = next((vsg for vsg in self.vsgs_list if vsg.id == gserver.vsg_id), None)
        if target_vsg:
            for sat in target_vsg.satellites:
                if sat.id not in self.congested_sat_ids:
                    tsl_delay_ms = sat.calculate_TSL_propagation_delay(gserver)
                    # TSL 엣지 추가 (weight=delay)
                    TG_temp.add_edge(sat.id, gserver_node_id, weight=tsl_delay_ms, link_type='tsl')

        return TG_temp

    def get_gserver_obj(self, node_id):
        """Gserver 노드 ID로부터 Gserver 객체를 반환합니다."""
        if self.is_gserver(node_id):
            gserver_id = node_id - NUM_SATELLITES
            return next((gserver for gserver in self.gserver_list if gserver.id == gserver_id), None)
        return None

    def get_selected_gserver_id(self, vnf_combo):
        for node_id in vnf_combo:
            if self.is_gserver(node_id):
                return node_id - NUM_SATELLITES
        return None

    def check_vsg_regions(self):
        vid = 0
        is_changed = False

        # lat_bins = np.arange(-1*POLAR_LATITUDE, POLAR_LATITUDE, self.lat_step)
        lat_bins = np.arange(LAT_RANGE[0], LAT_RANGE[1], LAT_STEP)
        lon_bins = np.arange(LON_RANGE[0], LON_RANGE[1], LON_STEP)

        for lat_min in lat_bins:
            lat_max = lat_min + LAT_STEP

            for lon_min in lon_bins:
                lon_max = lon_min + LON_STEP

                # 현재 그리드 셀 안에 속하는 위성 추출
                cell_sats = [
                    sat for sat in self.sat_list
                    if lat_min <= sat.lat < lat_max and lon_min <= sat.lon < lon_max
                ]

                if not cell_sats:
                    print("[WARNING] NO SATELLITES")
                    continue

                for sat in cell_sats:
                    sat.current_vsg_id = vid

                for vsg in self.vsgs_list:
                    if not is_changed:
                        is_changed = vsg.satellites != set(s.id for s in cell_sats)
                    vsg.satellites = cell_sats

                if not is_changed:
                    is_changed = self.vsgs_list[vid].satellites != set(s.id for s in cell_sats)
                    self.vsgs_list[vid].satellites = cell_sats
                vid += 1

        return is_changed

    def visualized_network_constellation(self, current_time):
        """
        기존의 시각화 로직을 사용하여 현재 프레임을 그리고 Artist 객체를 저장합니다.
        """
        # 매 time step마다 축을 초기화하여 이전 프레임의 잔상을 제거합니다.
        self.ax.clear()

        # 축 제목 및 한계 설정 (매번 다시 설정)
        self.ax.set_title(f"Satellite Network Constellation (Time: {current_time:.1f}s)")
        self.ax.set_xlim([-180, 180])
        self.ax.set_ylim([-90, 90])

        # 컬러맵 생성 (VSG별 색상)
        cmap = cm.get_cmap('tab20', len(self.vsgs_list))
        vsg_colors = {vsg.id: cmap(vsg.id) for vsg in self.vsgs_list}

        # ----------------------------------------------------------------------------------
        # 0. VSG 영역 표현 (패치(Patch)는 plt.gca().add_patch()를 통해 추가되므로,
        #    이들을 직접 Artists 리스트에 추가해야 합니다.)
        vsg_artists = []
        for vsg in self.vsgs_list:
            rect = Rectangle((vsg.lon_min, vsg.lat_min), LON_STEP, LAT_STEP,
                             linewidth=0.8, edgecolor=vsg_colors[vsg.id], facecolor=vsg_colors[vsg.id],
                             alpha=0.4, zorder=0)
            self.ax.add_patch(rect)
            vsg_artists.append(rect)  # Artist 리스트에 추가

        # 1. ISL (adjacency edge) 그리기
        isl_artists = []
        for sat in self.sat_list:
            for nbr_id in sat.adj_sat_index_list:
                # nbr_id를 인덱스로 사용하기 전에 범위 확인 필요
                if nbr_id < 0 or nbr_id >= len(self.sat_list):
                    continue

                nbr_sat = self.sat_list[int(nbr_id)]  # numpy 배열 때문에 int 변환 추가
                line, = self.ax.plot([sat.lon, nbr_sat.lon], [sat.lat, nbr_sat.lat],
                                     color='gray', linewidth=1.0, alpha=0.3, zorder=1)
                isl_artists.append(line)

        # 2. VSG 영역 위성 산점도
        scatter_artists = []
        for vsg in self.vsgs_list:
            for sat in vsg.satellites:
                edge = 'black'
                lw = 0.8
                if sat.id in self.congested_sat_ids:
                    edge = 'red'
                    lw = 3.0  # 좀 더 강조

                scatter = self.ax.scatter(sat.lon, sat.lat, s=100, color=vsg_colors[vsg.id], edgecolors=edge,
                                          linewidths=lw,
                                          alpha=0.6, zorder=2)
                scatter_artists.append(scatter)

        # 3. VNF 수행 위성 강조 및 주석
        vnf_scatter_artists = []
        vnf_annotate_artists = []
        for sat in self.sat_list:
            if sat.vnf_list:
                scatter = self.ax.scatter(sat.lon, sat.lat, marker='*', s=80, color='red', edgecolors='black',
                                          linewidths=0.8,
                                          zorder=4)
                vnf_scatter_artists.append(scatter)

                annotate = self.ax.annotate(f"VNF {sat.vnf_list}", (sat.lon + 8.0, sat.lat + 3.0),
                                            fontsize=13, color='darkred', alpha=0.8, zorder=12)
                vnf_annotate_artists.append(annotate)

        # 4. 위성 인덱스 모두 표시
        annotate_artists = []
        for sat in self.sat_list:
            annotate = self.ax.annotate(str(sat.id), (sat.lon, sat.lat), fontsize=13, alpha=0.7, zorder=5)
            annotate_artists.append(annotate)
        # ----------------------------------------------------------------------------------

        # 모든 Artist를 하나의 리스트로 통합하여 저장합니다.
        # list.extend()는 반환 값이 없으므로, 모든 리스트를 + 연산자로 합칩니다.
        current_frame_artists = (vsg_artists + isl_artists + scatter_artists +
                                 vnf_scatter_artists + vnf_annotate_artists + annotate_artists)

        self.figure_list.append(current_frame_artists)

    def save_constellation_animation(self, filename="network_constellation.mp4", fps=5):
        """
        저장된 Figure 리스트를 사용하여 MP4 애니메이션을 생성하고 저장합니다.

        :param filename: 저장할 MP4 파일 이름
        :param fps: 초당 프레임 수
        """

        if not self.figure_list:
            print("[WARNING] 저장할 프레임이 없습니다.")
            return

        print(f"[{len(self.figure_list)} 프레임을 사용하여 비디오를 생성합니다...]")

        # ArtistAnimation 객체 생성
        # self.fig: 애니메이션을 만들 Figure
        # self.figure_list: 각 time step의 Artist 리스트
        # interval: 프레임 간격 (ms) -> 1000/fps
        ani = ArtistAnimation(self.fig, self.figure_list, interval=1000 / fps, blit=False, repeat=False)

        # FFMpegWriter를 사용하여 MP4 파일로 저장
        Writer = animation.FFMpegWriter(fps=fps)

        try:
            # anim.save() 호출 시 FFMPEG이 프레임을 조합하여 비디오를 만듭니다.
            ani.save(filename, writer=Writer)
            print(f"✅ 위성 토폴로지 애니메이션이 성공적으로 '{filename}'에 저장되었습니다.")
        except ValueError as e:
            print(f"❌ 비디오 저장 중 오류 발생: {e}")
            print("FFMPEG이 설치되어 있고 환경 변수에 등록되었는지 확인하세요.")

        plt.close(self.fig)  # Figure 닫기

    def extract_sat_ids(self, path):
        ids = []
        for step in path:
            if isinstance(step, (list, tuple)):
                if len(step) == 0:
                    continue
                ids.append(step[0])
            else:
                ids.append(step)
        return ids

    # 현재 위성에서 vnf 처리하는지 확인
    def has_vnf_tag(self,x):
        if x is None:
            return False
        if isinstance(x, (list, tuple)):
            return any(isinstance(e, str) and 'vnf' in e.lower() for e in x)
        if isinstance(x, str):
            return 'vnf' in x.lower()
        return False

    # 현재 위성에서 vnf 처리하는지 확인
    def has_dst_tag(self, x):
        if x is None:
            return False
        if isinstance(x, (list, tuple)):
            return any(isinstance(e, str) and 'dst' in e.lower() for e in x)
        if isinstance(x, str):
            return 'dst' in x.lower()
        return False

    def set_initial_vnf(self):
        # 전체 위성 중 80%만 미리 셋팅
        vnf_set_sats = sorted(random.sample(range(0,NUM_SATELLITES), int(NUM_SATELLITES*0.8)))
        for sat in self.sat_list:
            if sat.id in vnf_set_sats:
                # 3개 이상 탑재 (최대 개수는 넘기지 않도록)
                vnf_per_sat = random.randint(3,NUM_VNFS_PER_SAT) # 흠...
                vnfs = sorted(random.sample(range(*VNF_TYPES_PER_VSG), vnf_per_sat))
                sat.vnf_list = vnfs

    def simulation_proceeding(self, mode='dd', data_processing_rate_pair=(10, 1000), proposed=True, results_dir=None):
        if proposed is None:
            IS_PROPOSED = True
        else:
            IS_PROPOSED = proposed

        vnf_size_mode = "VSG"  # VSG: vnf 사이즈 모두 100 // 이외: 50부터 200 랜덤 지정
        sat_data_rate = data_processing_rate_pair[0]
        gs_data_rate = data_processing_rate_pair[1]
        if results_dir is None:
            if IS_PROPOSED: self.results_dir = f"./results/{NUM_GSFC}/proposed_{mode}/{sat_data_rate / 1e6}sat_{gs_data_rate / 1e6}gs/"
            else: self.results_dir = f"./results/{NUM_GSFC}/{mode}/{sat_data_rate / 1e6}sat_{gs_data_rate / 1e6}gs/"
        else:
            self.results_dir = results_dir

        ## 1. Network architecture (위성, VSG, 지상국, VNF 할당)
        # 1-1. 토폴로지 초기화
        self.set_constellation(NUM_SATELLITES, NUM_ORBITS)  # 위성 위치 초기화
        self.initial_vsg_regions()  # VSG 영역, VSG 내 위성 및 지상국 초기화\
        # TODO 1. 각 위성 당 탑재 가능 VNF 수: 3개 이상, 네트워크 내 80% 이상은 vnf를 탑재하고 있을 것
        # self.set_initial_vnf()
        self.initial_vnfs_to_vsgs()  # VSG 당 VNF 할당
        # for vsg in self.vsgs_list:
        #     print(f"vsg_id: {vsg.id}, vsg assigned_vnf: {vsg.assigned_vnfs}, satellites: {(sat.id for sat in vsg.satellites)}, satellite vnf: {(sat.vnf_list for sat in vsg.satellites)}")

        # 1-2. congestion 위성 설정
        congestion_ratio = 0.0  # 현재 congestion 고려 안 함
        self.data_drop_threshold = 1 - congestion_ratio
        self.set_congested_satellites(self.data_drop_threshold)
        self.compute_all_pair_hop_counts(self.congested_sat_ids, mode) # mode==sd -> 지상국 포함 graph 생성

        congested_sat_ids = None #self.congested_sat_ids
        exclude_congested = False

        # # constellation 확인 함수
        # self.visualized_network_constellation()

        new_gsfc_id_start = 0
        t = 0

        while True:
            print(f"\n==================== TIME TICK {t} MS ====================")
            print(f"current generated sfc num: {len(self.gsfc_list)}, current completed sfc num: {len(self.csv_gsfc_set)}")

            # gsfc 생성
            if t < NUM_ITERATIONS:
                self.generate_gsfc(NUM_GSFC, vnf_size_mode)
            # gsfc 경로 설정 (새로 생성된 gsfc)
            for gsfc in self.gsfc_list[new_gsfc_id_start:]:
                # print(f"[GSFC GENERATION] Time {t} Mode {mode}: GSFC {gsfc.id} 생성 완료. 경로 탐색 시작.")

                if mode == "basic":
                    self.set_gsfc_flow_rule(gsfc)
                    self.set_vsg_path(gsfc)
                    self.set_satellite_path(gsfc)
                    # print(f"[PATH LOG] GSFC {gsfc.id}: BASIC VSG 경로 설정 완료. Path: {gsfc.basic_satellite_path}")

                elif "noname" in mode:
                    self.set_gsfc_flow_rule(gsfc)
                    self.set_vsg_path(gsfc)
                    self.set_satellite_path_noname(gsfc, mode)
                    print(f"[GSFC GENERATION] Time {t} Mode {mode}: GSFC {gsfc.id} 생성 완료. satellite_path: {gsfc.noname_satellite_path}, processed_path: {gsfc.noname_processed_satellite_path} 경로 탐색 시작.")

                elif mode == "dd":
                    if IS_PROPOSED:
                        # proposed
                        self.proposed_find_satellite_path(gsfc)
                        # print(f"[PATH LOG] GSFC {gsfc.id}: PROPOSED DD 경로 설정 완료. Path: {gsfc.dd_satellite_path}")
                    else:
                        # DD
                        self.find_shortest_satellite_vnf_path(gsfc)  # vsg_path 없이 satellite_path 생성
                        # print(f"[PATH LOG] GSFC {gsfc.id}: DD 경로 설정 완료. Path: {gsfc.dd_satellite_path}")

                elif mode == "sd":
                    if IS_PROPOSED:
                        # proposed
                        self.advanced_proposed_find_satellite_path(gsfc)
                        # print(f"[PATH LOG] GSFC {gsfc.id}: PROPOSED SD 경로 설정 완료. Path: {gsfc.sd_satellite_path}")
                    else:
                        # SD 경로 생성
                        self.find_shortest_satellite_path_between_src_dst(gsfc)
                        # print(f"[PATH LOG] GSFC {gsfc.id}: SD 경로 설정 완료. Path: {gsfc.sd_satellite_path}")

                ## 첫 위치 설정
                satellite_path_attr = f"{mode}_satellite_path"
                current_satellite_path = getattr(gsfc, satellite_path_attr)

                first_sat_id, first_vnf = current_satellite_path[0]
                first_sat = self.sat_list[first_sat_id]
                is_vnf = self.has_vnf_tag(first_vnf)
                # VNF 여부에 따라 process queue, transmit queue에 추가
                if is_vnf:  # --(예)--> 해당 위성 processing queue에 추가 (gsfc id, vnf id, vnf size)
                    first_sat.add_to_process_queue(gsfc, first_vnf)
                    # print(f"[QUEUE LOG] GSFC {gsfc.id} -> Sat {first_sat_id}: PROC Queue 진입 (VNF {vnf_id}). Size: {gsfc.vnf_sizes[vnf_id]}.")
                else:  # --(아니오)--> 해당 위성 transmitting queue에 추가 (gsfc id, vnf size), 해당 gsfc의 transmitting 변수 True로 설정
                    # 해당 gsfc의 다음 위성으로의 경로 찾기 + isl link queue 추가
                    processed_path_attr = f"{mode}_processed_satellite_path"
                    current_processed_path = getattr(gsfc, processed_path_attr)
                    current_processed_path.append(current_satellite_path[0])
                    first_sat.add_to_transmit_queue(gsfc, mode=mode)
                new_gsfc_id_start += 1

            # 종료 시점 파악 #
            # 모든 gsfc가 success or dropped이 될 때까지 #
            all_completed = True
            if self.gsfc_list:
                for gsfc in self.gsfc_list:
                    succeed_attr = f"{mode}_succeed"
                    dropped_attr = f"{mode}_dropped"

                    if not getattr(gsfc, succeed_attr, False) and not getattr(gsfc, dropped_attr, False):
                        all_completed = False
                        break

            if all_completed:  # GSFC 생성이 끝난 후 (t > 40), 모든 GSFC가 완료되면 루프 종료
                print(f"\n*** 모든 GSFC가 succeed 또는 dropped 상태로 완료되었습니다. 시뮬레이션을 종료합니다. *** {all_completed}")
                input("\n3. Press Enter to continue the simulation...\n")
                break

            # gsfc satellite path 연장 #
            for gsfc in self.gsfc_list:
                succeed_attr = f"{mode}_succeed"
                dropped_attr = f"{mode}_dropped"

                if getattr(gsfc, succeed_attr, False) or getattr(gsfc, dropped_attr, False):
                    continue

                if gsfc.sfc_type_idx == 1:
                    continue
                else:
                    remain_path = gsfc.get_remain_path(mode=mode)
                    if len(remain_path) < 1:
                        cur_sat_id = gsfc.noname_processed_satellite_path[-1][0]
                        cur_sat = self.sat_list[cur_sat_id]

                        # print(f"[PATH LOG] time {t} GSFC {gsfc.id} on Node {cur_sat.id} in VSG {cur_sat.current_vsg_id}: Destination reached. Success.")

                        self.set_satellite_path_noname(gsfc, mode)
                        cur_sat.add_to_transmit_queue(gsfc, mode=mode)

            # TODO 3. mMTC SFC 처리 로직 별도로 필요. 매 time마다 dst_vsg에 도착했는지 확인 --(YES)-->vnf도 처리 완료? --(YES)-->succeed. --(NO)-->지상국으로 내려서 처리.
            # TODO 3. reassign을 통해 mMTC SFC가 있는 위성이 vnf 처리 가능 상태가 되면 처리하기.
            for gsfc in self.gsfc_list:
                succeed_attr = f"{mode}_succeed"
                dropped_attr = f"{mode}_dropped"

                is_success = getattr(gsfc, succeed_attr, False)
                is_dropped = getattr(gsfc, dropped_attr, False)

                if is_success or is_dropped:
                    continue
                else:
                    if gsfc.is_keeping == True:
                        # dst vsg에 도착했는지 확인
                        satellite_path_attr = f"{mode}_satellite_path"
                        satellite_path = getattr(gsfc, satellite_path_attr, [])
                        cur_sat_id = satellite_path[0][0]
                        cur_sat = self.sat_list[cur_sat_id]

                        if cur_sat.current_vsg_id == gsfc.dst_vsg_id:
                            # 지상국 경로 추가하고 지상국으로의 TSL link queue 넣기
                            remain_path = gsfc.get_remain_path(mode=mode)

                            processed_satellite_path_attr = f"{mode}_processed_satellite_path"
                            processed_satellite_path = getattr(gsfc, processed_satellite_path_attr, [])
                            processed_satellite_path.append(remain_path[0])

                            cur_vsg = self.vsgs_list[cur_sat.current_vsg_id]
                            gserver = cur_vsg.gserver
                            gserver_id = gserver.id + NUM_SATELLITES
                            gsfc.gserver = gserver

                            if len(remain_path) >= 2:
                                processed_satellite_path_attr = f"{mode}_processed_satellite_path"
                                processed_satellite_path = getattr(gsfc, processed_satellite_path_attr, [])
                                remain_id = len(processed_satellite_path)

                                satellite_path.insert(remain_id, [gserver_id, 'vnf2'])
                            cur_sat.add_to_transmit_queue(gsfc, mode=mode)

                            gsfc.is_keeping = False

                        else: # dst vsg 도착 X -> 현재 위성 상태가 가능한 상태인지 확인
                            cur_vnf = satellite_path[1][1]
                            cur_sat.add_to_process_queue(gsfc, cur_vnf)

            ## processing 로직
            # processing queue가 있는 gserver 리스트 추출
            for gserver in self.gserver_list:
                if gserver.process_queue:
                    if (mode == "dd") or (mode == "basic"): #호옥쉬나~
                        print(f"[ERROR] what??????? gserver activate in dd?????")
                        return

                    # print(f"[PROCESSING START] Time {t}: {gserver.id} 지상국에서 VNF 처리 시작.")
                    gserver.process_vnfs(self.gsfc_list, mode=mode, processing_rate=gs_data_rate)

            # processing queue가 있는 위성 리스트 추출 - # Option 1. 위성마다 한 번에 실행할 수 있는 vnf 개수 설정
            for sat in self.sat_list:
                if sat.process_queue:
                    # print(f"[PROCESSING START] Time {t}: {sat.id} 위성에서 VNF 처리 시작.")
                    sat.process_vnfs(self.gsfc_list, mode=mode, processing_rate=sat_data_rate)

            ## tx, prop 로직
            for sat in self.sat_list:
                if sat.queue_TSL:
                    # print(f"[TRANSMISSION START] Time {t}: {sat.id} 위성에서 TSL 전송 시작.")
                    transmit_completed_gsfc_ids = sat.transmit_TSL_gsfcs(self.gsfc_list, mode=mode)

                    for gsfc_id in transmit_completed_gsfc_ids: # gserver로 전송 완료 --> gserver의 process queue에 넣음
                        gsfc = self.gsfc_list[gsfc_id]
                        gserver = gsfc.gserver
                        # print(f"[TRANS COMPLETE] Sat {sat.id} to gserver {gserver.id}: GSFC {gsfc.id} 전송 완료. Handover 시작.")

                        gserver.add_to_process_queue(gsfc.id, gsfc.num_completed_vnf, gsfc.vnf_sizes[gsfc.num_completed_vnf])

            for gserver in self.gserver_list:
                if gserver.queue_TSL:
                    transmit_completed_gsfc_ids = gserver.transmit_TSL_gsfcs(self.gsfc_list, self.sat_list, mode=mode) # 다시 돌아옴 -> sd_satellite_path 대로 전송

                    for gsfc_id in transmit_completed_gsfc_ids:
                        gsfc = self.gsfc_list[gsfc_id]
                        remain_path = gsfc.get_remain_path(mode=mode)

                        if remain_path == []:
                            satellite_path_attr = f"{mode}_satellite_path"
                            satellite_path = getattr(gsfc, satellite_path_attr, [])

                            was_dst = self.has_dst_tag(satellite_path[-1][1])
                            if was_dst:
                                setattr(gsfc, f"{mode}_succeed", True)
                                # print(f"[PATH LOG] GSFC {gsfc.id} on Sat {self.id}: Destination reached. Success.")

                        sat_id = remain_path[0][0]
                        next_sat = self.sat_list[sat_id]
                        # print(f"[TRANS COMPLETE] Gserver {gserver.id} to sat {sat_id}: GSFC {gsfc.id} 전송 완료. Handover 시작.")

                        next_function = remain_path[0][1]
                        is_vnf = self.has_vnf_tag(next_function)

                        if is_vnf:
                            next_sat.add_to_process_queue(gsfc, next_function)
                        else:  # --(아니오)--> 다음 위성 transmitting queue에 추가
                            processed_path_attr = f"{mode}_processed_satellite_path"
                            current_processed_path = getattr(gsfc, processed_path_attr)
                            current_processed_path.append(remain_path[0])

                            next_sat.add_to_transmit_queue(gsfc, mode=mode)

            # TODO. sat list에서 sat id가 오름차순이면 한번에 경로가 처리됨
            for sat in self.sat_list:
                if any(q for q in sat.queue_ISL):
                    transmit_completed_gsfc_ids = sat.transmit_gsfcs(self.gsfc_list, mode=mode)

                    # 다음 vnf 여부 확인
                    for gsfc_id in transmit_completed_gsfc_ids:
                        gsfc = self.gsfc_list[gsfc_id]
                        # print(f"[TRANS COMPLETE] Sat {sat.id}: GSFC {gsfc.id} 전송 완료. Handover 시작.")

                        remain_path = gsfc.get_remain_path(mode=mode)
                        if remain_path == []:
                            satellite_path_attr = f"{mode}_satellite_path"
                            satellite_path = getattr(gsfc, satellite_path_attr, [])

                            was_dst = self.has_dst_tag(satellite_path[-1][1])
                            if was_dst:
                                setattr(gsfc, f"{mode}_succeed", True)
                                # print(f"[PATH LOG] GSFC {gsfc.id} on Sat {self.id}: Destination reached. Success.")

                        next_sat_id = remain_path[0][0]
                        next_sat = self.sat_list[next_sat_id]
                        next_function = remain_path[0][1] if len(remain_path) > 1 else None
                        is_vnf = self.has_vnf_tag(next_function)

                        if is_vnf:
                            vnf_id = gsfc.num_completed_vnf  # vnf 종류가 아닌, 현 gsfc에서 실행되는 vnf 순서
                            if vnf_id >= len(gsfc.vnf_sizes):
                                print(f"[WARNING] gsfc_id: {gsfc.id}, vsg_path: {self.vsg_path[gsfc.id]} gsfc_satellite_path: {gsfc.noname_satellite_path} gsfc_processed: {gsfc.noname_processed_satellite_path}")

                            next_sat.add_to_process_queue(gsfc, next_function)
                        else:  # --(아니오)--> 다음 위성 transmitting queue에 추가
                            processed_path_attr = f"{mode}_processed_satellite_path"
                            current_processed_path = getattr(gsfc, processed_path_attr)
                            current_processed_path.append(remain_path[0])

                            next_sat.add_to_transmit_queue(gsfc, mode=mode)

            # 위성 이동
            for sat in self.sat_list:
                sat.time_tic(t)

            # self.check_vsg_regions() VSG 구역 재설정
            vid = 0
            is_changed = False

            lat_bins = np.arange(LAT_RANGE[0], LAT_RANGE[1], LAT_STEP)
            lon_bins = np.arange(LON_RANGE[0], LON_RANGE[1], LON_STEP)
            # print(f"DEBUG: lat_bins size: {len(lat_bins)}, lon_bins size: {len(lon_bins)}")

            for lat_min in lat_bins:
                lat_max = lat_min + LAT_STEP
                for lon_min in lon_bins:
                    lon_max = lon_min + LON_STEP

                    # 현재 그리드 셀 안에 속하는 위성 추출
                    cell_sats = [
                        sat for sat in self.sat_list
                        if lat_min <= sat.lat < lat_max and lon_min <= sat.lon < lon_max
                    ]

                    if not cell_sats:
                        print("[WARNING] NO SATELLITES")
                        continue

                    for sat in cell_sats:
                        sat.current_vsg_id = vid

                    if not is_changed:
                        is_changed = self.vsgs_list[vid].satellites != cell_sats
                        if is_changed:
                            print(f"CHANGE TOPOLOGY in VSG {vid}: time {t} before {[sat.id for sat in self.vsgs_list[vid].satellites]}, after {[sat.id for sat in cell_sats]}")
                    self.vsgs_list[vid].satellites = cell_sats
                    vid += 1

            # vnf consistency유지
            if is_changed:
                inconsistent_vsgs = {} #key: vsg, value: vnf_type
                for vsg in self.vsgs_list:
                    inconsistent_vsgs[vsg] = []
                    found_vnfs = set()
                    for sat in vsg.satellites:
                        if sat.vnf_list:
                            for vnf in sat.vnf_list:
                                found_vnfs.add(vnf)

                    # 한 개라도 없는 VNF가 있다면 재할당 필요
                    for vnf in vsg.assigned_vnfs:
                        if vnf not in found_vnfs:
                            print(f"[ALERT] time {t} VSG {vsg.id} lost VNF {vnf}. Reassigning VNFs in VSG {vsg.id}.")
                            inconsistent_vsgs[vsg].append(vnf)
                            break
                    if inconsistent_vsgs[vsg]:
                        print(f"[REASSIGN] VSG {vsg.id}, assigned vnfs: {vsg.assigned_vnfs}, ")
                        # TODO 2. 해당 VSG 내 위성들에 대해서, 가장 적은 load를 가지고 있는 VNF 종류 및 해당 위성 id 찾기, 더 이상 처리 못하게 된 sfc는 다른 위성으로 포워딩
                        self.reassign_vnfs_to_satellite(vsg)
                        print(f"[REASSIGN] time {t} VSG {vsg.id}, satellite with vnf {(sat.id for sat in vsg.satellites if sat.vnf_list is not [])}")

            self.visualized_network_constellation(t)

            # 결과 저장 및 그래프
            path_filename = "path_results_per_time.csv"
            self.write_results_csv(t, path_filename, mode, IS_PROPOSED)

            satellite_status_filename = "satellite_status_per_time.csv"
            self.write_satellite_status_csv(t, satellite_status_filename)

            t+=1

        # self.save_constellation_animation(filename="simulation_video.mp4", fps=10)

        success_filename = "success_results.csv"
        self.write_success_results_csv(success_filename, mode)
        plot_each_mean_e2e(self.results_dir, sat_data_rate, gs_data_rate, mode,
                           success_filename, out_png_filename="mean_e2e.png")
        plot_each_mean_stack_e2e_segment(self.results_dir, sat_data_rate, gs_data_rate, mode,
                           success_filename, out_png_filename="mean_e2e.png")