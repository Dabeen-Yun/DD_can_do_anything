from Params import *

import numpy as np
import networkx as nx

class VSG:
    def __init__(self, id, center_coords, lon_min, lat_min, satellites, gserver):
        self.id = id
        self.assigned_vnfs = []
        self.satellites = satellites
        self.center_coords = center_coords # lon, lat
        self.lon_min = lon_min
        self.lat_min = lat_min
        self.gserver = gserver

        self.satellite_vnf_resolution_table = {} #key: "VNF type", value: satellite_idx
        # self.update_satellite_vnf_resolution_table()
        # print("satellite_to_vnf", self.satellite_to_vnf)

        self.vnf_duration_table = {}
        # self.update_vnf_duration_table() # {vnf_type (str): countdown_time (int)}
        # print("duration", self.vnf_duration_table)

    def update_satellite_vnf_resolution_table(self):
        self.satellite_vnf_resolution_table = {}
        for vnf in self.assigned_vnfs: # vnf-> int
            self.satellite_vnf_resolution_table[vnf] = []

        for sat in self.satellites:
            if sat.vnf_list in self.assigned_vnfs:
                self.satellite_vnf_resolution_table[sat.vnf_list].append(sat.id)

    # countdown 계산 함수 (위성 위도와 VSG 중심 위도 차이 → 남은 거리 계산)
    def compute_duration_seconds(self, sat_lat, vsg_lat):
        delta_km = abs(sat_lat - vsg_lat) * DEGREE_TO_KM
        duration_sec = delta_km / SATELLITE_SPEED
        # return int(np.clip(duration_sec, 1, 300))  # 1초 이상, 5분 이하로 클립
        return duration_sec

    def update_vnf_duration_table(self, alpha=0.5):
        # print(f"vsg: {self.id} ===================before===============> {self.vnf_duration_table}")
        self.vnf_duration_table = {}
        #
        # # 1. geo-consistency가 유지되고 있는지 확인
        # missing_vnfs = [vnf for vnf in self.assigned_vnfs
        #                 if vnf not in self.satellite_to_vnf or not self.satellite_to_vnf[vnf]]
        # if missing_vnfs:
        #     print(f"[WARNING] VSG {self.id} geo-consistency broken! Missing VNFs: {missing_vnfs}")
        #     for vnf in self.assigned_vnfs:
        #         if vnf not in missing_vnfs:
        #             continue
        #
        #         candidate_sats = [
        #             sat for sat in self.satellites
        #             if sat.vnf_list not in str(self.assigned_vnfs)
        #         ]
        #
        #         max_time = max([sat.vsg_enter_time for sat in candidate_sats], default=1e-6)
        #         max_drop_rate = max([sat.drop_rate for sat in candidate_sats], default=1e-6)
        #
        #         best_sat = None
        #         best_efficiency = -1
        #
        #         for sat in candidate_sats:
        #             norm_time = sat.vsg_enter_time / max_time
        #             norm_drop_rate = sat.drop_rate / max_drop_rate
        #             efficiency = alpha * norm_time + (1-alpha) * (1 - norm_drop_rate)
        #
        #             if efficiency > best_efficiency:
        #                 best_efficiency = efficiency
        #                 best_sat = sat
        #
        #         if best_sat:
        #             best_sat.vnf_list = str(vnf)
        #             self.update_satellite_vnf_resolution_table()
        #
        #     for vnf in missing_vnfs:
        #         # TODO: geo-consistency가 깨졌을 때 대처
        #         self.vnf_duration_table[vnf] = 0  # 또는 -1 등으로 처리
        #     return

        # 2. duration countdown 계산
        # print(f"vsg: {self.id} assigned===============> {self.assigned_vnfs}")
        for vnf in self.assigned_vnfs:
            max_duration = 0
            for sat_id in self.satellite_vnf_resolution_table[vnf]:
                sat = next((s for s in self.satellites if s.id == sat_id), None)
                if sat is None:
                    continue

                duration = self.compute_duration_seconds(sat.lat, self.center_coords[1])

                # print("duration", sat_id, duration)
                max_duration = max(max_duration, duration)

            self.vnf_duration_table[vnf] = max_duration

        # print(f"vsg: {self.id} ===================after===============> {self.vnf_duration_table}")

    def build_local_graph(self):
        G = nx.Graph()
        for sat in self.satellites:
            for nbr_id in sat.adj_sat_index_list:
                if nbr_id == -1:
                    continue
                nbr_sat = next((s for s in self.satellites if s.id == nbr_id), None)
                if nbr_sat:
                    G.add_edge(sat.id, nbr_id, weight=1)
        return G
