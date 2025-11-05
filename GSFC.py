import random
from Params import *

class GSFC:
    def __init__(self, gsfc_id, src_vsg_id, dst_vsg_id, vnf_sequence, vnf_to_vsg, mode="VSG"):
        self.id = gsfc_id
        self.vnf_sequence = vnf_sequence
        self.vnf_to_vsg = vnf_to_vsg # 각 VNF가 포함된 VSG
        self.src_vsg_id = src_vsg_id
        self.dst_vsg_id = dst_vsg_id
        self.vnf_sizes = [] #value: total vnf sizes [bit]
        self.gserver = None

        # ~~ 구현 변수
        self.current_essential_path_id = 0 # 현재 vsg path를 만드는데 사용한 essential vsg idx
        self.noname_cur_vsg_path_id = 0 # 현재 satellite_path를 만드는데 사용한 vsg_idx
        self.noname_satellite_path = []
        self.noname_succeed = False
        self.noname_dropped = False
        self.noname_processed_satellite_path = []
        self.noname_is_transmitting = False
        self.noname_hop_count = 0
        self.noname_prop_delay_ms = 0
        self.noname_proc_delay_ms = 0
        self.noname_trans_delay_ms = 0
        self.noname_queue_delay_ms = 0
        self.noname_e2e_delay_ms = 0


        # VSG 논문 구현 변수
        self.basic_satellite_path = []
        self.basic_succeed = False
        self.basic_dropped = False
        self.basic_processed_satellite_path = []
        self.basic_is_transmitting = False
        self.basic_hop_count = 0
        self.basic_prop_delay_ms = 0
        self.basic_proc_delay_ms = 0
        self.basic_trans_delay_ms = 0
        self.basic_queue_delay_ms = 0
        self.basic_e2e_delay_ms = 0

        # DD, DD+DBPR 변수
        self.dd_satellite_path = []
        self.dd_processed_satellite_path = []
        self.num_completed_vnf = 0
        self.completed_vnfs_size = []  # value: processed vnf sizes [bit]
        self.dd_is_transmitting = False
        self.dd_succeed = False
        self.dd_dropped = False
        self.dd_hop_count = 0
        self.dd_proc_delay_ms = 0
        self.dd_queue_delay_ms = 0
        self.dd_trans_delay_ms = 0
        self.dd_prop_delay_ms = 0
        self.dd_e2e_delay_ms = 0

        # SD, SD+DBPR 변수
        self.sd_satellite_path = []
        self.sd_processed_satellite_path = []
        self.sd_is_transmitting = False
        self.sd_succeed = False
        self.sd_dropped = False
        self.sd_hop_count = 0
        self.sd_prop_delay_ms = 0
        self.sd_proc_delay_ms = 0
        self.sd_trans_delay_ms = 0
        self.sd_queue_delay_ms = 0
        self.sd_e2e_delay_ms = 0

        self.sd_total_vnf_size = 0
        self.sd_completed_sfc_size = 0 # [bit]

        if mode == "VSG":
            for vnf in self.vnf_sequence:
                self.vnf_sizes.append(VNF_SIZE)
                self.completed_vnfs_size.append(0)
        else:
            for vnf in self.vnf_sequence:
                self.vnf_sizes.append(random.randint(50*8,200*8))
                self.completed_vnfs_size.append(0)

    def get_remain_path(self, mode='dd'):
        def _sat_ids(path):
            """[sat_id, meta] 또는 sat_id 를 sat_id 리스트로 정규화"""
            ids = []
            for step in path:
                if isinstance(step, (list, tuple)):
                    if len(step) == 0:
                        continue
                    ids.append(step[0])
                else:
                    ids.append(step)
            return ids

        total_path = getattr(self, f"{mode}_satellite_path")
        processed_path = getattr(self, f"{mode}_processed_satellite_path")
        cur_ids = _sat_ids(total_path)
        processed_ids = _sat_ids(processed_path)
        idx = 0
        limit = min(len(cur_ids), len(processed_ids))
        while idx < limit and cur_ids[idx] == processed_ids[idx]:
            idx += 1

        remain_sat_path = total_path[idx:]
        return remain_sat_path

    def accumulate_queue_delay(self, mode='dd'):
        """Queueing Delay [ms] = Queue Buffer [bit] / (Data Rate [bit/ms] * Time Slot [ms])"""

        delay_attr = f"{mode}_queue_delay_ms"
        current_delay = getattr(self, delay_attr)
        setattr(self, delay_attr, current_delay + 1)

        # print(f"[DELAY LOG] MODE {mode} GSFC {self.id} | Q Delay +1ms. Total {mode} Queue Delay: {current_delay+1} ms")

    def accumulate_trans_delay(self, mode='dd'):
        """Transmission Delay [ms] = Packet Length [bit] / Data Rate [bit/ms]"""
        # # 전송 속도 상수는 SAT_DATA_RATE_BIT_PER_MS를 사용한다고 가정
        # # (위성-위성, 위성-지상국 모두 동일한 링크 속도를 사용한다고 가정)
        #
        # # 0으로 나누는 오류 방지
        # SAT_DATA_RATE_BIT_PER_MS = 25 * 1e6
        # if SAT_DATA_RATE_BIT_PER_MS == 0:
        #     return float('inf')
        #
        # # (packet_length_bits / SAT_DATA_RATE_BIT_PER_MS)는 [ms] 단위
        # return packet_length_bits / SAT_DATA_RATE_BIT_PER_MS


        delay_attr = f"{mode}_trans_delay_ms"
        current_delay = getattr(self, delay_attr)
        setattr(self, delay_attr, current_delay + 1)

        # print(f"[TRANS DELAY LOG] GSFC {self.id} | Trans Delay +1ms. Total {mode} Trans Delay: {current_delay+1} ms")

    def processing_vnf(self, processing_power, mode='dd'):
        """
        :param processing_power [bits/ms]
        """
        if self.num_completed_vnf >= len(self.vnf_sequence):
            print("[WARNING] All VNFs already processed.")
            return False

        vnf_total_size = self.vnf_sizes[self.num_completed_vnf]
        vnf_completed_size = self.completed_vnfs_size[self.num_completed_vnf]

        vnf_remaining_size = vnf_total_size - vnf_completed_size
        processed = min(processing_power, vnf_remaining_size)

        self.completed_vnfs_size[self.num_completed_vnf] += processed

        # option 1. 1ms 틱 전체를 처리 지연으로 간주
        delay_attr = f"{mode}_proc_delay_ms"
        current_delay = getattr(self, delay_attr)
        setattr(self, delay_attr, current_delay + 1)

        # print(f"[PROC LOG] MODE {mode} GSFC {self.id} | VNF {self.num_completed_vnf} processed {processed}. Total Proc Delay: {current_delay + 1} ms. Remaining: {vnf_remaining_size - processed}")

        # # option 2. 처리된 양에 비례하여 지연 시간 계산
        # self.dd_proc_delay_ms += (processed / processing_power) # *1ms
        # self.sd_proc_delay_ms += (processed / processing_power)  # *1ms

        if vnf_remaining_size - processed <= 0: # 현 위치에서 vnf 처리 완료
            remain = self.get_remain_path(mode=mode)
            if remain:
                processed_path_attr = f"{mode}_processed_satellite_path"
                current_processed_path = getattr(self, processed_path_attr)
                current_processed_path.append(remain[0])
            else:
                setattr(self, f"{mode}_succed", True)

            self.num_completed_vnf += 1
            # print(f"[PROC LOG] GSFC {self.id} | VNF {self.num_completed_vnf - 1} COMPLETED. Next VNF: {self.num_completed_vnf}")
            return True
        else:
            return False

    def processing_sfc(self, processing_power, mode='sd'):
        """
        :param processing_power [bits/ms]
        """

        total_size = getattr(self, f"{mode}_total_vnf_size")
        sfc_completed_size = getattr(self, f"{mode}_completed_sfc_size")

        sfc_remaining_size = total_size - sfc_completed_size
        processed = min(processing_power, sfc_remaining_size)

        setattr(self, f"{mode}_completed_sfc_size", sfc_completed_size+processed)

        # option 1. 1ms 틱 전체를 처리 지연으로 간주
        delay_attr = f"{mode}_proc_delay_ms"
        current_delay = getattr(self, delay_attr)
        setattr(self, delay_attr, current_delay + 1)

        # print(f"[PROC LOG] MODE {mode} GSFC {self.id} | processed {processed}. Total Proc Delay: {current_delay+1} ms. Remaining: {sfc_remaining_size - processed}")

        # # option 2. 처리된 양에 비례하여 지연 시간 계산
        # self.dd_proc_delay_ms += (processed / processing_power) # *1ms
        # self.sd_proc_delay_ms += (processed / processing_power)  # *1ms

        if sfc_remaining_size - processed <= 0: # 현 위치에서 vnf 처리 완료
            # print(f"[PROC LOG] {mode} - GSFC {self.id} COMPLETED.")
            return True
        else:
            return False
