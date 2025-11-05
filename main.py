# from Simulation import *
from Simulation import *
from tqdm import tqdm
from Plot import *

class Main:
    # ETRI VSG SOTA paper 구현 및 비교용
    scenario_num = 0 # 0: gif 출력, 1: congestion 위성 설정, 2: packet drop rate, 3: 제안하는 방식과의 비교 5: 성능 그래프 출력

    data_rate_pairs = [ # [sat, gs] 단위 bps
        # (200*8, 2000*8)
        # (10e6, 50e6),
        # (40e6, 160e6), # A 4배
        # # (40e6, 400e6),  # B 10배
        # (80e6, 320e6),  # C 10배
        # (160e6, 640e6),  # D 10배
        (320e6, 1280e6)
        # (20 * 8, 500 * 8),  # E 25배
        # (10 * 8, 1000 * 8),  # F 100배
    ]

    # data_rate_pairs = [(100, 200)]

    # TODO. data rate이 오를 수록 시간이 더 오래 걸림
    # TODO. proposed dd가 더 오래 걸림

    modes = ['noname'] # dd, sd, base
    proposed_list = [True, False]

    csv_path_basic = None
    csv_path_dd = None
    csv_path_proposed_dd = None

    for pair in tqdm(data_rate_pairs):
        for mode in modes: # basic일 땐 proposed_list 안 돌게
            if mode != 'dd':
                simulation = Simulation()
                simulation.simulation_proceeding(mode, pair, proposed=False)
                csv_path_basic = f"./results/{NUM_GSFC}/{mode}/{pair[0] / 1e6}sat_{pair[1] / 1e6}gs/success_results.csv"
            else:
                for proposed in proposed_list:
                    simulation = Simulation()
                    simulation.simulation_proceeding(mode, pair, proposed)

                    if proposed:
                        csv_path_proposed_dd = f"./results/{NUM_GSFC}/proposed_{mode}/{pair[0] / 1e6}sat_{pair[1] / 1e6}gs/success_results.csv"
                    else:
                        csv_path_dd = f"./results/{NUM_GSFC}/{mode}/{pair[0] / 1e6}sat_{pair[1] / 1e6}gs/success_results.csv"


        # plot_mean_e2e_stack_for_pair(pair[0], pair[1],
        #                              csv_paths=[
        #                                  csv_path_basic,
        #                                  csv_path_dd,
        #                                  csv_path_proposed_dd,
        #                              ],
        #                              labels=["BASIC", "DD", "PROPOSED DD"],
        #                              results_root=f"./results/{NUM_GSFC}/plot/{pair[0] / 1e6}sat_{pair[1] / 1e6}gs/",
        #                              out_png_filename="mean_stack_e2e.png"
        #                              )

        plot_mean_e2e_stack_for_pair(pair[0], pair[1],
                                     csv_paths=[f"./results/{NUM_GSFC}/noname/{pair[0] / 1e6}sat_{pair[1] / 1e6}gs/success_results.csv"],
                                     labels=["noname"],
                                     results_root=f"./results/{NUM_GSFC}/plot/{pair[0] / 1e6}sat_{pair[1] / 1e6}gs/",
                                     out_png_filename="mean_stack_e2e.png"
                                     )

    # plot_all_mean_stack_e2e_segment_in_mode(data_rate_pairs, 'dd', results_root = f"./results/{NUM_GSFC}/dd/")
    # plot_all_mean_stack_e2e_segment_in_mode(data_rate_pairs, 'sd', results_root = f"./results/{NUM_GSFC}/sd/")
