import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def ensure_results_dir(results_root):
    os.makedirs(results_root, exist_ok=True)

def mean_e2e_segment_from_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] failed to read {csv_path}: {e}")
        return None

    if ("propagation_delay_ms" or "processing_delay_ms" or "queueing_delay_ms" or "transmission_delay_ms") not in df.columns:
        print(f"[WARN] column e2e semgent not in {csv_path.name}")
        return None

    prop_vals = pd.to_numeric(df["propagation_delay_ms"], errors="coerce").dropna()
    proc_vals = pd.to_numeric(df["processing_delay_ms"], errors="coerce").dropna()
    queue_vals = pd.to_numeric(df["queueing_delay_ms"], errors="coerce").dropna()
    trans_vals = pd.to_numeric(df["transmission_delay_ms"], errors="coerce").dropna()

    if len(prop_vals) == 0:
        print(f"[WARN] no numeric values in 'propagation_delay_ms' for {csv_path.name}")
        return None
    if len(proc_vals) == 0:
        print(f"[WARN] no numeric values in 'processing_delay_ms' for {csv_path.name}")
        return None
    if len(queue_vals) == 0:
        print(f"[WARN] no numeric values in 'queueing_delay_ms' for {csv_path.name}")
        return None
    if len(trans_vals) == 0:
        print(f"[WARN] no numeric values in 'transmission_delay_ms' for {csv_path.name}")
        return None

    return float(prop_vals.mean()), float(proc_vals.mean()), float(queue_vals.mean()), float(trans_vals.mean())

def mean_e2e_from_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] failed to read {csv_path}: {e}")
        return None

    if "e2e_delay_ms" not in df.columns:
        print(f"[WARN] column 'e2e_delay_ms' not in {csv_path.name}")
        return None

    vals = pd.to_numeric(df["e2e_delay_ms"], errors="coerce").dropna()
    if len(vals) == 0:
        print(f"[WARN] no numeric values in 'e2e_delay_ms' for {csv_path.name}")
        return None
    return float(vals.mean())

def plot_all_mean_stack_e2e_segment_in_mode(data_rate_pairs, mode,
                                            results_root = "./results/plot/",
                                            out_png_filename="mean_e2e_stack_in_mode.png"):
    """
    특정 mode에서 sat, gs data rate 별 delay segment 비교
    """
    # Prepare data for plotting (skip None values gracefully)
    labels = []
    prop_vals = []
    proc_vals = []
    queue_vals = []
    trans_vals = []

    for sat_rate, gs_rate in data_rate_pairs:
        csv_path = f"results/{mode}/{sat_rate / 1e6}sat_{gs_rate / 1e6}gs/success_results.csv"

        means = mean_e2e_segment_from_csv(csv_path)

        if means is None:
            print(f"[WARN] Skipping data rate pair ({sat_rate}/{gs_rate}) for mode {mode.upper()} - Data not found or invalid.")
            continue

        mean_prop, mean_proc, mean_queue, mean_trans = means

        # 플롯 데이터 리스트에 추가
        labels.append(f"SAT:{sat_rate/1e6}Mbps/GS:{gs_rate/1e6}Mbps")
        prop_vals.append(0.0 if mean_prop is None else mean_prop)
        proc_vals.append(0.0 if mean_proc is None else mean_proc)
        queue_vals.append(0.0 if mean_queue is None else mean_queue)
        trans_vals.append(0.0 if mean_trans is None else mean_trans)

    if not labels:
        print(f"[ERROR] No valid data found for mode {mode.upper()} across all data rate pairs")
        return None

    # 2. 플로팅 준비
    x = np.arange(len(labels))  # X-좌표 설정

    # 스택 막대의 바닥(bottom) 위치 계산 (누적 합)
    bottom_prop = [0.0] * len(labels)
    bottom_proc = prop_vals
    bottom_queue = [p + c for p, c in zip(prop_vals, proc_vals)]
    bottom_trans = [b_q + q for b_q, q in zip(bottom_queue, queue_vals)]

    plt.figure(figsize=(10, 5))
    width = 0.8  # 막대 너비

    # 스택 막대 그리기
    bars_prop = plt.bar(x, prop_vals, width, label="Propagation", bottom=bottom_prop)
    bars_proc = plt.bar(x, proc_vals, width, bottom=bottom_proc, label="Processing")
    bars_queue = plt.bar(x, queue_vals, width, bottom=bottom_queue, label="Queueing")
    bars_trans = plt.bar(x, trans_vals, width, bottom=bottom_trans, label="Transmitting")

    # 3. 주석 (Annotation) 추가
    def autolabel_stack_simple(bars, segment_vals, bottom_start_vals):
        for bar, val, bottom_start in zip(bars, segment_vals, bottom_start_vals):
            if val > 0:
                y_center = bottom_start + val / 2.0
                plt.text(bar.get_x() + bar.get_width() / 2.0, y_center,
                         f"{val:.1f}", ha="center", va="center", fontsize=8)

    autolabel_stack_simple(bars_prop, prop_vals, bottom_prop)
    autolabel_stack_simple(bars_proc, proc_vals, bottom_proc)
    autolabel_stack_simple(bars_queue, queue_vals, bottom_queue)
    autolabel_stack_simple(bars_trans, trans_vals, bottom_trans)

    # 4. 플롯 마무리
    plt.xticks(x, labels, rotation=20, ha='right')  # 레이블 회전
    plt.ylabel("Mean delay (ms)")
    plt.title(f"Mean E2E Delay Segments for {mode.upper()} Mode - Data Rate Comparison")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    out_path = results_root + out_png_filename.format(mode=mode)
    ensure_results_dir(results_root)
    try:
        plt.savefig(out_path, dpi=150)
        print(f"[OK] saved: {out_path}")
    finally:
        plt.close()

    return out_path

def plot_mean_e2e_for_pair(sat_rate, gs_rate, dd_csv_path, sd_csv_path,
                           results_root = "./results/plot/", out_png_filename = "mean_e2e.png"):
    """
    end-to-end delay만으로 plot 형성
    각 data rate pair 쌍마다 dd mode와 sd mode의 E2E delay 그래프를 그려 수치 비교
    """

    mean_dd = mean_e2e_from_csv(dd_csv_path)
    mean_sd = mean_e2e_from_csv(sd_csv_path)

    # Prepare data for plotting (skip None values gracefully)
    labels = []
    values = []
    if mean_dd is not None:
        labels.append("DD")
        values.append(mean_dd)
    if mean_sd is not None:
        labels.append("SD")
        values.append(mean_sd)

    if not labels:
        print(f"[ERROR] no valid data to plot in {results_root}")
        return None

    # Plot: single bar chart (per tool requirements: 1 plot, no style/colors)
    plt.figure(figsize=(6, 4))
    x = range(len(labels))
    plt.bar(x, values)
    plt.xticks(list(x), labels, rotation=0)
    plt.ylabel("Mean E2E delay (ms)")
    plt.title(f"Mean E2E Delay — sat {sat_rate/1e6}Mbps / gs {gs_rate/1e6}Mbps")
    plt.tight_layout()

    out_path = results_root + out_png_filename
    ensure_results_dir(results_root)
    try:
        plt.savefig(out_path, dpi=150)
        print(f"[OK] saved: {out_path}")
    finally:
        plt.close()

    return out_path

def plot_mean_e2e_stack_for_pair(sat_rate, gs_rate, csv_paths, labels,
                           results_root = "./results/plot/", out_png_filename = "mean_stack_e2e.png"):
    """
    E2E delay의 구성요소를 각각 표시하여 plot 형성
    각 data rate pair 쌍마다 dd mode와 sd mode의 E2E delay segment 그래프를 그려 수치 비교
    """

    if len(csv_paths) != len(labels):
        print("[ERROR] csv_paths와 labels의 개수가 일치해야 합니다.")
        return None

    # 데이터 수집용 리스트 초기화
    prop_vals = []
    proc_vals = []
    queue_vals = []
    trans_vals = []

    valid_labels = []

    # 1. 모든 CSV 파일을 순회하며 데이터 추출
    for i, csv_path in enumerate(csv_paths):
        # mean_e2e_segment_from_csv 함수를 사용하여 각 지연 구성 요소 추출
        mean_prop, mean_proc, mean_queue, mean_trans = mean_e2e_segment_from_csv(csv_path)

        # 데이터가 하나라도 유효하면 리스트에 추가
        if mean_prop is not None or mean_proc is not None or mean_queue is not None or mean_trans is not None:
            valid_labels.append(labels[i])
            prop_vals.append(0.0 if mean_prop is None else mean_prop)
            proc_vals.append(0.0 if mean_proc is None else mean_proc)
            queue_vals.append(0.0 if mean_queue is None else mean_queue)
            trans_vals.append(0.0 if mean_trans is None else mean_trans)
        else:
            print(f"[WARNING] Data missing or invalid for {labels[i]} at {csv_path}. Skipping.")

    if not valid_labels:
        print(f"[ERROR] no valid data to plot in {results_root}")
        return None

    x = range(len(valid_labels))

    # 스택 막대의 바닥(bottom) 위치 계산 (누적 합)
    bottom_prop = [0.0] * len(valid_labels)
    bottom_proc = prop_vals  # Prop 위에 Process
    bottom_queue = [p + c for p, c in zip(prop_vals, proc_vals)]  # Prop + Process 위에 Queueing
    bottom_trans = [b_q + q for b_q, q in
                    zip(bottom_queue, queue_vals)]  # Prop + Process + Queueing 위에 Transmitting

    plt.figure(figsize=(7, 4.5))

    # 스택 막대 그리기
    bars_prop = plt.bar(x, prop_vals, label="Propagation", bottom=bottom_prop)
    bars_proc = plt.bar(x, proc_vals, bottom=bottom_proc, label="Processing")
    bars_queue = plt.bar(x, queue_vals, bottom=bottom_queue, label="Queueing")
    bars_trans = plt.bar(x, trans_vals, bottom=bottom_trans, label="Transmitting")

    # === 각 스택 '자기 가운데'에 자기 평균값 주석 ===

    # Annotation Helper Function
    def autolabel_stack_simple(bars, segment_vals, bottom_start_vals):
        for bar, val, bottom_start in zip(bars, segment_vals, bottom_start_vals):
            if val > 0:
                y_center = bottom_start + val / 2.0
                plt.text(bar.get_x() + bar.get_width() / 2.0, y_center,
                         f"{val:.1f}", ha="center", va="center", fontsize=8)

    autolabel_stack_simple(bars_prop, prop_vals, bottom_prop)
    autolabel_stack_simple(bars_proc, proc_vals, bottom_proc)
    autolabel_stack_simple(bars_queue, queue_vals, bottom_queue)
    autolabel_stack_simple(bars_trans, trans_vals, bottom_trans)


    # plt.ylim(0, 3000)
    plt.xticks(list(x), valid_labels, rotation=0)
    plt.ylabel("Mean delay(ms)")
    plt.title(f"Mean E2E Delay Segments Comparison — sat {sat_rate/1e6}Mbps / gs {gs_rate/1e6}Mbps")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    out_path = results_root + out_png_filename
    ensure_results_dir(results_root)
    try:
        plt.savefig(out_path, dpi=150)
        print(f"[OK] saved: {out_path}")
    finally:
        plt.close()

    return out_path

def plot_each_mean_e2e(results_root, sat_rate, gs_rate, mode,
                           filename = "path_results.csv", out_png_filename = "mean_e2e.png"):
    """
    특정 mode에서의 E2E delay 그래프 확인
    """

    csv_path = results_root + filename
    mean = mean_e2e_from_csv(csv_path)

    # Prepare data for plotting (skip None values gracefully)
    if mean is None:
        print(f"[ERROR] no valid data to plot in {results_root}")
        return None

    x_pos = [0]
    mean_val = [mean]
    label_val = [mode.upper()]

    # Plot: single bar chart (per tool requirements: 1 plot, no style/colors)
    plt.figure(figsize=(6, 4))
    plt.bar(x_pos, mean_val)

    plt.xticks(x_pos, label_val, rotation=0)
    plt.ylabel("Mean E2E delay (ms)")
    plt.title(f"Mean E2E Delay — MODE {mode} sat {sat_rate/1e6}Mbps / gs {gs_rate/1e6}Mbps")
    plt.tight_layout()

    out_path = results_root + out_png_filename
    ensure_results_dir(results_root)
    try:
        plt.savefig(out_path, dpi=150)
        print(f"[OK] saved: {out_path}")
    finally:
        plt.close()

    return out_path

def plot_each_mean_stack_e2e_segment(results_root, sat_rate, gs_rate, mode,
                           filename = "path_results.csv", out_png_filename = "mean_stack_e2e.png"):
    """
    특정 mode에서의 E2E delay segment 그래프 확인
    """

    csv_path = results_root + filename
    mean_prop, mean_proc, mean_queue, mean_trans = mean_e2e_segment_from_csv(csv_path)

    # Prepare data for plotting (skip None values gracefully)
    if mean_prop is None and mean_proc is None and mean_queue is None and mean_trans is None:
        print(f"[ERROR] no valid data to plot in {results_root}")
        return None

    prop = 0.0 if mean_prop is None else mean_prop
    proc = 0.0 if mean_proc is None else mean_proc
    queue = 0.0 if mean_queue is None else mean_queue
    trans = 0.0 if mean_trans is None else mean_trans

    # 플롯 데이터 (단일 요소 리스트)
    x_pos = [0]
    prop_vals = [prop]
    proc_vals = [proc]
    queue_vals = [queue]
    trans_vals = [trans]

    # Plot: single bar chart (per tool requirements: 1 plot, no style/colors)
    plt.figure(figsize=(7, 4.5))

    bottom_prop = [0.0]
    bottom_proc = prop_vals
    bottom_queue = [p + c for p, c in zip(prop_vals, proc_vals)]
    bottom_trans = [b_q + q for b_q, q in zip(bottom_queue, queue_vals)]

    # 스택 막대
    bars_prop = plt.bar(x_pos, prop_vals, label="Propagation", bottom=bottom_prop)
    bars_proc = plt.bar(x_pos, proc_vals, label="Processing", bottom=bottom_proc)
    bars_queue = plt.bar(x_pos, queue_vals, label="Queueing", bottom=bottom_queue)
    bars_trans = plt.bar(x_pos, trans_vals, label="Transmitting", bottom=bottom_trans)

    # === 각 스택 '자기 가운데'에 자기 평균값 주석 (단일 막대 처리) ===
    def autolabel_stack_simple(bars, segment_vals, bottom_start_vals):
        for bar, val, bottom_start in zip(bars, segment_vals, bottom_start_vals):
            if val > 0:
                y_center = bottom_start + val / 2.0
                plt.text(bar.get_x() + bar.get_width() / 2.0, y_center,
                         f"{val:.1f}", ha="center", va="center", fontsize=8)

    autolabel_stack_simple(bars_prop, prop_vals, bottom_prop)
    autolabel_stack_simple(bars_proc, proc_vals, bottom_proc)
    autolabel_stack_simple(bars_queue, queue_vals, bottom_queue)
    autolabel_stack_simple(bars_trans, trans_vals, bottom_trans)

    # plt.ylim(0, 3000)
    plt.xticks(x_pos, [mode.upper()], rotation=0)
    plt.ylabel("Mean delay (ms)")
    plt.title(f"Mean E2E Delay Segments ({mode.upper()} — sat {sat_rate/1e6}Mbps / gs {gs_rate/1e6}Mbps")
    plt.legend()
    plt.tight_layout()

    out_path = results_root + out_png_filename
    ensure_results_dir(results_root)
    try:
        plt.savefig(out_path, dpi=150)
        print(f"[OK] saved: {out_path}")
    finally:
        plt.close()

    return out_path
