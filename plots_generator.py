import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rc

# 設定中文字型
font_path = "TFT/MSGOTHIC.TTF"
prop = font_manager.FontProperties(fname=font_path)
rc('font', family=prop.get_name())

def ensure_plots_dir():
    """確保 plots 目錄存在"""
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    return plots_dir

def create_comparison_charts(model_results):
    """創建模型比較圖表"""
    if not model_results:
        print("沒有模型結果可供比較")
        return

    # 確保 plots 目錄存在
    plots_dir = ensure_plots_dir()

    # 獲取模型名稱和指標
    model_names = [name.upper() for name in model_results.keys()]
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'fpr']
    metric_labels = ['準確率', '精確率', '召回率', 'F1分數', '誤報率']

    # 1. 性能指標比較圖表
    plt.figure(figsize=(15, 10))

    # 主要指標比較 (準確率, 精確率, 召回率, F1)
    plt.subplot(2, 1, 1)
    x = np.arange(4)
    width = 0.35

    for i, model_name in enumerate(model_results.keys()):
        values = [model_results[model_name][metric] for metric in metrics[:4]]
        plt.bar(x + i * width, values, width, label=model_name.upper())

    plt.ylabel('分數')
    plt.title('模型性能比較')
    plt.xticks(x + width / 2, metric_labels[:4])
    plt.ylim(0, 1)
    plt.legend()

    # 誤報率比較
    plt.subplot(2, 1, 2)
    fpr_values = [model_results[model_name]['fpr'] for model_name in model_results.keys()]
    plt.bar(model_names, fpr_values, color=['skyblue', 'salmon'])
    plt.ylabel('誤報率')
    plt.title('誤報率比較 (越低越好)')
    plt.ylim(0, max(fpr_values) * 1.2)

    # 保存圖表
    plt.tight_layout()
    model_comparison_path = os.path.join(plots_dir, 'model_comparison.png')
    plt.savefig(model_comparison_path)
    plt.close()
    print(f"模型比較圖表已保存至 '{model_comparison_path}'")

    # 繪製混淆矩陣
    plt.figure(figsize=(15, 7))

    for i, (model_name, results) in enumerate(model_results.items()):
        plt.subplot(1, 2, i + 1)
        cm = results['confusion_matrix']

        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'混淆矩陣 - {model_name.upper()}')
        plt.colorbar()

        classes = ['正常', '攻擊']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        # 在混淆矩陣中添加數值標籤
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('實際類別')
        plt.xlabel('預測類別')

    plt.tight_layout()
    confusion_matrices_path = os.path.join(plots_dir, 'confusion_matrices.png')
    plt.savefig(confusion_matrices_path)
    plt.close()
    print(f"混淆矩陣比較圖表已保存至 '{confusion_matrices_path}'")

def plot_memory_usage(memory_usage, save_path='plots/memory_usage.png'):
    """繪製記憶體使用變化的圖表"""
    if not memory_usage:
        print("沒有記憶體使用數據可供繪製")
        return

    # 確保 plots 目錄存在
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # 提取數據
    times = [entry['time'] for entry in memory_usage]
    memory_mb = [entry['memory_mb'] for entry in memory_usage]
    stages = [entry['stage'] for entry in memory_usage]

    # 繪製圖表
    plt.figure(figsize=(12, 6))
    for stage in set(stages):
        stage_times = [times[i] for i in range(len(stages)) if stages[i] == stage]
        stage_memory = [memory_mb[i] for i in range(len(stages)) if stages[i] == stage]
        plt.plot(stage_times, stage_memory, label=f'{stage.capitalize()} 階段')

    plt.xlabel('時間 (秒)')
    plt.ylabel('記憶體使用量 (MB)')
    plt.title('記憶體使用變化')
    plt.legend()
    plt.grid(True)

    # 保存圖表
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"記憶體使用圖表已保存至 '{save_path}'")

def create_performance_comparison_chart(model_results, save_path='plots/performance_comparison.png'):
    """創建模型性能比較圖表（記憶體使用和執行時間）"""
    if not model_results:
        print("沒有模型結果可供比較")
        return

    # 確保 plots 目錄存在
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # 創建子圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 提取數據
    model_names = list(model_results.keys())
    memory_changes = [results['memory_change'] for results in model_results.values()]
    execution_times = [results['time'] for results in model_results.values()]

    # 繪製記憶體使用對比
    bars1 = ax1.bar(model_names, memory_changes, color=['skyblue', 'salmon'])
    ax1.set_title('記憶體使用變化對比')
    ax1.set_ylabel('記憶體變化 (MB)')
    # 在柱狀圖上添加數值標籤
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}MB',
                ha='center', va='bottom')

    # 繪製執行時間對比
    bars2 = ax2.bar(model_names, execution_times, color=['skyblue', 'salmon'])
    ax2.set_title('執行時間對比')
    ax2.set_ylabel('執行時間 (秒)')
    # 在柱狀圖上添加數值標籤
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s',
                ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"性能比較圖表已保存至 '{save_path}'")