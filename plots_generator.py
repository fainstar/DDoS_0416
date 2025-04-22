#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
圖表生成模塊 - 用於生成各類分析圖表
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc
import matplotlib as mpl
import platform

# 根據作業系統選擇適合的中文字體
def get_system_font():
    system = platform.system()
    if system == 'Windows':
        # Windows系統上常見的中文字體
        fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi', 'Arial Unicode MS']
    elif system == 'Darwin':  # macOS
        fonts = ['PingFang TC', 'Hiragino Sans GB', 'Apple LiGothic', 'STHeiti', 'Heiti TC', 'Apple LiSung']
    else:  # Linux或其他
        fonts = ['Noto Sans CJK TC', 'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'Droid Sans Fallback']
    
    # 檢查每個字體是否可用
    for font in fonts:
        try:
            if font.lower() in [f.name.lower() for f in font_manager.fontManager.ttflist]:
                print(f"使用系統字體: {font}")
                return font
        except:
            pass
    
    return None

# 設定中文字型 - 嘗試多種方法
system_font = get_system_font()
font_path = "TFT/MSGOTHIC.TTF"

# 首先嘗試使用系統字體
if system_font:
    mpl.rcParams['font.family'] = system_font
    plt.rcParams['font.family'] = system_font
    mpl.rcParams['axes.unicode_minus'] = False
    print(f"已設置系統中文字體: {system_font}")
# 如果沒有找到系統字體，嘗試使用指定的字體文件
elif os.path.exists(font_path):
    try:
        # 註冊字體
        font_prop = font_manager.FontProperties(fname=font_path)
        font_family_name = font_prop.get_name()
        
        # 將字體設為全局默認
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = [font_family_name, 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
        mpl.rcParams['axes.unicode_minus'] = False
        
        print(f"已成功配置中文字體文件: {font_family_name}")
    except Exception as e:
        print(f"配置中文字體時出錯: {str(e)}")
else:
    print("警告：未找到合適的中文字體，圖表可能無法正確顯示中文。")

# 將中文字體設置應用到 Seaborn
sns.set(font=mpl.rcParams['font.family'])


def ensure_plots_dir():
    """確保 plots 目錄存在"""
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    return plots_dir


def apply_chinese_font_to_figure(fig=None):
    """對特定圖形對象應用中文字體設置"""
    if fig is None:
        fig = plt.gcf()  # 獲取當前圖形
    
    font_path = "TFT/MSGOTHIC.TTF"
    if os.path.exists(font_path):
        font_prop = font_manager.FontProperties(fname=font_path)
        
        # 應用到所有的文字元素
        for ax in fig.get_axes():
            for text in ax.get_xticklabels() + ax.get_yticklabels():
                text.set_fontproperties(font_prop)
            
            # 標題和標籤
            if ax.get_title():
                ax.title.set_fontproperties(font_prop)
            if ax.get_xlabel():
                ax.xaxis.label.set_fontproperties(font_prop)
            if ax.get_ylabel():
                ax.yaxis.label.set_fontproperties(font_prop)
            
            # 圖例
            legend = ax.get_legend()
            if legend:
                for text in legend.get_texts():
                    text.set_fontproperties(font_prop)


def plot_from_training_csv(csv_file, model_name=None):
    """從訓練記錄 CSV 文件生成訓練過程圖表"""
    try:
        # 讀取 CSV 文件
        df = pd.read_csv(csv_file)
        
        # 設置模型名稱
        if model_name is None:
            model_name = os.path.basename(csv_file).replace('_training_log.csv', '')
        
        # 確保 plots 目錄存在
        plots_dir = ensure_plots_dir()
        
        # 設置圖形風格
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 創建一個 2x2 的子圖佈局
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 損失曲線
        ax1.plot(df['epoch'], df['train_loss'], label='訓練損失')
        ax1.plot(df['epoch'], df['val_loss'], label='驗證損失')
        ax1.set_xlabel('訓練週期')
        ax1.set_ylabel('損失值')
        ax1.set_title(f'{model_name} 訓練和驗證損失變化')
        ax1.legend()
        ax1.grid(True)
        
        # 2. 準確率曲線
        ax2.plot(df['epoch'], df['val_accuracy'], label='驗證準確率')
        ax2.set_xlabel('訓練週期')
        ax2.set_ylabel('準確率')
        ax2.set_title(f'{model_name} 驗證準確率變化')
        ax2.legend()
        ax2.grid(True)
        
        # 3. 執行時間分析
        ax3.bar(df['epoch'], df['time_seconds'], color='teal', alpha=0.7)
        ax3.set_xlabel('訓練週期')
        ax3.set_ylabel('執行時間 (秒)')
        ax3.set_title(f'{model_name} 各週期執行時間')
        ax3.grid(True)
        
        # 4. 記憶體使用分析
        ax4.plot(df['epoch'], df['memory_allocated_mb'], label='已分配記憶體')
        ax4.plot(df['epoch'], df['memory_reserved_mb'], label='已預留記憶體', linestyle='--')
        ax4.set_xlabel('訓練週期')
        ax4.set_ylabel('記憶體使用量 (MB)')
        ax4.set_title(f'{model_name} GPU 記憶體使用變化')
        ax4.legend()
        ax4.grid(True)
        
        # 應用中文字體設置到圖形
        apply_chinese_font_to_figure(fig)
        
        # 儲存圖表
        plt.tight_layout()
        save_path = os.path.join(plots_dir, f'{model_name}_training_history.png')
        plt.savefig(save_path)
        plt.close(fig)
        
        print(f"已從 CSV 生成訓練歷史圖表: '{save_path}'")
        return save_path
    
    except Exception as e:
        print(f"從 CSV 生成訓練歷史圖表時發生錯誤：{str(e)}")
        import traceback
        traceback.print_exc()
        return None


def plot_from_evaluation_csv(eval_csv_file, batch_csv_file=None, model_name=None):
    """從評估記錄 CSV 文件生成評估結果圖表"""
    try:
        # 讀取評估結果 CSV 文件
        eval_df = pd.read_csv(eval_csv_file)
        
        # 設置模型名稱
        if model_name is None:
            model_name = os.path.basename(eval_csv_file).replace('_evaluation_result.csv', '')
        
        # 確保 plots 目錄存在
        plots_dir = ensure_plots_dir()
        
        # 設置圖形風格
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 創建混淆矩陣圖
        fig = plt.figure(figsize=(10, 8))
        
        # 從 CSV 中提取混淆矩陣數據
        tn = eval_df['tn'].values[0]
        fp = eval_df['fp'].values[0]
        fn = eval_df['fn'].values[0]
        tp = eval_df['tp'].values[0]
        
        cm = np.array([[tn, fp], [fn, tp]])
        
        # 使用 seaborn 繪製混淆矩陣
        ax = sns.heatmap(
            cm, 
            annot=True, 
            fmt=',d', 
            cmap='Blues',
            xticklabels=['正常', '攻擊'],
            yticklabels=['正常', '攻擊']
        )
        plt.title(f'{model_name} 模型混淆矩陣')
        plt.ylabel('真實類別')
        plt.xlabel('預測類別')
        
        # 應用中文字體設置
        apply_chinese_font_to_figure(fig)
        
        # 儲存混淆矩陣圖
        cm_path = os.path.join(plots_dir, f'{model_name}_confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close(fig)
        
        # 創建性能指標圖
        fig = plt.figure(figsize=(12, 6))
        
        # 提取性能指標
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'fpr']
        values = [eval_df[metric].values[0] for metric in metrics]
        
        # 為誤報率反向顯示 (1-fpr)，使其與其他指標方向一致
        metrics[-1] = '1-FPR'
        values[-1] = 1 - values[-1]
        
        # 繪製條形圖
        bars = plt.bar(metrics, values, color='skyblue')
        plt.ylim(0, 1.1)
        plt.title(f'{model_name} 模型性能指標')
        plt.ylabel('分數')
        
        # 添加數值標籤
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.4f}', ha='center', va='bottom')
        
        # 應用中文字體設置
        apply_chinese_font_to_figure(fig)
        
        # 儲存性能指標圖
        metrics_path = os.path.join(plots_dir, f'{model_name}_performance_metrics.png')
        plt.savefig(metrics_path)
        plt.close(fig)
        
        # 如果有批次評估數據，生成批次級別的圖表
        if batch_csv_file and os.path.exists(batch_csv_file):
            batch_df = pd.read_csv(batch_csv_file)
            
            # 創建批次評估圖
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # 批次損失曲線
            ax1.plot(batch_df['batch'], batch_df['loss'], color='purple', marker='o', markersize=3, linestyle='-', alpha=0.7)
            ax1.set_title(f'{model_name} 批次損失變化')
            ax1.set_xlabel('批次')
            ax1.set_ylabel('損失')
            ax1.grid(True)
            
            # 批次準確率曲線
            ax2.plot(batch_df['batch'], batch_df['accuracy'], color='green', marker='o', markersize=3, linestyle='-', alpha=0.7)
            ax2.set_title(f'{model_name} 批次準確率變化')
            ax2.set_xlabel('批次')
            ax2.set_ylabel('準確率')
            ax2.grid(True)
            
            # 應用中文字體設置
            apply_chinese_font_to_figure(fig)
            
            # 儲存批次評估圖
            plt.tight_layout()
            batch_path = os.path.join(plots_dir, f'{model_name}_batch_evaluation.png')
            plt.savefig(batch_path)
            plt.close(fig)
            
            print(f"已生成批次評估圖表: '{batch_path}'")
        
        print(f"已從 CSV 生成評估結果圖表: '{cm_path}', '{metrics_path}'")
        return cm_path, metrics_path
    
    except Exception as e:
        print(f"從 CSV 生成評估結果圖表時發生錯誤：{str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def create_comparison_charts_from_csv(model_names=None):
    """從所有模型的 CSV 文件創建比較圖表"""
    try:
        # 確保 logs 和 plots 目錄存在
        logs_dir = 'logs'
        plots_dir = ensure_plots_dir()
        
        if not os.path.exists(logs_dir):
            print(f"找不到日誌目錄: {logs_dir}")
            return None
            
        # 如果沒有指定模型名稱，則獲取 logs 目錄中所有的評估結果文件
        if model_names is None:
            eval_files = [f for f in os.listdir(logs_dir) if f.endswith('_evaluation_result.csv')]
            model_names = [f.replace('_evaluation_result.csv', '') for f in eval_files]
        
        if not model_names:
            print("找不到任何模型的評估結果文件")
            return None
        
        # 讀取所有模型的評估結果
        models_data = []
        for model_name in model_names:
            eval_file = os.path.join(logs_dir, f'{model_name}_evaluation_result.csv')
            training_file = os.path.join(logs_dir, f'{model_name}_training_log.csv')
            
            if os.path.exists(eval_file):
                eval_df = pd.read_csv(eval_file)
                model_data = {'name': model_name, 'evaluation': eval_df}
                
                if os.path.exists(training_file):
                    training_df = pd.read_csv(training_file)
                    model_data['training'] = training_df
                
                models_data.append(model_data)
            else:
                print(f"找不到模型 {model_name} 的評估結果文件")
        
        # 如果沒有找到任何模型數據，則退出
        if not models_data:
            print("沒有找到任何模型的評估結果數據")
            return None
        
        # 設置圖形風格
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 創建性能比較圖表
        fig = plt.figure(figsize=(14, 10))
        
        # 準備數據
        model_names = [model['name'] for model in models_data]
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'fpr']
        metric_names = ['準確率', '精確率', '召回率', 'F1分數', '誤報率(FPR)']
        
        # 為每個指標繪製一個子圖
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            plt.subplot(2, 3, i+1)
            
            values = [model['evaluation'][metric].values[0] for model in models_data]
            
            # 對於誤報率，較低表示較好
            if metric == 'fpr':
                colors = ['red' if v > min(values) else 'green' for v in values]
            else:
                colors = ['red' if v < max(values) else 'green' for v in values]
            
            bars = plt.bar(model_names, values, color=colors)
            plt.title(metric_name)
            plt.ylabel('分數')
            plt.xticks(rotation=45)
            
            # 添加數值標籤
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 應用中文字體設置
        apply_chinese_font_to_figure(fig)
        
        plt.tight_layout()
        performance_path = os.path.join(plots_dir, 'performance_comparison.png')
        plt.savefig(performance_path)
        plt.close(fig)
        
        # 創建訓練時間和記憶體比較圖
        fig = plt.figure(figsize=(15, 6))
        
        # 訓練時間比較
        plt.subplot(1, 2, 1)
        
        time_data = []
        for model in models_data:
            if 'training' in model:
                # 計算總訓練時間
                total_time = model['training']['time_seconds'].sum()
                time_data.append((model['name'], total_time))
        
        if time_data:
            names, times = zip(*time_data)
            plt.bar(names, times, color='skyblue')
            plt.title('模型訓練時間比較')
            plt.ylabel('總訓練時間 (秒)')
            plt.xticks(rotation=45)
            
            # 添加數值標籤
            for i, v in enumerate(times):
                plt.text(i, v + 5, f"{v:.1f}", ha='center')
        
        # 記憶體使用比較
        plt.subplot(1, 2, 2)
        
        memory_data = []
        for model in models_data:
            if 'training' in model:
                # 取最大記憶體使用量
                max_memory = model['training']['memory_allocated_mb'].max()
                memory_data.append((model['name'], max_memory))
        
        if memory_data:
            names, memories = zip(*memory_data)
            plt.bar(names, memories, color='lightgreen')
            plt.title('模型最大記憶體使用比較')
            plt.ylabel('最大記憶體使用 (MB)')
            plt.xticks(rotation=45)
            
            # 添加數值標籤
            for i, v in enumerate(memories):
                plt.text(i, v + 2, f"{v:.1f}", ha='center')
        
        # 應用中文字體設置
        apply_chinese_font_to_figure(fig)
        
        plt.tight_layout()
        resource_path = os.path.join(plots_dir, 'resource_comparison.png')
        plt.savefig(resource_path)
        plt.close(fig)
        
        # 創建訓練過程比較圖
        if all('training' in model for model in models_data):
            fig = plt.figure(figsize=(16, 12))
            
            # 損失比較
            plt.subplot(2, 2, 1)
            for model in models_data:
                plt.plot(
                    model['training']['epoch'], 
                    model['training']['train_loss'], 
                    label=f"{model['name']} 訓練損失"
                )
            for model in models_data:
                plt.plot(
                    model['training']['epoch'], 
                    model['training']['val_loss'], 
                    linestyle='--',
                    label=f"{model['name']} 驗證損失"
                )
            plt.title('模型損失比較')
            plt.xlabel('訓練週期')
            plt.ylabel('損失值')
            plt.legend()
            plt.grid(True)
            
            # 準確率比較
            plt.subplot(2, 2, 2)
            for model in models_data:
                plt.plot(
                    model['training']['epoch'], 
                    model['training']['val_accuracy'], 
                    label=f"{model['name']}"
                )
            plt.title('模型驗證準確率比較')
            plt.xlabel('訓練週期')
            plt.ylabel('準確率')
            plt.legend()
            plt.grid(True)
            
            # 每週期時間比較
            plt.subplot(2, 2, 3)
            for model in models_data:
                plt.plot(
                    model['training']['epoch'], 
                    model['training']['time_seconds'], 
                    marker='o',
                    label=f"{model['name']}"
                )
            plt.title('每週期執行時間比較')
            plt.xlabel('訓練週期')
            plt.ylabel('時間 (秒)')
            plt.legend()
            plt.grid(True)
            
            # 記憶體使用比較
            plt.subplot(2, 2, 4)
            for model in models_data:
                plt.plot(
                    model['training']['epoch'], 
                    model['training']['memory_allocated_mb'], 
                    label=f"{model['name']}"
                )
            plt.title('GPU記憶體使用比較')
            plt.xlabel('訓練週期')
            plt.ylabel('已分配記憶體 (MB)')
            plt.legend()
            plt.grid(True)
            
            # 應用中文字體設置
            apply_chinese_font_to_figure(fig)
            
            plt.tight_layout()
            training_comparison_path = os.path.join(plots_dir, 'training_process_comparison.png')
            plt.savefig(training_comparison_path)
            plt.close(fig)
            print(f"已創建訓練過程比較圖: '{training_comparison_path}'")
            
        print(f"已創建模型性能比較圖: '{performance_path}'")
        print(f"已創建資源使用比較圖: '{resource_path}'")
        
        return performance_path, resource_path
    
    except Exception as e:
        print(f"創建比較圖表時發生錯誤：{str(e)}")
        import traceback
        traceback.print_exc()
        return None


# 保留舊版相容接口的函數 - 這些函數使用新的基於 CSV 的函數

def create_comparison_charts(model_results):
    """創建不同模型結果的比較圖表
    (舊版兼容接口，實際調用新的基於 CSV 的函數)
    """
    print("生成模型比較圖表...")
    
    # 獲取模型名稱列表
    model_names = list(model_results.keys())
    
    # 調用基於 CSV 的比較函數
    return create_comparison_charts_from_csv(model_names)


def create_confusion_matrices_chart(model_results):
    """創建混淆矩陣比較圖表 (維持兼容舊版接口)"""
    print("生成混淆矩陣比較圖表...")

    # 確保 plots 目錄存在
    plots_dir = ensure_plots_dir()
    
    # 設置圖形
    fig = plt.figure(figsize=(15, 7))
    
    # 提取模型名稱
    model_names = list(model_results.keys())
    
    # 為每個模型生成混淆矩陣子圖
    for i, model_name in enumerate(model_names):
        plt.subplot(1, len(model_names), i+1)
        
        # 從日誌讀取混淆矩陣數據
        logs_dir = 'logs'
        eval_file = os.path.join(logs_dir, f'{model_name}_evaluation_result.csv')
        
        if os.path.exists(eval_file):
            eval_df = pd.read_csv(eval_file)
            tn = eval_df['tn'].values[0]
            fp = eval_df['fp'].values[0]
            fn = eval_df['fn'].values[0]
            tp = eval_df['tp'].values[0]
            
            cm = np.array([[tn, fp], [fn, tp]])
            
            # 使用seaborn繪製混淆矩陣
            sns.heatmap(
                cm, 
                annot=True, 
                fmt=',d', 
                cmap='Blues',
                xticklabels=['正常', '攻擊'],
                yticklabels=['正常', '攻擊']
            )
            plt.title(f'{model_name.upper()} 混淆矩陣')
            plt.ylabel('真實類別')
            plt.xlabel('預測類別')
        else:
            # 從模型結果直接讀取
            if 'confusion_matrix' in model_results[model_name]:
                cm = model_results[model_name]['confusion_matrix']
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt=',d',
                    cmap='Blues',
                    xticklabels=['正常', '攻擊'],
                    yticklabels=['正常', '攻擊']
                )
                plt.title(f'{model_name.upper()} 混淆矩陣')
                plt.ylabel('真實類別')
                plt.xlabel('預測類別')
            else:
                plt.text(0.5, 0.5, "無混淆矩陣數據", ha='center')
                plt.title(f'{model_name.upper()} - 無數據')
    
    # 應用中文字體設置
    apply_chinese_font_to_figure(fig)
    
    plt.tight_layout()
    confusion_matrices_path = os.path.join(plots_dir, 'confusion_matrices.png')
    plt.savefig(confusion_matrices_path)
    plt.close(fig)
    
    print(f"混淆矩陣比較圖表已保存至 '{confusion_matrices_path}'")
    return confusion_matrices_path


def create_performance_comparison_chart(model_results):
    """創建性能指標比較圖表 (舊版兼容接口)"""
    # 直接調用基於 CSV 的比較函數
    return create_comparison_charts_from_csv(list(model_results.keys()))