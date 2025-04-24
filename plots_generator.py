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
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, auc  # 新增導入ROC曲線相關函數
import warnings

# 設定中文字型
font_path = "TFT/MSGOTHIC.TTF"
prop = font_manager.FontProperties(fname=font_path)
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['MSGOTHIC']
mpl.rcParams['axes.unicode_minus'] = False

# 完全禁用所有UserWarning類別的警告
warnings.simplefilter("ignore", UserWarning)

# 特別針對字體缺失的警告
warnings.filterwarnings("ignore", category=UserWarning, message="Glyph \d+ .*? missing from font.*?")
warnings.filterwarnings("ignore", module="matplotlib.backends.backend_agg", message="Glyph \d+ .*? missing from font.*?")
warnings.filterwarnings("ignore", module="matplotlib.font_manager", message="Glyph \d+ .*? missing from font.*?")

# 禁用特定的中文字符警告 - 完全屏蔽與字體相關的警告
warnings.filterwarnings("ignore", message=r".*?\\(\\\\N\\{CJK.*?\\).*?")

# 根據作業系統選擇適合的中文字體
def get_system_font():
    system = platform.system()
    if system == 'Windows':
        # Windows系統上常見的中文字體，擴展更多選項
        fonts = [
            'Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi', 
            'Arial Unicode MS', 'DengXian', 'MingLiU', 'PMingLiU', 'MS UI Gothic',
            'Yu Gothic', 'Meiryo', 'Microsoft JhengHei'
        ]
    elif system == 'Darwin':  # macOS
        fonts = ['PingFang TC', 'Hiragino Sans GB', 'Apple LiGothic', 'STHeiti', 'Heiti TC', 'Apple LiSung']
    else:  # Linux或其他
        fonts = ['Noto Sans CJK TC', 'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'Droid Sans Fallback']
    
    # 檢查每個字體是否可用
    available_fonts = []
    for font in fonts:
        try:
            font_names = [f.name for f in font_manager.fontManager.ttflist]
            for font_name in font_names:
                if font.lower() in font_name.lower():
                    print(f"找到系統字體: {font_name}")
                    available_fonts.append(font_name)
                    break
        except Exception as e:
            print(f"檢查字體 {font} 時出錯: {str(e)}")
    
    if available_fonts:
        print(f"使用系統字體: {available_fonts[0]}")
        return available_fonts[0]
    return None

# 檢查目錄中的自定義字體
def find_custom_fonts(font_dirs=['TFT', 'fonts']):
    custom_fonts = []
    
    for font_dir in font_dirs:
        if os.path.exists(font_dir) and os.path.isdir(font_dir):
            for file in os.listdir(font_dir):
                if file.lower().endswith(('.ttf', '.ttc', '.otf')):
                    font_path = os.path.join(font_dir, file)
                    try:
                        # 嘗試載入字體
                        font_prop = font_manager.FontProperties(fname=font_path)
                        custom_fonts.append((font_path, font_prop.get_name()))
                        print(f"找到自定義字體: {font_path} ({font_prop.get_name()})")
                    except Exception as e:
                        print(f"載入字體 {font_path} 時出錯: {str(e)}")
    
    return custom_fonts

# 強制註冊所有可用的字體
def register_all_fonts():
    # 重新掃描字體目錄 - 相容不同版本的 matplotlib
    try:
        # 嘗試使用新版 matplotlib 方法
        font_manager.fontManager.findfont('DejaVu Sans')  # 刷新字體快取
    except:
        try:
            # 嘗試使用舊版 _rebuild() 方法
            font_manager._rebuild()
        except:
            print("警告: 無法重新加載字體快取，將繼續使用已載入的字體。")
    
    # 註冊系統字體
    system_font = get_system_font()
    
    # 註冊自定義字體
    custom_fonts = find_custom_fonts()
    
    # 創建字體列表
    font_list = []
    if system_font:
        font_list.append(system_font)
    
    for font_path, font_name in custom_fonts:
        try:
            # 註冊字體
            font_manager.fontManager.addfont(font_path)
            font_list.append(font_name)
        except Exception as e:
            print(f"註冊字體 {font_path} 時出錯: {str(e)}")
    
    if font_list:
        # 設置全局字體配置
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = font_list + ['DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
        mpl.rcParams['axes.unicode_minus'] = False
        
        # 設置 Seaborn 字體
        sns.set(font=font_list[0] if font_list else 'sans-serif')
        
        print(f"已成功配置字體列表: {font_list}")
        return True
    else:
        print("警告：未找到合適的中文字體，圖表可能無法正確顯示中文。")
        return False

# 立即註冊所有字體
register_all_fonts()

def ensure_plots_dir():
    """確保 plots 目錄存在"""
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    return plots_dir

# 改進圖形應用中文字體的函數
def apply_chinese_font(fig=None):
    if fig is None:
        fig = plt.gcf()
    for ax in fig.get_axes():
        ax.set_title(ax.get_title(), fontproperties=prop)
        ax.set_xlabel(ax.get_xlabel(), fontproperties=prop)
        ax.set_ylabel(ax.get_ylabel(), fontproperties=prop)
        for label in ax.get_xticklabels():
            label.set_fontproperties(prop)
        for label in ax.get_yticklabels():
            label.set_fontproperties(prop)
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontproperties(prop)

# 自定義 Seaborn 繪圖函數，強制應用中文字體
def custom_heatmap(*args, **kwargs):
    """自定義熱圖函數，強制使用中文字體"""
    ax = sns.heatmap(*args, **kwargs)
    apply_chinese_font(ax.figure)
    return ax

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
        apply_chinese_font(fig)
        
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
        apply_chinese_font(fig)
        
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
        apply_chinese_font(fig)
        
        # 儲存性能指標圖
        metrics_path = os.path.join(plots_dir, f'{model_name}_performance_metrics.png')
        plt.savefig(metrics_path)
        plt.close(fig)
        
        # 如果有批次評估數據，生成批次級別的圖表
        if batch_csv_file和os.path.exists(batch_csv_file):
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
            apply_chinese_font(fig)
            
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
        apply_chinese_font(fig)
        
        plt.tight_layout()
        performance_path = os.path.join(plots_dir, 'performance_comparison.png')
        # 確保在保存前應用字體設置
        with plt.rc_context({'font.family': 'sans-serif', 
                            'font.sans-serif': mpl.rcParams['font.sans-serif']}):
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
        apply_chinese_font(fig)
        
        plt.tight_layout()
        resource_path = os.path.join(plots_dir, 'resource_comparison.png')
        # 確保在保存前應用字體設置
        with plt.rc_context({'font.family': 'sans-serif', 
                            'font.sans-serif': mpl.rcParams['font.sans-serif']}):
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
            apply_chinese_font(fig)
            
            plt.tight_layout()
            training_comparison_path = os.path.join(plots_dir, 'training_process_comparison.png')
            # 確保在保存前應用字體設置
            with plt.rc_context({'font.family': 'sans-serif', 
                               'font.sans-serif': mpl.rcParams['font.sans-serif'],
                               'font.size': 10}):
                # 直接指定字體文件路徑，確保能找到中文字體
                font_path = os.path.join('TFT', 'MSGOTHIC.TTF')
                if os.path.exists(font_path):
                    custom_font = font_manager.FontProperties(fname=font_path)
                    # 再次應用自定義字體到圖表所有文本元素
                    for ax in fig.get_axes():
                        for text in ax.get_xticklabels() + ax.get_yticklabels():
                            text.set_fontproperties(custom_font)
                        if ax.get_title():
                            ax.title.set_fontproperties(custom_font)
                        if ax.get_xlabel():
                            ax.xaxis.label.set_fontproperties(custom_font)
                        if ax.get_ylabel():
                            ax.yaxis.label.set_fontproperties(custom_font)
                        legend = ax.get_legend()
                        if legend:
                            for text in legend.get_texts():
                                text.set_fontproperties(custom_font)
                plt.savefig(training_comparison_path, bbox_inches='tight')
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
    apply_chinese_font(fig)
    
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

# 新增功能: 特徵重要性分析與可視化
def plot_feature_importance(X, y, feature_names, model_name="model", n_features=15, method="random_forest"):
    """
    生成特徵重要性圖表
    
    Args:
        X: 特徵數據 (numpy 數組或 pandas DataFrame)
        y: 標籤數據 (numpy 數組或 pandas Series)
        feature_names: 特徵名稱列表
        model_name: 模型名稱 (用於保存文件)
        n_features: 顯示的最重要特徵數量
        method: 特徵重要性計算方法 ('random_forest' 或 'permutation')
        
    Returns:
        importance_df: 包含特徵重要性分數的 DataFrame
    """
    print(f"正在生成特徵重要性圖表...")
    
    # 確保 plots 目錄存在
    plots_dir = ensure_plots_dir()
    
    # 如果 X 是 DataFrame，轉換為 numpy 數組並保存特徵名稱
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X = X.values
    
    # 如果特徵名稱為空或長度不對，創建默認名稱
    if not feature_names或len(feature_names) != X.shape[1]:
        feature_names = [f"特徵 {i+1}" for i in range(X.shape[1])]
    
    # 創建隨機森林模型計算特徵重要性
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # 獲取特徵重要性
    if method == "random_forest":
        # 使用隨機森林內建的特徵重要性
        importances = rf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
        importance_type = "基於不純度的重要性"
    else:  # permutation
        # 使用排列重要性（評估特徵對模型性能的影響）
        result = permutation_importance(rf, X, y, n_repeats=10, random_state=42)
        importances = result.importances_mean
        std = result.importances_std
        importance_type = "基於排列的重要性"
    
    # 創建特徵重要性 DataFrame
    importance_df = pd.DataFrame({
        '特徵': feature_names,
        '重要性': importances,
        '標準差': std
    })
    
    # 按重要性排序
    importance_df = importance_df.sort_values('重要性', ascending=False)
    
    # 選擇前 N 個最重要的特徵
    top_features = importance_df.head(n_features)
    
    # 繪製特徵重要性條形圖
    plt.figure(figsize=(12, 10))
    
    # 設置風格
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 繪製條形圖，水平方向
    bars = plt.barh(np.arange(n_features), top_features['重要性'], 
             xerr=top_features['標準差'], 
             align='center',
             color='skyblue',
             edgecolor='black',
             alpha=0.8)
    
    # 設置 Y 軸刻度為特徵名稱，並反轉順序（最重要的在頂部）
    plt.yticks(np.arange(n_features), top_features['特徵'])
    
    # 添加數值標籤到條形上
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{width:.3f}', ha='left', va='center', fontweight='bold')
    
    # 添加標題和標籤
    plt.title(f'{model_name} 模型特徵重要性分析', fontsize=16)
    plt.xlabel('特徵重要性分數', fontsize=14)
    plt.ylabel('特徵', fontsize=14)
    
    # 添加網格線，僅限 X 軸
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # 添加注釋說明使用的方法
    plt.figtext(0.5, 0.01, f'使用方法: {importance_type}', ha='center', fontsize=10, style='italic')
    
    # 應用中文字體設置
    apply_chinese_font(plt.gcf())
    
    # 調整布局
    plt.tight_layout(pad=3.0)
    
    # 保存圖片
    save_path = os.path.join(plots_dir, f'{model_name}_feature_importance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"特徵重要性圖表已保存至: {save_path}")
    
    # 顯示圖表
    plt.show()
    
    return importance_df

# 多模型特徵重要性比較函數
def plot_multiple_model_feature_importance(X, y, feature_names, model_names=None, n_features=10):
    """
    比較多個模型的特徵重要性
    
    Args:
        X: 特徵數據 (numpy 數組或 pandas DataFrame)
        y: 標籤數據 (numpy 數組或 pandas Series)
        feature_names: 特徵名稱列表
        model_names: 模型名稱列表，如 ['transformer', 'cnnlstm']
        n_features: 顯示的最重要特徵數量
    """
    if model_names is None:
        model_names = ['transformer', 'cnnlstm']
    
    print(f"正在生成多模型特徵重要性比較圖...")
    
    # 確保 plots 目錄存在
    plots_dir = ensure_plots_dir()
    
    # 如果 X 是 DataFrame，轉換為 numpy 數組並保存特徵名稱
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X = X.values
    
    # 創建子圖
    fig, axes = plt.subplots(1, len(model_names), figsize=(6*len(model_names), 10), sharey=True)
    
    # 如果只有一個模型，確保 axes 是列表
    if len(model_names) == 1:
        axes = [axes]
    
    importance_dfs = []
    
    # 為每個模型計算並繪製特徵重要性
    for i, model_name in enumerate(model_names):
        # 創建隨機森林模型計算特徵重要性
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # 獲取特徵重要性
        importances = rf.feature_importances_
        
        # 創建特徵重要性 DataFrame
        importance_df = pd.DataFrame({
            '特徵': feature_names,
            '重要性': importances
        })
        
        # 按重要性排序
        importance_df = importance_df.sort_values('重要性', ascending=False)
        importance_dfs.append(importance_df.copy())
        
        # 選擇前 N 個最重要的特徵
        top_features = importance_df.head(n_features)
        
        # 繪製條形圖，水平方向
        bars = axes[i].barh(np.arange(n_features), top_features['重要性'], 
                 align='center',
                 color='skyblue' if i == 0 else 'lightcoral',
                 edgecolor='black',
                 alpha=0.8)
        
        # 設置 Y 軸刻度為特徵名稱，並反轉順序（最重要的在頂部）
        axes[i].set_yticks(np.arange(n_features))
        axes[i].set_yticklabels(top_features['特徵'])
        
        # 添加數值標籤到條形上
        for bar in bars:
            width = bar.get_width()
            axes[i].text(width + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{width:.3f}', ha='left', va='center', fontweight='bold')
        
        # 添加標題和標籤
        axes[i].set_title(f'{model_name} 模型', fontsize=16)
        axes[i].set_xlabel('特徵重要性分數', fontsize=14)
        
        # 僅為第一個子圖添加 Y 軸標籤
        if i == 0:
            axes[i].set_ylabel('特徵', fontsize=14)
        
        # 添加網格線，僅限 X 軸
        axes[i].grid(axis='x', linestyle='--', alpha=0.6)
    
    # 添加總標題
    plt.suptitle('模型特徵重要性比較', fontsize=18)
    
    # 應用中文字體設置
    apply_chinese_font(fig)
    
    # 調整布局
    plt.tight_layout()
    
    # 保存圖片
    save_path = os.path.join(plots_dir, 'feature_importance_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"特徵重要性比較圖表已保存至: {save_path}")
    
    # 顯示圖表
    plt.show()
    
    return importance_dfs

def plot_feature_importance(data_path="AllMerged.csv", top_n=20, model_type="random_forest"):
    """生成特徵重要性圖表
    
    Args:
        data_path (str): 數據文件路徑
        top_n (int): 顯示的前N個重要特徵
        model_type (str): 使用的模型類型，可選 "random_forest", "permutation"
    """
    
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.inspection import permutation_importance
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 確保 plots 目錄存在
    plots_dir = ensure_plots_dir()
    
    print(f"讀取數據集: {data_path}")
    # 讀取數據
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"讀取數據時發生錯誤: {str(e)}")
        return
    
    # 查找標籤列
    label_col = None
    possible_label_cols = ['Label', ' Label', 'label', 'CLASS', 'class', 'target']
    
    for col in possible_label_cols:
        if (col in df.columns):
            label_col = col
            print(f"找到標籤列: '{label_col}'")
            break
    
    if label_col is None:
        # 如果找不到標籤列，假設最後一列是標籤
        label_col = df.columns[-1]
        print(f"找不到明確的標籤列，使用最後一列作為標籤: '{label_col}'")
    
    # 提取標籤
    y = df[label_col].values
    
    # 二分類處理：BENIGN(良性) -> 0, 其他攻擊 -> 1
    if isinstance(y[0], str):
        binary_y = np.array([0 if label == 'BENIGN' else 1 for label in y])
        print("將標籤轉換為二分類: BENIGN -> 0, 其他 -> 1")
    else:
        binary_y = y
        print("使用原始數值標籤")
    
    # 提取特徵，刪除標籤列
    X_df = df.drop(label_col, axis=1)
    
    # 處理類別特徵和文本特徵
    categorical_cols = X_df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"發現 {len(categorical_cols)} 個類別特徵，將其移除")
        X_df = X_df.drop(columns=categorical_cols)
    
    # 移除缺失值、無限值和極端值
    for col in X_df.columns:
        X_df[col] = X_df[col].replace([np.inf, -np.inf], np.nan)
        
    X_df = X_df.dropna(axis=1)  # 移除包含NaN的列
    print(f"數據預處理後的特徵數: {len(X_df.columns)}")
    
    # 備份特徵名稱
    feature_names = X_df.columns.tolist()
    
    # 標準化特徵
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    
    # 計算特徵重要性
    print(f"使用 {model_type} 計算特徵重要性...")
    
    plt.figure(figsize=(12, 10))
    
    if (model_type == "random_forest"):
        # 使用隨機森林計算特徵重要性
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_scaled, binary_y)
        
        # 獲取特徵重要性
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # 只顯示前N個重要特徵
        indices = indices[:top_n]
        
        # 繪製特徵重要性圖
        plt.title('隨機森林模型特徵重要性排序', fontsize=16)
        plt.barh(range(len(indices)), importances[indices], color='b', align='center', alpha=0.7)
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('相對重要性')
        
    elif (model_type == "permutation"):
        # 使用置換重要性
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X_scaled, binary_y)
        
        # 計算置換重要性
        result = permutation_importance(rf, X_scaled, binary_y, n_repeats=10, 
                                        random_state=42, n_jobs=-1)
        
        # 獲取重要性
        importances = result.importances_mean
        indices = np.argsort(importances)[::-1]
        
        # 只顯示前N個重要特徵
        indices = indices[:top_n]
        
        # 繪製特徵重要性圖
        plt.title('置換重要性排序', fontsize=16)
        plt.barh(range(len(indices)), importances[indices], color='g', align='center', alpha=0.7)
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('相對重要性')
    
    # 反轉Y軸，讓最重要的特徵顯示在頂部
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # 應用中文字體設置
    apply_chinese_font(plt.gcf())
    
    # 生成輸出文件名
    output_file = os.path.join(plots_dir, f'{model_type}_feature_importance.png')
    
    # 儲存圖片
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"特徵重要性圖表已保存到: {output_file}")
    
    # 顯示圖表（可選）
    plt.show()
    
    return output_file


def plot_feature_correlation(data_path="AllMerged.csv", top_n=15):
    """生成特徵相關性熱力圖
    
    Args:
        data_path (str): 數據文件路徑
        top_n (int): 顯示的前N個重要特徵
    """
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.ensemble import RandomForestClassifier
    
    # 確保 plots 目錄存在
    plots_dir = ensure_plots_dir()
    
    # 讀取數據
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"讀取數據時發生錯誤: {str(e)}")
        return
    
    # 查找標籤列
    label_col = None
    possible_label_cols = ['Label', ' Label', 'label', 'CLASS', 'class', 'target']
    
    for col in possible_label_cols:
        if col in df.columns:
            label_col = col
            print(f"找到標籤列: '{label_col}'")
            break
    
    if label_col is None:
        # 如果找不到標籤列，假設最後一列是標籤
        label_col = df.columns[-1]
        print(f"找不到明確的標籤列，使用最後一列作為標籤: '{label_col}'")
    
    # 提取特徵，刪除標籤列
    X_df = df.drop(label_col, axis=1)
    
    # 處理類別特徵和文本特徵
    categorical_cols = X_df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"發現 {len(categorical_cols)} 個類別特徵，將其移除")
        X_df = X_df.drop(columns=categorical_cols)
    
    # 移除缺失值、無限值和極端值
    for col in X_df.columns:
        X_df[col] = X_df[col].replace([np.inf, -np.inf], np.nan)
    
    # 移除包含NaN的列
    X_df = X_df.dropna(axis=1)
    
    # 計算隨機森林特徵重要性，以選擇要顯示的特徵
    y = df[label_col].values
    
    # 二分類處理：BENIGN(良性) -> 0, 其他攻擊 -> 1
    if isinstance(y[0], str):
        binary_y = np.array([0 if label == 'BENIGN' else 1 for label in y])
    else:
        binary_y = y
    
    # 使用隨機森林計算特徵重要性
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_df, binary_y)
    
    # 獲取特徵重要性
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # 選擇前N個重要特徵
    top_features = X_df.columns[indices[:top_n]].tolist()
    
    # 添加標籤列到相關性矩陣
    selected_df = X_df[top_features].copy()
    selected_df['Label'] = binary_y
    
    # 計算相關性矩陣
    corr = selected_df.corr()
    
    # 繪製熱力圖
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones像(corr, dtype=bool))  # 創建上三角遮罩
    
    # 設置顏色映射
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # 繪製熱力圖
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt=".2f", square=True, linewidths=.5)
    
    plt.title('重要特徵相關性熱力圖', fontsize=16)
    
    # 應用中文字體設置
    apply_chinese_font(plt.gcf())
    
    # 生成輸出文件名
    output_file = os.path.join(plots_dir, 'feature_correlation.png')
    
    # 儲存圖片
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"特徵相關性熱力圖已保存到: {output_file}")
    
    # 顯示圖表（可選）
    plt.show()
    
    return output_file

# 新增功能: 特徵相關性熱力圖
def plot_feature_correlation(data_path=None, X=None, feature_names=None, top_n=15):
    """
    生成特徵相關性熱力圖

    Args:
        data_path: 數據文件路徑 (CSV格式)
        X: 直接提供的特徵數據 (numpy數組或pandas DataFrame)
        feature_names: 特徵名稱列表
        top_n: 顯示的特徵數量
        
    Returns:
        圖表保存路徑
    """
    print(f"正在生成特徵相關性熱力圖...")
    
    # 確保 plots 目錄存在
    plots_dir = ensure_plots_dir()
    
    try:
        # 如果提供了數據路徑而不是數據
        if data_path is not None:
            # 通過 data_processor 讀取數據
            from data_processor import load_network_data, preprocess_data
            data = load_network_data(data_path)
            X, _ = preprocess_data(data)
            
            # 獲取特徵名稱
            if isinstance(data, pd.DataFrame):
                feature_names = data.columns[:-1].tolist()  # 假設最後一列是標籤列
        
        # 如果 X 是 DataFrame，保存特徵名稱並轉換為 numpy 數組
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X = X.values
        
        # 如果沒有特徵名稱，創建默認名稱
        if not feature_names or len(feature_names) != X.shape[1]:
            feature_names = [f"特徵 {i+1}" for i in range(X.shape[1])]
        
        # 創建 DataFrame 用於計算相關性
        df = pd.DataFrame(X, columns=feature_names)
        
        # 計算相關性矩陣
        corr_matrix = df.corr()
        
        # 如果要顯示的特徵數量少於總特徵數量，選擇相關性最高的特徵
        if top_n和top_n < len(feature_names):
            # 計算每個特徵與其他特徵的相關性總和的絕對值
            corr_sum = np.abs(corr_matrix.values).sum(axis=0) - 1  # 減去自身的相關性 (=1)
            # 選擇相關性總和最大的 top_n 個特徵
            top_indices = np.argsort(corr_sum)[-top_n:]
            top_features = [feature_names[i] for i in top_indices]
            # 子集化相關性矩陣
            corr_matrix = corr_matrix.loc[top_features, top_features]
        
        # 繪製相關性熱力圖
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones像(corr_matrix))  # 創建上三角形遮罩
        ax = sns.heatmap(
            corr_matrix, 
            mask=mask,
            annot=True, 
            cmap='coolwarm', 
            vmin=-1, 
            vmax=1, 
            fmt='.2f',
            square=True,
            linewidths=0.5
        )
        plt.title('特徵相關性熱力圖 (Pearson 相關係數)')
        
        # 應用中文字體設置
        apply_chinese_font(plt.gcf())
        
        # 調整圖表布局和字體大小
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 保存圖表
        save_path = os.path.join(plots_dir, 'feature_correlation_heatmap.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"特徵相關性熱力圖已保存至: {save_path}")
        return save_path
    
    except Exception as e:
        print(f"生成特徵相關性熱力圖時出錯: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def plot_roc_curve(y_true, y_pred_prob, model_name="model"):
    """
    為單個模型生成ROC曲線圖
    
    Args:
        y_true: 真實標籤
        y_pred_prob: 預測的正類機率 (sigmoid/softmax輸出)
        model_name: 模型名稱
    
    Returns:
        保存路徑
    """
    print(f"生成 {model_name} 的ROC曲線圖...")
    
    # 確保 plots 目錄存在
    plots_dir = ensure_plots_dir()
    
    # 計算ROC曲線的數據點
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # 繪製ROC曲線
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲線 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='隨機猜測')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('偽陽性率 (False Positive Rate)')
    plt.ylabel('真陽性率 (True Positive Rate)')
    plt.title(f'{model_name} 模型 ROC 曲線')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 應用中文字體設置
    apply_chinese_font(plt.gcf())
    
    # 保存圖表
    save_path = os.path.join(plots_dir, f'{model_name}_roc_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC曲線圖已保存至: {save_path}")
    return save_path, (fpr, tpr, roc_auc)  # 返回圖表路徑和ROC相關數據供後續使用

def plot_roc_comparison(models_data, title="模型ROC曲線比較"):
    """
    比較多個模型的ROC曲線
    
    Args:
        models_data: 包含模型數據的列表，每個元素為 (model_name, y_true, y_pred_prob) 或
                    (model_name, fpr, tpr, roc_auc) 元組
        title: 圖表標題
        
    Returns:
        圖表保存路徑
    """
    print(f"生成多模型ROC曲線比較圖...")
    
    # 確保 plots 目錄存在
    plots_dir = ensure_plots_dir()
    
    # 創建圖形
    plt.figure(figsize=(12, 10))
    
    # 使用不同顏色
    colors = ['darkorange', 'forestgreen', 'royalblue', 'darkviolet', 'crimson', 
              'darkturquoise', 'gold', 'brown', 'pink', 'gray']
    line_styles = ['-', '--', '-.', ':']  # 多種線型，可循環使用
    
    # 繪製每個模型的ROC曲線
    for i, model_data in enumerate(models_data):
        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]
        
        if len(model_data) == 3:
            # 如果提供的是模型名稱、真實值和預測機率
            model_name, y_true, y_pred_prob = model_data
            # 計算ROC曲線的數據點
            fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
            roc_auc = auc(fpr, tpr)
        elif len(model_data) == 4:
            # 如果提供的是已經計算好的ROC數據
            model_name, fpr, tpr, roc_auc = model_data
        else:
            print(f"錯誤：模型數據格式不正確: {model_data}")
            continue
        
        # 繪製ROC曲線
        plt.plot(fpr, tpr, color=color, lw=2, linestyle=line_style,
                 label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    # 添加隨機猜測的基準線
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='隨機猜測')
    
    # 設置圖表屬性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('偽陽性率 (False Positive Rate)')
    plt.ylabel('真陽性率 (True Positive Rate)')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 應用中文字體設置
    apply_chinese_font(plt.gcf())
    
    # 保存圖表
    save_path = os.path.join(plots_dir, 'roc_curves_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC曲線比較圖已保存至: {save_path}")
    return save_path

def load_model_predictions_for_roc(model_names=None):
    """
    從評估結果載入模型預測數據，用於生成ROC曲線
    
    Args:
        model_names: 模型名稱列表，如果為None則自動尋找

    Returns:
        模型預測數據列表，每個元素為 (model_name, y_true, y_pred_prob)
    """
    try:
        # 確保 logs 目錄存在
        logs_dir = 'logs'
        if not os.path.exists(logs_dir):
            print(f"找不到日誌目錄: {logs_dir}")
            return []
        
        # 如果沒有指定模型名稱，則獲取 logs 目錄中所有的評估結果文件
        if model_names is None:
            prediction_files = [f for f in os.listdir(logs_dir) if f.endswith('_predictions.csv')]
            model_names = [f.replace('_predictions.csv', '') for f in prediction_files]
        
        if not model_names:
            print("找不到任何模型的預測結果文件，請先執行模型評估並保存預測結果")
            return []
        
        # 讀取每個模型的預測結果
        models_data = []
        for model_name in model_names:
            pred_file = os.path.join(logs_dir, f'{model_name}_predictions.csv')
            
            if os.path.exists(pred_file):
                # 讀取預測結果文件
                pred_df = pd.read_csv(pred_file)
                
                # 檢查文件格式是否包含必要的列
                required_cols = ['true_label', 'pred_probability']
                if all(col in pred_df.columns for col in required_cols):
                    y_true = pred_df['true_label'].values
                    y_pred_prob = pred_df['pred_probability'].values
                    
                    # 添加到模型數據列表
                    models_data.append((model_name, y_true, y_pred_prob))
                    print(f"成功載入 {model_name} 模型的預測結果")
                else:
                    print(f"警告：{pred_file} 缺少必要的數據列 (true_label, pred_probability)")
            else:
                print(f"找不到模型 {model_name} 的預測結果文件：{pred_file}")
        
        return models_data
    
    except Exception as e:
        print(f"載入模型預測數據時發生錯誤：{str(e)}")
        import traceback
        traceback.print_exc()
        return []

def generate_roc_comparison_from_logs(model_names=None):
    """
    從日誌文件生成多個模型的ROC曲線比較圖
    
    Args:
        model_names: 模型名稱列表，如果為None則自動尋找所有可用模型
        
    Returns:
        比較圖的保存路徑, 模型數據字典
    """
    print("正在從日誌生成模型ROC曲線比較圖...")
    
    # 確保 logs 目錄存在
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        print(f"警告: 找不到日誌目錄 {logs_dir}")
        return None, None
    
    # 如果沒有提供模型名稱，則自動搜索所有預測結果文件
    if model_names is None:
        prediction_files = [f for f in os.listdir(logs_dir) if f.endswith('_predictions.csv')]
        model_names = [f.replace('_predictions.csv', '') for f in prediction_files]
    
    if not model_names:
        print("未找到任何模型預測數據。請先運行測試或提供有效的模型名稱。")
        return None, None
    
    print(f"將比較以下模型: {', '.join(model_names)}")
    
    # 收集所有模型的數據
    models_data = {}
    
    for model_name in model_names:
        pred_file = os.path.join(logs_dir, f"{model_name}_predictions.csv")
        
        if not os.path.exists(pred_file):
            print(f"找不到模型 {model_name} 的預測數據文件: {pred_file}")
            continue
        
        try:
            # 讀取預測結果
            df = pd.read_csv(pred_file)
            
            if 'true_label' not in df.columns or 'pred_probability' not in df.columns:
                print(f"警告: {pred_file} 缺少必要的列 ('true_label', 'pred_probability')")
                continue
            
            # 提取真實標籤和預測概率
            y_true = df['true_label'].values
            y_score = df['pred_probability'].values
            
            # 計算ROC曲線數據
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            
            # 存儲這個模型的數據
            models_data[model_name] = {
                'y_true': y_true,
                'y_score': y_score,
                'fpr': fpr,
                'tpr': tpr,
                'auc': roc_auc
            }
            
            print(f"已讀取 {model_name} 的預測結果 (AUC: {roc_auc:.4f})")
            
        except Exception as e:
            print(f"處理模型 {model_name} 的數據時出錯: {str(e)}")
    
    if not models_data:
        print("未能從任何模型讀取有效數據")
        return None, None
    
    # 繪製比較ROC曲線圖
    save_path = plot_multiple_roc_curves(models_data, save_name="roc_comparison")
    
    return save_path, models_data