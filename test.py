#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
測試與預測模塊 - 用於模型測試和實時預測
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import psutil
import time

# 導入自定義模塊
from data_processor import preprocess_data, NetworkTrafficDataset
from models import create_model
from trainer import evaluate_model

# 設定中文字型
font_path = "TFT/MSGOTHIC.TTF"
prop = font_manager.FontProperties(fname=font_path)
rc('font', family=prop.get_name())


def load_trained_model(model_path, model_type='mamba', input_dim=78):
    """載入訓練好的模型"""
    try:
        # 檢查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
            
        # 创建模型實例
        model = create_model(model_type, input_dim)
        
        # 載入模型狀態
        model.load_state_dict(torch.load(model_path))
        
        print(f"成功載入模型: {model_path}")
        return model
    except Exception as e:
        print(f"載入模型時出錯: {str(e)}")
        return None


def predict_sample(model, sample_data, device='cpu'):
    """預測單個樣本"""
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        # 確保輸入是 torch tensor 且維度正確
        if isinstance(sample_data, np.ndarray):
            sample_tensor = torch.tensor(sample_data, dtype=torch.float32).to(device)
        else:
            sample_tensor = sample_data.to(device)
        
        # 如果輸入是單個樣本，需要增加批次維度
        if len(sample_tensor.shape) == 1:
            sample_tensor = sample_tensor.unsqueeze(0)
        
        # 模型預測
        output = model(sample_tensor)
        
        # 使用閾值 0.5 將輸出轉換為二分類結果
        prediction = (output.squeeze() > 0.5).float().cpu().numpy()
        probability = output.squeeze().cpu().numpy()
        
    if len(prediction) == 1:
        prediction = prediction[0]
        probability = probability[0]
        
    return {"prediction": int(prediction), "probability": float(probability)}


def test_model_on_file(model_path, test_file_path, model_type='mamba', sample_size=None, output_file=None):
    """在測試文件上測試模型"""
    print(f"在 {test_file_path} 上測試模型...")
    
    try:
        # 加載和預處理測試數據
        df = pd.read_csv(test_file_path, low_memory=False)
        print(f"測試數據大小: {df.shape}")
        
        # 如果只使用部分樣本
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            print(f"使用 {sample_size} 個樣本進行測試")
        
        # 數據預處理
        X, y = preprocess_data(df)
        print(f"預處理後數據形狀: X={X.shape}, y={y.shape}")
        
        # 加載模型
        input_dim = X.shape[1]
        model = load_trained_model(model_path, model_type, input_dim)
        
        if model is None:
            print("模型加載失敗，無法進行測試")
            return None
        
        # 創建測試資料集
        test_dataset = NetworkTrafficDataset(X, y)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128)
        
        # 評估模型
        results = evaluate_model(model, test_loader)
        
        # 存儲結果（如果指定了輸出文件）
        if output_file:
            with open(output_file, 'w') as f:
                json.dump({
                    'accuracy': float(results['accuracy']),
                    'precision': float(results['precision']),
                    'recall': float(results['recall']),
                    'f1': float(results['f1']),
                    'fpr': float(results['fpr'])
                }, f, indent=4)
                print(f"測試結果已保存至 {output_file}")
        
        return results
    
    except Exception as e:
        print(f"測試過程中出錯: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def real_time_prediction(model, data_source, preprocess_fn=None):
    """實時預測功能"""
    # 這個函數可以擴展為從網絡捕獲流量進行即時分析
    # 目前只是一個基本示例，可以根據需求進一步開發
    print("進行實時預測...")
    
    try:
        # 如果 data_source 是文件，則加載文件
        if isinstance(data_source, str) and os.path.exists(data_source):
            df = pd.read_csv(data_source)
            print(f"從文件加載數據: {data_source}, 形狀: {df.shape}")
            
            # 應用預處理函數（如果提供）
            if preprocess_fn:
                X, _ = preprocess_fn(df)
            else:
                # 假設數據已經預處理過或只需要特徵列
                X = df.drop(df.columns[-1], axis=1).values
                X = StandardScaler().fit_transform(X)
            
            # 轉換為 PyTorch 張量進行批量預測
            tensor_data = torch.tensor(X, dtype=torch.float32)
            
            # 批量預測
            batch_size = 32
            predictions = []
            
            for i in range(0, len(tensor_data), batch_size):
                batch = tensor_data[i:i+batch_size]
                result = predict_sample(model, batch)
                predictions.extend(result["prediction"] if isinstance(result["prediction"], list) else [result["prediction"]])
            
            # 顯示結果摘要
            attack_count = sum(predictions)
            benign_count = len(predictions) - attack_count
            print(f"預測結果: 共 {len(predictions)} 個樣本")
            print(f"  - 正常流量 (BENIGN): {benign_count} ({benign_count/len(predictions)*100:.2f}%)")
            print(f"  - 攻擊流量 (ATTACK): {attack_count} ({attack_count/len(predictions)*100:.2f}%)")
            
            return predictions
        
        else:
            print("不支持的數據源。請提供有效的文件路徑。")
            return None
            
    except Exception as e:
        print(f"實時預測時出錯: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def visualize_results(predictions, labels, timestamps=None):
    """視覺化預測結果"""
    # 確保 plots 目錄存在
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # 設置中文字體
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 準備數據
    if timestamps is None:
        timestamps = range(len(predictions))
    
    # 創建時間序列圖
    plt.figure(figsize=(15, 8))
    
    # 繪製真實標籤和預測值
    plt.plot(timestamps, labels, 'b.', label='實際值', alpha=0.5, markersize=10)
    plt.plot(timestamps, predictions, 'r.', label='預測值', alpha=0.5, markersize=10)
    
    plt.xlabel('時間')
    plt.ylabel('分類結果 (0:正常, 1:攻擊)')
    plt.title('DDoS 攻擊檢測結果')
    plt.legend()
    plt.grid(True)
    
    # 添加背景顏色以區分正確和錯誤的預測
    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            color = 'green'
        else:
            color = 'red'
        plt.axvspan(i-0.5, i+0.5, alpha=0.1, color=color)
    
    plt.tight_layout()
    detection_results_path = os.path.join(plots_dir, 'detection_results.png')
    plt.savefig(detection_results_path)
    plt.close()
    
    print(f"檢測結果圖表已保存至 '{detection_results_path}'")


def get_memory_usage():
    """獲取當前進程的記憶體使用情況"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # 轉換為 MB


if __name__ == "__main__":
    # 執行獨立測試
    model_path = "ddos_detection_model.pth"
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        results = test_model_on_file(model_path, test_file, output_file="test_results.json")
    else:
        print("使用方法: python test.py <test_data_file.csv>")