#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DDoS 攻擊檢測系統主程序 - 模組化結構
日期: 2025-04-17
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import argparse
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import psutil

# 設定中文字型
font_path = "TFT/MSGOTHIC.TTF"
prop = font_manager.FontProperties(fname=font_path)
rc('font', family=prop.get_name())

# 導入自定義模塊
from data_processor import load_network_data, preprocess_data, prepare_data_loaders
from models import create_model
from trainer import train_model, evaluate_model
from test import load_trained_model, real_time_prediction
from plots_generator import create_comparison_charts


def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='DDoS 攻擊檢測系統')
    
    # 基本參數
    parser.add_argument('--data', type=str, default='AllMerged.csv', help='數據文件路徑')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'test', 'predict', 'compare'], help='運行模式')
    parser.add_argument('--model_type', type=str, default='mamba', 
                        choices=['mamba', 'cnnlstm', 'dnn'], help='模型類型')
    parser.add_argument('--model_path', type=str, default='ddos_detection_model.pth', help='模型保存/載入路徑')
    
    # 訓練參數
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--epochs', type=int, default=5, help='訓練輪數')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='學習率')
    parser.add_argument('--max_samples', type=int, default=500000, help='最大樣本數量 (0 表示使用所有數據)')
    
    # 其他參數
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='訓練設備')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    
    args = parser.parse_args()
    
    # 自動選擇設備
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return args


def set_seed(seed):
    """設置隨機種子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_memory_usage():
    """獲取當前進程的記憶體使用情況"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # 轉換為 MB


def train_ddos_model(args):
    """訓練 DDoS 檢測模型"""
    print(f"\n{'='*50}")
    print(f"DDoS 攻擊檢測系統 - 訓練 {args.model_type.upper()} 模型")
    print(f"{'='*50}")
    
    # 載入數據
    data = load_network_data(args.data)
    
    # 數據預處理
    X, y = preprocess_data(data)
    
    # 如果指定了最大樣本數，進行采樣
    if args.max_samples > 0 and len(X) > args.max_samples:
        print(f"為了提高效率，隨機抽樣 {args.max_samples} 條數據...")
        indices = np.random.choice(len(X), args.max_samples, replace=False)
        X = X[indices]
        y = y[indices]
    
    # 準備數據加載器
    train_loader, val_loader, test_loader = prepare_data_loaders(
        X, y, batch_size=args.batch_size
    )
    
    # 創建模型
    input_dim = X.shape[1]
    try:
        model = create_model(args.model_type, input_dim)
        print(f"成功初始化 {args.model_type} 模型")
    except Exception as e:
        print(f"模型初始化失敗，錯誤: {e}")
        return None, None
    
    # 定義損失函數和優化器
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 訓練模型
    print("\n開始訓練模型...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=args.epochs,
        device=args.device,
        model_save_path=args.model_path
    )
    
    # 評估模型
    print("\n評估模型性能...")
    results = evaluate_model(model, test_loader, device=args.device)
    
    print(f"\n訓練完成! 模型已保存至 '{args.model_path}'")
    
    return model, results


def test_ddos_model(args):
    """測試 DDoS 檢測模型"""
    print(f"\n{'='*50}")
    print("DDoS 攻擊檢測系統 - 測試模式")
    print(f"{'='*50}")
    
    # 首先確認模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"錯誤: 模型文件 '{args.model_path}' 不存在。請先訓練模型或指定正確的模型路徑。")
        return None
    
    # 載入數據
    data = load_network_data(args.data)
    
    # 數據預處理
    X, y = preprocess_data(data)
    
    # 如果指定了最大樣本數，進行采樣
    if args.max_samples > 0 and len(X) > args.max_samples:
        print(f"為了提高效率，隨機抽樣 {args.max_samples} 條數據...")
        indices = np.random.choice(len(X), args.max_samples, replace=False)
        X = X[indices]
        y = y[indices]
    
    # 創建測試數據加載器
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size
    )
    
    # 加載模型
    input_dim = X.shape[1]
    model = load_trained_model(args.model_path, args.model_type, input_dim)
    
    if model is None:
        print("模型加載失敗，無法進行測試")
        return None
    
    # 評估模型性能
    results = evaluate_model(model, test_loader, device=args.device)
    
    # 保存測試結果
    output_file = "test_results.json"
    try:
        import json
        with open(output_file, 'w') as f:
            json.dump({
                'accuracy': float(results['accuracy']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1': float(results['f1']),
                'fpr': float(results['fpr']),
                'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'test_samples': len(test_dataset)
            }, f, indent=4)
        print(f"測試結果已保存至 {output_file}")
    except Exception as e:
        print(f"保存測試結果時出錯: {str(e)}")
        
    return results


def predict_flow(args):
    """使用模型進行預測"""
    print(f"\n{'='*50}")
    print("DDoS 攻擊檢測系統 - 預測模式")
    print(f"{'='*50}")
    
    # 檢查模型文件
    if not os.path.exists(args.model_path):
        print(f"錯誤: 模型文件 '{args.model_path}' 不存在。請先訓練模型或指定正確的模型路徑。")
        return
    
    # 檢查數據文件
    if not os.path.exists(args.data):
        print(f"錯誤: 數據文件 '{args.data}' 不存在。")
        return
    
    # 載入模型
    # 我們需要知道輸入維度，先讀取數據
    df = pd.read_csv(args.data, low_memory=False, nrows=1)
    estimated_input_dim = len(df.columns) - 1  # 假設最後一列是標籤
    
    model = load_trained_model(args.model_path, args.model_type, estimated_input_dim)
    
    if model is None:
        print("模型加載失敗，無法進行預測")
        return
    
    # 進行預測
    from data_processor import preprocess_data
    results = real_time_prediction(model, args.data, preprocess_data)
    
    # 這裡可以添加更多代碼來處理預測結果
    # 例如，生成報告，發送警報等
    
    print("\n預測完成!")


def compare_models(args):
    """比較不同模型架構的性能"""
    print(f"\n{'='*50}")
    print("DDoS 攻擊檢測系統 - 模型比較模式")
    print(f"{'='*50}")

    # 保存原始參數
    original_model_type = args.model_type
    original_model_path = args.model_path

    model_types = ['mamba', 'cnnlstm']
    model_paths = ['mamba_model.pth', 'cnnlstm_model.pth']
    model_results = {}

    # 依次訓練並評估每個模型
    for i, model_type in enumerate(model_types):
        print(f"\n{'='*50}")
        print(f"訓練模型 {model_type.upper()}")
        print(f"{'='*50}")

        # 更新模型類型和保存路徑
        args.model_type = model_type
        args.model_path = model_paths[i]

        # 記錄開始時間和記憶體
        start_time = time.time()
        start_memory = get_memory_usage()

        # 訓練模型
        _, results = train_ddos_model(args)

        # 記錄結束時間和記憶體
        end_time = time.time()
        end_memory = get_memory_usage()

        if results is not None:
            model_results[model_type] = results
            model_results[model_type].update({
                'time': end_time - start_time,
                'memory_change': end_memory - start_memory
            })

    # 恢復原始參數
    args.model_type = original_model_type
    args.model_path = original_model_path

    # 生成比較圖表
    from plots_generator import create_comparison_charts, create_performance_comparison_chart
    create_comparison_charts(model_results)
    create_performance_comparison_chart(model_results)

    # 輸出比較結果摘要
    print("\n模型性能比較摘要:")
    for model_type, results in model_results.items():
        print(f"\n{model_type.upper()} 模型:")
        print(f"  訓練時間: {results['time']:.2f} 秒")
        print(f"  記憶體變化: {results['memory_change']:.2f} MB")
        print(f"  準確率 (Accuracy): {results['accuracy']:.4f}")
        print(f"  精確率 (Precision): {results['precision']:.4f}")
        print(f"  召回率 (Recall): {results['recall']:.4f}")
        print(f"  F1 分數: {results['f1']:.4f}")
        print(f"  誤報率 (FPR): {results['fpr']:.4f}")

    return model_results


def ensure_plots_dir():
    """確保 plots 目錄存在"""
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    return plots_dir


def main():
    """主函數"""
    # 使用命令行參數
    args = parse_args()
    
    # 設置隨機種子
    set_seed(args.seed)
    
    # 根據指定模式執行相應功能
    if args.mode == 'train':
        train_ddos_model(args)
    elif args.mode == 'test':
        test_ddos_model(args)
    elif args.mode == 'predict':
        predict_flow(args)
    elif args.mode == 'compare':
        compare_models(args)
    else:
        print(f"不支持的模式: {args.mode}")


if __name__ == "__main__":
    main()