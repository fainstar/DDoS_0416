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
from models import create_model, count_parameters
from trainer import train_model, evaluate_model
from test import load_trained_model, real_time_prediction
from plots_generator import create_comparison_charts, plot_feature_importance, plot_multiple_model_feature_importance, plot_feature_correlation


def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='DDoS 攻擊檢測系統')
    
    # 基本參數
    parser.add_argument('--data', type=str, default='AllMerged.csv', help='數據文件路徑')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'test', 'predict', 'compare', 'feature_importance'], help='運行模式')
    parser.add_argument('--model_type', type=str, default='transformer', 
                        choices=['transformer', 'mamba', 'cnnlstm', 'dnn', 'compare', 'random_forest', 'permutation'], help='模型類型')
    parser.add_argument('--model_path', type=str, default='ddos_detection_model.pth', help='模型保存/載入路徑')
    
    # 訓練參數
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--epochs', type=int, default=5, help='訓練輪數')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='學習率')
    parser.add_argument('--max_samples', type=int, default=500000, help='最大樣本數量 (0 表示使用所有數據)')
    
    # 特徵重要性參數
    parser.add_argument('--n_features', type=int, default=15, help='顯示的特徵數量')
    parser.add_argument('--importance_method', type=str, default='random_forest', 
                       choices=['random_forest', 'permutation'], help='特徵重要性計算方法')
    
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


def get_gpu_memory():
    """獲取當前 GPU 記憶體使用情況"""
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # 轉換為 MB
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024  # 轉換為 MB
        return {
            'allocated': gpu_memory_allocated,
            'reserved': gpu_memory_reserved
        }
    return None


def train_ddos_model(args):
    """訓練 DDoS 檢測模型"""
    print(f"\n{'='*50}")
    # 顯示 TRANSFORMER 而不是 MAMBA
    model_display_name = "TRANSFORMER" if args.model_type.lower() == "mamba" else args.model_type.upper()
    print(f"DDoS 攻擊檢測系統 - 訓練 {model_display_name} 模型")
    print(f"{'='*50}")
    
    # 記錄初始 GPU 記憶體
    initial_gpu_memory = get_gpu_memory()
    if initial_gpu_memory:
        print(f"初始 GPU 記憶體使用：")
        print(f"已分配：{initial_gpu_memory['allocated']:.2f} MB")
        print(f"已預留：{initial_gpu_memory['reserved']:.2f} MB")
    
    # 載入數據
    data = load_network_data(args.data)
    
    # 數據預處理
    X, y = preprocess_data(data)
    
    # 如果指定了最大樣本數，進行采樣
    if args.max_samples > 0 and len(X) > args.max_samples:
        print(f"為了提高效率，順序抽樣 {args.max_samples} 條數據...")
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
    
    # 評估模型性能和記錄最終 GPU 記憶體使用
    print("\n評估模型性能...")
    results = evaluate_model(model, test_loader, device=args.device)
    
    final_gpu_memory = get_gpu_memory()
    if final_gpu_memory:
        print(f"\nGPU 記憶體使用情況：")
        print(f"已分配：{final_gpu_memory['allocated']:.2f} MB")
        print(f"已預留：{final_gpu_memory['reserved']:.2f} MB")
        print(f"訓練期間增加：{final_gpu_memory['allocated'] - initial_gpu_memory['allocated']:.2f} MB")
    
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

    model_types = ['transformer', 'cnnlstm']  # 使用 transformer 替代原來的 mamba
    model_paths = ['transformer_model.pth', 'cnnlstm_model.pth']
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

        # 訓練模型並獲取模型實例和結果
        model, results = train_ddos_model(args)

        # 計算模型參數量
        if model is not None:
            total_params, trainable_params = count_parameters(model)
        else:
            total_params = trainable_params = 0

        # 記錄結束時間和記憶體
        end_time = time.time()
        end_memory = get_memory_usage()

        if results is not None:
            model_results[model_type] = results
            model_results[model_type].update({
                'time': end_time - start_time,
                'memory_change': end_memory - start_memory,
                'total_params': total_params,
                'trainable_params': trainable_params
            })

    # 恢復原始參數
    args.model_type = original_model_type
    args.model_path = original_model_path

    # 從 CSV 文件生成訓練和評估圖表
    from plots_generator import plot_from_training_csv, plot_from_evaluation_csv, create_comparison_charts_from_csv
    
    # 確保 logs 目錄存在
    logs_dir = 'logs'
    if os.path.exists(logs_dir):
        # 檔名映射表，處理可能的模型類型名稱與實際檔案名稱不符的情況
        file_name_mapping = {
            'transformer': 'transformer',
            'mamba': 'transformer',  # mamba 實際使用 transformer
            'cnnlstm': 'cnnlstm'
        }
        
        # 檢查已存在的檔案
        existing_files = os.listdir(logs_dir)
        
        # 如果 cnnlstm 的檔案不存在，但 unknown 檔案存在，則使用 unknown
        if not any(f.startswith('cnnlstm_') for f in existing_files) and any(f.startswith('unknown_') for f in existing_files):
            file_name_mapping['cnnlstm'] = 'unknown'
        
        # 為每個模型生成單獨的訓練和評估圖表
        for model_type in model_types:
            file_prefix = file_name_mapping.get(model_type, model_type)
            training_log = os.path.join(logs_dir, f'{file_prefix}_training_log.csv')
            eval_result = os.path.join(logs_dir, f'{file_prefix}_evaluation_result.csv')
            batch_eval = os.path.join(logs_dir, f'{file_prefix}_batch_evaluation.csv')
            
            if os.path.exists(training_log):
                plot_from_training_csv(training_log, model_type)
                print(f"已從 {training_log} 生成訓練圖表")
            else:
                print(f"找不到模型 {model_type} 的訓練記錄文件: {training_log}")
            
            if os.path.exists(eval_result):
                if os.path.exists(batch_eval):
                    plot_from_evaluation_csv(eval_result, batch_eval, model_type)
                else:
                    plot_from_evaluation_csv(eval_result, model_name=model_type)
                print(f"已從 {eval_result} 生成評估圖表")
            else:
                print(f"找不到模型 {model_type} 的評估結果文件: {eval_result}")
        
        # 使用映射後的檔案名稱列表進行比較圖表生成
        mapped_model_types = [file_name_mapping.get(mt, mt) for mt in model_types]
        
        # 檢查是否所有模型的評估結果文件都存在
        all_models_exist = all(os.path.exists(os.path.join(logs_dir, f'{mt}_evaluation_result.csv')) for mt in mapped_model_types)
        
        if all_models_exist:
            create_comparison_charts_from_csv(mapped_model_types)
            print(f"已從 CSV 文件生成所有比較圖表")
        else:
            print(f"無法生成比較圖表，因為部分評估結果文件不存在")
    else:
        print(f"找不到日誌目錄: {logs_dir}")
    
    # 保留兼容舊版本的行為
    from plots_generator import create_comparison_charts, create_confusion_matrices_chart, create_performance_comparison_chart
    create_comparison_charts(model_results)
    create_confusion_matrices_chart(model_results)
    create_performance_comparison_chart(model_results)

    # 輸出比較結果摘要
    print("\n模型性能比較摘要:")
    for model_type, results in model_results.items():
        print(f"\n{model_type.upper()} 模型:")
        print(f"  訓練時間: {results['time']:.2f} 秒")
        print(f"  記憶體變化: {results['memory_change']:.2f} MB")
        print(f"  總參數量: {results['total_params']:,}")
        print(f"  可訓練參數量: {results['trainable_params']:,}")
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


def generate_feature_importance(args):
    """生成特徵重要性圖表"""
    print(f"\n{'='*50}")
    print("DDoS 攻擊檢測系統 - 特徵重要性分析")
    print(f"{'='*50}")
    
    # 載入數據
    print("加載數據...")
    data = load_network_data(args.data)
    
    # 數據預處理
    print("預處理數據...")
    X, y = preprocess_data(data)
    
    # 獲取特徵名稱列表
    if isinstance(data, pd.DataFrame):
        # 假設最後一列是標籤列，之前的列都是特徵列
        feature_names = data.columns[:-1].tolist()
    else:
        # 如果沒有列名，就生成默認列名
        feature_names = [f'特徵 {i+1}' for i in range(X.shape[1])]
    
    # 如果指定了最大樣本數，進行采樣
    if args.max_samples > 0 and len(X) > args.max_samples:
        print(f"為了提高效率，隨機抽樣 {args.max_samples} 條數據...")
        indices = np.random.choice(len(X), args.max_samples, replace=False)
        X = X[indices]
        y = y[indices]
    
    # 根據指定的模型生成特徵重要性
    if args.model_type.lower() == 'compare':
        print("為多個模型生成特徵重要性比較...")
        plot_multiple_model_feature_importance(
            X, y, feature_names, 
            model_names=['transformer', 'cnnlstm'], 
            n_features=args.n_features
        )
    else:
        print(f"為 {args.model_type} 模型生成特徵重要性...")
        importance_df = plot_feature_importance(
            X, y, feature_names, 
            model_name=args.model_type,
            n_features=args.n_features,
            method=args.importance_method
        )
        
        # 打印特徵重要性排名
        print("\n特徵重要性排名:")
        for i, (feature, importance) in enumerate(zip(importance_df['特徵'].head(args.n_features), 
                                                    importance_df['重要性'].head(args.n_features))):
            print(f"{i+1}. {feature}: {importance:.4f}")
    
    print("\n特徵重要性分析完成！圖表已保存至 plots 目錄")


def analyze_feature_importance(args):
    """分析特徵重要性"""
    print(f"\n{'='*50}")
    print(f"DDoS 攻擊檢測系統 - 特徵重要性分析")
    print(f"{'='*50}")
    
    start_time = time.time()
    initial_memory = get_memory_usage()
    
    print(f"數據文件: {args.data}")
    print(f"特徵重要性計算方法: {args.importance_method}")
    print(f"顯示特徵數量: {args.n_features}")
    print(f"開始分析特徵重要性...")
    
    # 生成特徵重要性圖表
    importance_file = plot_feature_importance(
        data_path=args.data,
        top_n=args.n_features,
        model_type=args.importance_method
    )
    
    # 生成特徵相關性熱力圖
    correlation_file = plot_feature_correlation(
        data_path=args.data,
        top_n=args.n_features
    )
    
    # 計算執行時間和記憶體使用
    execution_time = time.time() - start_time
    final_memory = get_memory_usage()
    memory_used = final_memory - initial_memory
    
    print(f"\n{'='*50}")
    print(f"特徵重要性分析完成！")
    print(f"執行時間: {execution_time:.2f} 秒")
    print(f"記憶體使用: {memory_used:.2f} MB")
    print(f"特徵重要性圖表已保存至: {importance_file}")
    print(f"特徵相關性熱力圖已保存至: {correlation_file}")
    print(f"{'='*50}\n")


def main():
    """主函數"""
    args = parse_args()
    set_seed(args.seed)
    
    # 根據運行模式選擇操作
    if args.mode == 'train':
        train_ddos_model(args)
    elif args.mode == 'test':
        test_ddos_model(args)
    elif args.mode == 'predict':
        predict_flow(args)
    elif args.mode == 'compare':
        compare_models(args)
    elif args.mode == 'feature_importance':
        analyze_feature_importance(args)
    else:
        print(f"錯誤：未知的運行模式 '{args.mode}'")
        sys.exit(1)


if __name__ == "__main__":
    main()