#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
訓練與評估模塊 - 處理模型訓練與評估功能
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import psutil
import time
import os
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import gc
import torch
from tqdm import tqdm

# 設定中文字型
font_path = "TFT/MSGOTHIC.TTF"
prop = font_manager.FontProperties(fname=font_path)
rc('font', family=prop.get_name())

def get_memory_usage():
    """獲取當前進程的記憶體使用情況"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # 轉換為MB

def clear_memory():
    """清理記憶體，釋放未使用的資源"""
    # 記錄清理前的記憶體使用量
    before_clear = get_memory_usage()
    
    # 清理 Python 的垃圾回收
    gc.collect()
    
    # 如果有 CUDA 可用，清理 GPU 記憶體
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 記錄清理後的記憶體使用量
    after_clear = get_memory_usage()
    memory_freed = before_clear - after_clear
    
    if memory_freed > 0:
        print(f"記憶體清理完成：釋放 {memory_freed:.2f} MB")
    else:
        print("記憶體清理完成：無可釋放空間")

def get_gpu_memory_usage():
    """獲取 GPU 記憶體使用情況"""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024 / 1024,  # MB
            'reserved': torch.cuda.memory_reserved() / 1024 / 1024     # MB
        }
    return None

def train_model(model, train_loader, val_loader, criterion=None, optimizer=None, 
               epochs=10, device='cuda', model_save_path=None):
    """訓練 DDoS 檢測模型"""
    import csv
    import os
    
    if criterion is None:
        criterion = nn.BCELoss()
    
    if optimizer is None:
        optimizer = optim.Adam(model.parameters())
    
    model = model.to(device)
    train_losses = []
    val_losses = []
    val_accuracies = []
    memory_usage = []
    start_time = time.time()
    
    # 確保 logs 目錄存在
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # 創建 CSV 文件保存訓練記錄
    model_type = getattr(model, 'model_type', 'unknown')
    csv_filename = os.path.join(logs_dir, f'{model_type}_training_log.csv')
    
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # 寫入 CSV 頭部
        csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_accuracy', 
                             'time_seconds', 'memory_allocated_mb', 'memory_reserved_mb'])
    
    print(f"初始 GPU 記憶體使用: {get_gpu_memory_usage()['allocated']:.2f} MB")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            target = target.view(-1, 1)  # 調整目標張量維度為 [batch_size, 1]
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                gpu_mem = get_gpu_memory_usage()
                memory_usage.append({
                    'time': time.time() - start_time,
                    'memory_allocated': gpu_mem['allocated'],
                    'memory_reserved': gpu_mem['reserved'],
                    'epoch': epoch,
                    'batch': batch_idx,
                    'stage': 'training'
                })
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 驗證階段
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                target = target.view(-1, 1)  # 調整目標張量維度為 [batch_size, 1]
                output = model(data)
                val_loss += criterion(output, target).item()
                predicted = (output > 0.5).float()
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        accuracy = correct / total
        val_accuracies.append(accuracy)
        
        gpu_mem = get_gpu_memory_usage()
        memory_usage.append({
            'time': time.time() - start_time,
            'memory_allocated': gpu_mem['allocated'],
            'memory_reserved': gpu_mem['reserved'],
            'epoch': epoch,
            'stage': 'validation'
        })
        
        epoch_time = time.time() - epoch_start_time
        
        # 保存這個 epoch 的訓練記錄到 CSV
        with open(csv_filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([
                epoch + 1,
                avg_train_loss,
                avg_val_loss,
                accuracy,
                epoch_time,
                gpu_mem["allocated"],
                gpu_mem["reserved"]
            ])
        
        print(f'Epoch {epoch+1}/{epochs} - 訓練損失: {avg_train_loss:.4f} - 驗證損失: {avg_val_loss:.4f} - '
              f'驗證準確率: {accuracy:.4f} - 執行時間: {epoch_time:.2f}秒 - '
              f'記憶體使用: {gpu_mem["allocated"]:.2f}MB')
        
        clear_memory()
        
        # 移除詢問用戶是否繼續訓練的代碼
    
    if model_save_path:
        torch.save(model.state_dict(), model_save_path)
        print(f"模型已保存至 '{model_save_path}'")
        print(f"訓練記錄已保存至 '{csv_filename}'")
    
    return model, {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_acc': val_accuracies,
        'memory_usage': memory_usage
    }


def evaluate_model(model, data_loader, criterion=None, device='cuda', return_metrics=False, return_predictions=False):
    """評估模型性能"""
    import csv
    import os
    
    if criterion is None:
        criterion = nn.BCELoss()
    
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []  # 新增，用於保存預測概率
    val_loss = 0.0
    val_correct = 0
    
    # 確保 logs 目錄存在
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # 獲取模型類型
    model_type = getattr(model, 'model_type', 'unknown')
    
    # 記錄評估開始時間
    start_time = time.time()
    start_memory = get_memory_usage()
    
    # 逐批次評估
    batch_metrics = []
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向傳播
            outputs = model(inputs)
            
            # 計算損失
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item() * inputs.size(0)
            
            # 計算準確率
            predicted = (outputs.squeeze() > 0.5).float()
            batch_correct = (predicted == labels).sum().item()
            val_correct += batch_correct
            
            # 保存預測結果和真實標籤
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.squeeze().cpu().numpy())  # 新增保存預測概率
            
            # 記錄每個批次的指標
            batch_accuracy = batch_correct / len(labels)
            batch_time = time.time() - start_time
            gpu_mem = get_gpu_memory_usage() if torch.cuda.is_available() else {"allocated": 0, "reserved": 0}
            
            batch_metrics.append({
                'batch': batch_idx,
                'loss': loss.item(),
                'accuracy': batch_accuracy,
                'time': batch_time,
                'memory_allocated': gpu_mem['allocated'] if gpu_mem else 0,
                'memory_reserved': gpu_mem['reserved'] if gpu_mem else 0
            })
    
    # 計算評估總時間
    end_time = time.time()
    evaluation_time = end_time - start_time
    
    end_memory = get_memory_usage()
    memory_change = end_memory - start_memory
    
    # 計算平均損失和準確率
    val_loss = val_loss / len(data_loader.dataset)
    val_accuracy = val_correct / len(data_loader.dataset)
    
    # 如果只需要返回指標，不打印詳細信息
    if return_metrics:
        return val_loss, val_accuracy
    
    # 如果需要返回預測結果用於繪製ROC曲線
    if return_predictions:
        results = {
            'accuracy': val_accuracy,
            'loss': val_loss,
            'memory_change': memory_change,
            'evaluation_time': evaluation_time
        }
        return results, all_preds, all_labels, all_probs
    
    # 計算評估指標
    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    
    # 打印評估結果
    print(f"\n模型評估結果:")
    print(f"準確率: {accuracy:.4f}")
    print(f"評估過程記憶體使用變化: {memory_change:.2f}MB")
    print("\n混淆矩陣:")
    print(conf_matrix)
    print("\n分類報告:")
    print(report)
    
    # 計算更多指標
    tn, fp, fn, tp = conf_matrix.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 計算誤報率 (False Positive Rate)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"\n偵測性能指標:")
    print(f"精確率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1 分數: {f1:.4f}")
    print(f"誤報率 (FPR): {fpr:.4f}")
    
    # 保存評估結果到 CSV
    csv_filename = os.path.join(logs_dir, f'{model_type}_evaluation_result.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            'accuracy', 'precision', 'recall', 'f1', 'fpr',
            'evaluation_time', 'memory_change', 'tn', 'fp', 'fn', 'tp'
        ])
        csv_writer.writerow([
            accuracy, precision, recall, f1, fpr,
            evaluation_time, memory_change, tn, fp, fn, tp
        ])
    
    # 保存批次級別的評估指標
    batch_csv_filename = os.path.join(logs_dir, f'{model_type}_batch_evaluation.csv')
    with open(batch_csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            'batch', 'loss', 'accuracy', 'time', 'memory_allocated', 'memory_reserved'
        ])
        for metric in batch_metrics:
            csv_writer.writerow([
                metric['batch'], 
                metric['loss'], 
                metric['accuracy'], 
                metric['time'], 
                metric['memory_allocated'],
                metric['memory_reserved']
            ])
    
    print(f"評估結果已保存至 '{csv_filename}'")
    print(f"批次評估數據已保存至 '{batch_csv_filename}'")
    
    # 返回詳細結果
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'confusion_matrix': conf_matrix,
        'classification_report': report,
        'memory_change': memory_change,
        'evaluation_time': evaluation_time
    }


def plot_training_history(train_losses, val_losses, val_accuracies, memory_usage):
    """繪製訓練歷史圖表"""
    # 確保 plots 目錄存在
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    plt.style.use('seaborn-v0_8-darkgrid')

    # 設定中文字型
    font_path = "TFT/MSGOTHIC.TTF"
    prop = font_manager.FontProperties(fname=font_path)
    rc('font', family=prop.get_name())

    # 創建一個 2x2 的子圖佈局
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 損失曲線
    ax1.plot(train_losses, label='訓練損失')
    ax1.plot(val_losses, label='驗證損失')
    ax1.set_xlabel('訓練週期')
    ax1.set_ylabel('損失值')
    ax1.set_title('訓練和驗證損失變化')
    ax1.legend()

    # 2. 準確率曲線
    ax2.plot(val_accuracies, label='驗證準確率')
    ax2.set_xlabel('訓練週期')
    ax2.set_ylabel('準確率')
    ax2.setTitle('驗證準確率變化')
    ax2.legend()

    # 3. 記憶體使用量（按階段）
    times = [m['time'] for m in memory_usage]
    memory_allocated = [m['memory_allocated'] for m in memory_usage]
    memory_reserved = [m['memory_reserved'] for m in memory_usage]
    stages = [m['stage'] for m in memory_usage]

    for stage in ['training', 'validation']:
        stage_indices = [i for i, s in enumerate(stages) if s == stage]
        if stage_indices:
            stage_times = [times[i] for i in stage_indices]
            stage_allocated = [memory_allocated[i] for i in stage_indices]
            stage_reserved = [memory_reserved[i] for i in stage_indices]
            label_allocated = f'已分配記憶體 ({stage})'
            label_reserved = f'保留記憶體 ({stage})'
            ax3.plot(stage_times, stage_allocated, label=label_allocated, alpha=0.7)
            ax3.plot(stage_times, stage_reserved, label=label_reserved, linestyle='--', alpha=0.5)

    ax3.set_xlabel('執行時間（秒）')
    ax3.set_ylabel('記憶體使用量（MB）')
    ax3.set_title('GPU 記憶體使用變化')
    ax3.legend()

    # 4. 平均記憶體使用量（按epoch）
    epochs = sorted(list(set(m['epoch'] for m in memory_usage)))
    avg_allocated_by_epoch = []
    avg_reserved_by_epoch = []

    for epoch in epochs:
        epoch_allocated = [m['memory_allocated'] for m in memory_usage if m['epoch'] == epoch]
        epoch_reserved = [m['memory_reserved'] for m in memory_usage if m['epoch'] == epoch]
        avg_allocated_by_epoch.append(np.mean(epoch_allocated))
        avg_reserved_by_epoch.append(np.mean(epoch_reserved))

    x = np.arange(len(epochs))
    width = 0.35
    ax4.bar(x - width/2, avg_allocated_by_epoch, width, label='已分配記憶體')
    ax4.bar(x + width/2, avg_reserved_by_epoch, width, label='保留記憶體')
    ax4.set_xlabel('訓練週期')
    ax4.set_ylabel('平均記憶體使用量（MB）')
    ax4.set_title('每個週期的平均 GPU 記憶體使用量')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'Epoch {e}' for e in epochs])
    ax4.legend()

    plt.tight_layout()
    history_path = os.path.join(plots_dir, 'training_history.png')
    plt.savefig(history_path)
    plt.close()

    print(f"訓練歷史圖表已保存至 '{history_path}'")

def plot_training_metrics(history):
    """繪製訓練指標圖表"""
    # 確保 plots 目錄存在
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    plt.figure(figsize=(12, 8))

    # 設定中文字型
    font_path = "TFT/MSGOTHIC.TTF"
    prop = font_manager.FontProperties(fname=font_path)
    rc('font', family=prop.get_name())

    # 繪製損失曲線
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='訓練損失')
    plt.plot(history['val_loss'], label='驗證損失')
    plt.title('模型訓練過程中的損失變化')
    plt.xlabel('訓練輪次')
    plt.ylabel('損失值')
    plt.legend()

    # 繪製準確率曲線
    plt.subplot(2, 1, 2)
    plt.plot(history['train_acc'], label='訓練準確率')
    plt.plot(history['val_acc'], label='驗證準確率')
    plt.title('模型訓練過程中的準確率變化')
    plt.xlabel('訓練輪次')
    plt.ylabel('準確率')
    plt.legend()

    plt.tight_layout()
    metrics_path = os.path.join(plots_dir, 'training_metrics.png')
    plt.savefig(metrics_path)
    plt.close()
    print(f"訓練指標圖表已保存至 '{metrics_path}'")