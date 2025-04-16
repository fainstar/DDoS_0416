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

# 設定中文字型
font_path = "TFT/MSGOTHIC.TTF"
prop = font_manager.FontProperties(fname=font_path)
rc('font', family=prop.get_name())

def get_memory_usage():
    """獲取當前進程的記憶體使用情況"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # 轉換為MB

def train_model(model, train_loader, val_loader, criterion=None, optimizer=None, 
               epochs=10, device='cuda', model_save_path=None):
    """訓練 DDoS 檢測模型"""
    
    # 如果未提供損失函數和優化器，則創建默認值
    if criterion is None:
        criterion = nn.BCELoss()
    
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 檢查 GPU 可用性
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print("使用 GPU 訓練")
    else:
        device = torch.device('cpu')
        print("使用 CPU 訓練")
    
    model.to(device)
    
    # 跟蹤訓練過程
    train_losses = []
    val_losses = []
    val_accuracies = []
    memory_usage = []
    best_val_accuracy = 0.0
    
    # 記錄開始時間
    start_time = time.time()
    
    for epoch in range(epochs):
        # 訓練階段
        model.train()
        train_loss = 0.0
        epoch_start_time = time.time()
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 前向傳播
            outputs = model(inputs)
            
            # 計算損失
            loss = criterion(outputs.squeeze(), labels)
            
            # 反向傳播
            loss.backward()
            
            # 更新權重
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            # 記錄記憶體使用
            memory_usage.append({
                'epoch': epoch + 1,
                'stage': 'training',
                'memory_mb': get_memory_usage(),
                'time': time.time() - start_time
            })
        
        # 計算平均訓練損失
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 驗證階段
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device, return_metrics=True)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # 記錄驗證階段的記憶體使用
        memory_usage.append({
            'epoch': epoch + 1,
            'stage': 'validation',
            'memory_mb': get_memory_usage(),
            'time': time.time() - start_time
        })
        
        # 如果達到更好的驗證準確率，保存模型
        if model_save_path and val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"發現更好的模型，保存到 {model_save_path}")
            torch.save(model.state_dict(), model_save_path)
        
        # 計算每個 epoch 的執行時間
        epoch_time = time.time() - epoch_start_time
        
        # 打印當前 epoch 的訓練狀態
        print(f"Epoch {epoch+1}/{epochs} - "
              f"訓練損失: {train_loss:.4f} - "
              f"驗證損失: {val_loss:.4f} - "
              f"驗證準確率: {val_accuracy:.4f} - "
              f"執行時間: {epoch_time:.2f}秒 - "
              f"記憶體使用: {get_memory_usage():.2f}MB")
    
    # 繪製訓練過程圖表
    plot_training_history(train_losses, val_losses, val_accuracies, memory_usage)
    
    # 如果沒有在訓練過程中保存最佳模型，則保存最終模型
    if model_save_path and val_accuracies[-1] > best_val_accuracy:
        torch.save(model.state_dict(), model_save_path)
        print(f"最終模型已保存到 {model_save_path}")
    
    return model, {
        'train_losses': train_losses, 
        'val_losses': val_losses, 
        'val_accuracies': val_accuracies,
        'memory_usage': memory_usage
    }


def evaluate_model(model, data_loader, criterion=None, device='cuda', return_metrics=False):
    """評估模型性能"""
    
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
    val_loss = 0.0
    val_correct = 0
    
    start_memory = get_memory_usage()
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向傳播
            outputs = model(inputs)
            
            # 計算損失
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item() * inputs.size(0)
            
            # 計算準確率
            predicted = (outputs.squeeze() > 0.5).float()
            val_correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    end_memory = get_memory_usage()
    memory_change = end_memory - start_memory
    
    # 計算平均損失和準確率
    val_loss = val_loss / len(data_loader.dataset)
    val_accuracy = val_correct / len(data_loader.dataset)
    
    # 如果只需要返回指標，不打印詳細信息
    if return_metrics:
        return val_loss, val_accuracy
    
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
    
    # 返回詳細結果
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'confusion_matrix': conf_matrix,
        'classification_report': report,
        'memory_change': memory_change
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
    ax2.set_title('驗證準確率變化')
    ax2.legend()

    # 3. 記憶體使用量（按階段）
    times = [m['time'] for m in memory_usage]
    memory_mb = [m['memory_mb'] for m in memory_usage]
    stages = [m['stage'] for m in memory_usage]

    for stage in ['training', 'validation']:
        stage_indices = [i for i, s in enumerate(stages) if s == stage]
        if stage_indices:
            stage_times = [times[i] for i in stage_indices]
            stage_memory = [memory_mb[i] for i in stage_indices]
            label = '訓練階段' if stage == 'training' else '驗證階段'
            ax3.plot(stage_times, stage_memory, label=label, alpha=0.7)

    ax3.set_xlabel('執行時間（秒）')
    ax3.set_ylabel('記憶體使用量（MB）')
    ax3.set_title('記憶體使用變化')
    ax3.legend()

    # 4. 平均記憶體使用量（按epoch）
    epochs = sorted(list(set(m['epoch'] for m in memory_usage)))
    avg_memory_by_epoch = []

    for epoch in epochs:
        epoch_memory = [m['memory_mb'] for m in memory_usage if m['epoch'] == epoch]
        avg_memory_by_epoch.append(np.mean(epoch_memory))

    ax4.bar(epochs, avg_memory_by_epoch)
    ax4.set_xlabel('訓練週期')
    ax4.set_ylabel('平均記憶體使用量（MB）')
    ax4.set_title('每個週期的平均記憶體使用量')

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