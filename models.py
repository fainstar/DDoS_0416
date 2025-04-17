#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型定義模塊 - 包含神經網絡模型架構定義
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mambapy.mamba import Mamba, MambaConfig


def count_parameters(model):
    """計算模型的參數量
    Args:
        model: PyTorch 模型
    Returns:
        total_params: 總參數量
        trainable_params: 可訓練參數量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


class DDoSDetectionModel(nn.Module):
    """基於 Mamba 的 DDoS 攻擊檢測模型"""
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=1):
        super(DDoSDetectionModel, self).__init__()
        
        # 模型結構
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 使用 MambaConfig 來配置 Mamba 模型
        mamba_config = MambaConfig(
            d_model=hidden_dim,
            d_state=32,  # 增加狀態空間維度
            d_conv=16,   # 增加卷積維度
            expand_factor=4,
            n_layers=4   # 增加層數
        )
        
        # 使用 config 對象來初始化 Mamba
        self.sequence_model = Mamba(config=mamba_config)
        self.model_type = 'mamba'
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 確保數據在正確的設備上
        device = x.device
        
        # 確保模型參數在與輸入相同的設備上
        if next(self.parameters()).device != device:
            self.to(device)
        
        # 改變輸入形狀以適應序列模型: [batch_size, seq_len, d_model]
        x = x.unsqueeze(1)  # 將特徵視為序列長度為1: [batch_size, 1, feature_dim]
        
        # 特徵投影到 hidden_dim
        x = self.input_projection(x)  # [batch_size, 1, hidden_dim]
        
        # 序列模型處理
        x = self.sequence_model(x)  # [batch_size, 1, hidden_dim]

        # 取最後序列位置的輸出
        x = x.squeeze(1)  # [batch_size, hidden_dim]
        
        # 輸出層
        x = self.output_layer(x)  # [batch_size, output_dim]
        x = self.sigmoid(x)
        
        return x


class CNNLSTMModel(nn.Module):
    """基於 CNN-LSTM 架構的 DDoS 攻擊檢測模型"""
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=1, seq_len=10):
        super(CNNLSTMModel, self).__init__()
        
        # 特徵轉換層 - 將輸入特徵轉換為適合 CNN 處理的格式
        self.seq_len = seq_len
        self.feature_transform = nn.Linear(input_dim, seq_len)
        self.batch_norm = nn.BatchNorm1d(seq_len)
        
        # CNN 層
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)  # 新增
        self.residual = nn.Conv1d(32, 64, kernel_size=1)  # shortcut 讓維度對齊
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.4)
        
        # 計算卷積後的序列長度
        cnn_out_len = seq_len // 2  # 經過池化層後長度減半
        
        # LSTM 層
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.4,
            bidirectional=True
        )
        
        # 全連接層
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 因為是雙向LSTM
        self.dropout2 = nn.Dropout(0.3)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 確保數據在正確的設備上
        device = x.device
        
        # 確保模型參數在與輸入相同的設備上
        if next(self.parameters()).device != device:
            self.to(device)
        
        # 首先將特徵轉換為CNN能處理的序列形式
        x = self.feature_transform(x)  # [batch_size, seq_len]
        x = self.batch_norm(x)
        
        # 添加通道維度
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len]
        
        # CNN層
        x = F.relu(self.conv1(x))  # [batch_size, 32, seq_len]
        x = self.pool(x)  # [batch_size, 32, seq_len/2]
        x = F.relu(self.conv2(x))  # [batch_size, 64, seq_len/2]
        x = self.pool(x)  # [batch_size, 64, seq_len/4]
        x = F.relu(self.conv3(x))  # [batch_size, 128, seq_len/4]
        x = self.pool(x)  # [batch_size, 128, seq_len/8]
        x = self.dropout1(x)
        
        # 重塑以適應LSTM層 [batch_size, channels, seq_len] -> [batch_size, seq_len, channels]
        x = x.transpose(1, 2)  # [batch_size, seq_len/2, 64]
        
        # LSTM層
        x, _ = self.lstm(x)  # [batch_size, seq_len/2, hidden_dim*2]
        
        # 取最後一個時間步的輸出
        x = x[:, -1, :]  # [batch_size, hidden_dim*2]
        
        # 全連接層
        x = F.relu(self.fc(x))
        x = self.dropout2(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        
        return x


class DeepNeuralNetDDoSModel(nn.Module):
    """基於 DNN 的 DDoS 攻擊檢測模型替代方案"""
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], output_dim=1, dropout_rate=0.3):
        super(DeepNeuralNetDDoSModel, self).__init__()
        
        # 構建多層神經網絡
        layers = []
        
        # 輸入層
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.Dropout(dropout_rate))
        
        # 隱藏層
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.Dropout(dropout_rate))
        
        # 輸出層
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


def create_model(model_type, input_dim):
    """根據指定的模型類型創建模型實例"""
    if model_type == 'mamba':
        try:
            model = DDoSDetectionModel(input_dim=input_dim)
            total_params, trainable_params = count_parameters(model)
            print(f"\n{model_type.upper()} 模型參數統計：")
            print(f"總參數量: {total_params:,}")
            print(f"可訓練參數量: {trainable_params:,}")
            return model
        except Exception as e:
            print(f"無法創建 Mamba 模型: {str(e)}，使用 DNN 替代")
            model = DeepNeuralNetDDoSModel(input_dim=input_dim)
            total_params, trainable_params = count_parameters(model)
            print(f"\nDNN 模型參數統計：")
            print(f"總參數量: {total_params:,}")
            print(f"可訓練參數量: {trainable_params:,}")
            return model
    elif model_type == 'cnnlstm':
        model = CNNLSTMModel(input_dim=input_dim)
        total_params, trainable_params = count_parameters(model)
        print(f"\n{model_type.upper()} 模型參數統計：")
        print(f"總參數量: {total_params:,}")
        print(f"可訓練參數量: {trainable_params:,}")
        return model
    elif model_type == 'dnn':
        model = DeepNeuralNetDDoSModel(input_dim=input_dim)
        total_params, trainable_params = count_parameters(model)
        print(f"\n{model_type.upper()} 模型參數統計：")
        print(f"總參數量: {total_params:,}")
        print(f"可訓練參數量: {trainable_params:,}")
        return model
    else:
        raise ValueError(f"不支持的模型類型: {model_type}")