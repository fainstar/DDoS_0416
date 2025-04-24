#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型定義模塊 - 包含神經網絡模型架構定義
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class TransformerModel(nn.Module):
    """基於 Transformer 的 DDoS 攻擊檢測模型"""
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=1, nhead=8, num_layers=4, dropout=0.2):
        super(TransformerModel, self).__init__()
        
        # 模型結構
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer 編碼器層
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        # Transformer 編碼器
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.model_type = 'transformer'
        
        # 輸出層
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
        
        # Transformer 編碼器處理
        x = self.transformer_encoder(x)  # [batch_size, 1, hidden_dim]

        # 取最後序列位置的輸出
        x = x.squeeze(1)  # [batch_size, hidden_dim]
        
        # 輸出層
        x = self.output_layer(x)  # [batch_size, output_dim]
        x = self.sigmoid(x)
        
        return x


class CNNLSTMModel(nn.Module):
    """基於 CNN-LSTM 混合架構的 DDoS 攻擊檢測模型
    
    此模型結合了卷積神經網絡 (CNN) 和雙向長短期記憶網絡 (BiLSTM)，
    能夠有效捕捉網絡流量特徵中的時空模式。CNN 用於提取局部特徵，
    BiLSTM 用於捕捉長期依賴關係。
    
    架構流程:
    1. 將輸入特徵轉換為序列形式
    2. 通過多層卷積提取局部特徵
    3. 使用雙向LSTM捕捉序列中的時間依賴關係
    4. 全連接層進行最終分類
    """
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=1, seq_len=10):
        super(CNNLSTMModel, self).__init__()
        
        # 設置模型類型標識
        self.model_type = 'cnnlstm'
        
        # 特徵轉換層 - 將輸入特徵轉換為適合 CNN 處理的格式
        self.seq_len = seq_len
        self.feature_transform = nn.Linear(input_dim, seq_len)
        self.batch_norm = nn.BatchNorm1d(seq_len)
        
        # CNN 層 - 提取局部特徵
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.residual = nn.Conv1d(32, 64, kernel_size=1)  # shortcut 讓維度對齊
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.4)
        
        # 計算卷積後的序列長度
        cnn_out_len = seq_len // 8  # 經過三次池化層後長度變為原始的 1/8
        
        # LSTM 層 - 捕捉時間依賴關係
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.4,
            bidirectional=True  # 雙向LSTM可以同時考慮前後文信息
        )
        
        # 全連接層 - 最終分類
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
        
        # 特徵轉換 - 將輸入轉換為序列形式
        x = self.feature_transform(x)  # [batch_size, seq_len]
        x = self.batch_norm(x)
        
        # 添加通道維度用於CNN處理
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len]
        
        # CNN特徵提取 - 三層卷積堆疊
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)  # 使用LeakyReLU以避免死神經元
        x = self.pool(x)  # [batch_size, 32, seq_len/2]
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        x = self.pool(x)  # [batch_size, 64, seq_len/4]
        x = F.leaky_relu(self.conv3(x), negative_slope=0.1)
        x = self.pool(x)  # [batch_size, 128, seq_len/8]
        x = self.dropout1(x)
        
        # 重塑以適應LSTM層 [batch_size, channels, seq_len] -> [batch_size, seq_len, channels]
        x = x.transpose(1, 2)  # [batch_size, seq_len/8, 128]
        
        # BiLSTM處理 - 捕捉時間依賴關係
        x, _ = self.lstm(x)  # [batch_size, seq_len/8, hidden_dim*2]
        
        # 取最後一個時間步的輸出 - 包含整個序列的信息
        x = x[:, -1, :]  # [batch_size, hidden_dim*2]
        
        # 分類部分
        x = F.leaky_relu(self.fc(x), negative_slope=0.1)
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
    if model_type == 'transformer' or model_type == 'mamba':  # 支持兩種名稱，向後兼容
        try:
            model = TransformerModel(input_dim=input_dim)
            total_params, trainable_params = count_parameters(model)
            print(f"\nTransformer 模型參數統計：")
            print(f"總參數量: {total_params:,}")
            print(f"可訓練參數量: {trainable_params:,}")
            return model
        except Exception as e:
            print(f"無法創建 Transformer 模型: {str(e)}，使用 DNN 替代")
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