#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
數據處理模塊 - 負責數據加載、預處理和特徵工程
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from matplotlib import font_manager, rc

# 設定中文字型
font_path = "TFT/MSGOTHIC.TTF"
prop = font_manager.FontProperties(fname=font_path)
rc('font', family=prop.get_name())


class NetworkTrafficDataset(Dataset):
    """網絡流量數據集類"""
    
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        
        # 確保 y 是數值型別，對於字符串標籤需要先轉換
        if isinstance(y[0], str):
            self.y = torch.tensor(LabelEncoder().fit_transform(y), dtype=torch.float32)
        else:
            self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_network_data(file_path="AllMerged.csv"):
    """載入網絡流量數據"""
    print(f"載入數據集: {file_path}")
    
    try:
        # 載入 CSV 文件
        df = pd.read_csv(file_path, low_memory=False)
        
        # 數據清理和預處理
        print(f"原始數據集大小: {df.shape}")
        
        # 處理缺失值
        df = df.dropna()
        print(f"移除缺失值後數據集大小: {df.shape}")
        
        # 處理無限或極大/極小值
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            # 將無限值替換為 NaN
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            # 將超出範圍的值設定為列的分位值
            if not df[col].isna().all():  # 確保列不全是 NaN
                upper_limit = df[col].quantile(0.999)
                lower_limit = df[col].quantile(0.001)
                df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
        
        # 再次移除可能出現的 NaN 值
        df = df.dropna()
        print(f"數據清理後數據集大小: {df.shape}")
        
        return df
    
    except Exception as e:
        print(f"載入數據時發生錯誤: {str(e)}")
        print("生成模擬數據作為替代...")
        
        # 生成模擬數據
        n_samples = 1000
        normal_traffic = np.random.normal(loc=10, scale=2, size=(n_samples // 2, 10))
        normal_labels = np.zeros(n_samples // 2)
        ddos_traffic = np.random.normal(loc=30, scale=5, size=(n_samples // 2, 10))
        ddos_labels = np.ones(n_samples // 2)
        
        # 合併數據
        X = np.vstack([normal_traffic, ddos_traffic])
        y = np.concatenate([normal_labels, ddos_labels])
        
        # 特徵名稱
        feature_names = [
            'packet_rate', 'byte_rate', 'flow_duration',
            'tcp_flags', 'udp_length', 'unique_ports',
            'ip_entropy', 'protocol_distribution', 'packet_size_std', 'ttl_variance'
        ]
        
        # 創建 DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['Label'] = y
        
        return df


def preprocess_data(data):
    """數據預處理函數 - 增強版"""
    print("進行數據預處理和特徵工程...")
    
    # 查找標籤列
    label_col = None
    possible_label_cols = ['Label', ' Label', 'label', 'CLASS', 'class', 'target']
    
    for col in possible_label_cols:
        if col in data.columns:
            label_col = col
            print(f"找到標籤列: '{label_col}'")
            break
    
    if label_col is None:
        # 如果找不到標籤列，假設最後一列是標籤
        label_col = data.columns[-1]
        print(f"找不到明確的標籤列，使用最後一列作為標籤: '{label_col}'")
    
    # 提取標籤
    y = data[label_col].values
    
    # 二分類處理：BENIGN(良性) -> 0, 其他攻擊 -> 1
    binary_y = np.array([0 if label == 'BENIGN' else 1 for label in y])
    print(f"二分類標籤分布: {np.unique(binary_y, return_counts=True)}")
    
    # 提取特徵，刪除標籤列
    X = data.drop(label_col, axis=1)
    original_feature_count = X.shape[1]
    print(f"原始特徵數量: {original_feature_count}")
    
    # 顯示所有可能的特徵名稱
    print(f"可用特徵列表: {X.columns.tolist()}")
    
    # 檢測並移除常數特徵 (所有值相同的特徵)
    constant_features = []
    for col in X.columns:
        if X[col].nunique() <= 1:
            constant_features.append(col)
    
    if constant_features:
        print(f"移除 {len(constant_features)} 個常數特徵: {constant_features}")
        X = X.drop(columns=constant_features)
    
    # 處理類別特徵和文本特徵
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"發現 {len(categorical_cols)} 個類別特徵，進行轉換")
        
        # 嘗試將可轉換的類別特徵轉為數值
        for col in categorical_cols:
            try:
                X[col] = pd.to_numeric(X[col])
                print(f"  - 成功將 '{col}' 轉換為數值")
            except ValueError:
                print(f"  - 無法將 '{col}' 轉換為數值，將其刪除")
                X = X.drop(col, axis=1)
    
    # 檢測並處理高度相關特徵
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 1:
        try:
            # 計算相關係數矩陣
            correlation_matrix = X[numeric_cols].corr().abs()
            
            # 找出高度相關的特徵對 (相關係數大於 0.95)
            upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
            
            if to_drop:
                print(f"移除 {len(to_drop)} 個高度相關特徵: {to_drop}")
                X = X.drop(columns=to_drop)
        except Exception as e:
            print(f"計算相關係數時發生錯誤: {str(e)}，跳過相關性分析")
    
    # 檢查剩餘特徵
    print(f"預處理後特徵數量: {X.shape[1]}")
    
    # 將特徵轉換為 numpy 數組
    # 備份列名，以便後續分析
    feature_names = X.columns.tolist()
    X_values = X.values
    
    # 標準化特徵 (重要！確保所有特徵在同一尺度)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_values)
    
    # 使用主成分分析（PCA）進行降維（可選）
    # 當特徵數量非常大時，可以考慮使用 PCA
    use_pca = False
    if use_pca and X_scaled.shape[1] > 50:
        try:
            from sklearn.decomposition import PCA
            print("使用 PCA 降維...")
            
            # 保留 95% 的方差
            pca = PCA(n_components=0.95)
            X_pca = pca.fit_transform(X_scaled)
            
            explained_var = np.sum(pca.explained_variance_ratio_)
            n_components = X_pca.shape[1]
            
            print(f"PCA 降維: {X_scaled.shape[1]} 個特徵 -> {n_components} 個主成分 "
                  f"(保留 {explained_var:.2%} 的方差)")
            
            X_final = X_pca
        except Exception as e:
            print(f"PCA 降維失敗: {str(e)}，使用原始特徵")
            X_final = X_scaled
    else:
        X_final = X_scaled
    
    # 嘗試執行特徵重要性分析（基於樹模型）
    try:
        from sklearn.ensemble import RandomForestClassifier
        # 使用一個小型隨機森林模型來快速評估特徵重要性
        # 只在樣本數量不太大時執行
        if len(binary_y) < 50000:
            print("執行特徵重要性分析...")
            
            # 創建一個小型隨機森林模型
            rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
            rf.fit(X_scaled, binary_y)
            
            # 獲取特徵重要性
            importances = rf.feature_importances_
            
            # 將特徵重要性與特徵名稱匹配
            feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            
            # 顯示前 10 個最重要特徵
            print("最重要的 10 個特徵:")
            for i, (feature, importance) in enumerate(feature_importance[:10]):
                print(f"  {i+1}. {feature}: {importance:.4f}")
    except Exception as e:
        print(f"特徵重要性分析失敗: {str(e)}")
    
    print(f"最終特徵形狀: {X_final.shape}, 標籤形狀: {binary_y.shape}")
    
    return X_final, binary_y


def prepare_data_loaders(X, y, batch_size=128, test_size=0.3, val_size=0.5, random_state=42):
    """準備數據加載器 - 使用順序切割
    
    將數據按照時間順序分成訓練集、驗證集和測試集，以便模擬真實環境中的模型評估情境。
    
    Args:
        X (numpy.ndarray): 預處理後的特徵矩陣
        y (numpy.ndarray): 二分類標籤向量
        batch_size (int): 批次大小
        test_size (float): 測試集佔總數據的比例
        val_size (float): 驗證集佔測試數據的比例
        random_state (int): 隨機種子
        
    Returns:
        tuple: (訓練數據加載器, 驗證數據加載器, 測試數據加載器)
    """
    total_samples = len(X)
    
    # 計算各部分的樣本數
    test_samples = int(total_samples * test_size)
    temp_samples = total_samples - test_samples
    val_samples = int(test_samples * val_size)
    
    # 順序切割數據 - 訓練集使用較早的數據，測試集使用較新的數據
    X_train = X[:temp_samples]
    y_train = y[:temp_samples]
    
    X_val = X[temp_samples:temp_samples+val_samples]
    y_val = y[temp_samples:temp_samples+val_samples]
    
    X_test = X[temp_samples+val_samples:]
    y_test = y[temp_samples+val_samples:]
    
    print(f"訓練集大小: {X_train.shape}, 驗證集大小: {X_val.shape}, 測試集大小: {X_test.shape}")
    
    # 創建數據加載器
    train_dataset = NetworkTrafficDataset(X_train, y_train)
    val_dataset = NetworkTrafficDataset(X_val, y_val)
    test_dataset = NetworkTrafficDataset(X_test, y_test)
    
    # 注意：訓練集不再進行隨機打亂，保持時間順序
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader