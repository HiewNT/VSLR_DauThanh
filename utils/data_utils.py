import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

class DataProcessor:
    def __init__(self, data_dir: str = "data"):
        """
        Khởi tạo DataProcessor
        
        Args:
            data_dir: Thư mục chứa dữ liệu
        """
        self.data_dir = data_dir
        self.keypoints_dir = os.path.join(data_dir, "keypoints")
        self.raw_videos_dir = os.path.join(data_dir, "raw_videos")
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(self.keypoints_dir, exist_ok=True)
        os.makedirs(self.raw_videos_dir, exist_ok=True)
        
        # Các lớp dấu thanh
        self.classes = ['huyen', 'sac', 'hoi', 'nga', 'nang']
        self.class_names = {
            'huyen': 'huyen',
            'sac': 'sac', 
            'hoi': 'hoi',
            'nga': 'nga',
            'nang': 'nang'
        }
        
        # Label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.classes)
    
    def save_keypoints(self, keypoints: List[np.ndarray], class_name: str, sample_id: int):
        """
        Lưu keypoints vào file
        
        Args:
            keypoints: List các keypoints arrays
            class_name: Tên lớp (huyen, sac, hoi, nga, nang)
            sample_id: ID của mẫu
        """
        if class_name not in self.classes:
            raise ValueError(f"Class {class_name} không hợp lệ. Các lớp hợp lệ: {self.classes}")
        
        # Tạo thư mục cho lớp
        class_dir = os.path.join(self.keypoints_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Lưu keypoints
        file_path = os.path.join(class_dir, f"sample_{sample_id:04d}.npy")
        np.save(file_path, np.array(keypoints))
        
        # Lưu metadata
        metadata = {
            'class': class_name,
            'sample_id': sample_id,
            'num_frames': len(keypoints),
            'keypoints_shape': np.array(keypoints).shape
        }
        
        metadata_path = os.path.join(class_dir, f"sample_{sample_id:04d}_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def load_keypoints(self, class_name: str = None) -> Tuple[List[np.ndarray], List[str]]:
        """
        Tải keypoints từ thư mục
        
        Args:
            class_name: Tên lớp cụ thể hoặc None để tải tất cả
            
        Returns:
            Tuple (keypoints_list, labels_list)
        """
        keypoints_list = []
        labels_list = []
        
        classes_to_load = [class_name] if class_name else self.classes
        
        for cls in classes_to_load:
            if cls not in self.classes:
                continue
                
            class_dir = os.path.join(self.keypoints_dir, cls)
            if not os.path.exists(class_dir):
                continue
            
            # Tải tất cả file .npy trong thư mục lớp
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.npy'):
                    file_path = os.path.join(class_dir, file_name)
                    keypoints = np.load(file_path)
                    keypoints_list.append(keypoints)
                    labels_list.append(cls)
        
        return keypoints_list, labels_list
    
    def prepare_training_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Chuẩn bị dữ liệu cho huấn luyện
        
        Args:
            test_size: Tỷ lệ dữ liệu test
            random_state: Random seed
            
        Returns:
            Tuple (X_train, X_test, y_train, y_test, label_encoder)
        """
        # Tải dữ liệu
        keypoints_list, labels_list = self.load_keypoints()
        
        if not keypoints_list:
            raise ValueError("Không tìm thấy dữ liệu keypoints")
        
        # Chuẩn hóa dữ liệu
        X = self.normalize_keypoints(keypoints_list)
        y = self.label_encoder.transform(labels_list)
        
        # Chia dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, self.label_encoder
    
    def normalize_keypoints(self, keypoints_list: List[np.ndarray]) -> np.ndarray:
        """
        Chuẩn hóa keypoints
        
        Args:
            keypoints_list: List các keypoints arrays
            
        Returns:
            Array keypoints đã chuẩn hóa
        """
        # Tìm độ dài tối đa
        max_frames = max(kp.shape[0] for kp in keypoints_list)
        num_features = keypoints_list[0].shape[1] if keypoints_list else 0
        
        # Padding hoặc truncate để có cùng độ dài
        normalized_keypoints = []
        for kp in keypoints_list:
            if kp.shape[0] < max_frames:
                # Padding với zeros
                padding = np.zeros((max_frames - kp.shape[0], num_features))
                kp_padded = np.vstack([kp, padding])
            else:
                # Truncate
                kp_padded = kp[:max_frames]
            
            normalized_keypoints.append(kp_padded)
        
        return np.array(normalized_keypoints)
    
    def get_data_statistics(self) -> Dict:
        """
        Lấy thống kê dữ liệu
        
        Returns:
            Dictionary chứa thống kê
        """
        stats = {}
        
        for cls in self.classes:
            class_dir = os.path.join(self.keypoints_dir, cls)
            if os.path.exists(class_dir):
                num_samples = len([f for f in os.listdir(class_dir) if f.endswith('.npy')])
                stats[cls] = num_samples
            else:
                stats[cls] = 0
        
        stats['total'] = sum(stats.values())
        stats['classes'] = self.classes
        
        return stats
    
    def save_label_encoder(self, file_path: str):
        """Lưu label encoder"""
        with open(file_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
    
    def load_label_encoder(self, file_path: str):
        """Tải label encoder"""
        with open(file_path, 'rb') as f:
            self.label_encoder = pickle.load(f) 