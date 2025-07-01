import cv2
import numpy as np
import tensorflow as tf
import pickle
import os
import time
from collections import deque
from utils.mediapipe_utils import MediaPipeHandTracker
from utils.data_utils import DataProcessor

class TonePredictor:
    def __init__(self):
        """Khởi tạo TonePredictor"""
        self.hand_tracker = MediaPipeHandTracker()
        self.data_processor = DataProcessor()
        self.classes = self.data_processor.classes
        self.class_names = self.data_processor.class_names
        
        # Tham số dự đoán
        self.sequence_length = 30  # Chính xác 30 frame
        self.prediction_threshold = 0.7  # Ngưỡng tin cậy
        
        # Trạng thái hiện tại
        self.is_predicting = False
        self.prediction_frames = []
        self.frame_count = 0
        self.current_prediction = None
        self.current_confidence = 0.0
        
        # Tải mô hình
        self.model = None
        self.label_encoder = None
        self.model_type = None
        
        # Tự động tải mô hình
        self.auto_load_model()
    
    def auto_load_model(self):
        """Tự động tải mô hình tốt nhất"""
        model_dir = "trained_models"
        
        if not os.path.exists(model_dir):
            print("Không tìm thấy thư mục trained_models!")
            return
        
        # Tìm mô hình tốt nhất
        best_model = None
        
        for model_type in ['lstm', 'cnn']:
            model_path = os.path.join(model_dir, f"{model_type}_model_final.h5")
            if os.path.exists(model_path):
                # Thử tải mô hình để kiểm tra
                try:
                    model = tf.keras.models.load_model(model_path)
                    # Ưu tiên LSTM
                    if model_type == 'lstm':
                        best_model = model_path
                        self.model_type = 'lstm'
                        break
                    elif best_model is None:
                        best_model = model_path
                        self.model_type = 'cnn'
                except:
                    continue
        
        if best_model:
            self.load_model(best_model)
        else:
            print("Không tìm thấy mô hình đã huấn luyện!")
            print("Vui lòng chạy train_model.py trước!")
    
    def load_model(self, model_path: str):
        """
        Tải mô hình và label encoder
        
        Args:
            model_path: Đường dẫn đến file mô hình
        """
        try:
            print(f"Đang tải mô hình từ: {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            
            # Tải label encoder
            encoder_path = model_path.replace('_final.h5', '_label_encoder.pkl')
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
            else:
                # Sử dụng label encoder từ data processor
                self.label_encoder = self.data_processor.label_encoder
            
            print(f"Mô hình {self.model_type.upper()} đã được tải thành công!")
            print(f"Input shape: {self.model.input_shape}")
            print(f"Số lớp: {len(self.classes)}")
            
        except Exception as e:
            print(f"Lỗi khi tải mô hình: {e}")
            self.model = None
            self.label_encoder = None
    
    def start_prediction(self):
        """Bắt đầu dự đoán"""
        if self.is_predicting or self.model is None:
            return
        
        self.is_predicting = True
        self.prediction_frames = []
        self.frame_count = 0
        print(f"\n🎯 Bắt đầu dự đoán dấu thanh...")
    
    def stop_prediction(self):
        """Dừng dự đoán và thực hiện dự đoán"""
        if not self.is_predicting:
            return
        
        self.is_predicting = False
        
        if len(self.prediction_frames) == self.sequence_length:
            # Thực hiện dự đoán
            prediction, confidence, probabilities = self.predict_tone(self.prediction_frames)
            
            if prediction is not None and confidence >= self.prediction_threshold:
                self.current_prediction = prediction
                self.current_confidence = confidence
                print(f"🎯 Dự đoán: {self.class_names[self.current_prediction]} (Tin cậy: {self.current_confidence:.2f})")
            else:
                print("❌ Độ tin cậy quá thấp hoặc không thể dự đoán!")
        else:
            print(f"❌ Dự đoán không thành công! Chỉ có {len(self.prediction_frames)} frames")
        
        self.prediction_frames = []
        self.frame_count = 0
    
    def preprocess_keypoints(self, keypoints_sequence: list) -> np.ndarray:
        """
        Tiền xử lý keypoints cho dự đoán
        
        Args:
            keypoints_sequence: Danh sách keypoints
            
        Returns:
            Array keypoints đã chuẩn hóa
        """
        # Chuẩn hóa
        keypoints_array = np.array(keypoints_sequence)
        
        # Reshape cho mô hình
        if self.model_type == 'lstm':
            return keypoints_array.reshape(1, self.sequence_length, -1)
        else:  # CNN
            return keypoints_array.reshape(1, self.sequence_length, -1)
    
    def predict_tone(self, keypoints_sequence: list) -> tuple:
        """
        Dự đoán dấu thanh
        
        Args:
            keypoints_sequence: Danh sách keypoints
            
        Returns:
            Tuple (predicted_class, confidence, all_probabilities)
        """
        if self.model is None or len(keypoints_sequence) != self.sequence_length:
            return None, 0.0, None
        
        # Tiền xử lý
        X = self.preprocess_keypoints(keypoints_sequence)
        
        # Dự đoán
        predictions = self.model.predict(X, verbose=0)
        probabilities = predictions[0]
        
        # Lấy kết quả
        predicted_class_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_class_idx]
        
        if self.label_encoder:
            predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        else:
            predicted_class = self.classes[predicted_class_idx]
        
        return predicted_class, confidence, probabilities
    
    def process_frame(self, frame):
        """Xử lý frame và trích xuất keypoints"""
        # Trích xuất keypoints
        keypoints = self.hand_tracker.extract_keypoints(frame)
        
        # Vẽ landmarks nếu có
        if keypoints is not None:
            frame = self.hand_tracker.draw_landmarks(frame, keypoints)
            if self.is_predicting:
                self.prediction_frames.append(keypoints)
                self.frame_count += 1
        else:
            if self.is_predicting:
                # Nếu không phát hiện tay, thêm frame trống
                self.prediction_frames.append(np.zeros(63))
                self.frame_count += 1
        
        return frame
    
    def draw_ui(self, frame):
        """Vẽ giao diện người dùng lên frame"""
        # Thông tin mô hình
        if self.model is not None:
            cv2.putText(frame, f"Model: {self.model_type.upper()}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "No model!", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Kết quả dự đoán
        if self.current_prediction:
            color = (0, 255, 0) if self.current_confidence > 0.8 else (0, 255, 255)
            cv2.putText(frame, f"Result: {self.class_names[self.current_prediction]}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Confidence: {self.current_confidence:.2f}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Trạng thái dự đoán
        if self.is_predicting:
            cv2.putText(frame, f"Predicting: {self.frame_count}/{self.sequence_length}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Vẽ progress bar
            progress = self.frame_count / self.sequence_length
            bar_width = 200
            bar_height = 20
            bar_x, bar_y = 10, 150
            
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (128, 128, 128), 2)
            fill_width = int(bar_width * progress)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (0, 255, 0), -1)
            
            if self.frame_count >= self.sequence_length:
                cv2.putText(frame, "Complete!", (10, 190), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Press SPACE to predict", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Danh sách các lớp
        y_offset = 220
        for i, (cls, name) in enumerate(self.class_names.items()):
            color = (128, 128, 128)
            cv2.putText(frame, f"{i+1}. {name}", (10, y_offset + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Hướng dẫn
        y_offset = 370
        cv2.putText(frame, "CONTROLS:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "SPACE: Start prediction", (10, y_offset + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, "Q: Exit", (10, y_offset + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def run_prediction(self):
        """Chạy dự đoán real-time"""
        if self.model is None:
            print("Không có mô hình để dự đoán!")
            return
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Không thể mở camera!")
            return
        
        # Thiết lập camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 20)
        
        print("=== HỆ THỐNG DỰ ĐOÁN DẤU THANH ===")
        print("Nhấn 'q' để thoát, 'space' để dự đoán")
        print("Thực hiện dấu thanh trong 30 frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Xử lý frame
            frame = self.process_frame(frame)
            
            # Vẽ giao diện
            self.draw_ui(frame)
            
            # Kiểm tra hoàn thành dự đoán
            if self.is_predicting and self.frame_count >= self.sequence_length:
                self.stop_prediction()
            
            cv2.imshow('Tone Recognition', frame)
            
            # Xử lý phím nhấn
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space
                if not self.is_predicting:
                    self.start_prediction()
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Hàm chính"""
    print("=== HỆ THỐNG DỰ ĐOÁN DẤU THANH ===")
    
    # Khởi tạo predictor và chạy ngay lập tức
    predictor = TonePredictor()
    predictor.run_prediction()

if __name__ == "__main__":
    main() 