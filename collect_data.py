import cv2
import os
import time
import numpy as np
from utils.mediapipe_utils import MediaPipeHandTracker
from utils.data_utils import DataProcessor

class DataCollector:
    def __init__(self):
        """Khởi tạo DataCollector"""
        self.hand_tracker = MediaPipeHandTracker()
        self.data_processor = DataProcessor()
        
        # Các tham số thu thập
        self.total_frames = 30  # Chính xác 30 frame
        self.fps = 20  # FPS của video
        
        # Các lớp dấu thanh
        self.classes = self.data_processor.classes
        self.class_names = self.data_processor.class_names
        
        # Trạng thái hiện tại
        self.current_class = 0  # Index của lớp hiện tại
        self.is_recording = False
        self.recording_frames = []
        self.frame_count = 0
        
        # Thống kê
        self.stats = self.data_processor.get_data_statistics()
        
        print("=== HỆ THỐNG THU THẬP DỮ LIỆU DẤU THANH ===")
        print(f"Số frame: {self.total_frames}")
        print(f"FPS: {self.fps}")
        print("Các lớp dấu thanh:")
        for i, (cls, name) in enumerate(self.class_names.items(), 1):
            print(f"  {i}. {cls}: {name}")
        print("\nĐIỀU KHIỂN:")
        print("  1-5: Chọn dấu thanh")
        print("  SPACE: Bắt đầu thu thập")
        print("  Q: Thoát")
    
    def get_next_sample_id(self, class_name: str) -> int:
        """Lấy ID mẫu tiếp theo cho lớp"""
        class_dir = os.path.join(self.data_processor.keypoints_dir, class_name)
        if not os.path.exists(class_dir):
            return 0
        
        existing_samples = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        return len(existing_samples)
    
    def start_recording(self):
        """Bắt đầu thu thập video"""
        if self.is_recording:
            return
        
        self.is_recording = True
        self.recording_frames = []
        self.frame_count = 0
        self.sample_id = self.get_next_sample_id(self.classes[self.current_class])
        print(f"\n🎬 Bắt đầu thu thập {self.class_names[self.classes[self.current_class]]} (ID: {self.sample_id})")
    
    def stop_recording(self):
        """Dừng thu thập video"""
        if not self.is_recording:
            return
        self.is_recording = False
        if len(self.recording_frames) == self.total_frames:
            class_name = self.classes[self.current_class]
            # Sử dụng numpy array để giảm overhead khi lưu
            frames_array = np.stack(self.recording_frames)
            self.data_processor.save_keypoints(frames_array, class_name, self.sample_id)
            self.stats[class_name] += 1
            self.stats['total'] += 1
            print(f"✅ Đã lưu {len(self.recording_frames)} frames cho {self.class_names[class_name]} (ID: {self.sample_id})")
        else:
            print(f"❌ Thu thập không thành công! Chỉ có {len(self.recording_frames)} frames")
        self.recording_frames.clear()
        self.frame_count = 0

    def process_frame(self, frame):
        """Xử lý frame và trích xuất keypoints"""
        keypoints = self.hand_tracker.extract_keypoints(frame)
        # Sử dụng biến static cho frame trống để tránh tạo mới liên tục
        if not hasattr(self, '_empty_keypoints'):
            self._empty_keypoints = np.zeros(63)
        if keypoints is not None:
            frame = self.hand_tracker.draw_landmarks(frame, keypoints)
            if self.is_recording:
                self.recording_frames.append(keypoints)
                self.frame_count += 1
        else:
            if self.is_recording:
                self.recording_frames.append(self._empty_keypoints)
                self.frame_count += 1
        return frame
    
    def draw_ui(self, frame):
        """Vẽ giao diện người dùng lên frame"""
        # Thông tin lớp hiện tại
        current_class_name = self.class_names[self.classes[self.current_class]]
        cv2.putText(frame, f"Dau thanh: {current_class_name}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Thống kê lớp hiện tại
        current_count = self.stats.get(self.classes[self.current_class], 0)
        cv2.putText(frame, f"Label: {current_count}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Trạng thái thu thập
        if self.is_recording:
            cv2.putText(frame, f"Thu thập: {self.frame_count}/{self.total_frames}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Vẽ progress bar
            progress = self.frame_count / self.total_frames
            bar_width = 200
            bar_height = 20
            bar_x, bar_y = 10, 150
            
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (128, 128, 128), 2)
            fill_width = int(bar_width * progress)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (0, 255, 0), -1)
            
            if self.frame_count >= self.total_frames:
                cv2.putText(frame, "Hoàn thành!", (10, 190), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Enter Space to begin:", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Danh sách các lớp
        y_offset = 220
        for i, (cls, name) in enumerate(self.class_names.items()):
            color = (0, 255, 0) if i == self.current_class else (128, 128, 128)
            count = self.stats.get(cls, 0)
            cv2.putText(frame, f"{i+1}. {name}: {count} mẫu", (10, y_offset + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        
        # Tổng thống kê
        cv2.putText(frame, f"Total: {self.stats['total']} mau", (10, 470), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    def run(self):
        """Chạy hệ thống thu thập dữ liệu"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Không thể mở camera!")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        print("Bắt đầu hiển thị camera...")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = self.process_frame(frame)
                self.draw_ui(frame)
                if self.is_recording and self.frame_count >= self.total_frames:
                    self.stop_recording()
                cv2.imshow('Thu thập dữ liệu dấu thanh', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    if not self.is_recording:
                        self.start_recording()
                elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
                    class_idx = key - ord('1')
                    if 0 <= class_idx < len(self.classes):
                        self.current_class = class_idx
                        print(f"Đã chọn: {self.class_names[self.classes[self.current_class]]}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\n=== THỐNG KÊ CUỐI CÙNG ===")
            for cls in self.classes:
                print(f"  {self.class_names[cls]}: {self.stats[cls]} mẫu")
            print(f"Tổng cộng: {self.stats['total']} mẫu")

def main():
    """Hàm chính"""
    collector = DataCollector()
    collector.run()

if __name__ == "__main__":
    main()