import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional

class MediaPipeHandTracker:
    def __init__(self):
        """Khởi tạo MediaPipe Hand Tracking"""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Khởi tạo MediaPipe Hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Theo dõi tối đa 2 bàn tay
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Trích xuất keypoints từ frame
        
        Args:
            frame: Frame video (BGR format)
            
        Returns:
            Keypoints array hoặc None nếu không phát hiện tay
        """
        # Chuyển BGR sang RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Xử lý frame
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Lấy keypoints từ bàn tay đầu tiên
            hand_landmarks = results.multi_hand_landmarks[0]
            keypoints = np.empty(63, dtype=np.float32)
            for idx, landmark in enumerate(hand_landmarks.landmark):
                keypoints[idx*3:idx*3+3] = [landmark.x, landmark.y, landmark.z]
            return keypoints
        
        return None
    
    def extract_keypoints_from_video(self, video_path: str, max_frames: int = 30) -> List[np.ndarray]:
        """
        Trích xuất keypoints từ video
        
        Args:
            video_path: Đường dẫn đến video
            max_frames: Số frame tối đa cần trích xuất
            
        Returns:
            List các keypoints arrays
        """
        cap = cv2.VideoCapture(video_path)
        keypoints_list = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            keypoints = self.extract_keypoints(frame)
            if keypoints is not None:
                keypoints_list.append(keypoints)
            else:
                # Thêm frame trống nếu không phát hiện tay
                if not hasattr(self, '_empty_keypoints'):
                    self._empty_keypoints = np.zeros(63, dtype=np.float32)
                keypoints_list.append(self._empty_keypoints)
            
            frame_count += 1
        
        cap.release()
        return keypoints_list
    
    def draw_landmarks(self, frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """
        Vẽ landmarks lên frame
        
        Args:
            frame: Frame gốc
            keypoints: Keypoints đã trích xuất
            
        Returns:
            Frame với landmarks được vẽ
        """
        # Chuyển BGR sang RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Xử lý frame
        results = self.hands.process(rgb_frame)
        
        # Vẽ landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return frame
    
    def get_hand_gesture_info(self, keypoints: np.ndarray) -> dict:
        """
        Phân tích thông tin cử chỉ tay từ keypoints
        
        Args:
            keypoints: Keypoints array
            
        Returns:
            Dictionary chứa thông tin cử chỉ
        """
        if keypoints is None:
            return {}
        
        # Chuyển keypoints thành landmarks
        landmarks = []
        for i in range(0, len(keypoints), 3):
            landmarks.append({
                'x': keypoints[i],
                'y': keypoints[i+1],
                'z': keypoints[i+2]
            })
        
        # Tính toán các thông số cơ bản
        info = {
            'num_landmarks': len(landmarks),
            'hand_center': {
                'x': np.mean([lm['x'] for lm in landmarks]),
                'y': np.mean([lm['y'] for lm in landmarks]),
                'z': np.mean([lm['z'] for lm in landmarks])
            },
            'hand_span': {
                'x': max([lm['x'] for lm in landmarks]) - min([lm['x'] for lm in landmarks]),
                'y': max([lm['y'] for lm in landmarks]) - min([lm['y'] for lm in landmarks])
            }
        }
        
        return info
    
    def __del__(self):
        """Giải phóng tài nguyên"""
        if hasattr(self, 'hands'):
            self.hands.close()