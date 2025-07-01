# Hệ thống nhận dạng dấu thanh trong ngôn ngữ ký hiệu

Hệ thống sử dụng MediaPipe để nhận dạng 5 dấu thanh: huyền, sắc, hỏi, ngã, nặng thông qua các hành động tay.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cấu trúc dự án

```
VLSR_DauThanh/
├── collect_data.py      # Thu thập dữ liệu video
├── train_model.py       # Huấn luyện mô hình
├── predict.py          # Dự đoán dấu thanh
├── utils/
│   ├── __init__.py
│   ├── mediapipe_utils.py  # Tiện ích MediaPipe
│   └── data_utils.py       # Tiện ích xử lý dữ liệu
├── models/
│   └── __init__.py
├── data/
│   ├── raw_videos/     # Video thô
│   └── keypoints/      # Keypoints đã trích xuất
└── trained_models/     # Mô hình đã huấn luyện
```

## Sử dụng

### 1. Thu thập dữ liệu
```bash
python collect_data.py
```

### 2. Huấn luyện mô hình
```bash
python train_model.py
```

### 3. Dự đoán
```bash
python predict.py
```

## Các dấu thanh được hỗ trợ
- Huyền (`huyen`)
- Sắc (`sac`) 
- Hỏi (`hoi`)
- Ngã (`nga`)
- Nặng (`nang`) 