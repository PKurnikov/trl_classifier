# config.py

import torch

# Основные настройки
data_dir = "./data/traffic_lights"
batch_size = 32
num_epochs = 100
learning_rate = 0.001
val_split = 0.8
img_size = (128, 128)
log_file = "./logs/training.log"
model_save_path = "./models/model.pth"
onnx_model_path = "./models/model.onnx"
num_classes = 7  # Количество классов

# Настройки архитектуры
architecture = {
    "type": "SimpleCNN",  # ResNet и т.д.
    "conv_layers": [
        {"out_channels": 16, "kernel_size": 3},
        {"out_channels": 32, "kernel_size": 3},
        {"out_channels": 64, "kernel_size": 3},
    ],
    "fc_layers": [
        {"out_features": 128}
    ]
}

# Настройки экспорта в ONNX
export_onnx = True  # Флаг для экспорта в ONNX
best_epoch = 5      # Эпоха для экспорта модели (по умолчанию последняя)
eval_step = 5

# Настройка устройства (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")