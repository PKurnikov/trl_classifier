import os
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
# from src.dataset import TrafficLightDataset
from src.dataset import train_loader, valid_loader
from tqdm.auto import tqdm

from src.model import CustomModel
from src.utils import save_model, log_results, export_to_onnx

def train(cfg):
    # Данные и классы
    classes = [d for d in os.listdir(cfg.data_dir) if os.path.isdir(os.path.join(cfg.data_dir, d))]
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    # dataset = TrafficLightDataset(root_dir=cfg.data_dir, class_to_idx=class_to_idx, img_size=cfg.img_size)
    # train_size = int((1 - cfg.val_split) * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    model = CustomModel(architecture=cfg.architecture, num_classes=cfg.num_classes).to(cfg.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_val_accuracy = 0.0
    best_epoch = 0

    for epoch in range(cfg.num_epochs):
        model.train()
        train_predictions, train_targets = [], []
        # for images, labels in train_loader:
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            
            images, labels = data
            images = images.to(cfg.device)
            labels = labels.to(cfg.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs, 1)
            train_predictions.extend(predicted.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
        
        if epoch % cfg.eval_step == 0:
            train_accuracy = accuracy_score(train_targets, train_predictions)
            val_accuracy = evaluate(cfg, model, valid_loader)
        
            log_results(cfg.log_file, epoch, train_accuracy, val_accuracy)
            print(f"Epoch {epoch+1}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = epoch + 1

    save_model(model, cfg.model_save_path)

    if cfg.export_onnx:
        export_to_onnx(model, cfg.onnx_model_path, cfg.img_size, best_epoch)

def evaluate(cfg, model, val_loader):
    model.eval()
    val_predictions, val_targets = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(cfg.device), labels.to(cfg.device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_predictions.extend(predicted.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())
    return accuracy_score(val_targets, val_predictions)
