import torch

def save_model(model, path):
    torch.save(model.state_dict(), path)

def log_results(log_file, epoch, train_acc, val_acc):
    with open(log_file, 'a') as f:
        f.write(f"Epoch {epoch+1}, Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}\n")

def export_to_onnx(model, onnx_path, img_size, best_epoch):
    model.eval()
    dummy_input = torch.randn(1, 3, img_size[0], img_size[1]).to(next(model.parameters()).device)
    torch.onnx.export(model, dummy_input, onnx_path, export_params=True, opset_version=11,
                      do_constant_folding=True, input_names=['input'], output_names=['output'])
    print(f"Model exported to ONNX format at {onnx_path} for epoch {best_epoch}.")