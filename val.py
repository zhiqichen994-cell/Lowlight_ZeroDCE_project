import torch
import os
import torchvision
from models.zerodce import ZeroDCE
from PIL import Image
import numpy as np


def run_val():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ZeroDCE().to(device)

    # 1. 自动定位最新的最佳模型
    ckpt_dir = "checkpoints"
    ckpt_list = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pth")])
    if not ckpt_list:
        print("未找到模型文件，请先运行 train.py");
        return

    checkpoint_path = os.path.join(ckpt_dir, ckpt_list[-1])
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    model.eval()
    print(f"成功加载模型: {checkpoint_path}")

    # 2. 设置验证集路径和结果保存路径
    val_path = "data/val"
    save_path = "runs/val_results"
    os.makedirs(save_path, exist_ok=True)

    # 3. 推理并保存
    with torch.no_grad():
        for img_name in os.listdir(val_path):
            if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')): continue

            # 加载并转换图片
            img_raw = Image.open(os.path.join(val_path, img_name)).convert('RGB')
            img_tensor = torch.from_numpy(np.array(img_raw) / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(device)

            # 推理：Zero-DCE 通常返回迭代过程，取最后一个
            output = model(img_tensor)
            enhanced_img = output[-1] if isinstance(output, (list, tuple)) else output

            # 维度修复：强制取最后3通道 (RGB)
            if enhanced_img.shape[1] > 3:
                enhanced_img = enhanced_img[:, -3:, :, :]

            # 保存
            torchvision.utils.save_image(enhanced_img, os.path.join(save_path, img_name))
            print(f"验证图片已处理: {img_name}")


if __name__ == "__main__":
    run_val()