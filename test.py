import torch
import os
import torchvision
from models.zerodce import ZeroDCE
from PIL import Image
import numpy as np


def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ZeroDCE().to(device)

    ckpt_list = sorted([f for f in os.listdir("checkpoints") if f.endswith(".pth")])
    if not ckpt_list:
        print("错误：找不到权重文件！");
        return

    model.load_state_dict(torch.load(f"checkpoints/{ckpt_list[-1]}", map_location=device, weights_only=False))
    model.eval()

    test_path = "data/test"
    save_path = "runs/test_results"
    os.makedirs(save_path, exist_ok=True)

    for img_name in os.listdir(test_path):
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')): continue

        img_raw = Image.open(os.path.join(test_path, img_name)).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img_raw) / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            # 核心修复：处理 tuple 且只取最后 3 个通道
            if isinstance(output, (list, tuple)):
                enhanced_img = output[-1]
            else:
                enhanced_img = output

            if enhanced_img.shape[1] > 3:
                enhanced_img = enhanced_img[:, -3:, :, :]  # 只要最后 3 层

        torchvision.utils.save_image(enhanced_img, os.path.join(save_path, img_name))
        print(f"测试完成并修复维度: {img_name}")

if __name__ == "__main__":
    run_test()