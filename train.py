import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
from datetime import datetime

# 你的自定义模块
from models.zerodce import ZeroDCE
from datasets.lowlight_dataset import LowLightDataset
from losses.loss import MSELoss
from config import get_args


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for img in val_loader:
            img = img.to(device)
            # 假设 ZeroDCE 返回的是列表，取最后一个输出
            output = model(img)[-1]
            loss = criterion(output, img)
            total_loss += loss.item()
    return total_loss / len(val_loader) if len(val_loader) > 0 else 0


def log_info(content, log_file):
    """同时打印到控制台并写入文件，确保实时性"""
    print(content, flush=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(content + "\n")
        f.flush()  # 确保写入磁盘


def train():
    # 0. 参数与设备初始化
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建必要的文件夹
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("runs", exist_ok=True)  # 专门存放日志和曲线

    # 1. 初始化日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join("runs", f"train_log_{timestamp}.txt")
    curve_file = os.path.join("runs", f"loss_curve_{timestamp}.png")

    log_header = [
        "=" * 50,
        f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Device: {device}",
        f"Epochs: {args.epochs}, LR: {args.lr}, Batch Size: {args.batch_size}",
        "=" * 50
    ]
    for line in log_header: log_info(line, log_file)

    # 2. 数据与模型准备
    train_dataset = LowLightDataset("data/train")
    val_dataset = LowLightDataset("data/val")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = ZeroDCE().to(device)
    criterion = MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 3. 训练循环
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    start_total_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        start_epoch_time = time.time()

        for batch_idx, img in enumerate(train_loader):
            img = img.to(device)
            optimizer.zero_grad()

            # 前向传播 (注意：只跑一次，避免显存浪费)
            outputs = model(img)
            output = outputs[-1] if isinstance(outputs, list) else outputs

            loss = criterion(output, img)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # 计算平均损失
        train_avg_loss = epoch_loss / len(train_loader)
        val_avg_loss = validate(model, val_loader, criterion, device)

        train_losses.append(train_avg_loss)
        val_losses.append(val_avg_loss)
        scheduler.step()

        epoch_time = time.time() - start_epoch_time

        # 打印/记录当前 Epoch 结果
        log_content = (f"Epoch [{epoch + 1:03d}/{args.epochs}] | "
                       f"Train Loss: {train_avg_loss:.6f} | "
                       f"Val Loss: {val_avg_loss:.6f} | "
                       f"Time: {epoch_time:.2f}s")
        log_info(log_content, log_file)

        # 保存最优模型
        if val_avg_loss < best_loss:
            best_loss = val_avg_loss
            model_path = os.path.join("checkpoints", f"best_model_{timestamp}.pth")
            torch.save(model.state_dict(), model_path)
            log_info(f"  --> 发现更优模型，已保存至: {model_path}", log_file)

    # 4. 训练总结与绘图
    total_duration = (time.time() - start_total_time) / 60
    summary = [
        "\n" + "=" * 50,
        "训练完成！",
        f"总耗时: {total_duration:.2f} 分钟",
        f"最佳验证损失: {best_loss:.6f}",
        "=" * 50
    ]
    for line in summary: log_info(line, log_file)

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(curve_file)
        plt.close()
        log_info(f"损失曲线图已保存: {curve_file}", log_file)
    except Exception as e:
        log_info(f"绘图失败: {str(e)}", log_file)


if __name__ == "__main__":
    # 确保在正确的目录下运行
    print(f"工作目录: {os.getcwd()}")
    train()