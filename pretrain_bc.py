import argparse
import json
from pathlib import Path

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from net_actor import NetDQN


class DemoDataset(Dataset):
    def __init__(self, demo_dirs):
        self.demo_dirs = [Path(demo_dir) for demo_dir in demo_dirs]
        self.records = []

        for demo_dir in self.demo_dirs:
            actions_path = demo_dir / "actions.jsonl"
            frames_dir = demo_dir / "frames"
            if not actions_path.exists():
                raise FileNotFoundError(f"Missing actions.jsonl in {demo_dir}")
            if not frames_dir.exists():
                raise FileNotFoundError(f"Missing frames/ directory in {demo_dir}")

            with open(actions_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    record["_frames_dir"] = frames_dir
                    self.records.append(record)
        if not self.records:
            raise ValueError("No records found in provided demo directories.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        frame_path = Path(record["_frames_dir"]) / record["frame"]
        image = cv2.imread(str(frame_path))
        if image is None:
            raise FileNotFoundError(f"Missing frame: {frame_path}")
        image = cv2.resize(image, (640, 640))
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        action = torch.tensor(record["action"], dtype=torch.long)
        return image, action


def parse_args():
    parser = argparse.ArgumentParser(description="Behavior cloning pretraining.")
    parser.add_argument(
        "--data_dir",
        type=str,
        action="append",
        required=True,
        help=(
            "Demo directory or parent directory. Repeat this flag to provide multiple demos. "
            "If a parent directory is given, all subfolders containing actions.jsonl will be used."
        ),
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--pin_memory", action="store_true", help="Pin memory for faster GPU transfer.")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training.")
    parser.add_argument("--output", type=str, default="src/wzry_ai.pt", help="Model output path.")
    return parser.parse_args()


def resolve_demo_dirs(data_dirs):
    resolved = []
    for raw_dir in data_dirs:
        path = Path(raw_dir)
        if (path / "actions.jsonl").exists():
            resolved.append(path)
            continue

        if not path.exists():
            raise FileNotFoundError(f"Data directory not found: {path}")

        for child in sorted(path.iterdir()):
            if not child.is_dir():
                continue
            if (child / "actions.jsonl").exists():
                resolved.append(child)

    if not resolved:
        raise FileNotFoundError("No valid demo directories found.")
    return resolved


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    demo_dirs = resolve_demo_dirs(args.data_dir)
    dataset = DemoDataset(demo_dirs)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    model = NetDQN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for images, actions in loader:
            images = images.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(images)
                losses = []
                for i, logit in enumerate(logits):
                    losses.append(criterion(logit, actions[:, i]))
                loss = sum(losses)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        print(f"Epoch {epoch}/{args.epochs} - loss: {avg_loss:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved model to {output_path}")


if __name__ == "__main__":
    main()
