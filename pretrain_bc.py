import argparse
import json
import os
from pathlib import Path

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from net_actor import NetDQN


ACTION_SIZES = [2, 360, 9, 11, 3, 360, 100, 5]


class DemoDataset(Dataset):
    def __init__(self, demo_dir):
        self.demo_dir = Path(demo_dir)
        self.frames_dir = self.demo_dir / "frames"
        self.actions_path = self.demo_dir / "actions.jsonl"
        if not self.actions_path.exists():
            raise FileNotFoundError(f"Missing actions.jsonl in {demo_dir}")
        if not self.frames_dir.exists():
            raise FileNotFoundError(f"Missing frames/ directory in {demo_dir}")

        self.records = []
        with open(self.actions_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                self.records.append(record)
        if not self.records:
            raise ValueError(f"No records found in {self.actions_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        frame_path = self.frames_dir / record["frame"]
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
        required=True,
        help="Path to demo directory (e.g. data/demos/demo_YYYYMMDD_HHMMSS).",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--output", type=str, default="src/wzry_ai.pt", help="Model output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DemoDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = NetDQN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for images, actions in loader:
            images = images.to(device)
            actions = actions.to(device)

            logits = model(images)
            losses = []
            for i, logit in enumerate(logits):
                losses.append(criterion(logit, actions[:, i]))
            loss = sum(losses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        print(f"Epoch {epoch}/{args.epochs} - loss: {avg_loss:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved model to {output_path}")


if __name__ == "__main__":
    main()
