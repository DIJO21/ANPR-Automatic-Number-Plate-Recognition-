import os
import random
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from utils.gpu import get_device, clear_gpu_memory
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ForensicANPR.TrainOCR")

# Character set definition (Indian plate characters + blank for CTC)
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ↑^"
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(ALPHABET)}
IDX_TO_CHAR = {idx + 1: char for idx, char in enumerate(ALPHABET)}
IDX_TO_CHAR[0] = "" # CTC Blank token

class CRNN(nn.Module):
    """Convolutional Recurrent Neural Network for License Plate OCR."""

    def __init__(self, img_h: int = 32, nc: int = 1, nclass: int = len(ALPHABET) + 1, nh: int = 256):
        super(CRNN, self).__init__()
        assert img_h % 16 == 0, "img_h must be a multiple of 16"

        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64 x 16 x w/2
            
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 128 x 8 x w/4
            
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # 256 x 4 x w/4
            
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # 512 x 2 x w/4
            
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True)  # 512 x 1 x w/4 - 1
        )

        # Map to Sequence and Bidirectional RNN
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # conv features
        features = self.cnn(x)
        b, c, h, w = features.size()
        assert h == 1, "the height of conv features must be 1"
        features = features.squeeze(2)  # [b, c, w]
        features = features.permute(2, 0, 1)  # [w, b, c] (seq_len, batch_size, input_size)

        # rnn features
        output = self.rnn(features)
        return output

class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM Layer wrapper."""

    def __init__(self, nIn: int, nHidden: int, nOut: int):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recurrent, _ = self.rnn(x)
        t, b, h = recurrent.size()
        t_rec = recurrent.view(t * b, h)
        output = self.embedding(t_rec)  # [t * b, nOut]
        output = output.view(t, b, -1)
        return output

class SyntheticIndianPlatesDataset(Dataset):
    """Generates synthetic license plate images on-the-fly for OCR training."""

    def __init__(self, size: int = 1000, img_w: int = 100, img_h: int = 32):
        self.size = size
        self.img_w = img_w
        self.img_h = img_h
        
        # Indian Plate format options
        self.states = ["MH", "DL", "KA", "TN", "AP", "UP", "HR", "GJ", "TS"]
        self.letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __len__(self) -> int:
        return self.size

    def _generate_plate_text(self) -> str:
        """Generates random standard Indian plate text strings."""
        state = random.choice(self.states)
        zone = f"{random.randint(1, 99):02d}"
        series = "".join(random.choices(self.letters, k=2))
        num = f"{random.randint(1, 9999):04d}"
        return f"{state}{zone}{series}{num}"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text = self._generate_plate_text()
        
        # Render text to image
        img = Image.new('L', (self.img_w, self.img_h), color=255)
        draw = ImageDraw.Draw(img)
        
        # Draw text at center
        font_size = 14
        try:
            font = ImageFont.load_default()
        except IOError:
            font = None
            
        draw.text((8, 8), text, fill=0, font=font)
        
        # Convert to numpy array, apply normalize
        img_np = np.array(img, dtype=np.float32) / 255.0
        # Add channel dimension
        img_tensor = torch.tensor(img_np).unsqueeze(0)  # [1, H, W]
        
        # Encode targets
        target = [CHAR_TO_IDX[char] for char in text]
        target_tensor = torch.tensor(target, dtype=torch.long)
        target_length = torch.tensor(len(target), dtype=torch.long)

        return img_tensor, target_tensor, target_length

def collate_fn(batch):
    """Pads targets dynamically to align shape in loaders."""
    imgs, targets, lengths = zip(*batch)
    imgs = torch.stack(imgs, 0)
    lengths = torch.cat(lengths, 0)
    targets = torch.cat(targets, 0)
    return imgs, targets, lengths

def decode_prediction(preds: torch.Tensor) -> List[str]:
    """Decodes CRNN logits sequence using greedy CTC decoding."""
    # preds: [seq_len, batch_size, num_classes]
    preds = preds.argmax(2) # [seq_len, batch_size]
    preds = preds.transpose(1, 0) # [batch_size, seq_len]
    
    decoded_texts = []
    for batch_idx in range(preds.size(0)):
        char_list = []
        prev_char = 0
        for char_idx in preds[batch_idx]:
            val = char_idx.item()
            if val != 0:
                if val != prev_char:
                    char_list.append(IDX_TO_CHAR[val])
            prev_char = val
        decoded_texts.append("".join(char_list))
    return decoded_texts

def train(args):
    device = get_device()
    clear_gpu_memory()

    # Create model output directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize synthetics loader
    train_dataset = SyntheticIndianPlatesDataset(size=args.dataset_size, img_w=args.img_w, img_h=args.img_h)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )

    # Create model
    model = CRNN(img_h=args.img_h, nc=1, nclass=len(ALPHABET)+1, nh=256).to(device)
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    logger.info(f"Model initialized. Total classes: {len(ALPHABET) + 1}")
    logger.info("Starting CRNN OCR training loop...")
    
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for imgs, targets, target_lengths in train_loader:
            imgs = imgs.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            preds = model(imgs)  # [seq_len, batch_size, num_classes]
            
            # Prepare dimensions for CTCLoss
            seq_len, batch_size, _ = preds.size()
            input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
            
            loss = criterion(preds.log_softmax(2), targets, input_lengths, target_lengths)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_size

        scheduler.step()
        epoch_loss /= len(train_dataset)
        
        # Log progress and print a sample decode
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                sample_img, sample_target, _ = train_dataset[0]
                sample_pred = model(sample_img.unsqueeze(0).to(device))
                decoded = decode_prediction(sample_pred)[0]
                actual = "".join([IDX_TO_CHAR[val.item()] for val in sample_target])
                logger.info(f"Epoch {epoch+1}/{args.epochs} - Loss: {epoch_loss:.4f} | Sample Target: {actual} -> Predicted: {decoded}")
            model.train()
            
            # Save intermediate checkpoints
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"ocr_crnn_epoch_{epoch+1}.pth"))
            
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "ocr_crnn_final.pth"))
    logger.info("OCR CRNN training process finished. Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CRNN+CTC License Plate OCR Model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--dataset_size", type=int, default=1000, help="Synthesized dataset capacity")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--img_w", type=int, default=100, help="Input patch width")
    parser.add_argument("--img_h", type=int, default=32, help="Input patch height")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/ocr", help="Checkpoint directory")
    
    args = parser.parse_args()
    train(args)
