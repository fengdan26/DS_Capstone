import argparse
import os
import pickle
import numpy as np
import cv2
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from alive_progress import alive_bar
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
from torchvision.models import resnet50
import timm


# Enable segmented GPU memory management to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Path to the main dataset directory
data_folder = r"datasets_per_stock"

# Check if the data directory exists
if not os.path.exists(data_folder):
    raise ValueError(f"Data folder {data_folder} does not exist. Please check the path.")

# Parameter settings
last_checkpoint_file = "./saved_models/last_checkpoint.txt"
resume_path = None

if os.path.exists(last_checkpoint_file):
    with open(last_checkpoint_file, "r") as f:
        resume_path = f.read().strip()

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=8, help="Size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="Adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="Adam: decay of first-order momentum")
parser.add_argument("--b2", type=float, default=0.999, help="Adam: decay of second-order momentum")
parser.add_argument("--img_size", type=int, default=48, help="Size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
parser.add_argument("--save_dir", type=str, default="./saved_models", help="Directory to save models")
parser.add_argument("--accumulate_batches", type=int, default=4, help="Number of batches for gradient accumulation")
parser.add_argument("--resume", type=str, default=resume_path, help="Path to resume training from a checkpoint")
parser.add_argument("--noise_dim", type=int, default=10, help="Dimensionality of the noise input for the generator")
opt = parser.parse_args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable cuDNN benchmark
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    print("cuDNN benchmark enabled")

img_shape = (opt.channels, opt.img_size, opt.img_size)


def load_and_split_datasets(folder_path, test_split_ratio=0.2):
    """
    Load all .pkl files from the specified folder and randomly split them into training and testing datasets.
    :param folder_path: Path to the dataset folder
    :param test_split_ratio: Proportion of data allocated for testing
    :return: Lists of training and testing file paths
    """
    file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pkl")]
    if not file_list:
        raise ValueError(f"No .pkl files found in {folder_path}. Please check the folder path.")

    random.shuffle(file_list)
    split_idx = int(len(file_list) * (1 - test_split_ratio))
    train_files = file_list[:split_idx]
    test_files = file_list[split_idx:]
    print(f"Total files: {len(file_list)}, Train files: {len(train_files)}, Test files: {len(test_files)}")

    return train_files, test_files


class RGBStickDataset(IterableDataset):
    def __init__(self, file_list, img_shape):
        """
        Initialize the dataset.
        :param file_list: List of file paths
        :param img_shape: Target image dimensions (channels, height, width)
        """
        self.file_list = file_list
        self.img_shape = img_shape
        self.total_samples = 0
        for file_path in self.file_list:
            self.total_samples += self.count_samples(file_path)
        print(f"Dataset initialized with {len(self.file_list)} files and {self.total_samples} total samples.")

    def count_samples(self, file_path):
        """
        Count the number of samples in a file.
        """
        try:
            with open(file_path, "rb") as f:
                file_data = pickle.load(f)
                return len(file_data["images"])
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return 0

    def parse_file(self, file_path):
        """
        Load samples from a file and verify data integrity.
        """
        try:
            with open(file_path, "rb") as f:
                file_data = pickle.load(f)
                samples = file_data["images"]
                valid_samples = [
                    sample for sample in samples if "condition" in sample and "target" in sample
                ]
                return valid_samples
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []

    def parse_sample(self, sample):
        """
        Parse a single sample and validate its consistency.
        """
        try:
            conditions = np.array(sample["condition"]).squeeze()
            targets = np.array(sample["target"]).squeeze()

            conditions = self.resize_to_shape(conditions, self.img_shape)
            targets = self.resize_to_shape(targets, self.img_shape)

            return (
                torch.tensor(conditions, dtype=torch.float32),
                torch.tensor(targets, dtype=torch.float32),
            )
        except Exception as e:
            print(f"Error parsing sample: {e}")
            return None

    def resize_to_shape(self, array, shape):
        """
        Resize an array to the specified shape (channels, height, width).
        """
        try:
            channels, height, width = shape
            if array.shape != (channels, height, width):
                resized = []
                for c in range(array.shape[0]):
                    resized_channel = cv2.resize(array[c], (width, height), interpolation=cv2.INTER_LINEAR)
                    resized.append(resized_channel)
                array = np.stack(resized, axis=0)
            return array
        except Exception as e:
            print(f"Error resizing array: {e}")
            return np.zeros(shape, dtype=np.float32)

    def __iter__(self):
        """
        Dataset iterator.
        """
        for file_path in self.file_list:
            samples = self.parse_file(file_path)
            for sample in samples:
                parsed_sample = self.parse_sample(sample)
                if parsed_sample:
                    yield parsed_sample

    def __len__(self):
        """
        Total number of samples in the dataset.
        """
        return self.total_samples

# Model Definition

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Define the upsampling block
        def block(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2)
            )

        # LSTM for learning condition-target differences during training or generating features during testing
        self.lstm_input_dim = opt.channels + opt.noise_dim
        self.lstm_hidden_dim = 128
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Map LSTM output to per-pixel channels
        self.lstm_fc = nn.Linear(self.lstm_hidden_dim * 2, opt.channels)

        # Image generation module
        self.slice_generator = nn.Sequential(
            block(opt.channels, 64, 4, 2, 1),   # 48 -> 96
            block(64, 128, 4, 2, 1),           # 96 -> 192
            nn.ConvTranspose2d(128, opt.channels, kernel_size=4, stride=2, padding=1),  # 192 -> 384
            nn.Conv2d(opt.channels, opt.channels, kernel_size=3, stride=1, padding=1),  # Final layer adjustment
            nn.Tanh()
        )

    def forward(self, conditions, noise, targets=None):
        """
        conditions: Input condition images (batch_size, channels, height, width)
        noise: Random noise (batch_size, noise_dim, height, width)
        targets: Target images (used only during training)
        """
        batch_size = conditions.size(0)
        height, width = conditions.size(2), conditions.size(3)

        # Combine conditions and noise
        combined_input = torch.cat([conditions, noise], dim=1)

        # Reshape for LSTM input
        combined_input = combined_input.permute(0, 2, 3, 1).contiguous()
        combined_input = combined_input.view(batch_size * height, width, -1)

        # Pass through LSTM
        lstm_output, _ = self.lstm(combined_input)
        lstm_output = self.lstm_fc(lstm_output)

        # Reshape back to 2D image format
        lstm_output = lstm_output.view(batch_size, height, width, -1).permute(0, 3, 1, 2)

        # Generate the output image
        generated_output = self.slice_generator(lstm_output)

        # Adjust width to 1/5 of input width
        target_width = width // 5
        generated_output = F.interpolate(generated_output, size=(height, target_width), mode="bilinear", align_corners=False)

        return generated_output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Define the convolutional block
        def block(in_channels, out_channels, kernel_size=2, stride=1, padding=0):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )

        # Compute the dimensions of the convolutional output
        def compute_conv_dim(input_size, kernel_size, stride, padding, num_layers):
            size = input_size
            for _ in range(num_layers):
                size = (size + 2 * padding - kernel_size) // stride + 1
            return size

        # Updated convolutional output dimensions
        conv_height = compute_conv_dim(opt.img_size, kernel_size=2, stride=1, padding=0, num_layers=3)
        conv_width = compute_conv_dim(opt.img_size // 5, kernel_size=2, stride=1, padding=0, num_layers=3)

        # Convolutional module
        self.conv = nn.Sequential(
            block(opt.channels, 64, kernel_size=2, stride=1, padding=0),
            block(64, 128, kernel_size=2, stride=1, padding=0),
            block(128, 256, kernel_size=2, stride=1, padding=0),
        )

        # LSTM module
        self.lstm_input_dim = 256 * conv_height
        self.lstm_hidden_dim = 128
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim * 2 * conv_width, 1),
            nn.Sigmoid()
        )

    def forward(self, input_image):
        """
        input_image: Image tensor (batch_size, channels, height, width)
        """
        # Validate input width matches the expected 1/5 of original width
        expected_width = opt.img_size // 5
        assert input_image.size(-1) == expected_width, (
            f"Discriminator input width mismatch: expected {expected_width}, got {input_image.size(-1)}"
        )

        # Extract convolutional features
        conv_features = self.conv(input_image)

        # Reshape for LSTM input
        batch_size, channels, height, width = conv_features.size()
        conv_features = conv_features.permute(0, 3, 1, 2).reshape(batch_size, width, -1)

        # Pass through LSTM
        lstm_output, _ = self.lstm(conv_features)

        # Flatten and pass through fully connected layer
        lstm_output = lstm_output.reshape(batch_size, -1)
        validity = self.fc(lstm_output)

        return validity


# Initialize models and optimizers
generator = Generator().to(device)
discriminator = Discriminator().to(device)
adversarial_loss = nn.BCELoss().to(device)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr * 0.5, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr * 0.5, betas=(opt.b1, opt.b2))
l1_loss = torch.nn.MSELoss().to(device)

# Load ResNet50 as the base model for SwAV
swav = resnet50()

# Load SwAV pretrained weights
state_dict = torch.hub.load_state_dict_from_url(
    "https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar",
    map_location="cpu"
)
# print(state_dict.keys())  # Inspect top-level keys
# Clean up key names in the SwAV state dictionary
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

# Load weights into the model
swav.load_state_dict(state_dict, strict=False)
swav = swav.to(device).eval()

# Freeze SwAV weights
for param in swav.parameters():
    param.requires_grad = False

# Remove the fully connected layer, keeping only convolutional features
swav_features = nn.Sequential(*list(swav.children())[:-2]).to(device)
swav_features.eval()

# Define SwAV perceptual loss
def perceptual_loss_swav(fake, real):
    # Extract features
    fake_features = swav_features(fake)
    real_features = swav_features(real)
    # Compute MSE loss
    return 2000 * F.mse_loss(fake_features, real_features)

def create_dynamic_mask(condition, target, threshold=0.99):
    """
    Dynamically generate a mask based on condition and target:
    - Ignore loss (Mask = 0) when both condition and target are completely white.
    - Compute loss (Mask = 1) otherwise.
    """
    condition_mask = (condition < threshold).float()  # Non-white areas in condition
    target_mask = (target < threshold).float()        # Non-white areas in target

    # Ignore only when both condition and target are completely white
    dynamic_mask = (condition_mask + target_mask > 0).float()  # Valid regions as 1
    # Enhance: Assign higher weights to non-white regions in the target
    dynamic_mask = dynamic_mask * (1 + 10.0 * target_mask)
    return dynamic_mask

def masked_l1_loss(gen_targets, targets, mask):
    """
    Compute masked L1 loss.
    - gen_targets: Generated images
    - targets: Target images
    - mask: Dynamically generated binary mask
    """
    masked_diff = mask * torch.abs(gen_targets - targets)
    return torch.sum(masked_diff) / torch.sum(mask)  # Normalize the loss

def masked_perceptual_loss(gen_features, target_features, mask):
    """
    Compute masked perceptual loss.
    - gen_features: Features from the generator
    - target_features: Target features
    - mask: Dynamically generated binary mask
    """
    # Ensure channel counts match
    if gen_features.size(1) != target_features.size(1):
        raise RuntimeError(
            f"Channel mismatch: gen_features has {gen_features.size(1)} channels, "
            f"but target_features has {target_features.size(1)} channels"
        )

    # Resize mask to match the H, W of gen_features and target_features
    mask_resized = F.interpolate(mask, size=gen_features.shape[-2:], mode="nearest")

    # Ensure mask's channels match gen_features
    if mask_resized.size(1) == 1:
        # Repeat single-channel mask to match gen_features
        mask_resized = mask_resized.repeat(1, gen_features.size(1), 1, 1)
    elif mask_resized.size(1) == 3:
        # Adjust 3-channel mask to match gen_features
        repeat_times = gen_features.size(1) // mask_resized.size(1)
        extra_channels = gen_features.size(1) % mask_resized.size(1)
        mask_resized = mask_resized.repeat(1, repeat_times, 1, 1)
        if extra_channels > 0:
            extra_mask = mask_resized[:, :extra_channels, :, :]
            mask_resized = torch.cat([mask_resized, extra_mask], dim=1)
    else:
        raise RuntimeError(
            f"Unexpected mask channels: mask has {mask_resized.size(1)} channels, "
            f"but gen_features requires {gen_features.size(1)} channels"
        )

    # Compute weighted loss
    masked_diff = mask_resized * (gen_features - target_features) ** 2
    return torch.sum(masked_diff) / torch.sum(mask_resized)


# Load datasets
train_files, test_files = load_and_split_datasets(data_folder, test_split_ratio=0.2)

# Data loaders
img_shape = (3, 48, 48)  # Target image dimensions
train_dataset = RGBStickDataset(train_files, img_shape)
test_dataset = RGBStickDataset(test_files, img_shape)
train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

# Function to save checkpoints
def save_checkpoint(epoch):
    """
    Save generator and discriminator models, optimizers, and current epoch as checkpoint files.
    """
    checkpoint_path = os.path.join(opt.save_dir, f"checkpoint_epoch_{epoch}.pth")
    generator_path = os.path.join(opt.save_dir, f"generator_epoch_{epoch}.pth")
    discriminator_path = os.path.join(opt.save_dir, f"discriminator_epoch_{epoch}.pth")

    # Save full checkpoint
    torch.save({
        "epoch": epoch,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_G": optimizer_G.state_dict(),
        "optimizer_D": optimizer_D.state_dict(),
    }, checkpoint_path)

    # Save generator and discriminator separately
    torch.save(generator.state_dict(), generator_path)
    torch.save(discriminator.state_dict(), discriminator_path)

    # Save the latest checkpoint path
    with open(last_checkpoint_file, "w") as f:
        f.write(checkpoint_path)

    print(f"Checkpoint saved: {checkpoint_path}")
    print(f"Generator model saved: {generator_path}")
    print(f"Discriminator model saved: {discriminator_path}")

# Resume training logic
if opt.resume:
    print(f"Attempting to resume training from checkpoint: {opt.resume}")
    if os.path.isfile(opt.resume):
        checkpoint = torch.load(opt.resume, map_location=device)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_G.load_state_dict(checkpoint["optimizer_G"])
        optimizer_D.load_state_dict(checkpoint["optimizer_D"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed training from epoch {start_epoch}.")
    else:
        print(f"Checkpoint file not found: {opt.resume}")
        start_epoch = 0
else:
    print("No resume checkpoint provided. Starting training from scratch.")
    start_epoch = 0

# Load loss history if resuming from a checkpoint
if opt.resume and os.path.exists(opt.save_dir):
    loss_history_path = os.path.join(opt.save_dir, "loss_history.pth")
    if os.path.exists(loss_history_path):
        loss_history = torch.load(loss_history_path)
        train_g_losses = loss_history["train_g_losses"]
        train_d_losses = loss_history["train_d_losses"]
        test_g_losses = loss_history["test_g_losses"]
        test_d_losses = loss_history["test_d_losses"]
        print("Loss history loaded.")
    else:
        # Initialize loss lists
        train_g_losses, train_d_losses = [], []
        test_g_losses, test_d_losses = [], []
else:
    # Initialize loss lists
    train_g_losses, train_d_losses = [], []
    test_g_losses, test_d_losses = [], []

# Denormalize tensor to [0, 1] range
def denormalize(tensor):
    return (tensor + 1) / 2

# Model evaluation function
def evaluate_model(generator, discriminator, dataloader, device, epoch, save_dir):
    generator.eval()
    discriminator.eval()
    g_loss_total, d_loss_total = 0, 0
    image_count = 0  # Limit the number of images to save

    with torch.no_grad():
        for conditions, targets in dataloader:
            conditions, targets = conditions.to(device), targets.to(device)

            # Generate images with a width equal to one-fifth of the target's width
            noise = torch.randn(
                conditions.size(0),
                opt.noise_dim,
                conditions.size(2),
                conditions.size(3),
                device=device
            )
            gen_segments = generator(conditions, noise)

            # Ensure each generated segment matches the target's batch dimension
            gen_segments = [
                segment.expand(targets.size(0), -1, -1, -1) if segment.size(0) == 1 else segment
                for segment in gen_segments
            ]

            # Extract target segments corresponding to generated slices
            gen_slice_width = gen_segments[0].shape[-1]
            target_segments = [
                targets[:, :, :, slice_idx * gen_slice_width: (slice_idx + 1) * gen_slice_width]
                for slice_idx in range(len(gen_segments))
            ]

            # Compute generator and discriminator losses
            g_loss_total_batch, d_loss_total_batch = 0, 0
            for gen_segment, target_segment in zip(gen_segments, target_segments):
                # Ensure shape consistency
                assert target_segment.shape == gen_segment.shape, (
                    f"Shape mismatch: target_segment {target_segment.shape}, gen_segment {gen_segment.shape}"
                )
                # Generator loss
                validity = discriminator(gen_segment)
                g_adv_loss = adversarial_loss(validity, torch.ones_like(validity))
                g_perceptual_loss = perceptual_loss_swav(gen_segment, target_segment) / 100.0
                g_self_loss = l1_loss(gen_segment, target_segment)
                gen_to_target_diff = swav_features(gen_segment) - swav_features(target_segment)
                perceptual_contrast_loss = F.mse_loss(gen_to_target_diff, torch.zeros_like(gen_to_target_diff))

                g_loss = (
                    advers_weight * g_adv_loss
                    + percept_weight * g_perceptual_loss
                    + 0.2 * g_self_loss
                    + contrast_weight * perceptual_contrast_loss
                )
                g_loss_total_batch += g_loss.item()

                # Discriminator loss
                validity_real = discriminator(target_segment)
                validity_fake = discriminator(gen_segment.detach())
                d_loss_real = adversarial_loss(validity_real, torch.ones_like(validity_real))
                d_loss_fake = adversarial_loss(validity_fake, torch.zeros_like(validity_fake))
                d_loss = (d_loss_real + d_loss_fake) / 2
                d_loss_total_batch += d_loss.item()

            g_loss_total += g_loss_total_batch / len(gen_segments)
            d_loss_total += d_loss_total_batch / len(gen_segments)

            # Save comparisons between generated and target images
            if image_count < 40:
                batch_size = conditions.size(0)
                for idx in range(min(40 - image_count, batch_size)):
                    gen_image = gen_segments[0]  # Take the first generated segment

                    if gen_image.dim() == 3:  # Add batch dimension if missing
                        gen_image = gen_image.unsqueeze(0)

                    if gen_image.size(0) != batch_size:  # Adjust batch size
                        gen_image = gen_image.expand(batch_size, -1, -1, -1)

                    # Select the image for the current index
                    gen_image = gen_image[idx]

                    # Convert to numpy for visualization
                    gen_image = denormalize(gen_image).cpu().numpy().transpose(1, 2, 0)
                    condition_image = denormalize(conditions[idx]).cpu().numpy().transpose(1, 2, 0)
                    target_image = denormalize(target_segments[0][idx]).cpu().numpy().transpose(1, 2, 0)

                    # Plot and save comparisons
                    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
                    axes[0].imshow(condition_image)
                    axes[0].set_title("Condition")
                    axes[0].axis("off")
                    axes[1].imshow(gen_image)
                    axes[1].set_title("Generated")
                    axes[1].axis("off")
                    axes[2].imshow(target_image)
                    axes[2].set_title("Target Segment")
                    axes[2].axis("off")

                    save_path = os.path.join(save_dir, f"epoch_{epoch}_comparison_{image_count + 1}.png")
                    plt.savefig(save_path)
                    print(f"Saved comparison image: {save_path}")
                    plt.close()

                    image_count += 1
                    if image_count >= 40:
                        break

    return g_loss_total / len(dataloader), d_loss_total / len(dataloader)

# Real-time loss plotting function
def plot_losses(epoch):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_g_losses) + 1), train_g_losses, label='Train G Loss', marker='o')
    plt.plot(range(1, len(train_d_losses) + 1), train_d_losses, label='Train D Loss', marker='o')
    plt.plot(range(1, len(test_g_losses) + 1), test_g_losses, label='Test G Loss', linestyle='--', marker='x')
    plt.plot(range(1, len(test_d_losses) + 1), test_d_losses, label='Test D Loss', linestyle='--', marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Generator and Discriminator Loss")
    plt.legend()
    plt.grid()
    # Save the plot
    plot_path = os.path.join(opt.save_dir, f"loss_plot_epoch_{len(train_g_losses)}.png")
    plt.savefig(plot_path)
    print(f"Loss plot saved: {plot_path}")
    # Non-blocking display of the plot, closes automatically after 10 seconds
    plt.show(block=False)
    time.sleep(10)
    plt.close()

# Warm-up settings
warmup_epochs = 7
max_lr_G = opt.lr * 20  # Target learning rate for the generator
max_lr_D = opt.lr * 2   # Target learning rate for the discriminator

# Custom nonlinear warm-up functions
def cosine_warmup(epoch, max_lr, warmup_epochs):
    """Cosine warm-up function."""
    return max_lr * (1 - np.cos(np.pi * epoch / warmup_epochs)) / 2

def dynamic_cosine_warmup(epoch, max_lr, warmup_epochs):
    """Dynamic cosine warm-up function."""
    return max_lr * (1 - np.cos(np.pi * epoch / warmup_epochs)) / 2

def nonlinear_warmup(epoch):
    return cosine_warmup(epoch, max_lr=1.0, warmup_epochs=warmup_epochs)

# Warm-up schedulers
warmup_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=nonlinear_warmup)
warmup_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=nonlinear_warmup)


# Training loop

# Initialize perceptual loss weight and EMA smoothing coefficients
percept_weight = 0.5
ema_alpha = 0.8
smoothed_d_g_ratio = 1.0

# Initialize adversarial loss weight
advers_weight = 1.0
ema_alpha_adv = 0.8
contrast_weight = 0.5

# Learning rate schedulers
scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=5, verbose=True)
scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, patience=5, verbose=True)

# Initialize previous loss values
d_loss_prev, g_loss_prev = float("inf"), float("inf")

for epoch in range(start_epoch, opt.n_epochs):
    current_epoch = epoch if start_epoch != 0 else epoch + 1  # Human-readable epoch count
    print(f"Epoch {current_epoch}/{opt.n_epochs} starting...")

    generator.train()
    discriminator.train()

    train_g_loss, train_d_loss = 0, 0
    accumulation_steps = opt.accumulate_batches

    with alive_bar(len(train_dataloader), title=f"Epoch {current_epoch}") as bar:
        for i, (conditions, targets) in enumerate(train_dataloader):
            conditions, targets = conditions.to(device), targets.to(device)

            # Generate image slices
            noise = torch.randn(
                conditions.size(0),
                opt.noise_dim,
                conditions.size(2),
                conditions.size(3),
                device=device
            )
            gen_segments = generator(conditions, noise, targets)

            # Extract corresponding target slices
            gen_slice_width = gen_segments[0].shape[-1]
            target_segments = [
                targets[:, :, :, slice_idx * gen_slice_width: (slice_idx + 1) * gen_slice_width]
                for slice_idx in range(len(gen_segments))
            ]

            # Train Generator
            g_loss_total = 0
            for gen_segment, target_segment in zip(gen_segments, target_segments):
                if gen_segment.dim() == 3:  # Ensure batch dimension
                    gen_segment = gen_segment.expand(target_segment.size(0), -1, -1, -1)

                assert target_segment.shape == gen_segment.shape, (
                    f"Shape mismatch: target_segment {target_segment.shape}, gen_segment {gen_segment.shape}"
                )

                validity = discriminator(gen_segment)

                # Compute generator losses
                g_adv_loss = adversarial_loss(validity, torch.ones_like(validity))
                g_perceptual_loss = perceptual_loss_swav(gen_segment, target_segment) / 100.0
                g_self_loss = l1_loss(gen_segment, target_segment)
                gen_to_target_diff = swav_features(gen_segment) - swav_features(target_segment)
                perceptual_contrast_loss = F.mse_loss(
                    gen_to_target_diff, torch.zeros_like(gen_to_target_diff)
                )

                g_loss = (
                    advers_weight * g_adv_loss
                    + percept_weight * g_perceptual_loss
                    + 0.2 * g_self_loss
                    + contrast_weight * perceptual_contrast_loss
                )
                g_loss_total += g_loss / accumulation_steps
            train_g_adv_loss, train_g_perceptual_loss = 0, 0
            g_loss_total.backward()

            # Update optimizer for generator
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                optimizer_G.step()
                optimizer_G.zero_grad()

            # Train Discriminator
            d_loss_total = 0
            for gen_segment, target_segment in zip(gen_segments, target_segments):
                if gen_segment.dim() == 3:  # Ensure batch dimension
                    gen_segment = gen_segment.unsqueeze(0)

                validity_real = discriminator(target_segment)
                validity_fake = discriminator(gen_segment.detach())

                # Label smoothing for real labels
                real_labels = torch.full_like(validity_real, 0.9, device=device)
                fake_labels = torch.zeros_like(validity_fake, device=device)

                d_loss_real = adversarial_loss(validity_real, real_labels)
                d_loss_fake = adversarial_loss(validity_fake, fake_labels)
                d_loss = (d_loss_real + d_loss_fake) / 2
                d_loss_total += d_loss / accumulation_steps

            d_loss_total.backward()

            # Update optimizer for discriminator
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=5.0)
                optimizer_D.step()
                optimizer_D.zero_grad()

            # Accumulate losses
            train_g_loss += g_loss_total.item() * accumulation_steps
            train_d_loss += d_loss_total.item() * accumulation_steps
            bar()

    # Calculate average losses
    avg_train_g_loss = train_g_loss / len(train_dataloader)
    avg_train_d_loss = train_d_loss / len(train_dataloader)
    avg_train_g_adv_loss = train_g_adv_loss / len(train_dataloader)
    avg_train_g_perceptual_loss = train_g_perceptual_loss / len(train_dataloader)
    print(f"Epoch {current_epoch}: Train G Loss = {avg_train_g_loss:.6f}, Train D Loss = {avg_train_d_loss:.6f}")
    print(
        f"Epoch {current_epoch}: Train G Adv Loss = {avg_train_g_adv_loss:.6f}, Train G Perceptual Loss = {avg_train_g_perceptual_loss:.6f}")

    # Calculate loss changes
    d_loss_change = d_loss_prev - avg_train_d_loss if d_loss_prev is not None else 0
    g_loss_change = avg_train_g_loss - g_loss_prev if g_loss_prev is not None else 0

    # Update previous loss values
    d_loss_prev = avg_train_d_loss
    g_loss_prev = avg_train_g_loss

    # Dynamic adjustment of percept_weight (commented out for now)
    '''
    d_g_ratio = avg_train_d_loss / (avg_train_g_loss + 1e-6)
    new_percept_weight = max(0.05, min(0.5, d_g_ratio))  # Limit to range [0.05, 0.5]
    ema_alpha = 0.5
    percept_weight = ema_alpha * percept_weight + (1 - ema_alpha) * new_percept_weight
    print(f"Updated percept_weight: {percept_weight:.4f}")
    '''

    # Dynamic adjustment of adversarial loss weight (commented out for now)
    '''
    current_d_g_ratio = avg_train_d_loss / (avg_train_g_loss + 1e-6)
    smoothed_d_g_ratio = ema_alpha_adv * smoothed_d_g_ratio + (1 - ema_alpha_adv) * current_d_g_ratio
    advers_weight = max(0.1, min(2.0, smoothed_d_g_ratio))  # Limit to range [0.1, 2.0]
    print(f"Updated adversarial loss weight: {advers_weight:.4f}")
    '''

    # Warm-up adjustments for early epochs
    if current_epoch <= warmup_epochs:
        if g_loss_change > 0 and d_loss_change < 0:
            generator_growth_rate = max(1.0,
                                        min(3.0, 1 + (g_loss_change - d_loss_change) / max(abs(d_loss_change), 1e-6)))
            discriminator_growth_rate = max(0.5, 1.0)
        elif g_loss_change < 0 and d_loss_change > 0:
            generator_growth_rate = max(0.5, 1.0)
            discriminator_growth_rate = max(1.0, min(3.0, 1 + (d_loss_change - g_loss_change) / max(abs(g_loss_change),
                                                                                                    1e-6)))
        else:
            generator_growth_rate = max(1.0,
                                        min(3.0, 1 + (g_loss_change - d_loss_change) / max(abs(d_loss_change), 1e-6)))
            discriminator_growth_rate = max(1.0, min(3.0, 1 + (d_loss_change - g_loss_change) / max(abs(g_loss_change),
                                                                                                    1e-6)))

        max_lr_G_dynamic = max_lr_G * generator_growth_rate
        max_lr_D_dynamic = max_lr_D * discriminator_growth_rate

        warmup_scheduler_G.lr_lambda = lambda epoch: dynamic_cosine_warmup(epoch, max_lr=max_lr_G_dynamic,
                                                                           warmup_epochs=warmup_epochs)
        warmup_scheduler_D.lr_lambda = lambda epoch: dynamic_cosine_warmup(epoch, max_lr=max_lr_D_dynamic,
                                                                           warmup_epochs=warmup_epochs)

        warmup_scheduler_G.step()
        warmup_scheduler_D.step()

        print(f"Warm-Up Epoch {current_epoch}: Generator LR = {optimizer_G.param_groups[0]['lr']:.6f}")
        print(f"Warm-Up Epoch {current_epoch}: Discriminator LR = {optimizer_D.param_groups[0]['lr']:.6f}")

    # Adjust learning rates using the scheduler
    scheduler_G.step(avg_train_g_loss)
    scheduler_D.step(avg_train_d_loss)
    print(
        f"Adjusted LR after Scheduler (Epoch {current_epoch}): Generator LR = {optimizer_G.param_groups[0]['lr']:.8f}")
    print(
        f"Adjusted LR after Scheduler (Epoch {current_epoch}): Discriminator LR = {optimizer_D.param_groups[0]['lr']:.8f}")

    # Evaluate model on the test dataset
    test_g_loss, test_d_loss = evaluate_model(generator, discriminator, test_dataloader, device, current_epoch,
                                              opt.save_dir)
    print(f"Epoch {current_epoch}: Test G Loss = {test_g_loss:.6f}, Test D Loss = {test_d_loss:.6f}")

    # Save losses
    train_g_losses.append(avg_train_g_loss)
    train_d_losses.append(avg_train_d_loss)
    test_g_losses.append(test_g_loss)
    test_d_losses.append(test_d_loss)

    # Save loss history
    loss_history_path = os.path.join(opt.save_dir, "loss_history.pth")
    torch.save({
        "train_g_losses": train_g_losses,
        "train_d_losses": train_d_losses,
        "test_g_losses": test_g_losses,
        "test_d_losses": test_d_losses,
    }, loss_history_path)

    # Plot loss curves
    plot_losses(len(train_g_losses))

    # Save model checkpoint
    if current_epoch % 1 == 0:
        save_checkpoint(current_epoch)

    # Save final model and loss plot
    save_checkpoint(opt.n_epochs)
    plot_losses(opt.n_epochs)



