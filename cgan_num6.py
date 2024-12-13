import argparse
import os
import pickle
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
from alive_progress import alive_bar
import pandas as pd
from torch.optim.lr_scheduler import StepLR


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--data_folder", type=str, default="hs300_stock_pickle", help="path to dataset folder")
parser.add_argument("--save_dir", type=str, default="./saved_models_num", help="directory to save models")
parser.add_argument("--lr_schedule", type=str, default="step", help="Learning rate schedule: 'step', 'manual', or 'none'")
parser.add_argument("--lr_gamma", type=float, default=0.5, help="Factor to reduce the learning rate")
parser.add_argument("--lr_step_size", type=int, default=3, help="Number of epochs for step schedule")

opt = parser.parse_args()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def extract_trend_from_target(target_image, cut_ratio=0.5):
    channels, height, width = target_image.shape
    cut_height = int(cut_ratio * height)

    # Extract red, green, and blue channels from the top portion
    red_channel = target_image[0, :cut_height, :]
    green_channel = target_image[1, :cut_height, :]
    blue_channel = target_image[2, :cut_height, :]

    # Define thresholds for identifying dominant colors
    threshold = 0.5

    # Identify valid red pixels: High red, low green, low blue, not white
    red_pixels = (
        (red_channel > threshold)
        & (green_channel < threshold)
        & (blue_channel < threshold)
    ).sum()

    # Identify valid green pixels: High green, low red, low blue, not white
    green_pixels = (
        (green_channel > threshold)
        & (red_channel < threshold)
        & (blue_channel < threshold)
    ).sum()

    # Exclude white pixels (all channels high)
    white_pixels = (
        (red_channel > threshold)
        & (green_channel > threshold)
        & (blue_channel > threshold)
    ).sum()

    # Correct the red and green counts by excluding white pixels
    corrected_red_pixels = red_pixels - white_pixels
    corrected_green_pixels = green_pixels - white_pixels

    # Determine the signal based on corrected counts
    signal = (
        -1
        if corrected_green_pixels > corrected_red_pixels
        else 1
        if corrected_red_pixels > corrected_green_pixels
        else 0
    )

    # Debug outputs
    # print("Total Red Pixels (Corrected):", corrected_red_pixels)
    # print("Total Green Pixels (Corrected):", corrected_green_pixels)
    # print("Excluded White Pixels:", white_pixels)
    # print("Signal:", signal)

    return signal


class RGBStickDataset(IterableDataset):
    def __init__(self, file_list, img_shape):
        super(RGBStickDataset, self).__init__()
        self.file_list = file_list
        self.img_shape = img_shape
        self.total_samples = sum(self.count_samples(file_path) for file_path in self.file_list)

    def count_samples(self, file_path):
        """
        Count the number of samples by iterating over all `images` data.
        """
        try:
            with open(file_path, "rb") as f:
                file_data = pickle.load(f)
                return sum(len(image_data["condition"]) for image_data in file_data["images"])
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return 0

    def __iter__(self):
        """
        Iterator to process `data['images'][i]['condition'][j]` format data.
        """
        for file_path in self.file_list:
            try:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                    for image_idx, image_data in enumerate(data["images"]):
                        try:
                            # Iterate through `condition` and `target` data
                            for condition_idx in range(len(image_data["condition"])):
                                # Extract `condition` and `target` images
                                condition = torch.tensor(
                                    image_data["condition"][condition_idx], dtype=torch.float32
                                ).squeeze(0)
                                target_image = torch.tensor(
                                    image_data["target"][condition_idx], dtype=torch.float32
                                ).squeeze(0)

                                # Extract total trend values from `target_image`
                                total_trend = extract_trend_from_target(target_image)

                                # Convert to Tensor
                                numerical_target = torch.tensor(
                                    total_trend, dtype=torch.float32
                                ).unsqueeze(0)

                                yield condition, target_image, numerical_target
                        except Exception as sample_error:
                            print(f"Skipping invalid sample in {file_path}, Image {image_idx}: {sample_error}")
            except Exception as file_error:
                print(f"Skipping file {file_path}: {file_error}")

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return self.total_samples


# Generator
class Generator(nn.Module):
    def __init__(self, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        # Output logits for 3 categories: -1, 0, 1
        self.model = nn.Sequential(
            nn.Linear(np.prod(img_shape), 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)  # Output logits for three categories
        )

    def forward(self, condition):
        flat_condition = condition.view(condition.size(0), -1)
        logits = self.model(flat_condition)  # Logits for -1, 0, 1
        probabilities = F.softmax(logits, dim=1)  # Convert logits to probabilities
        return probabilities


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)  # Sigmoid activation removed
        )

    def forward(self, numerical_target):
        return self.model(numerical_target)


# Wasserstein loss function
def wasserstein_loss(y_true, y_pred):
    return torch.mean(y_true * y_pred)

# Load dataset and split into training and testing
def load_dataset(folder_path, test_split_ratio=0.2):
    file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pkl")]
    random.shuffle(file_list)
    split_idx = int(len(file_list) * (1 - test_split_ratio))
    return file_list[:split_idx], file_list[split_idx:]

# Save model checkpoints
def save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, epoch, save_dir):
    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")
    os.makedirs(save_dir, exist_ok=True)

    torch.save({
        "epoch": epoch,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_G": optimizer_G.state_dict(),
        "optimizer_D": optimizer_D.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at: {checkpoint_path}")

# Save final model weights
def save_models(generator, discriminator, save_dir, epoch):
    os.makedirs(save_dir, exist_ok=True)
    generator_path = os.path.join(save_dir, f"generator_epoch{epoch}.pth")
    discriminator_path = os.path.join(save_dir, f"discriminator_epoch{epoch}.pth")
    torch.save(generator.state_dict(), generator_path)
    torch.save(discriminator.state_dict(), discriminator_path)
    print(f"Generator model saved at: {generator_path}")
    print(f"Discriminator model saved at: {discriminator_path}")

# Save visualization of conditions, target images, and model output
def save_visualization(epoch, batch_idx, conditions, target_images, numerical_targets, generated_values, save_dir):
    """
    Save combined images and numerical values for visualization.
    """
    os.makedirs(save_dir, exist_ok=True)
    num_samples = min(conditions.size(0), 5)  # Save up to 5 samples per batch

    for i in range(num_samples):
        fig, axes = plt.subplots(1, 4, figsize=(10, 2.5))

        # Plot condition image
        condition_img = conditions[i].cpu().numpy().transpose(1, 2, 0)
        axes[0].imshow(condition_img)
        axes[0].set_title("Condition Image")
        axes[0].axis("off")

        # Plot target image
        target_img = target_images[i].cpu().numpy().transpose(1, 2, 0)
        axes[1].imshow(target_img)
        axes[1].set_title("Target Image")
        axes[1].axis("off")

        # Display target value as text
        target_value = numerical_targets[i].item()
        axes[2].text(
            0.5, 0.5, f"Target Value:\n{target_value:.2f}", fontsize=14, ha="center", va="center"
        )
        axes[2].axis("off")

        # Display generated value as text
        generated_value = generated_values[i].item()
        axes[3].text(
            0.5, 0.5, f"Generated Value:\n{generated_value:.2f}", fontsize=14, ha="center", va="center"
        )
        axes[3].axis("off")

        # Save visualization
        visualization_path = os.path.join(
            save_dir, f"epoch_{epoch}_batch_{batch_idx}_sample_{i}.png"
        )
        plt.tight_layout(pad=1.0)
        plt.savefig(visualization_path)
        plt.close()
        print(f"Visualization saved at: {visualization_path}")

# Compute gradient penalty for Wasserstein GAN
def gradient_penalty(discriminator, real_data, fake_data, conditions, lambda_gp=10):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(real_data)

    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.requires_grad_(True)

    disc_interpolates = discriminator(interpolates.unsqueeze(-1))

    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()

    return gradient_penalty

# Initialize lists to store training and testing losses
train_g_losses = []
train_d_losses = []
test_g_losses = []
test_d_losses = []

# Global DataFrame to store metrics
metrics_df = pd.DataFrame(columns=["epoch", "batch_idx", "target_value", "generated_value", "d_loss", "g_loss"])

def train(generator, discriminator, train_loader, test_loader, optimizer_G, optimizer_D, n_epochs, save_dir, lambda_gp=10):
    # Initialize lists to store losses and accuracy
    train_g_losses, train_d_losses = [], []
    test_g_losses, test_d_losses = [], []
    accuracies = []

    # Initialize optimizers and learning rate schedulers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)
    scheduler_G = StepLR(optimizer_G, step_size=opt.lr_step_size, gamma=opt.lr_gamma)
    scheduler_D = StepLR(optimizer_D, step_size=opt.lr_step_size, gamma=opt.lr_gamma)

    global metrics_df  # Use the global DataFrame to store metrics

    for epoch in range(1, n_epochs + 1):
        print(f"Epoch {epoch}/{n_epochs}")
        generator.train()
        discriminator.train()

        g_loss_total, d_loss_total = 0, 0

        with alive_bar(len(train_loader), title=f"Training Epoch {epoch}") as bar:
            for batch_idx, (conditions, target_images, numerical_targets) in enumerate(train_loader):
                # Move data to the appropriate device
                conditions, target_images, numerical_targets = (
                    conditions.to(device),
                    target_images.to(device),
                    numerical_targets.to(device),
                )

                # Generate fake numerical targets
                fake_numerical = generator(conditions).detach()

                # Train Discriminator
                validity_real = discriminator(numerical_targets.unsqueeze(-1))
                validity_fake = discriminator(fake_numerical.unsqueeze(-1))
                gp = gradient_penalty(discriminator, numerical_targets, fake_numerical, conditions, lambda_gp)
                d_loss = -(torch.mean(validity_real) - torch.mean(validity_fake)) + gp

                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()

                d_loss_total += d_loss.item()

                # Train Generator
                fake_numerical = generator(conditions)
                validity = discriminator(fake_numerical.unsqueeze(-1))
                g_loss = -torch.mean(validity)

                optimizer_G.zero_grad()
                g_loss.backward()
                optimizer_G.step()

                g_loss_total += g_loss.item()

                # Save visualizations periodically
                if batch_idx % 100 == 0:
                    epoch_dir = os.path.join(save_dir, f"epoch_{epoch}")
                    os.makedirs(epoch_dir, exist_ok=True)
                    save_visualization(
                        epoch,
                        batch_idx,
                        conditions[:5],
                        target_images[:5],
                        numerical_targets[:5],
                        fake_numerical[:5],
                        epoch_dir,
                    )

                # Log metrics for the batch
                batch_metrics = pd.DataFrame({
                    "epoch": [epoch],
                    "batch_idx": [batch_idx],
                    "target_value": [numerical_targets[0].item()],
                    "generated_value": [fake_numerical[0].item()],
                    "d_loss": [d_loss.item()],
                    "g_loss": [g_loss.item()],
                })

                # Append batch metrics to the global DataFrame
                if not batch_metrics.empty:
                    if metrics_df.empty:
                        metrics_df = batch_metrics
                    else:
                        metrics_df = pd.concat([metrics_df, batch_metrics], ignore_index=True)

                bar()

                # Compute average losses for the epoch
                train_g_loss = g_loss_total / len(train_loader)
                train_d_loss = d_loss_total / len(train_loader)
                train_g_losses.append(train_g_loss)
                train_d_losses.append(train_d_loss)

                # Evaluate the model on the test set
                test_g_loss, test_d_loss = evaluate(generator, discriminator, test_loader)
                test_g_losses.append(test_g_loss)
                test_d_losses.append(test_d_loss)

                print(
                    f"Epoch [{epoch}/{n_epochs}] - Train G Loss: {train_g_loss:.4f}, Train D Loss: {train_d_loss:.4f}")
                print(f"Epoch [{epoch}/{n_epochs}] - Test G Loss: {test_g_loss:.4f}, Test D Loss: {test_d_loss:.4f}")

                # Save metrics to CSV
                metrics_file = os.path.join(save_dir, f"training_metrics_epoch{epoch}.csv")
                metrics_df.to_csv(metrics_file, index=False)
                print(f"Metrics saved to {metrics_file}")

                # Plot losses after each epoch
                plot_losses(train_g_losses, train_d_losses, test_g_losses, test_d_losses, save_dir, epoch)

                # Save models after each epoch
                save_models(generator, discriminator, save_dir, epoch)

                # Update learning rate schedulers
                scheduler_G.step()
                scheduler_D.step()

                # Log learning rates for debugging
                current_lr_G = optimizer_G.param_groups[0]['lr']
                current_lr_D = optimizer_D.param_groups[0]['lr']
                print(f"Epoch {epoch} - Current LR for Generator: {current_lr_G}, Discriminator: {current_lr_D}")

        def plot_losses(train_g_losses, train_d_losses, test_g_losses, test_d_losses, save_dir, epoch):
            """
            Plot generator and discriminator losses for training and testing, and save the plot.
            """
            plt.figure(figsize=(12, 6))
            plt.plot(train_g_losses, label="Train G Loss")
            plt.plot(train_d_losses, label="Train D Loss")
            plt.plot(test_g_losses, label="Test G Loss")
            plt.plot(test_d_losses, label="Test D Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("GAN Training and Testing Losses")
            plt.legend()
            plt.grid(True)
            loss_plot_path = os.path.join(save_dir, f'loss_plot_epoch{epoch}.png')
            plt.savefig(loss_plot_path)
            plt.close()
            print(f"Loss plot saved at {loss_plot_path}")

        def evaluate(generator, discriminator, dataloader):
            generator.eval()
            discriminator.eval()
            g_loss_total, d_loss_total = 0, 0

            with torch.no_grad():
                for conditions, target_images, numerical_targets in dataloader:
                    conditions, numerical_targets = conditions.to(device), numerical_targets.to(device)

                    # Forward pass through the generator
                    fake_numerical = generator(conditions).detach()

                    # Compute generator loss
                    validity_fake = discriminator(fake_numerical.unsqueeze(-1))
                    g_loss = -torch.mean(validity_fake)  # Wasserstein Generator Loss
                    g_loss_total += g_loss.item()

                    # Compute discriminator loss
                    validity_real = discriminator(numerical_targets.unsqueeze(-1))
                    validity_fake = discriminator(fake_numerical.unsqueeze(-1))
                    d_loss = -(torch.mean(validity_real) - torch.mean(validity_fake))  # Wasserstein Discriminator Loss
                    d_loss_total += d_loss.item()

            g_loss_avg = g_loss_total / len(dataloader)
            d_loss_avg = d_loss_total / len(dataloader)

            return g_loss_avg, d_loss_avg


# Prepare data
train_files, test_files = load_dataset(opt.data_folder)
img_shape = (opt.channels, opt.img_size, opt.img_size)
train_dataset = RGBStickDataset(train_files, img_shape)
test_dataset = RGBStickDataset(test_files, img_shape)
train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size)

# Validate dataset output
for condition, target_image, numerical_target in train_dataset:
    print(f"Condition shape: {condition.shape}, Target value: {numerical_target.item()}")
    break

# Initialize models
generator = Generator(img_shape).to(device)
discriminator = Discriminator().to(device)

# Optimizers and loss
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)
adversarial_loss = nn.BCELoss()

# Train the model
train(
    generator=generator,
    discriminator=discriminator,
    train_loader=train_dataloader,
    test_loader=test_dataloader,
    optimizer_G=optimizer_G,
    optimizer_D=optimizer_D,
    n_epochs=opt.n_epochs,
    save_dir=opt.save_dir,
    lambda_gp=10,  # Gradient penalty weight
)
