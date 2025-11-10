import argparse
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from marigold.marigoldDepth import MarigoldDepth


def parse_args():
    parser = argparse.ArgumentParser("Inference with Marigold Depth")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--ensemble", type=int, default=10, help="Ensemble size (median aggregation)")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def infer(model, image_path, device, steps, ensemble):
    # Load and preprocess
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.45, 0.45, 0.45],
                             std=[0.225, 0.225, 0.225]),
    ])
    inp = transform(img).unsqueeze(0).to(device)

    # Inference with diffusion denoising
    model.eval()
    model.to(device)
    pred_depth = model.predict_depth(inp, num_inference_steps=steps, ensemble_size=ensemble)

    depth = pred_depth.squeeze().detach().cpu().numpy()
    depth -= depth.min()
    depth /= (depth.max() + 1e-8)

    # Resize back to original
    depth_resized = np.array(Image.fromarray((depth * 255).astype(np.uint8)).resize((orig_w, orig_h)))
    return img, depth_resized


def visualize(img, depth_resized, save_path=None, show=False):
    rgb_np = np.array(img)
    depth_colormap = plt.cm.inferno(depth_resized / 255.0)
    depth_rgb = (depth_colormap[:, :, :3] * 255).astype(np.uint8)
    stacked = np.vstack((rgb_np, depth_rgb))

    if show:
        plt.figure(figsize=(8, 10))
        plt.imshow(stacked)
        plt.axis("off")
        plt.show()

    if save_path:
        Image.fromarray(stacked).save(save_path)
        print(f"Saved visualization to {save_path}")

    return stacked


def main():
    args = parse_args()
    print(f"Loading model from {args.checkpoint}")
    model = MarigoldDepth.load_from_checkpoint(args.checkpoint, map_location=args.device)
    model.eval()

    img, depth_resized = infer(model, args.input, args.device, args.steps, args.ensemble)
    visualize(img, depth_resized, save_path=args.save_path, show=args.show)


if __name__ == "__main__":
    main()
