from tqdm import tqdm
import cv2
import numpy as np
import os
import torch
from torchvision import transforms
from time import time

# Video URL
vid_url = 'burning_forest_video.mp4'
output_video_path = 'output_video.avi'

# Device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
img_size = 224
model_path = os.path.join(os.getcwd(), "checkpoints/best_vgg_small_model_jit.pth")
model = torch.jit.load(model_path, map_location=device)
model.to(device)
model.eval()

# Model transform
model_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

# Open video
cap = cv2.VideoCapture(vid_url)
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

# Get video properties
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Class labels
class_names = ['No Fire', 'Start Fire', 'Fire']

# Loop through video with progress bar
for _ in tqdm(range(frame_count), desc="Processing video", ncols=100):
    ret, frame = cap.read()
    if not ret:
        break

    # Apply model transform
    input_tensor = model_transform(frame).unsqueeze(0).to(device)

    # take start time
    start = time()

    # Predict class
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        label = class_names[predicted_class]

    computation_time = time() - start

    # Draw label in the center
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2
    cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    #frame rate
    frame_rate = f"{1/computation_time:.1f} FPS"

    text_size = cv2.getTextSize(frame_rate, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = 3*(frame.shape[1] - text_size[0]) // 4
    text_y = 3*(frame.shape[0] + text_size[1]) // 4
    cv2.putText(frame, frame_rate, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Write the frame to output
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved as {output_video_path}")
