Crowd Counting and Density Estimation
This repository contains code and tools for crowd counting and density estimation using deep learning techniques. The model estimates the number of people in crowded scenes and generates density maps from input images.

📌 Features
Predict crowd counts from single images.

Generate high-quality density maps.

Support for benchmark datasets (ShanghaiTech, UCF-QNRF, etc.).

Training and evaluation pipelines.

Pretrained models and visualization tools.

🧠 Model Architecture
This project implements and supports various CNN-based and transformer-based architectures, such as:

MCNN (Multi-column CNN)

CSRNet (Dilated CNN)

CAN (Context-aware Network)

[Optional] ViT-based crowd estimators

📁 Directory Structure
bash
Copy
Edit
crowd_counting/
├── datasets/           # Data loaders and preprocessors
├── models/             # Model definitions
├── utils/              # Utility scripts (visualization, metrics)
├── train.py            # Training script
├── evaluate.py         # Evaluation script
├── inference.py        # Inference on new images
├── config.yaml         # Configuration file
└── README.md
🗂️ Datasets
Supported datasets:

ShanghaiTech Part A/B

UCF-QNRF

UCF-CC-50

Download datasets and place them under datasets/. Preprocessing is handled automatically.

🚀 Getting Started
Prerequisites
Python 3.8+

PyTorch or TensorFlow

OpenCV, NumPy, Matplotlib

tqdm, SciPy

Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Train the Model
bash
Copy
Edit
python train.py --config config.yaml
Evaluate the Model
bash
Copy
Edit
python evaluate.py --checkpoint checkpoints/best_model.pth
Inference
bash
Copy
Edit
python inference.py --image_path test.jpg
📊 Metrics
Mean Absolute Error (MAE)

Mean Squared Error (MSE)

PSNR/SSIM for density map quality (optional)

📷 Visualization
Generate and visualize density maps:

bash
Copy
Edit
python inference.py --image_path sample.jpg --visualize
Example:

Input Image	Density Map
