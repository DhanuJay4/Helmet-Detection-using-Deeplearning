# Helmet-Detection-using-Deeplearning
A comprehensive real-time object detection system built with TensorFlow and PyTorch to detect whether individuals are wearing safety helmets. This project is designed for industrial safety monitoring and traffic compliance.
🚀 Key Features
Hybrid Framework Support: Implementation using both TensorFlow Object Detection API and PyTorch-based YOLO architectures.
Real-Time Analysis: Optimized for processing live video feeds from CCTVs or IP cameras.
Dual-Class Detection: Accurately distinguishes between "Helmet" and "No Helmet" classes.
Data Visualization: Includes tools to plot accuracy, loss metrics, and confusion matrices.
🛠️ Tech Stack
Deep Learning: TensorFlow 2.x, PyTorch.
Computer Vision: OpenCV for image processing and video stream handling.
Languages: Python 3.8+.
Models: Support for MobileNet SSD (TensorFlow) and YOLOv5/v8 (PyTorch).
📋 Installation
Clone the Repository:
bash
git clone https://github.com
cd Helmet-Detection-using-Deeplearning
Use code with caution.

Set Up Environment:
bash
conda create -n helmet-det python=3.8 -y
conda activate helmet-det
Use code with caution.

Install Dependencies:
bash
pip install -r requirements.txt
Use code with caution.

📂 Project Structure
text
├── data/               # Training and validation datasets
├── models/             # Saved .h5, .pb, or .pt model weights
├── notebooks/          # Training scripts (TensorFlow/PyTorch)
├── src/                # Core detection and helper scripts
├── output/             # Detection results and evaluation graphs
└── requirements.txt    # List of required Python packages
Use code with caution.

🚦 Usage
1. Training the Model
To start training on your custom dataset using the TensorFlow or PyTorch pipeline:
bash
python train.py --config training_config.yaml
Use code with caution.

2. Running Detection
To run real-time detection using a webcam or video file:
bash
python detect.py --source 0  # For Webcam
python detect.py --source video.mp4  # For Video File
Use code with caution.

📊 Results
The system provides immediate feedback with bounding boxes:
Green Box: Helmet detected.
Red Box: No helmet detected.
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an Issue to improve detection accuracy or add new model architectures.
Suggested Next Steps:
requirements.txt: Create this file and include tensorflow, torch, torchvision, opencv-python, matplotlib, and pillow.
Dataset: Ensure you have your images labeled in Pascal VOC or YOLO format before starting the training
