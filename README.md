# Cold Drinks Inventory Detection using YOLOv5

This project performs **object detection** on a cold drinks inventory dataset using **YOLOv5**, trained on a custom dataset of various beverage brands. The notebook is developed and tested in a **Kaggle environment**, with data processing, model training, and inference steps provided.

---

## 🚀 Project Overview

- Object detection is implemented using **YOLOv5**, a state-of-the-art object detection model.
- The dataset consists of images of popular cold drink brands like Coca Cola, Sprite, Pepsi, etc.
- The trained model detects objects and visualizes results with bounding boxes and confidence scores.

---

## 📂 Dataset Structure

The dataset is structured as follows:

```
/cold-drinks-inventory-dataset/
├── Finalize/
│   ├── images/
│   │   ├── train/  # Training images
│   │   ├── test/   # Testing images
│   └── labels/     # Corresponding annotation labels
```

---

## ⚙️ Installation

Follow these steps to set up the environment and dependencies:

1. Clone the YOLOv5 repository:

   ```bash
   git clone https://github.com/ultralytics/yolov5.git
   cd yolov5
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Install PyTorch and torchvision:

   ```bash
   pip install torch torchvision
   ```

---

## 🛠️ Model Training

To train the YOLOv5 model with the provided dataset:

```bash
python train.py --img 640 --batch 16 --epochs 100 --data /kaggle/working/dataset.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt --name yolo_custom
```

---

## 📊 Object Detection and Visualization

The `detect_and_visualize` function in the code loads a trained YOLOv5 model and performs object detection on test images.

Example usage:

```python
model_path = '/kaggle/input/previous-version-output/yolov5/runs/train/yolo_custom/weights/best.pt'
image_path = '/kaggle/input/testdrinks/image_2024_02_23T12_06_03_052Z.png'
detect_and_visualize(model_path, image_path)
```

### **Detected Classes:**
1. Coca Cola
2. Sprite
3. Pepsi
4. Mountain Dew
5. 7UP
6. Fanta

---

## 📁 Project Structure

```
/Cold-Drinks-Detection/
│-- dataset.yaml                 # Dataset configuration file
│-- yolov5/                       # YOLOv5 cloned repository
│-- model_training.ipynb          # Training script notebook
│-- object_detection.py           # Object detection script
│-- test_images/                   # Test images
│-- README.md                      # Project documentation
```

---

## 🔍 Results

- Model trained on the custom dataset successfully detects cold drink products in images.
- Bounding boxes and confidence scores are visualized using OpenCV and Matplotlib.

Example output:

```
Detected objects: 3
Label Coca Cola: 1
Label Sprite: 2
```

---

## 📚 References

- [Kaggle Notebook](https://www.kaggle.com/)
- [YOLOv5 GitHub](https://github.com/ultralytics/yolov5)

---

## 💜 License

This project is licensed under the MIT License.

---

## 💡 Contributing

Contributions are welcome! Feel free to submit issues or pull requests to improve the project.

---

## 💎 Contact

For any questions or suggestions, please reach out via:

- GitHub Issues
- Email: veritasanalyticas@gmail.com

