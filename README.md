# 🔥fire-detection-using-cnn
CNN-based Fire Detection System using Python and TensorFlow. Detects fire in images and videos with real-time output.
Features
# CNN-based Fire Detection System

## Features
- **Real-Time Fire Detection:** Detects fire in images and videos using a CNN model.
- **Web Interface:** Simple website to upload images or videos and view detection results.
- **Accuracy & Performance Analysis:** Includes accuracy graphs and confusion matrix.
- **Automated Output:** Shows fire detection results with bounding boxes and predictions.
- **Flexible Deployment:** Run locally or host on a server for real-time detection.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- TensorFlow / Keras
- OpenCV
- Flask (for web interface)

---

### Installation

#### Clone the Repository
```bash
git clone <your-repo-link>
cd fire-detection-cnn
```
### 1.Install Dependencies
```
pip install -r requirements.txt

```

### 2.Run the Application
```
python app.py

```
### 3.Open the Website
```
 http://127.0.0.1:5000

```
# 📁 Project Structure
```fire-detection-cnn/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
├── app.py                    # Web interface entry point
├── model/                    # Trained CNN model files
│   └── fire_cnn_model.h5
├── src/                      # Source code
│   ├── cnn/                  # CNN model and training
│   │   ├── train.py
│   │   ├── model.py
│   │   └── utils.py
│   ├── preprocessing/        # Image/video preprocessing
│   │   └── preprocess.py
│   └── evaluation/           # Accuracy and performance analysis
│       ├── plot_accuracy.py
│       └── confusion_matrix.py
├── images/                   # Sample images
├── videos/                   # Demo videos of output
└── results/                  # Accuracy graphs, confusion matrix images
 ```
# 🧪 Testing
Run the test suite:
 ``` python src/evaluation/plot_accuracy.py
     python src/evaluation/confusion_matrix.py
 ```
# 📖 Usage Examples
### Usage Example: Demo Video
Click on the image below to watch the fire detection demo:
```
[![Fire Detection Demo](images/demo_thumbnail.png)](https://screenapp.io/app/#/shared/uNYgBb1SxQ)
```
## 🤝 Contributing

We welcome contributions to improve the CNN-based Fire Detection project! You can help by adding new features, improving accuracy, fixing bugs, or enhancing the web interface.

### Steps to Contribute

1. **Fork the repository**  
   Click the “Fork” button at the top-right of this repository to create your own copy.

2. **Clone your fork locally**  
```bash
git clone <your-forked-repo-link>
cd fire-detection-cnn
```
## 📝 License

This CNN-based Fire Detection project is licensed under the **MIT License**.  
You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of this project, provided that you include the original copyright notice and this license in all copies or substantial portions of the software.  

See the [LICENSE](LICENSE) file for full details.
## 🚨 Troubleshooting

### Common Issues

#### Model Loading Issues
- Ensure that the CNN model file (`fire_cnn_model.h5`) exists in the `model/` folder.  
- Verify that all required dependencies are installed (`TensorFlow`, `Keras`, `OpenCV`, `Flask`).  
- If the model still fails to load, check that the file path in `app.py` or training scripts is correct.

#### Web Interface Issues
- Make sure Flask is installed and running.  
- Check that port 5000 is free. If not, modify `app.py` to use another port:  
```bash
python app.py --port 5001
```
## 🔗 Links

- **TensorFlow Documentation:** [https://www.tensorflow.org](https://www.tensorflow.org)  
- **Keras Documentation:** [https://keras.io](https://keras.io)  
- **Flask Documentation:** [https://flask.palletsprojects.com](https://flask.palletsprojects.com)  
- **OpenCV Documentation:** [https://opencv.org](https://opencv.org)  
- **ScreenApp Demo Video:** [https://screenapp.io/app/#/shared/uNYgBb1SxQ](https://screenapp.io/app/#/shared/uNYgBb1SxQ)

## 💡 Roadmap

- Integrate real-time video streaming with webcam for live fire detection.  
- Add alert notifications (email/SMS) when fire is detected.  
- Optimize the CNN model for faster predictions and lower resource usage.  
- Expand the dataset to improve model generalization across different environments.  
- Add support for multi-camera or drone-based fire monitoring.  
- Deploy the application on cloud or edge devices for public access.  
- Enhance the web interface with better UI/UX and detailed analytics.  


