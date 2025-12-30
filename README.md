# ML Shape Recognizer 🎨🤖

An interactive machine learning application that recognizes hand-drawn shapes in real-time using computer vision and Random Forest classification.

## 🌟 Features

- **Interactive Drawing Interface**: Draw shapes directly with your mouse
- **Machine Learning Recognition**: Uses Random Forest Classifier to identify shapes
- **Two Modes**:
  - **Training Mode**: Collect custom training data by drawing labeled examples
  - **Testing Mode**: Real-time shape recognition with confidence scores
- **5 Shape Classes**: Circle, Square, Triangle, Line, and Star
- **Feature Extraction**: Analyzes geometric properties like aspect ratio, corners, circularity, and extent
- **Model Persistence**: Save and load trained models for reuse

## 🛠️ Technologies Used

- **Python 3.x**
- **OpenCV**: Real-time computer vision and drawing interface
- **scikit-learn**: Machine learning (Random Forest Classifier)
- **NumPy**: Numerical computations
- **Pickle**: Model serialization

## 📋 Requirements

```bash
pip install opencv-python
pip install scikit-learn
pip install numpy
```

Or install all at once:
```bash
pip install opencv-python scikit-learn numpy
```

## 🚀 Getting Started

### 1. Train the Model

First, you need to create training data:

```bash
python training.py
```

**Training Workflow:**
1. Press `1-5` to select a shape label:
   - `1` - Circle
   - `2` - Square
   - `3` - Triangle
   - `4` - Line
   - `5` - Star
2. Draw the shape with your mouse (click and drag)
3. The shape is automatically saved when you release the mouse
4. Repeat for **at least 10 examples per shape** (more is better!)
5. Press `T` to train the model
6. Model is saved as `shape_model.pkl`

**Training Tips:**
- Draw varied examples (different sizes, orientations)
- Aim for 15-20 examples per shape for best results
- The model trains faster with more diverse data

### 2. Test the Model

Once trained, test your model:

```bash
python main.py
```

**Testing Mode:**
- Simply draw a shape with your mouse
- The model predicts the shape automatically
- See the prediction and confidence score in real-time
- Press `C` to clear and draw again
- Press `Q` to quit

## 📊 How It Works

### Feature Extraction

The model extracts 5 key geometric features from each drawing:

1. **Aspect Ratio**: Width/Height ratio of bounding box
2. **Number of Corners**: Detected using polygon approximation
3. **Circularity**: Measures how close the shape is to a perfect circle
4. **Extent**: Ratio of shape area to bounding box area
5. **Is Closed**: Whether the start and end points are close together

### Classification

- **Algorithm**: Random Forest Classifier (100 estimators, max depth 10)
- **Training**: 80/20 train-test split (for datasets ≥20 examples)
- **Output**: Predicted shape class + confidence percentage

## 🎮 Controls

### Training Mode (`training.py`)
| Key | Action |
|-----|--------|
| `1-5` | Set label (Circle/Square/Triangle/Line/Star) |
| `Mouse Drag` | Draw shape |
| `T` | Train model on collected examples |
| `C` | Clear canvas |
| `Q` | Quit and save training data |

### Testing Mode (`main.py`)
| Key | Action |
|-----|--------|
| `Mouse Drag` | Draw shape to recognize |
| `C` | Clear canvas |
| `Q` | Quit application |

## 📁 File Structure

```
├── training.py           # Training mode script
├── main.py              # Testing mode script
├── shape_model.pkl      # Trained model (generated after training)
├── training_data.pkl    # Raw training data (auto-saved)
└── README.md           # This file
```

## 🎯 Example Workflow

1. **Collect Training Data**:
   ```bash
   python training.py
   # Press 1, draw 15 circles
   # Press 2, draw 15 squares
   # Press 3, draw 15 triangles
   # Press 4, draw 15 lines
   # Press 5, draw 15 stars
   # Press T to train
   ```

2. **Test Recognition**:
   ```bash
   python main.py
   # Draw any shape and see prediction!
   ```

## 📈 Model Performance

- **Typical Accuracy**: 85-95% (depends on training data quality)
- **Training Time**: < 1 second for 75 examples
- **Prediction Time**: Real-time (< 10ms per prediction)

## 🔮 Future Improvements

- [ ] Add more shape classes (pentagon, hexagon, etc.)
- [ ] Implement deep learning (CNN) for better accuracy
- [ ] Add shape rotation normalization
- [ ] Export predictions to file
- [ ] Add undo/redo functionality
- [ ] Implement data augmentation
- [ ] Create GUI with buttons instead of keyboard shortcuts

## 🐛 Troubleshooting

**"No model found" error:**
- Run `training.py` first to train and save a model

**Low accuracy:**
- Collect more training examples (aim for 20+ per shape)
- Draw more varied examples (different sizes, styles)
- Ensure shapes are drawn clearly and distinctly

**Model won't train:**
- Need at least 10 total examples
- Need at least 2 different shape classes
- Check console for specific error messages

## 📝 License

MIT License - Feel free to use and modify!

## 👤 Author

**Ashita Sharma**
- GitHub: [@Ashita-Sharma](https://github.com/Ashita-Sharma)
- LinkedIn: [Ashita Sharma](https://www.linkedin.com/in/ashita-sharma-a867b4384)

## 🙏 Acknowledgments

- Built with OpenCV for real-time drawing interface
- Uses scikit-learn's Random Forest for robust classification
- Inspired by gesture recognition and computer vision research

---

⭐ **Star this repo if you found it helpful!**
