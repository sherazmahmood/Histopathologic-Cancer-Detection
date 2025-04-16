# CNN Cancer Detection - Mini Project (Kaggle Histopathologic Dataset)

This project demonstrated deep convolutional neural networks in the early detection of cancer using histopathologic image data. Through extensive experimentation and model tuning, we compared two prominent architectures — VGGNet and ResNet — across a range of hyperparameters. Our findings showed that ResNet models significantly outperformed VGGNet in terms of validation accuracy, recall, and F1-score, highlighting the advantage of residual connections in training deeper and more accurate models.

Throughout the process, we encountered and addressed several real-world challenges, including GPU memory limitations and the need to balance model complexity with computational efficiency. By incorporating memory estimation techniques, careful data preprocessing, and systematic hyperparameter tuning, we were able to train models that achieved high performance while remaining resource-aware.

While our best-performing ResNet configuration achieved a validation accuracy of 94.6% and a recall of over 93%, this work lays the foundation for further improvement. Future directions may include incorporating transfer learning, advanced data augmentation, and ensembling strategies.

Overall, this study reinforces the power of deep learning in medical imaging and sets the stage for developing robust, scalable, and interpretable models for cancer detection.

## Project Structure

```
.
├── CNN_CancerDetectionKaggle_MiniProject.ipynb   # Main analysis notebook
├── /vgg_top_models                               # Saved VGG models and training history
├── /resnet_top_models                            # Saved ResNet models and training history
```

---

## Overview

The objective of this mini-project was to:

- Compare two CNN architectures (VGGNet and ResNet)
- Tune hyperparameters to optimize performance
- Evaluate using validation accuracy, recall, and F1-score
- Handle GPU memory limits and training stability
- Generate predictions on test data for submission

---

## Dataset

- Source: [Kaggle Histopathologic Cancer Detection](https://www.kaggle.com/competitions/histopathologic-cancer-detection)
- Total Images: 220,025 (50x50 RGB `.tif`)
- Binary Classification: `0` (non-cancer) vs `1` (cancer)

---

## ⚙Models

### ResNet
- Custom residual blocks with variable depth `(2, 2, 2)`, `(3, 4, 6)`, etc.
- Hyperparameter tuning across:
  - `filters_start`, `dense_units`, `dropout_rate`, `block_depths`
- Output: Best model achieved **94.6% validation accuracy**

### VGGNet
- Simplified sequential CNN with stacked `Conv2D` layers
- Tuned across similar hyperparameters
- Lower generalization compared to ResNet (best val accuracy ~92.7%)

---

## Evaluation Metrics

- Accuracy
- Recall
- F1 Score
- Loss curves visualized via `history.h5.json`

---

## Saved Artifacts

Both model folders contain:

- `*.h5`: Trained Keras models
- `history_*.h5.json`: Training history per model
- CSV logs of top-performing configurations

---

## Reproducibility

To reproduce:

1. Run `CNN_CancerDetectionKaggle_MiniProject.ipynb`
2. Mount Google Drive
3. Ensure folder structure is maintained
4. Train or load saved models, generate predictions

---

## Dependencies

- Python 3.7+
- TensorFlow / Keras
- Pandas, NumPy, Matplotlib, Seaborn
- Google Colab Pro (A100 GPU)

---

## Acknowledgments

- Kaggle for the dataset
- [ossaamamahmoud/Sports-Image-Classification](https://github.com/ossaamamahmoud/Sports-Image-Classification) for initial VGGNet reference

---

## Author

Sheraz Mahmood  
M.S. in Computer Science Candidate  
University of Colorado Boulder
