# Histopathologic-Cancer-Detection

This project demonstrated deep convolutional neural networks in the early detection of cancer using histopathologic image data. Through extensive experimentation and model tuning, we compared two prominent architectures — VGGNet and ResNet — across a range of hyperparameters. Our findings showed that ResNet models significantly outperformed VGGNet in terms of validation accuracy, recall, and F1-score, highlighting the advantage of residual connections in training deeper and more accurate models.

Throughout the process, we encountered and addressed several real-world challenges, including GPU memory limitations and the need to balance model complexity with computational efficiency. By incorporating memory estimation techniques, careful data preprocessing, and systematic hyperparameter tuning, we were able to train models that achieved high performance while remaining resource-aware.

While our best-performing ResNet configuration achieved a validation accuracy of 94.6% and a recall of over 93%, this work lays the foundation for further improvement. Future directions may include incorporating transfer learning, advanced data augmentation, and ensembling strategies.

Overall, this study reinforces the power of deep learning in medical imaging and sets the stage for developing robust, scalable, and interpretable models for cancer detection.
