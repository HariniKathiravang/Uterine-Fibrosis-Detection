"# Uterine-Fibrosis-Detection" 
# Deep Learning-Based Image Classification using Transfer Learning and Cross Validation

Overview

This project was created as part of learning and understanding the fundamentals of transfer learning in deep learning. It implements an image classification pipeline using a pre-trained ResNet50 model and basic fine-tuning techniques.

The primary goal of this project is educational rather than achieving state-of-the-art performance. It explores how pre-trained models can be adapted to a new dataset and how cross-validation can be used to evaluate model performance more reliably. This does not, by any means aim to replace any medical workers and associated practitioners of this field.

Performance is evaluated using Accuracy, Precision, Recall, and Confusion Matrix analysis.

Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

---

## Model Architecture

The project uses the pre-trained ResNet50 convolutional neural network.

Transfer Learning Strategy

* Load pre-trained ImageNet weights.
* Freeze most of the network layers.
* Fine-tune selected layers near the end of the network.
* Replace the final classification layer to match the target classes.

This setup was chosen to demonstrate the basic workflow of transfer learning rather than to maximize model performance.

The following preprocessing techniques are applied:

* Image resizing
* Tensor conversion
* Normalization using ImageNet statistics

These preprocessing steps are required for compatibility with the ResNet50 architecture.

Training Methodology

5-Fold Cross Validation

The dataset is divided into five folds:

1. Four folds are used for training.
2. One fold is used for validation.
3. The process repeats five times.
4. Performance metrics are averaged across all folds.

Cross-validation was included to provide a more reliable estimate of model performance than a single train-test split.

Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* Confusion Matrix

These metrics provide a basic overview of classification performance.

Running the Project

Open the notebook:

```bash
jupyter notebook FinalNNDL.ipynb
```

Run all cells sequentially to:

1. Load the dataset
2. Preprocess images
3. Train the model
4. Perform cross-validation
5. Evaluate results

Some observations from the implementation:

* Pre-trained weights allow the model to learn faster than training from scratch.
* Fine-tuning selected layers can improve adaptation to a new dataset.
* Cross-validation provides a more stable estimate of performance.
* Multiple evaluation metrics help identify strengths and weaknesses in predictions.

Results should be viewed in the context of learning and experimentation rather than as a benchmark against other models.

This project has several limitations typical of a learning-focused implementation:

* Minimal data augmentation.
* No hyperparameter tuning.
* No learning rate scheduling.
* No model checkpointing.
* No comparison against alternative architectures.
* Limited experiment tracking and reproducibility features.
* Notebook-based implementation rather than a modular project structure.

---


