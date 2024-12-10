# Multi-Modal Image Classification with EfficientNetB5

A deep learning project that combines image and numeric features using EfficientNetB5 architecture for binary classification tasks.

## Project Overview

This project implements a multi-modal neural network that processes both image data and numeric features simultaneously. It utilizes EfficientNetB5 as the backbone for image feature extraction and combines it with a separate numeric data processing branch.

## Architecture

- **Image Branch**: EfficientNetB5 with custom top layers
- **Numeric Branch**: Multi-layer neural network
- **Combined Architecture**: Fusion of image and numeric features with skip connections
- **Training Stability**: Implements gradient clipping and batch normalization

## Key Features

- Dual-input processing (images + numeric data)
- Data augmentation pipeline
- Class weight balancing
- Advanced regularization techniques
- Learning rate scheduling
- Early stopping with best weights restoration

## Requirements

```
tensorflow>=2.0.0
numpy
scikit-learn
pillow
```

## Model Performance

Current model achieves:
- Accuracy: 65% (validation)
- Stable training dynamics
- No overfitting observed
- Room for further optimization

## Usage

### Data Preparation

```python
# Prepare your data in the following format:
X_image_train  # Image data (N, height, width, channels)
X_numeric_train  # Numeric features (N, num_features)
y_train  # Labels
```

### Training

```python
# Import required modules
from model import create_model

# Initialize and train the model
model = create_model(input_shape_image, input_shape_numeric)
history = model.fit(
    [X_image_train, X_numeric_train_norm],
    y_train,
    validation_data=([X_image_val, X_numeric_val_norm], y_val),
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    class_weight=class_weight_dict
)
```

## Training Configuration

- Batch Size: 32
- Initial Learning Rate: 1e-4
- Optimizer: Adam with gradient clipping
- Loss Function: Binary Cross-entropy

## Results

The model demonstrates:
- Consistent improvement in accuracy from 35% to 65%
- Stable loss curves
- Validation metrics outperforming training metrics
- Potential for further improvement with extended training

## Future Improvements

1. Extended training duration
2. Reduced regularization strength
3. Architecture modifications for better feature extraction
4. Experimentation with different backbone models

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

[MIT License](LICENSE)

## Contact

[Your Contact Information]

## Citation

If you use this code in your research, please cite:

```
@misc{multimodal-classification,
  author = {[Your Name]},
  title = {Multi-Modal Image Classification with EfficientNetB5},
  year = {2024},
  publisher = {GitHub},
  url = {[Your Repository URL]}
}
```