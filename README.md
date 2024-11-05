# AI Image Detection: Real vs AI-Generated Image Classification

## Overview
This project focuses on developing a machine learning model to distinguish between real and AI-generated images. Using the CASIA dataset from Kaggle, which includes a variety of real and manipulated images, we aim to build a robust AI model capable of identifying AI-generated content. 

To enhance the detection of subtle differences in image compression and manipulation, we apply **Error Level Analysis (ELA)** as a preprocessing technique. This approach helps expose the inconsistencies often present in AI-generated or altered images. The project further explores data augmentation and transfer learning to improve model performance in the face of limited computational resources and dataset constraints.

## Objectives
- **Primary Objective**: Develop a model to accurately classify real and AI-generated images.
- **Secondary Objectives**:
  - Utilize ELA preprocessing to highlight subtle image manipulations.
  - Incorporate data augmentation to enhance the model's generalizability.
  - Use transfer learning with pre-trained models to boost performance with limited data.

## Features
- **Dataset**: [CASIA dataset on Kaggle](https://www.kaggle.com/datasets/sophatvathana/casia-dataset), containing real and manipulated (AI-generated) images.
- **Techniques Used**:
  - **Error Level Analysis (ELA)** to highlight areas of the image with compression differences.
  - **Image Augmentation** (height/width shifts, flipping) to increase dataset diversity.
  - **Transfer Learning** with models such as ResNet-50V3 to leverage pre-trained weights.

## Model Architecture
- **Base Model**: Convolutional Neural Network with:
  - Two convolutional layers with ReLU activation and MaxPooling.
  - Dropout layers to reduce overfitting.
  - Fully connected layers for classification.
- **Transfer Learning (Optional)**: Option to use `ResNet50V3` from TensorFlow or `ResNet-50` from PyTorch to leverage pre-trained weights.

## Performance Metrics
After training, the model's performance is evaluated using:
- **Accuracy**: Classification accuracy on test and validation sets.
- **Loss**: Binary cross-entropy loss for measuring classification performance.
- **Confusion Matrix**: Visualization of true vs. predicted classifications.

## Visualization
- **Error Level Analysis**: Visualizations of original vs. ELA images.
- **Data Augmentation**: Examples of augmented images, including height/width shifts and flips.
- **Model Training History**: Loss and accuracy plots to monitor training and validation performance over epochs.

## Key Findings
1. **Model Limitations**: Although effective on many samples, the model struggles with some AI-generated images, especially from advanced generators like MidJourney and Stable Diffusion.
2. **Importance of ELA**: ELA preprocessing significantly improves the model’s ability to differentiate real vs. AI images by highlighting manipulated regions.
3. **Resource Constraints**: Limited computational resources and dataset size impacted performance; expanding the dataset and using more powerful resources could further improve accuracy.

## Future Work
- **Enhanced Dataset**: Using larger, more diverse datasets for improved generalization.
- **Advanced Architectures**: Testing advanced architectures such as Vision Transformers.
- **Synthetic Image Sources**: Expanding to include images from newer AI generation models to improve robustness.

## License
This project is open-source and available for use and modification under the MIT License.
