# Background-Remover-with-Handling-Different-Input-Types
## Model Details
### Model Summary
Performs salient object detection using U2-Net architecture to remove background from portrait pictures.
Trained on Privacy-Preserving Portrait Matting Dataset.
### Usage
- Input: Color image of shape (512, 512, 3). Color channels should be in the order BGR (the default ordering provided by the OpenCV library).
- Output: Probability map of shape (512, 512, 1) where the last dimension represents the probability that a pixel belongs to the foreground.
### System
- Standalone model
- Input image should be resized to 512 x 512 pixels with 3 color channels. Each pixel value should then be rescaled by dividing them by 255.0

# Model was trained from scratch.
![Model_page-0001](https://github.com/user-attachments/assets/cae110c7-c217-45ea-b525-e7b2759d57c0)
