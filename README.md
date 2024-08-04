# Background-Remover-with-Handling-Different-Input-Types
---
# Background Remover with U2-Net

This project leverages the U2-Net model for background removal. It supports different input types: URL links, file uploads, and drag-and-drop images. You can use this tool to remove the background from images provided through various methods.

## Features

- **URL Input**: Remove background from images provided via a URL.
- **File Upload**: Upload image files to process and remove backgrounds.
- **Drag and Drop**: Drag and drop image files directly for background removal.

## Setup

### Prerequisites

- Python 3.x
- TensorFlow 2.x
- OpenCV
- Requests
- Matplotlib
- ipywidgets

You can install the required packages using:

```bash
pip install tensorflow opencv-python requests matplotlib ipywidgets
```

### Model Setup

Download the U2-Net model from Kaggle:

[U2-Net Model](https://www.kaggle.com/models/vaishaknair456/u2-net-portrait-background-remover/tensorFlow2/40_saved_model/1)

Place the model in your working directory.

## Usage

### Import Required Libraries

```python
import tensorflow_hub as hub
import tensorflow as tf
import cv2
import base64
import numpy as np
from urllib.request import Request, urlopen
import matplotlib.pyplot as plt
import time
from ipywidgets import FileUpload
import ipywidgets as widgets
from IPython.display import display, HTML
from google.colab import output as colab_output
```

### Load U2-Net Model

Load the U2-Net model for background removal:

```python
model = hub.KerasLayer("https://www.kaggle.com/models/vaishaknair456/u2-net-portrait-background-remover/tensorFlow2/40_saved_model/1")
```

### Background Removal Function

Define a function to process images and remove the background:

```python
def process_image(image):
    # Input image should be of height x width: 512 x 512 with 3 color channels:
    INPUT_IMG_HEIGHT = 512
    INPUT_IMG_WIDTH = 512
    INPUT_CHANNEL_COUNT = 3
    
    #image width and height
    h, w, channel_count = image.shape

    # Preprocess input image:
    if channel_count > INPUT_CHANNEL_COUNT:  # png images will have an alpha channel. Remove it:
        image = image[..., :INPUT_CHANNEL_COUNT]

    x = cv2.resize(image, (INPUT_IMG_WIDTH, INPUT_IMG_HEIGHT))  # Resize input image to 512 x 512 x 3
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)

    # Generate the pixel-wise probability mask:
    probability = model(x)[0].numpy()
    probability = cv2.resize(probability, dsize=(w, h))  # Resize the probability mask from (512, 512, 1) to (h, w)
    probability = np.expand_dims(probability, axis=-1)  # Reshape the probability mask from (h, w) to (h, w, 1)

    alpha_image = np.insert(image, 3, 255.0, axis=2)  # Add an opaque alpha channel to the input image

    PROBABILITY_THRESHOLD = 0.7  # Pixels with probability values less than or equal to the threshold belong to the background class.

    # Apply the probability mask by making pixels with probability value <= PROBABILITY_THRESHOLD transparent in the output image:
    masked_image = np.where(probability > PROBABILITY_THRESHOLD, alpha_image, 0.0)

    return masked_image.astype(np.uint8)
```

### Handling Different Inputs

#### File Upload & Display Output

Use the following function to handle file uploads:

```python
def handle_upload(change):
    for name, file in upload_widget.value.items():
        # Read the uploaded file
        image = np.asarray(bytearray(file['content']), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        start_time = time.time()

            # Process the image
        masked_image = process_image(image)

            # End timing the image processing
        end_time = time.time()
        processing_time = end_time - start_time
            # Display processing time
        print(f"Processing time: {processing_time:.2f} seconds")

        # Display input and output images using matplotlib
        plt.figure(figsize=(15, 7))

        # Display input image
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying with matplotlib
        plt.axis('off')
        plt.title('Input Image')

        # Display output image
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGRA2RGBA))  # Convert BGRA to RGBA
        plt.axis('off')
        plt.title('Processed Image Output')

        plt.show()
        print("File processed successfully!")
```

#### URL Input & Display Output

Use this function to handle image URLs:

```python
urls = [
    "https://w0.peakpx.com/wallpaper/287/473/HD-wallpaper-captain-america-mqvel.jpg",
    "https://e1.pxfuel.com/desktop-wallpaper/160/565/desktop-wallpaper-hinata-hyuga-aesthetic-hinata-thumbnail.jpg",
    "https://e0.pxfuel.com/wallpapers/881/1007/desktop-wallpaper-nami-anime-one-piece-manga-thumbnail.jpg"
]
for i, url in enumerate(urls):
    # Fetch and process the image
    image = get_image_from_url(url)
    start_time = time.time()
    # Process the image
    masked_image = process_image(image)
    # End timing the image processing
    end_time = time.time()
    processing_time = end_time - start_time
    # Display processing time
    print(f"Processing time: {processing_time:.2f} seconds")
    # Save the output
    output_filename = f"./output_{i+1}.png"
    cv2.imwrite(output_filename, masked_image)

    # Display input and output images using matplotlib
    plt.figure(figsize=(15, 7))

    # Display input image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying with matplotlib
    plt.axis('off')
    plt.title('Input Image')

    # Display output image
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGRA2RGBA))  # Convert BGRA to RGBA
    plt.axis('off')
    plt.title('Background_Removed Image Output')

    plt.show()
    print("File processed successfully!")
```

#### Drag and Drop

Here’s the HTML and JavaScript to handle drag-and-drop functionality:

```python
status_label = widgets.Label("Drag and drop an image file here")
output_widget = widgets.Output()
def handle_file(file):
    with output_widget:
        output_widget.clear_output()
        # Read the uploaded image
        img_bytes = file.split(",")[1]
        img_array = np.frombuffer(bytearray(base64.b64decode(img_bytes)), dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        start_time = time.time()
        # Process the image to remove the background
        processed_image = process_image(image)
        # End timing the image processing
        end_time = time.time()
        processing_time = end_time - start_time
            # Display processing time
        print(f"Processing time: {processing_time:.2f} seconds")

        # Display input and output images
        fig, axs = plt.subplots(1, 2, figsize=(15, 7))
        
        # Display input image
        axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[0].axis('off')
        axs[0].set_title('Input Image')
        
        # Display processed (background removed) image
        axs[1].imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGRA2RGBA))
        axs[1].axis('off')
        axs[1].set_title('Processed Image Output')
        
        plt.show()
    status_label.value = "File processed successfully!"

# HTML and JavaScript for drag-and-drop interface
html_code = """
<div id="drop_zone" style="border: 2px dashed lightgray; width: 50%; padding: 20px; text-align: center;">
  Drag and drop an image file here
</div>
<script>
  var dropZone = document.getElementById('drop_zone');
  dropZone.addEventListener('dragover', function(event) {
    event.preventDefault();
    event.stopPropagation();
    dropZone.style.borderColor = 'blue';
  }, false);
  dropZone.addEventListener('dragleave', function(event) {
    event.preventDefault();
    event.stopPropagation();
    dropZone.style.borderColor = 'lightgray';
  }, false);
  dropZone.addEventListener('drop', function(event) {
    event.preventDefault();
    event.stopPropagation();
    dropZone.style.borderColor = 'lightgray';
    var file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      var reader = new FileReader();
      reader.onload = function(e) {
        var dataUrl = e.target.result;
        google.colab.kernel.invokeFunction('notebook.handle_file', [dataUrl], {});
      };
      reader.readAsDataURL(file);
    } else {
      alert('Please drop an image file.');
    }
  }, false);
</script>
"""
```

### Register Callbacks 

Register the functions to Drag and Drop the image:

```python
colab_output.register_callback('notebook.handle_file', handle_file)
```

### Display the Interface

```python
display(HTML(html_code))
display(status_label)
display(output_widget)
```

## Troubleshooting

- **Multiple Inputs**: We may run multiple input method is active at a time, But ensure Handling multiple drag-and-drop inputs may require adjustments to the interface and backend processing.
- **Model Path**: Verify that the path to the U2-Net model is correct.
- **Dependencies**: Ensure all dependencies are installed and properly configured.
---

## Model Details
- Model link="https://www.kaggle.com/models/vaishaknair456/u2-net-portrait-background-remover/tensorFlow2/40_saved_model/1"
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
