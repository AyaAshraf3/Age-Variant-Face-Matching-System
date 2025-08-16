# Age-Variant Face Matching System

This project presents a robust system for verifying if two images of a person, taken at significantly different ages, belong to the same individual. It leverages a novel hybrid approach that combines a custom-trained deep learning model for age estimation with the FaceNet model for verification, all within a modern Keras 3 environment.

## Features
- **Keras 3 Age Estimation:** A ResNet50-based regression model trained to predict a person's age from a facial image.
- **FaceNet Verification:** Utilizes the powerful `keras-facenet` library for high-accuracy face matching.
- **MediaPipe Detection:** Employs Google's fast and lightweight MediaPipe library for initial face detection, ensuring compatibility with the Keras 3 ecosystem.
- **Dynamic Thresholding:** The system's core innovation. It uses the predicted age difference between the two images to dynamically adjust the similarity threshold, making the verification process more intelligent and context-aware.

## System Architecture

The verification pipeline operates as follows:

1.  **Input:** Two images (Image A, Image B) are provided to the system.
2.  **Face Detection & Cropping:** The `MediaPipe` library detects and extracts aligned face crops from both images.
3.  **Age Estimation:** The custom-trained Keras 3 age model predicts the age for each cropped face.
4.  **Face Verification:** The `keras-facenet` library generates a 512-dimensional embedding for each face. The cosine distance between these embeddings is calculated to determine similarity.
5.  **Dynamic Logic:**
    * The system calculates the absolute age difference from the predictions.
    * Based on this difference, it sets a dynamic similarity threshold (a larger age gap results in a more lenient threshold).
    * The final decision ("Same Person" or "Different People") is made by comparing the cosine distance to this dynamic threshold.
6.  **Output:** The system presents the two images, their predicted ages, and the final verification result with a clear title.


## Deliverables in this Repository
- **age_invariant_verification.py:** The main executable Python script for the final system.

- **AgeEstimation.ipynb:** The Jupyter Notebook containing the complete end-to-end process. In this notebook, you can find the training code and architecture for the age estimation model, inference code on a batch of images from the test set, and the development of the whole system pipeline.

- **Age-Variant Face Matching System Report.pdf:** A detailed report on the project's methodology, architecture, and performance.

- **README.md:** This file.

- **requirements.txt:** A list of required packages for installation.

## Setup and Installation

### Prerequisites
- Python 3.8+
- `pip` package installer

### Installation
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/AyaAshraf3/Age-Variant-Face-Matching-System.git
    cd Age-Variant-Face-Matching-System
    ```

2.  **Install Dependencies:**
    Install all the required Python packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

3. **Download The Age Estimation Model:**
    Download the age estimation model in `.keras` format from this drive url :
    https://drive.google.com/file/d/16QL8aPsKRTCHZFJqgfopptngWarm43yB/view?usp=sharing
    and put it in your project folder path

## How to Run

### 1. Prepare Your Age Model
This system requires your custom-trained age estimation model, saved in the modern `.keras` format.
- `model_epoch_20_valLoss_2.2699.keras` (or your chosen checkpoint)

You can generate this file by running the provided `AgeEstimation.ipynb` notebook. Place this file in a known directory.

### 2. Run the Verification Script
Execute the `age_invariant_verification.py` script from your terminal, providing paths to the two images and your age model file.

```bash
python age_invariant_verification.py \
    --image1 "/path/to/person_young.jpg" \
    --image2 "/path/to/person_old.jpg" \
    --age_model "/path/to/your_age_model.keras"
```
The script will process the images and display the final result plot.

### NOTE:
Instead of running the Verification Script, you can also run the last cell in `Age-Variant Face Matching System.ipynb` after adding the Path for the age estimation model and two images.