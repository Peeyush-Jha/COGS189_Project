# ADHD EEG Classification Project

This project aims to classify ADHD vs. Control subjects using EEG data and machine learning techniques. The approach extracts spectral power features from EEG signals and uses Support Vector Machines (SVM) for classification.

## Project Structure

- `adhd_eeg_classification.ipynb`: Main Python script with step-by-step implementation
- `requirements.txt`: List of required Python packages
- `ADHD_data/`: Directory containing EEG data files
  - `ADHD_part1/`, `ADHD_part2/`: Raw EEG data from ADHD subjects
  - `Control_part1/`, `Control_part2/`: Raw EEG data from Control subjects
  - `*_preprocessed/`: Directories containing preprocessed EEG data
  - `Channel_Labels.docx`: Information about EEG channel labels
  - `Standard-10-20-Cap19new.ced`: EEG cap configuration file

## Setup Instructions

1. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Ensure the data directory structure is correct:
   - The script expects preprocessed EEG data in the `ADHD_data` directory
   - Each `.mat` file should contain a variable named `preprocessedData` with shape (channels, time_points)
   - Each `.mat` file should also contain a variable named `fs` with the sampling frequency



## Feature Extraction

The script extracts the following features:
- Spectral power in delta (0.5-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), and beta (13-30 Hz) bands for each EEG channel
- Theta/Beta ratio for frontal channels, which is a common biomarker for ADHD

## Model Training

The classification model uses:
- StandardScaler for feature normalization
- Support Vector Machine (SVM) for classification
- GridSearchCV for hyperparameter tuning

## Results Analysis

The script provides:
- Classification accuracy on the test set
- Confusion matrix
- Detailed classification report (precision, recall, F1-score)
- Feature importance analysis (for linear kernel)

## Next Steps


