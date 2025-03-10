import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import entropy

# Define the EEG frequency bands
freq_bands = {
    'delta': (0.5, 4),    # Delta band: 0.5-4 Hz
    'theta': (4, 8),      # Theta band: 4-8 Hz
    'alpha': (8, 13),     # Alpha band: 8-13 Hz
    'beta': (13, 30),     # Beta band: 13-30 Hz
    'gamma': (30, 45)     # Gamma band: 30-45 Hz
}

def extract_eeg_features(patient_data, channel_names, sampling_rate=128):
    """
    Extract EEG features from a single patient's data
    
    Parameters:
    -----------
    patient_data : DataFrame
        EEG data for a single patient
    channel_names : list
        Names of EEG channels
    sampling_rate : int
        Sampling rate of the EEG data in Hz
        
    Returns:
    --------
    features : ndarray
        Extracted features vector
    """
    # Get only EEG channel data
    eeg_data = patient_data[channel_names].values
    
    # Initialize features list
    features = []
    
    # 1. Time domain features
    # Calculate statistical measures for each channel
    for i, channel in enumerate(channel_names):
        # Get channel data
        channel_data = eeg_data[:, i]
        
        # Statistical features
        features.append(np.mean(channel_data))           # Mean
        features.append(np.std(channel_data))            # Standard deviation
        features.append(np.max(channel_data))            # Maximum value
        features.append(np.min(channel_data))            # Minimum value
        features.append(np.median(channel_data))         # Median
        features.append(pd.Series(channel_data).skew())  # Skewness
        features.append(pd.Series(channel_data).kurtosis()) # Kurtosis
    
    # 2. Frequency domain features
    for i, channel in enumerate(channel_names):
        # Get channel data
        channel_data = eeg_data[:, i]
        
        # Compute power spectral density using Welch's method
        freqs, psd = signal.welch(channel_data, fs=sampling_rate, nperseg=min(256, len(channel_data)))
        
        # Calculate absolute power in each frequency band
        for band_name, (low_freq, high_freq) in freq_bands.items():
            idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
            band_power = np.sum(psd[idx_band])
            features.append(band_power)
            
        # Calculate relative power in each frequency band
        total_power = np.sum(psd)
        for band_name, (low_freq, high_freq) in freq_bands.items():
            idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
            band_power = np.sum(psd[idx_band])
            rel_band_power = band_power / total_power if total_power > 0 else 0
            features.append(rel_band_power)
            
        # Add band power ratios (commonly used in ADHD EEG analysis)
        # Theta/Beta ratio (typically higher in ADHD)
        theta_idx = np.logical_and(freqs >= 4, freqs <= 8)
        theta_power = np.sum(psd[theta_idx])
        
        beta_idx = np.logical_and(freqs >= 13, freqs <= 30)
        beta_power = np.sum(psd[beta_idx])
        
        theta_beta_ratio = theta_power / beta_power if beta_power > 0 else 0
        features.append(theta_beta_ratio)
    
    # 3. Entropy measures for each channel (complexity measure)
    for i in range(eeg_data.shape[1]):
        channel_data = eeg_data[:, i]
        # Shannon entropy
        hist, _ = np.histogram(channel_data, bins=20)
        hist = hist / np.sum(hist)
        channel_entropy = entropy(hist)
        features.append(channel_entropy)
    
    return np.array(features)

def train_adhd_classifier(file_path):
    """
    Train a general SVM classifier to detect ADHD from EEG data
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing EEG data
        
    Returns:
    --------
    model : sklearn Pipeline
        Trained SVM classifier
    """
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Extract EEG channel names
    channel_names = data.columns.tolist()
    channel_names.remove('Class')
    channel_names.remove('ID')
    
    # Get unique patient IDs
    patient_ids = data['ID'].unique()
    print(f"Number of unique patients: {len(patient_ids)}")
    print(f"Class distribution: {data['Class'].value_counts().to_dict()}")
    
    # Initialize feature matrix and labels
    X = []
    y = []
    
    # Extract features for each patient
    print("Extracting features for each patient...")
    for patient_id in patient_ids:
        patient_data = data[data['ID'] == patient_id]
        
        # Skip patients with very little data
        if len(patient_data) < 10:
            print(f"Skipping patient {patient_id} due to insufficient data")
            continue
        
        # Extract features
        features = extract_eeg_features(patient_data, channel_names)
        X.append(features)
        
        # Get label (assuming all rows for a patient have the same label)
        label = patient_data['Class'].iloc[0]
        label_numeric = 1 if label == 'ADHD' else 0
        y.append(label_numeric)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Handle any NaN or infinite values
    if np.isnan(X).any() or np.isinf(X).any():
        print("Warning: NaN or infinite values detected in features. Replacing with zeros.")
        X = np.nan_to_num(X)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Class distribution: ADHD={np.sum(y)}, Control={len(y)-np.sum(y)}")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create a pipeline with preprocessing and SVM classifier
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize features
        ('svm', SVC(kernel='rbf', probability=True, random_state=42))  # SVM with RBF kernel
    ])
    
    # Train the SVM model
    print("Training SVM model...")
    svm_pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = svm_pipeline.predict(X_test)
    y_pred_prob = svm_pipeline.predict_proba(X_test)[:, 1]
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Control', 'ADHD'])
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Print results
    print(f"SVM Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Control', 'ADHD'], 
                yticklabels=['Control', 'ADHD'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    return svm_pipeline

def predict_adhd(model, patient_data, channel_names):
    """
    Predict if a patient has ADHD based on their EEG data
    
    Parameters:
    -----------
    model : sklearn Pipeline
        Trained SVM classifier
    patient_data : DataFrame
        EEG data for a single patient
    channel_names : list
        Names of EEG channels
        
    Returns:
    --------
    prediction : int
        1 for ADHD, 0 for Control
    probability : float
        Probability of ADHD
    """
    # Extract features
    features = extract_eeg_features(patient_data, channel_names)
    features = features.reshape(1, -1)  # Reshape for single sample prediction
    
    # Make prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0, 1]  # Probability of ADHD
    
    return prediction, probability

# Main execution
if __name__ == "__main__":
    file_path = "/Users/optimus/.cache/kagglehub/datasets/danizo/eeg-dataset-for-adhd/versions/1/adhdata.csv"
    
    # Train the classifier
    model = train_adhd_classifier(file_path)
    
    # Load data for a test patient
    data = pd.read_csv(file_path)
    channel_names = data.columns.tolist()
    channel_names.remove('Class')
    channel_names.remove('ID')
    
    # Select a test patient
    test_id = 'v10p'  # Example ID, replace with actual ID
    test_patient = data[data.ID == test_id]
    
    # Make prediction
    actual_class = test_patient['Class'].iloc[0]
    pred, prob = predict_adhd(model, test_patient, channel_names)
    
    print(f"\nTest patient ID: {test_id}")
    print(f"Actual class: {actual_class}")
    print(f"Predicted class: {'ADHD' if pred == 1 else 'Control'}")
    print(f"Probability of ADHD: {prob:.4f}")