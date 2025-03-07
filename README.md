# COGS189 ADHD EEG Classification Project

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
   conda create -n nameyouchoose python=3.9
   conda activate nameyouchoose
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


## References

1. **Spectral Power Features**  
   - *Duan, L., Liu, Q., Zhao, X., Li, P., Huang, Y., & Dai, W. (2020).* Investigation of EEG power spectral slope in medication-naive children with ADHD. *Frontiers in Psychiatry*, 10, 926.  
     [PMCID: PMC6966317](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6966317/)

2. **Hjorth Parameters**  
   - *Azami, H., et al. (2023).* Time-domain features including Hjorth parameters for discrimination between ADHD patients and healthy controls. In *EUSIPCO 2023 Proceedings*, 1065–1069.  
     [Link](https://eurasip.org/Proceedings/Eusipco/Eusipco2023/pdfs/0001065.pdf)

3. **Sample Entropy**  
   - *Meng, L., et al. (2021).* Comparison of different entropy estimators for ADHD and control EEG signals. *Biomedical Signal Processing and Control*, 69, 102890.  
     [PMID: 34424101](https://pubmed.ncbi.nlm.nih.gov/34424101/)

4. **Connectivity Measures**  
   - *Tomescu, M. I., et al. (2022).* EEG microstate dynamics and frequency features in ADHD. *European Child & Adolescent Psychiatry*, 31, 689–701.  
     [DOI: 10.1007/s00787-022-02068-6](https://link.springer.com/article/10.1007/s00787-022-02068-6)

5. **Increased Alpha Power and Steeper Spectral Slopes in Medication-Naive Children with ADHD**  
   [PMC6966317](https://pmc.ncbi.nlm.nih.gov/articles/PMC6966317/)  
   This study investigated EEG power spectral slope in children with ADHD, finding that medication-naive children exhibited higher alpha power and steeper spectral slopes. Their findings support the use of frequency-specific power as a potential biomarker for ADHD.



---