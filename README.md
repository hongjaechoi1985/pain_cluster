# pain_cluster

# Clustering Cardiac Surgery Patients Based on Multi-dimensional Pain Information

This repository contains the analysis code for the study: **"Clustering Cardiac Surgery Patients Based on Multi-dimensional Pain Information to Predict Clinical Outcomes."**

## 📌 Project Overview
The goal of this research is to identify distinct pain trajectories or clusters in patients undergoing cardiac surgery using nursing records and machine learning techniques. By clustering multi-dimensional pain information, we aim to bridge the gap between clinical guidelines and real-world treatment outcomes prediction.

## 📊 Data Availability
The data used in this study were derived from clinical nursing records. Due to Institutional Review Board (IRB) restrictions and the sensitive nature of patient health information, the raw dataset is not publicly available.

- **Data Access:** Researchers interested in the data extraction protocol or variable definitions may contact the corresponding author (H-JC) upon reasonable request.
- **Protocol:** The inclusion/exclusion criteria and operational definitions of pain clusters are documented within the analysis scripts.

## 🛠 Environment & Requirements
The analysis was performed using Python 3.12.12 

### Key Libraries:
- **Data Analysis:** `pandas`, `numpy`
- **Machine Learning & Clustering:** `scikit-learn`, `scipy` (for hierarchical clustering or K-means)
- **Visualization:** `matplotlib`, `seaborn`
- **Statistical Analysis:** `statsmodels`

Install dependencies via pip:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib statsmodels
