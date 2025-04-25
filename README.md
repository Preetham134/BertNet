# Student Graduation Outcome Prediction

This project implements and compares Tabnet and Vanilla Transformer models to predict student graduation outcomes using both static and sequential educational data. The Transformer-based approach achieved 57% accuracy compared to Tabnet's 42%.

## Overview

Educational institutions are increasingly seeking data-driven methods to identify at-risk students and improve graduation rates. This project leverages modern deep learning approaches to analyze both static student information (demographics, entry qualifications, etc.) and sequential data (semester-by-semester performance, attendance patterns, etc.) to predict graduation outcomes and identify key influencing factors.

## Features

- Integration of static and sequential educational data
- Implementation of Tabnet and Vanilla Transformer models
- Explainable AI analysis using SHAP and LIME frameworks
- Identification of 5 critical success predictors
- Visualization tools for model interpretation

## Installation

```bash
git clone https://github.com/yourusername/student-graduation-prediction.git
cd student-graduation-prediction
```

## Data

The dataset used in this project includes:

- **Static data**: Student demographics, high school GPA, standardized test scores, socioeconomic indicators
- **Sequential data**: Semester-by-semester GPA, course load, attendance records, participation metrics

## Models

### Tabnet

TabNet is a deep learning architecture designed specifically for tabular data, allowing for interpretable decision-making by learning which features to use at each decision step.

### Vanilla Transformer

Our custom implementation of a Transformer model processes both static features and the sequential educational timeline data, capturing temporal patterns in student performance.


## Results

- **Transformer model**: 57% accuracy
- **Tabnet model**: 42% accuracy
