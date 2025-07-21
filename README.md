MSCS_634_Lab_2: KNN and RNN Classification Analysis


Purpose of Lab Work

This lab explores the performance characteristics of two neighbor-based classification algorithms:

1. K-Nearest Neighbors (KNN) - A instance-based learning algorithm that classifies data points based on the majority class of their k nearest neighbors
2. Radius Neighbors (RNN) - A variant that considers all neighbors within a fixed radius for classification

Objectives:
- Compare classification performance across different parameter values
- Analyze the impact of hyperparameter tuning on model accuracy
- Understand when each algorithm is most effective
- Develop skills in model evaluation and parameter optimization
- Gain practical experience with sklearn's neighbor-based classifiers

Technical Implementation:
- Dataset: Wine Dataset (178 samples, 13 features, 3 classes)
- Train/Test Split: 80% training, 20% testing (stratified)
- KNN Parameters: k ∈ {1, 5, 11, 15, 21}
- RNN Parameters: radius ∈ {350, 400, 450, 500, 550, 600}
- Evaluation Metrics: Accuracy, Classification Report, Confusion Matrix

Key Insights and Observations

KNN Performance Trends:
- Optimal k-value: k=5 achieved the highest accuracy (80.56%)
- Low k-values (k=1): Showed overfitting with 77.78% accuracy, sensitive to noise
- High k-values (k=5-21): Interestingly showed stable performance at 80.56% across k=5,11,15,21
- Sweet spot: k=5-21 range provided consistently strong performance (80.56%)
- Bias-Variance Trade-off: Clear demonstration showing k=1 underfitting, with optimal balance at k≥5

RNN Performance Analysis:
- Performance Range: RNN worked successfully but with decreasing performance as radius increased
- Best Performance: radius=350 achieved 72.22% accuracy
- Performance Degradation: Accuracy declined from 72.22% (r=350) to 66.67% (r=600)
- Root Cause: Larger radius values include more distant, less relevant neighbors
- Learning Insight: RNN requires careful radius selection, with smaller radii performing better for this dataset

Dataset Characteristics:
- High Separability: Wine dataset classes are well-separated in feature space (80.56% KNN accuracy)
- Feature Scale Variation: Different features have varying numerical ranges affecting distance calculations
- Class Balance: Relatively balanced classes (Class 0: 59, Class 1: 71, Class 2: 48)
- Classification Difficulty: Class 2 showed lower performance in both algorithms (67% f1-score KNN, 18% f1-score RNN)

Comparative Analysis:
- KNN Robustness: More consistent performance across parameter range (80.56% for k≥5)
- RNN Performance: Successfully implemented but achieved lower accuracy (72.22% at best)
- Performance Gap: KNN outperformed RNN by ~8.3 percentage points
- Practical Preference: KNN demonstrated superior accuracy and consistency for this dataset

Challenges Faced and Design Decisions

1. RNN Parameter Selection and Performance
Observation: RNN successfully executed with radius values 350-600 but showed declining performance  
Decision: Systematic testing revealed optimal radius=350 with 72.22% accuracy  
Performance Pattern: Larger radius values (400-600) included too many distant neighbors, reducing accuracy  
Learning: RNN requires careful radius tuning, with smaller radii often performing better

2. Data Preprocessing Strategy
Decision: Used raw features without standardization  
Rationale: To observe algorithm behavior under real-world conditions  
Impact: Demonstrated the importance of preprocessing for distance-based algorithms  
Future Improvement: Implement standardization for fair RNN comparison

3. Visualization Design
Challenge: Presenting failed RNN results meaningfully  
Solution: Created informative error visualizations with explanatory text  
Benefit: Educational value in showing algorithm limitations  
Design Choice: Maintained consistent plot structure for easy comparison

4. Parameter Range Selection
KNN k-values: Chose odd numbers to avoid tie-breaking issues  
RNN radius: Initially followed assignment suggestions, adapted based on results  
Validation Strategy: Comprehensive testing across reasonable parameter ranges

5. Error Handling Implementation
Approach: Try-catch blocks for RNN predictions  
Benefit: Robust code that handles algorithm failures gracefully  
Documentation: Clear error messages explaining why failures occurred

Technical Implementation Details

Libraries Used:
- sklearn: Machine learning algorithms and datasets
- pandas: Data manipulation and analysis
- matplotlib/seaborn: Data visualization
- numpy: Numerical computations

Code Structure:
1. Data Loading & Exploration - Comprehensive dataset analysis
2. Model Implementation - Systematic parameter testing
3. Performance Evaluation - Multiple metrics and visualizations
4. Comparative Analysis - Side-by-side algorithm comparison

Reproducibility:
- Fixed random state (RANDOM_STATE = 42)
- Stratified train-test split
- Documented parameter choices
- Comprehensive commenting

Results Summary

| Algorithm | Best Parameters | Accuracy | Performance Notes |
|-----------|-----------------|----------|-------------------|
| KNN | k = 5 | 80.56% | Consistent performance across k≥5 |
| RNN | radius = 350 | 72.22% | Declining performance with larger radius |

Key Findings:
- KNN achieved 80.56% accuracy outperforming RNN's 72.22%
- Parameter tuning impact varies by algorithm - KNN stable across k≥5, RNN sensitive to radius
- Radius selection critical for RNN - smaller radii (350) performed better than larger ones (600)
- Algorithm selection significantly impacts results - 8.3% accuracy difference between methods

Future Improvements

1. Feature Standardization: Implement z-score normalization to potentially improve RNN performance
2. Extended RNN Parameter Search: Test smaller radius values (50-300 range) for potentially better RNN results  
3. Cross-Validation: Implement k-fold CV for more robust performance estimates
4. Feature Selection: Analyze impact of dimensionality reduction on neighbor-based methods
5. Distance Metrics: Experiment with Manhattan, Minkowski distances to improve classification
6. Class-Specific Analysis: Investigate why Class 2 showed lower performance in both algorithms

Repository Structure

```
MSCS_634_Lab_2/
├── README.md                    # This file
├── KNN_RNN_Analysis.ipynb      # Main Jupyter notebook with complete analysis

```

Setup and Execution

Prerequisites:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

Running the Analysis:
1. Clone this repository
2. Open KNN_RNN_Analysis.ipynb in Jupyter
3. Run all cells sequentially
4. Review generated visualizations and analysis
