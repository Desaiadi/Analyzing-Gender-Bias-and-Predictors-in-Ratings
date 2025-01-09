# Assessing Professor Effectiveness (APE)

## Project Overview
This project investigates gender bias in professor ratings and identifies key predictors of teaching effectiveness using data from RateMyProfessor.com (RMP). We analyze disparities in ratings, examine descriptive tags, and build predictive models to uncover patterns in evaluations. The findings aim to improve fairness in academia and provide actionable insights for educators and institutions.

---

## Data Collection
The dataset was created using the following steps:
1. **Web Scraping**:
   - Scraped professor profiles, ratings, tags, and institutional data from RMP using Python libraries like BeautifulSoup and Selenium.
2. **Data Collation**:
   - Aggregated individual ratings into average scores (e.g., quality, difficulty) and cleaned duplicates.
3. **Anonymization**:
   - Removed identifying information such as professor names to preserve privacy.
4. **Validation**:
   - Cross-checked with a manually collected sample, achieving a 98% match rate for accuracy.

---

## Dataset Description 


## Preprocessing
### Steps and Rationale
- **Outlier Removal**: 
  - Filtered professors with fewer than 5 ratings to prevent skewed averages.
- **Handling Missing Data**:
  - Used mean imputation for numerical features and excluded rows missing critical data.
- **Tag Normalization**:
  - Converted raw tag counts to proportions to ensure fair comparisons.
- **Gender Classification**:
  - Created binary columns to handle ambiguous gender entries.
- **Feature Scaling**:
  - Standardized numerical features for consistent model performance.

---

## Objectives
1. **Gender Bias Analysis**:
   - Investigate disparities in ratings, variability, and descriptive tags by gender.
2. **Predictive Modeling**:
   - Identify predictors of average ratings and difficulty.
   - Build models to predict "pepper" badges.
3. **Exploratory Insights**:
   - Link qualitative data (e.g., university, major) with numerical and tag metrics.

---

## Methods
### Statistical Analysis
- Mann-Whitney U Test, Welch’s T-Test, and Levene’s Test were used to compare distributions, means, and variances.
- Confidence intervals (95%) calculated for rating effects.

### Regression Models
1. **Numerical Predictors**:
   - \( R^2 = 0.48 \), RMSE = 0.68.
   - Difficulty ratings negatively correlated with average ratings.
2. **Tag Predictors**:
   - \( R^2 = 0.72 \), RMSE = 0.50.
   - "Tough Grader" strongly influenced ratings negatively.

### Classification Model
- Logistic regression with SMOTE for class imbalance:
  - AUROC: 0.78, Accuracy: 71%.
  - Influential predictors: "Amazing Lectures," "Good Feedback."

---

## Results
### Gender Bias in Ratings
- Male professors had slightly higher average ratings than females:
  - Male: 4.2, Female: 4.1 (1.4% higher).
- Female professors showed higher variability in ratings.

### Key Predictors of Ratings
- "Tough Grader" and "Inspirational" were significant predictors.
- Higher difficulty ratings correlated with lower average ratings.

### Exploratory Insights
- Aerospace Engineering professors had the highest difficulty (~5.0) but lowest ratings (~1.8).
- Office Technology professors achieved perfect average ratings (~5.0).

---

## Deliverables
1. **Report**:
   - Detailed PDF summarizing statistical tests, models, and visualizations.
2. **Code**:
   - Python scripts for data collection, preprocessing, analysis, and modeling.

---

## Conclusion
This project reveals subtle gender biases in ratings and highlights key predictors of teaching effectiveness. The results can guide educators and institutions toward fairer evaluation practices and improved teaching strategies.

---

## References
- Benjamin, D. J., et al. (2018). *Redefine statistical significance*. Nature Human Behaviour.
- Centra, J. A., & Gaubatz, N. B. (2000). *Is there gender bias in student evaluations of teaching?* Journal of Higher Education.
- MacNell, L., et al. (2015). *What’s in a name: Exposing gender bias in student ratings*. Innovative Higher Education.