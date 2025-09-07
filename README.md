1. Polars vs Pandas

Polars was chosen as the first option for reading CSV files because:
It is faster and memory-efficient than Pandas, especially for large datasets.
Provides lazy evaluation for performance.
Pandas can be a bottleneck for large datasets as it processes data sequentially.

Fallback to Pandas:
Some downstream libraries (like ydata-profiling) work best with Pandas DataFrames.
Therefore, data is converted (.to_pandas()) when advanced operations are needed.
Pandas has a large community, extensive documentation, and a mature ecosystem with many integrations.

2. Data Profiling Integration
   
ydata-profiling was used to:
ydata-profiling is a package for data profiling that automates and standardises the generation of detailed reports, complete with statistics and visualisations
ydata-profiling streamlines EDA (Exploratory Data Analysis), provides comprehensive insights automatically and enhances data quality
Summarise distributions, missing values, correlations, and warnings.
Integration with Streamlit was done using streamlit-pandas-profiling, which allows the profiling report to be displayed inside the Streamlit app.
Profiling report can also be downloaded as HTML for offline use.

3. ML Model Use-Case

A basic supervised ML module is included:
Classification (Logistic Regression) if the target column is categorical (object dtype or <10 unique values).
Regression (Linear Regression) if the target column is numeric with more unique values.

Example datasets where it works well:

Classification: Logistic Regression.
It classifies data points into discrete categories by estimating the probability of an event occurring.
Example: Predicting whether a customer will click on an ad (yes/no).
Output: A probability value between 0 and 1, which is then used to classify the outcome.
Mathematical Model: Uses a sigmoid (S-shaped) curve to transform a linear equation into a probability, ensuring the output stays within the 0 to 1 range
It utilises the sigmoid function and a threshold value to categorise data. 
 
Regression: Linear Regression
It predicts a continuous numerical value.
Example: Predicting house prices based on size, number of bedrooms, and location.
Output: A continuous numerical value on a scale
Mathematical Model: Uses a linear equation to model the relationship between variables, like Y = a + bX
It creates a "best-fit" straight line through the data.

The trained model can:
Show evaluation metrics: Accuracy (classification) or MSE (regression).
Accept user input values for features and make live predictions.

4. Performance Tips (Large Files & Optimisation)

Polars is preferred for reading and initial cleaning because it is multi-threaded and much faster than Pandas.
Caching with st.cache_data:
Profiling report generation is cached to avoid recomputation when data doesn’t change.

Lazy operations:
Polars supports lazy execution, which minimises memory overhead for very large CSVs.
