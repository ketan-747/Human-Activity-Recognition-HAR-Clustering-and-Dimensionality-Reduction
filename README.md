# Human-Activity-Recognition-HAR-Clustering-and-Dimensionality-Reduction 🚶‍♀️📊
This repository explores clustering and dimensionality reduction techniques on the UCI Human Activity Recognition (HAR) dataset. The goal is to identify distinct activity groups within the sensor data and visualize the results in a reduced-dimensional space.

## Dataset 📋
The UCI Human Activity Recognition dataset contains measurements from smartphone sensors captured during various activities. The raw time-series data has been pre-processed, resulting in 561 features that describe different aspects of the sensor dynamics. The training set, which is the focus of this analysis, comprises 7352 individual samples. There are 6 distinct activity classes in the dataset.

## Project Overview 🗺️

This project demonstrates a typical machine learning workflow involving unsupervised learning (clustering) and dimensionality reduction which aims to:

#### 1: Pre-process the high-dimensional sensor data.
#### 2: Determine an appropriate number of clusters for the data using the Elbow Method.
#### 3: Apply K-Means clustering to group similar activities.
#### 4: Evaluate the quality of the clustering using a suitable metric.
#### 5: Reduce the dimensionality of the data using Principal Component Analysis (PCA) for visualization purposes.
#### 6: Visualize the clustered data and compare it with the ground truth labels in 2D and 3D plots.

## Methodology 🔬

### 1: Data Loading and Initial Exploration
The dataset UCI_HAR.npz is loaded, containing the training features (x_train) and corresponding labels (y_train).
The training set consists of 7352 samples, each with 561 features. There are 6 unique classes in the dataset.


### 2: Data Standardization
StandardScaler from sklearn.preprocessing is used to standardize the x_train data. This is an important step for K-Means clustering as it is sensitive to the scale of the features.


### 3: Determining Optimal Clusters (K-Means Elbow Method)
The Elbow Method is employed to find an appropriate number of clusters (K) for K-Means. K-Means models are trained for k values ranging from 1 to 10, and the inertia (sum of squared distances of samples to their closest cluster center) is calculated for each k. The elbow curve is then plotted to visually identify the optimal K.
Based on the plot, a value of K=5 was chosen as a good balance, as the decrease in inertia starts to slow down significantly after this point.

### 4: K-Means Clustering
K-Means clustering is applied to the standardized training data (x_train_scaled) with n_clusters=5. n_init is set to 10 and random_state to 42 for reproducibility. The fit_predict method is used to obtain the cluster labels for each sample.

### 5: Cluster Quality Analysis (Cluster Purity)
- To assess the quality of the clustering, the cluster purity metric is calculated. The cluster_purity function computes a contingency matrix between the true labels (y_train) and the predicted cluster labels (cluster_labels), then calculates purity as the sum of the maximum values in each column (cluster) of the contingency matrix, divided by the total number of samples.


- The calculated cluster purity is approximately 0.43. While a value closer to 1 indicates better quality, this result, combined with the elbow method's suggestion, indicates that K=5 is a reasonable choice. Cluster purity was chosen for its ease of interpretation and its direct reliance on true and clustered labels.

### 6: Dimensionality Reduction (PCA)

- Principal Component Analysis (PCA) is employed to reduce the 561 features of the dataset down to 3 components. This reduction is crucial for visualizing the high-dimensional data. PCA is chosen for its computational efficiency and effectiveness in retaining data variability in a reduced space.
- The transformed data, x_train_pca, will have a shape of (7352, 3).

### 7: Visualization of Results
- The project includes both 2D and 3D visualizations of the clustered data and the ground truth labels in the PCA-reduced space.

- 2D Plots: Scatter plots are generated to show the relationships between Feature 1 vs Feature 2, Feature 2 vs Feature 3, and Feature 1 vs Feature 3 for both the K-Means clusters and the ground truth labels.
- 3D Plots: A 3D scatter plot is generated for the K-Means clusters using Feature 1, Feature 2, and Feature 3. An animation function update_rotation is included to allow for interactive viewing of the 3D plot.


## Results 🎯

- The visualizations show that while the clusters formed by K-Means with K=5 are sometimes close to each other, they generally exhibit distinct groups with minimal overlap. 
- This contrasts with the ground truth labels, which, despite having 6 unique classes, show more considerable overlap in the reduced 3D space. Visually, using 5 clusters appears to provide a clearer separation than attempting to perfectly adhere to the 6 ground truth labels.
- Although increasing the number of clusters might improve the purity score, it could also lead to overfitting and make it harder for the algorithm to differentiate between clusters.


## Conclusion ✅
- The analysis demonstrates the application of K-Means clustering and PCA for understanding and visualizing high-dimensional sensor data from human activities.
- Despite a moderate cluster purity score, the visual analysis of the 2D and 3D plots suggests that clustering with 5 clusters provides a meaningful and interpretable grouping of the activities in the reduced PCA space.

## Dependencies 🧩
The code in this repository requires the following Python libraries:
- numpy 
- matplotlib 
- scikit-learn (specifically KMeans, StandardScaler, adjusted_rand_score, normalized_mutual_info_score, PCA, contingency_matrix)


## Contact 🤝
- Name : Ketan Kulkarni 
- LinkedIn: https://www.linkedin.com/in/ketan-b-kulkarni/

