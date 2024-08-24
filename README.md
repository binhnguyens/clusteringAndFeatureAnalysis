# clusteringAndFeatureAnalysis

# Paper title: Clustering and Feature Analysis of Smartphone Data for Depression Monitoring

https://ieeexplore.ieee.org/document/9629737

# Abstract
Modern advancements have allowed society to be at the most innovative stages of technology which involves the possibility of multimodal data collection. Dartmouth dataset is a rich dataset collected over 10 weeks from 60 participants. The dataset includes different types of data but this paper focuses on 10 different smartphone sensor data and a Patient Health Questionnaire (PHQ) 9 survey that monitors the severity of depression. This paper extracts key features from smartphone data to identify depression. A multi-view bi-clustering (MVBC) algorithm is applied to categorize homogeneous behaviour subgroups. MVBC takes multiple views of sensing data as input. The algorithm inputs three views: average, trend, and location views. MVBC categorizes the subjects to low, medium and high PHQ-9 scores. Real-world data collection may have fewer sensors, allowing for less features to be extracted. This creates a focus on prioritization of features. In this body of work, minimum redundancy maximum relevance (mRMR) is applied to the sensing features to prioritize the features that better distinguish the different groups. The resulting MVBC are compared to literature to support the categorized clusters. Decision Tree (DT) 10-fold cross validation shows that our method can classify individuals into the correct subgroups using a reduced number of features to achieve an overall accuracy of 94.7±1.62%. Achieving high accuracies with reduced features allows for focus on low power analysis and edge computing applications for long-term mental health monitoring using a smartphone.

# Tools used
Python, Machine learning, Clustering (supervised and unsupervised), multi view biclustering, Feature ranking (mRMR)

# Other
Pandas, Numpy, SciKitLearn, wavelet 
