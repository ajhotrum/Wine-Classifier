# Wine Classifier

This project was developed as an assignment for Georgia Tech’s CS7641 Machine Learning program. The report (atorreshotrum3-analysis.pdf) analyzes the performance of five supervised learning techniques as applied to two independent datasets. The learning algorithms were assessed based on their ability to correctly classify data using 5-fold cross-validation. Experiments were conducted to gain an understanding of how varying hyperparameter values
affects the learner’s ability to correctly classify the datasets. 

The wine dataset consisted of information on 4898 white wines and 11 measurable attributes (e.g. PH values) for each. The wines were all given a score between 0 (very bad) and 10 (very excellent) by wine experts. The learner’s objective was to predict a wine’s score based on the 11 measurable attributes. The dataset was balanced and cleaned, then fed into five supervised learning techniques - Decision Tree, Artificial Neural Network, K-Nearest Neighbor, Support Vector Machine and Boosting. The results are shown in the figure below. Several of the learning techniques performed similarly well, with SVM ultimately reaching the highest accuracy of over 82%.


<p align="center">
  <img src="https://github.com/ajhotrum/Wine-Classifier/blob/main/images/results.PNG?raw=true"/>
</p>


