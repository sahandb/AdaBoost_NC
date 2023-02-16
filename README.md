# AdaBoost_NC
Implementation of the AdaBoost.NC algorithm which is an ensemble learning algorithm suitable for imbalanced data classification.

AdaBoost.NC penalizes classification errors and encourages ensemble diversity sequentially with the AdaBoost training framework.

In step 3 of the algorithm, a penalty term 𝑝𝑡 is calculated for each training example, in which 𝑎𝑚𝑏𝑡 assesses the disagreement degree of the classification within the ensemble at the current iteration 𝑡. It is defined as:

___
𝑎𝑚𝑏𝑡 = 1/𝑡 [𝑡 Σ 𝑖=1](‖𝐻𝑡 = 𝑦‖ − ‖ℎ𝑖 = 𝑦‖  )
___

where 𝐻𝑡 is the class label given by the ensemble composed of the existing 𝑡 classifiers. The magnitude of 𝑎𝑚𝑏𝑡 indicates a “pure” disagreement. 𝑝𝑡 is introduced into the weight-updating step (step 5). By doing so, training examples with small |𝑎𝑚𝑏𝑡| will gain more attention. The expression of 𝛼𝑡 in step 4 bounds the overall training error. The predefined parameter 𝜆 controls the strength of applying 𝑝𝑡. The optimal 𝜆 depends on problem domains and base learners. In general, (0, 4] is deemed to be a conservative range of setting 𝜆. As 𝜆 becomes larger, there could be either a further performance improvement or a performance degradation.

The pseudo code for AdaBoost.NC algorithm is provided in the figure below.

![image](https://user-images.githubusercontent.com/24508376/219425853-6781720f-b28b-42fd-9644-3d35d51507e4.png)

The penalty strength 𝜆 in AdaBoost.NC tuned for the given data set empirically to achieve the best results. For example, 𝜆 = 2 is a relatively conservative setting to show if AdaBoost.NC can make a performance improvement, and 𝜆 = 9 encourages ensemble diversity aggressively. we used C4.5 decision tree with default parameters as the base learner. The iteration number 𝑇 selected from the set 𝑇 ∈ {11, 21, 31, 41, 51}. Reported the results for each value of 𝑇.


Since the data set is imbalanced, you should use Precision, Recall, F-measure, AUC, and G-mean measures to evaluate the performance of your implemented algorithm. It is worth mentioning that Precision, Recall, F-measure measures are one-class measures; in other words, you should compute these metrics just for the minority class. For more information regarding the AdaBoost.NC algorithm, please see the attached paper.

Repeated all our experiments for AdaBoost.NC, and for AdaBoost, and Bagging with the noisy version of the data set and report the results. The explanation regarding adding noise to the discrete data set is provided in the attached file. Read it carefully and implement the code to add noise to the data set.

After that initialize weights equal to 1/n and every time use c4.5 classifier for train and then calculate the penalty value for every example and after that calculate alpha T by error and penalty and update data weight and obtain new weights by error and penalty and then predict the efficiency of algorithm and then calculate the metrics for 11 and 21 and 31 and 41 and 51

# Some beneficial questions
Why is AdaBoost.NC algorithm suitable for imbalanced data classification?
Because AdaBoost.NC penalizes classification errors and encourages ensemble diversity sequentially with the AdaBoost training framework

Diversity is important in ensemble learning. How does AdaBoost.NC incorporate diversity in its model?
AdaBoost.NC that combines the strength of negative correlation learning, and boosting. It emphasizes ensemble diversity explicitly during training and shows very encouraging empirical results in both effectiveness and efficiency

What  are  the  differences  between  Precision,  Recall,  F-measure,  AUC,  and  G-mean measures?
Precision means the percentage of your results which are relevant
Recall refers to the percentage of total relevant results correctly classified by your algorithm
F-measure considers both the precision p and the recall r of the test to compute the score: p is the number of correct positive results divided by the number of all positive results returned by the classifier, and r is the number of correct positive results divided by the number of all relevant samples (all samples that should have been identified as positive). 
While the F-measure is the harmonic mean of recall and precision, the G-measure is the geometric mean
AUC - ROC curve is a performance measurement for classification problem at various thresholds settings. ROC is a probability curve and AUC represents degree or measure of separability. It tells how much model is capable of distinguishing between classes
The ROC curve is plotted with TPR against the FPR where TPR is on y-axis and FPR is on the x-axis.

How does noise influence the classification results? AreAdaBoost.NC, AdaBoost, and Bagging robust to the added noise?
All of the are robust against noise and don’t change too much

