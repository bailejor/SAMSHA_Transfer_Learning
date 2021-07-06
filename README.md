# Transfer Learning using SuperTML method from Sun et al. (2019) (https://arxiv.org/abs/1903.06246) on SAMSHA substance use data

Background: Data in machine learning are often in tabular format, but deep neural networks are typically not best-in-class with tabular data. Transfer learning, which involves repurposing a model for a task, may allow for better predictions with this kind of data. This approach may also be useful for small clinical datasets.
<br>
Objectives: The current study examined methods of classifying substance treatment success using transfer learning. 
Methods: Transfer learning was used to classify data from a nationwide database. In Experiment 1, a general substance use dataset (n = 622,825, 38% female) was divided into training, validation, and test sets. Two transfer learning model configurations and a tuned random forest (RF) were compared. In experiment 2 a neural network was trained on a heroin use treatment dataset, then tested on a small opioid use treatment dataset to see whether transfer could occur across treatment involving substances of the same class. 
<br>
Results: In Experiment 1 the RF (F0.5: 0.67, PR AUC: 0.74) performed better than the transfer (F0.5: 0.56, PR AUC: 0.67) or base CNN (F0.5: 0.64, PR AUC: 0.71). In Experiment 2 the transfer model (F0.5: 0.38, PR AUC: 0.49) outperformed the RF model (F0.5: 0.34, PR AUC: 0.43). The neural networks took 20 times longer to train than the RF and required 10 times as much storage space. 
Conclusion: These findings suggest transfer learning may be effective in predicting substance use treatment outcomes. It is possible to achieve a score that performs better than RF, but this method is highly inefficient. 
 

<hr>
<br>
There are four files used in two experiments. <br>
The first is called General_sub_resnet_50.ipynb. It contains the experiment that was run on the general substance dataset using transfer learning with the weights transferred from ResNet50. 
The second model is General_sub_no_transfer. This model is a CNN that was directly trained on the general substance dataset. It has randomly initialized weights. 
The third model is the General_substance_RandomForest. This is a tuned random forest used to compare the other models to. 

In experiment two there is one other file. This file is named Heroin_transfer_opioid.ipynb. In this file a tuned random forest is compared with a model that is trained on a large heroin dataset and then trained and tested on a small non-heroin opioid dataset. 



