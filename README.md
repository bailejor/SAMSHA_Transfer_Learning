# SAMSHA_Transfer_Learning

Background: Efficacious treatments for substance use disorder do not result in successful treatment for all patients. Machine learning can be used to identify where treatment may fail. This allows resources to be targeted toward those cases. Data in machine learning are often in structured format (rows and columns) deep neural networks are often not best-in-class with this format of data. Typically tree-based models perform better. Transfer learning, which involves repurposing a model on a task it was not trained for, may be an appropriate alternative to tree-based methods for structured (tabular) substance use data. This may allow training on large nationwide datasets to be leveraged when dealing with smaller clinical datasets.

Objectives: The current study examined multiple methods of classifying substance use treatment success using transfer learning. 

Methods: Transfer learning and image embedding were used to classify data from a nationwide database. 28 categorical variables were embedded onto 2D images as inputs. Features were embedded as strings or numeric values. Missing data were embedded as question marks. In experiment one a randomly selected general substance use dataset 70% (n = 435,974, 38% female) of nationwide discharges were used as a training set. Three model configurations were tested (a) a transfer model using ResNet50 as the base classifier, (b) a transfer model using fine-tuning (c) a randomly initialized-weight convolutional neural network (CNN). All models were compared with a tuned random forest (RF). In experiment two the randomly initialized-weight CNN from experiment one was trained and tested on a much smaller target dataset consisting of those in opioid use treatment. This model was compared with a tuned RF. Comparisons were made using an F0.5 metric (favoring precision over recall) and precision-recall curves. 

Results: In experiment one the RF performed better than all other models F0.5 = 0.67. The CNN with randomly initialized weights had the second highest F score (F0.5 = 0.64). The two types of transfer learning had the lowest performance (transfer: F0.5 = 0.56, fine-tuning: F0.5 = 0.55). In experiment the RF outperformed the transfer model (RF: F0.5 = 0.38, CNN: F0.5 = 0.23).


Conclusion: These findings suggest that an approach to leveraging the power of large SUD datasets for transfer learning by embedding text onto 2D images may not be an effective approach. Although it is possible to achieve a score that is close to RF using 2D embeddings, it is highly inefficient. 


