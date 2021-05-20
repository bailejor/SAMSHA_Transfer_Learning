# SAMSHA_Transfer_Learning

Background: Efficacious treatments for substance use disorder do not result in successful treatment for all patients. Machine learning can be used to identify where treatment may fail. This allows resources to be targeted toward those cases. Data in machine learning are often in structured format (rows and columns) and tree-based algorithms are often best-in-class with this format of data. Transfer learning, which involves repurposing a model on a task it was not trained for, may be an appropriate alternative to these methods. 

Objectives: The current study examined a method of classifying substance use treatment success using transfer learning. 

Methods: Transfer learning and image embedding were used to classify data from a nationwide database. 28 categorical variables were embedded onto 2D images as inputs. Features were embedded as strings or numeric values. Missing data were embedded as question marks. Three model configurations were tested (a) a transfer model using ResNet50 as the base classifier, fine-tuning with ResNet50, and a randomly initialized-weight convolutional neural network (CNN) trained on the general substance dataset The most performant model was compared with a tuned random forest. 


 A randomly selected 70% (n = 435,974, 38% female) of nationwide discharges were used. The transfer learning model was compared with other models. All were evaluated using precision-recall curves. 
