# Design Report 

## Initial Problem Break-Down 
At first glance of the _kinase_JAK.csv_ file, a few details stood out: 
* The measurement types are not equally distributed, but their relative compositions of kinases are.
* There are four output targets, JAK1, JAK2, JAK3, and TYK, but they are not independent.
* There are repeated rows of SMILES strings, meaning that a data structure must hold multiple targets per molecule. 

Predicting Inactives vs. Actives is a challenging task with many variations of approaches -- which usually simplifies to predicting binary classification or predicting the pIC50/pKi values directly. Both are not without their weaknesses. Binary classification ignores any information the measurement reveals about the molecular structure, and regression tasks do not learn to penalize features of inactive molecules directly since there is no 'negative pIC50' value. Generating random values will certainly induce noise into the model.  

I chose to tackle this problem in two ways: the first is to explore merging two separate models into one since the dataset is relatively small and the models could be made quite lean, and two, making one model that is trained to predict both class and pIC50 values at the same time.

## Data Analysis and Exploration 

My analysis and exploration of the data were performed in [Data_Analysis.ipynb](Data_Analysis.ipynb), and some conclusions influenced my model design:
* The dataset is comparatively small when the task is compressed into predicting both class and pIC50 values.
* Making an individual model for each target would yield a lopsided training dataset for TYK2 since it is _under-represented_.
* The deviations between pIC50 and pKi make them incompatible with being used together, and artificially generating data will tamper with model predictions.

To strike a balance between the risk of over and underfitting, I chose to use the pIC50 values for all targets and make a model that simultaneously predicts both class and pIC50 values. Upon evaluating the GNN model, it was clear it could predict reasonably well on OOD data, but I would be more confident that it is production ready if it had a larger dataset.

## Methodology of Model Construction

With only a week, I had to limit the number of models I could build while also balancing my work on deployment and documentation. Two of the most prevalent models in chemistry are Mulit-Layer Perceptrons and Graph Neural Networks. Thus I first chose to explore a smaller MLP model's efficiency at predicting pIC50 values and then a GNN model to predict both class and pIC50 values simultaneously. 

Starting with the MLP, the initial approach was to build an adversarial autoencoder (AAE) followed by a multi-layer perceptron (MLP) model. The AAE model was used to train an encoder and discriminator on active and inactive molecules, represented as Morgan Fingerprint bit vectors. Upon evaluating the AAE's encoder, it showed it could distinguish active and inactive molecules with high accuracy. However, when the MLP model was attached and used to predict precise pIC50 values, it often converged to the mean value of the training set and showed poor correlation abilities. I tried to address this issue by converting the scalar output into a probability distribution (Mixture of Gaussians). While this did help spread out the predictions (and provide confidence estimations), it also reduced the model's overall precision.

As a result, I considered the GNN next, which involved using the knowledge of the well-formed encoder to influence the latent space. I know GNNs can better handle molecule structures for this particular problem, and to leverage that, the spatial pharmacophoric features were also included in the molecule's atom vectors. The molecule's Morgan Fingerprints were still used to train the encoder portion of the GNN, as MLP testing revealed it provided useful embeddings. This led to a final model with explainable embeddings that could allude to indicators of target activity.  


I used standard optimizers, loss functions, and learning rate schedulers to stabilize the training procedure. The trained models are included in the kinpred directory, and the web app uses them to make predictions. 


## Results and Model Evaluation

The results from each model are contained within their respective jupyter notebooks, [AAE_MLP_Training.ipynb](AAE_MLP_Training.ipynb) and [GNN_Training.ipynb](GNN_Training.ipynb). One interesting experiment I performed was testing the GNN on 5 OOD molecules that it had never seen before. The results are shown below:


| Molecule Name | JAK1 | JAK2 | JAK3 | TYK2 |
| --- | --- | --- | --- | --- |
| WP1066 | **0.23** | **0.90** | **0.12** | **0.02** |
| Ruxolitinib | **0.99** | **0.75** | 0.994 | **0.01** |
| AZD1480 | 0.98 | **0.99** | **0.003** | **0.006** |
| AT9283 | **0.02** | **0.97** | **0.99** | **0.02** |
| Deucravacitinib | 0.97 | 0.97 | 0.80 | 0.04 |

The model achieved an AUC of 0.78 on class label prediction! Unfortunately, I did not find the literature for the pIC50 values of these molecules in time, so I could not evaluate the direct pIC50 predictions on OOD data. 

## Deployment 

I chose to build out a small Flask-based application to show the ease of this model's deployment. It can be hosted locally or on a server and exposes an API without needing to visit the frontend directly.  

The full use case is explored in the [webapp](webapp) folder, and the API can be used with the following curl command:
```bash
curl -X POST http://127.0.0.1:5000/api_predict -d "smiles=[INSERT SMILES]"
```

## Future Work

Given more time, I would have liked to explore AWS deployment and dataset balancing. I think the model could be improved by evening the distribution of samples to their respective targets. However, this method would need to consider molecules with multiple targets, which provides a unique challenge. Would the molecule be counted for each target once or multiple times? How does the distribution of active molecules with multiple targets skew the model's predictions on molecules with only one target? 

Another task I would like to explore is uncovering the interpretability of the GNN model. I wonder which pharmacophoric features and morgan fingerprint bits are most important and if they can describe the local pocket structure, which could aid in the search for novel bioisosteres.    

Lastly, I believe this model could be improved significantly when given the 3D local pocket structure. The receptor environment plays a pivotal role in determining binding affinity, and this could enhance the precision and specificity of prediction tasks.

## Package Requirements 

A couple of quick notes about package requirements:

I used the standard torch/torch_geometric, rdkit, and NumPy/pandas frameworks to build the models and handle data preparation. Matplotlib, scikit-learn, and scipy were used for data analysis and visualization—finally, the web app leverages flask for the API construction and deployment.
