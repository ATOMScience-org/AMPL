# AMPL pipeline parameters (options)
The AMPL pipeline contains many parameters and options to fit models and make predictions. The parameters have been organized in the following sections:

## Table of contents
- [Training Dataset Parameters](#Training-Dataset-Parameters)
- [Model Building Parameters](#Model-Building-Parameters)
  - [Autoencoders](#Autoencoders)
  - [Classifiers](#Classifiers)
  - [Descriptors](#Descriptors)
  - [ECFP](#ECFP)
  - [General](#General)
  - [Graph Convolution](#Graph-Convolution)
  - [Mordred](#Mordred)
  - [Neural Networks](#Neural-Networks)
  - [Random Forests](#Random-Forests)
  - [Hybrid model](#Hybrid-model)
  - [Splitting](#Splitting)
  - [Transformers](#Transformers)
  - [UMAP](#UMAP)
  - [XGBoost](#XGBoost)
  - [Additional DeepChem Models](#Auto-DCModels)
- [Model Saving](#Model-Saving)
- [Model Metadata](#Model-Metadata)
- [Miscellaneous](#Miscellaneous)
- [Hyperparameter Optimization](#Hyperparameter-Optimization)
  - [Bayesian Optimization](#Bayesian-Optimization)


<a name="Training-Dataset-Parameters"></a>
# Training Dataset Parameters

- **bucket**  
  
|||
|-|-|
|*Description:*|Name of datastore bucket. Specific to LLNL datastore system.|
|*Default:*|gsk\_ml|
  
- **dataset\_key**  
  
|||
|-|-|
|*Description:*|Datastore key (LLNL system) or file path for dataset.|
  
- **dataset\_name**  
  
|||
|-|-|
|*Description:*|Parameter for overriding the output files/dataset object names. Default is set within model\_pipeline.|
  
- **dataset\_oid**  
  
|||
|-|-|
|*Description:*|OID of the model dataset inserted into the datastore. Specific to LLNL datastore system.|
  
- **datastore**  
  
|||
|-|-|
|*Description:*|Boolean flag for using an input file from the LLNL specific datastore system based on a key of dataset\_key|
|*Default:*|FALSE|
|*Type:*|Bool|
  
- **id\_col**  
  
|||
|-|-|
|*Description:*|Name of column containing compound IDs. Will default to "compound\_id" if not specified|
|*Default:*|compound\_id|
  
- **min\_compound\_number**  
  
|||
|-|-|
|*Description:*|Minimum number of dataset compounds considered adequate for model training. A warning message will be issued if the dataset size is less than this.|
|*Default:*|200|
|*Type:*|int|
  
- **response\_cols**  
  
|||
|-|-|
|*Description:*|name of column(s) containing response values. Will default to last column if not specified. Can be input as a string of comma separated values or as a comma separated list (e.g. 'column1','column2'). Multitask models will be generated when multiple columns are specified.|
  
- **save\_results**  
  
|||
|-|-|
|*Description:*|Save model results to MongoDB. LLNL model\_tracker system specific|
|*Default:*|FALSE|
|*Type:*|BOOL|
  
- **smiles\_col**  
  
|||
|-|-|
|*Description:*|Name of column containing SMILES strings. Will default to "rdkit\_smiles" if not specified|
|*Default:*|rdkit\_smiles|
  
---

<a name="Model-Building-Parameters"></a>
# Model Building Parameters  

---

<a name="Autoencoders"></a>
## Autoencoders  

- **autoencoder\_bucket**  
  
|||
|-|-|
|*Description:*|datastore bucket for the autoencoder file. Specific to LLNL datastore system. TODO: Not yet implemented|
|*Default:*|gsk\_ml|
  
- **autoencoder\_key**  
  
|||
|-|-|
|*Description:*|Base of key for the autoencoder. TODO: Not yet implemented|
  
- **autoencoder\_type**  
  
|||
|-|-|
|*Description:*|Type of autoencoder being used as features. TODO: not yet implemented|
|*Default:*|molvae|
  
- **mol\_vae\_model\_file**  
  
|||
|-|-|
|*Description:*|Trained model HDF5 file path, only needed for MolVAE featurizer|
  
---

<a name="Classifiers"></a>
## Classifiers  

- **class\_name**  
  
|||
|-|-|
|*Description:*|User specified list of names of each class|
  
- **class\_number**  
  
|||
|-|-|
|*Description:*|User specified number of classes. This is required for NN models but inferred for RF and XGBoost models.|
|*Default:*|2|
|*Type:*|int|
  
---

<a name="Descriptors"></a>
## Descriptors  

- **descriptor\_bucket**  
  
|||
|-|-|
|*Description:*|datastore bucket for the descriptor file. Specific to LLNL datastore system.|
|*Default:*|gskdata|
  
- **descriptor\_key**  
  
|||
|-|-|
|*Description:*|Base of key for descriptor table file. Subset files will be prepended with "subset" and appended with the dataset name. Specific to LLNL datastore system.|
  
- **descriptor\_oid**  
  
|||
|-|-|
|*Description:*|dataset\_oid for the descriptor file in the datastore. Specific to LLNL datastore system.|
  
- **descriptor\_spec\_bucket**  
  
|||
|-|-|
|*Description:*|Bucket where descriptor specification is located for a descriptor type. Specific to LLNL datastore system.|
|*Default:*|public|
  
- **descriptor\_spec\_key**  
  
|||
|-|-|
|*Description:*|Datastore key or file path for a table specifying descriptor columns for each descriptor type. Specific to LLNL datastore system.|
|*Default:*|descriptor\_sets\_sources\_by\_descr\_type.csv|
  
- **descriptor\_type**  
  
|||
|-|-|
|*Description:*|Type of descriptors being used as features, e.g. moe, dragon7, used when featurizer = "computed_descriptors". Sets the subclass within featurizer.py|
|*Default:*|moe|
|*Options:*|'moe', 'mordred_filtered', and 'rdkit_raw' are recommended. See atomsci/ddm/data/descriptor_sets_sources_by_descr_type.csv for more.|
  
---

<a name="ECFP"></a>
## ECFP  

- **ecfp\_radius**  
  
|||
|-|-|
|*Description:*|Radius used for ECFP generation|
|*Default:*|2|
|*Type:*|int|
  
- **ecfp\_size**  
  
|||
|-|-|
|*Description:*|Size of ECFP bit vectors|
|*Default:*|1024|
|*Type:*|int|
  
---

<a name="General"></a>
## General  

- **featurizer**  
  
|||
|-|-|
|*Description:*|Type of featurizer to use on chemical structures. Current supported options: ["ecfp","graphconv","molvae","computed\_descriptors","descriptors"]. Further information on descriptors are in descriptor\_type. Options are used to set the featurization subclass in the create\_featurization method of featurization.py. Can be input as a comma separated list for hyperparameter search (e.g. 'ecfp','molvae')|
|*Type:*|str|
  
- **model\_choice\_score\_type**  
  
|||
|-|-|
|*Description:*|Type of score function used to choose best epoch and/or hyperparameters (defaults to "roc\_auc" for classification and "r2" for regression). |
  
- **model\_type**  
  
|||
|-|-|
|*Description:*|Type of model to fit (NN, RF, or xgboost). The model\_type sets the model subclass in model\_wrapper. Can be input as a comma separated list for hyperparameter search (e.g. 'NN','RF','xgboost')|
|*Type:*|str|
  
- **prediction\_type**  
  
|||
|-|-|
|*Description:*|Sets the prediction type of the model to a choice between ["regression","classification"]. Used as a flag for model behavior throughout the pipeline.|
|*Default:*|regression|
|*Type:*|choice|
  
- **previously\_featurized**  
  
|||
|-|-|
|*Description:*|Boolean flag for loading in previously featurized data files. If set to True, the method get\_featurized\_data within model\_datasets will attempt to load the featurized dataset associated with the given dataset\_oid parameter|
|*Default:*|TRUE|
|*Type:*|Bool|
  
- **uncertainty**  
  
|||
|-|-|
|*Description:*|Boolean flag for computing uncertainty estimates for regression model predictions. Will also change the default values for dropouts if set to True.|
|*Default:*|TRUE|
|*Type:*|Bool|
  
- **verbose**  
  
|||
|-|-|
|*Description:*|True/False flag for setting verbosity|
|*Default:*|FALSE|
|*Type:*|Bool|

- **seed**  
  
|||
|-|-|
|*Description:*|Seed used for initializing a random number generator to ensure results are reproducible. Default is None and a random seed will be generated.|
|*Default:*|None|
|*Type:*|int|
  
- **production**  
  
|||
|-|-|
|*Description:*|True/False flag for training models in production mode. The entire dataset is used in training, validation, and test. If using training epocs
the model will train for max_epochs regardless of validation error.|
|*Default:*|FALSE|
|*Type:*|Bool|
  

---

<a name="Graph-Convolution"></a>
## Graph Convolution  

- **optimizer\_type**  
  
|||
|-|-|
|*Description:*|Optimizer specific for graph conv, defaults to "adam"|
|*Default:*|adam|
  
---

<a name="Mordred"></a>
## Mordred  

- **mordred\_cpus**  
  
|||
|-|-|
|*Description:*|Max number of CPUs to use for Mordred descriptor computations. None means use all available|
|*Type:*|int|
  
---

<a name="Neural-Networks"></a>
## Neural Networks  

- **baseline\_epoch**  
  
|||
|-|-|
|*Description:*|Baseline epoch at which to evaluate performance for DNN models|
|*Default:*|30|
|*Type:*|int|
  
- **batch\_size**  
  
|||
|-|-|
|*Description:*|Sets the model batch size within model\_wrapper|
|*Default:*|50|
|*Type:*|int|
  
- **bias\_init\_consts**  
  
|||
|-|-|
|*Description:*|Comma-separated list of initial bias parameters per layer for dense NN models with conditional values.  Defaults to [1.0]*len(layer\_sizes). Must be same length as layer\_sizes. Can be input as a space-separated list of comma-separated lists for hyperparameters. Hyperparameter example: '1.0,1.0 0.9,0.9 0.8,0.9' Default behavior is set within \_\_init\_\_ method of DCNNModelWrapper.  Defaults: all:[1.0,1.0]|
  
- **dropouts**  
  
|||
|-|-|
|*Description:*|Comma-separated list of dropout rates per layer for NN models with default values conditional on featurizer. Default behavior is controlled in model\_wrapper.py. Must be same length as layer\_sizes. Can be input as a space-separated list of comma-separated lists for hyperparameters (e.g. '0.4,0.4 0.2,0.2 0.3,0.3'). Default behavior is set within \_\_init\_\_ method of DCNNModelWrapper. Defaults: graphconv: [0,0,0], non-graphconv:[0.40,0.40]|
|*Type:*|list|
  
- **layer\_sizes**  
  
|||
|-|-|
|*Description:*|Comma-separated list of layer sizes for NN models with default values conditional on featurizer. Must be same length as layer\_sizes. Can be input as a space-separated list of comma-separated lists for hyperparameters (e.g. '64,16 200,100 1000,500'). Default behavior is set within \_\_init\_\_ method of DCNNModelWrapper. Defaults: graphconv: [64,64,128], ecfp: [1000,500], descriptors: [200,100]|
|*Type:*|list|
  
- **learning\_rate**  
  
|||
|-|-|
|*Description:*|Learning rate for dense NN models. Input as comma separated floats for hyperparameters (e.g. '0.0005,0.0004,0.0003')|
|*Default:*|0.0005|
  
- **max\_epochs**  
  
|||
|-|-|
|*Description:*|Maximum number of training epochs to run for DNN models. Default 30.|
|*Default:*|30|
|*Type:*|int|
  
- **weight\_decay\_penalty**  
  
|||
|-|-|
|*Description:*|weight\_decay\_penalty: float. The magnitude of the weight decay penalty to use. Can be input as a comma separated list of strings for hyperparameter search (e.g. '0.0001,0.0002,0.0003') default 0.0001|
|*Default:*|0.0001|
  
- **weight\_decay\_penalty\_type**  
  
|||
|-|-|
|*Description:*|weight\_decay\_penalty\_type: str. The type of penalty to use for weight decay, either "l1" or "l2". Can be input as a comma separated list for hyperparameter search (e.g. 'l1,l2') default: "l2"|
|*Default:*|l2|
|*Type:*|str|
  
- **weight\_init\_stddevs**  
  
|||
|-|-|
|*Description:*|Comma-separated list of standard deviations per layer for initializing weights in dense NN models with conditional values. Must be same length as layer\_sizes. Can be input as a space-separated list of comma-separated lists for hyperparameters (e.g. '0.001,0.001 0.002,0.002 0.03,003'). Default behavior is set within \_\_init\_\_ method of DCNNModelWrapper. Defaults: all: [0.02,0.02]|
|*Default:*|[0.02]*len(param.layer\_size)|
  
---

<a name="Random-Forests"></a>
## Random Forests  

- **rf\_estimators**  
  
|||
|-|-|
|*Description:*|Number of estimators to use in random forest models. Hyperparameter searching requires 3 inputs: start, end, step when used with search\_type geometric or grid (example: '100,500,100') or can be input as a list of possible values for search\_type user\_specified (example: '100,200,300,400,500')|
|*Default:*|500|
  
- **rf\_max\_depth**  
  
|||
|-|-|
|*Description:*|The maximum depth of a decision tree in the random forest.  Hyperparameter searching requires 3 inputs: start, end, step when used with search\_type geometric or grid (example: '4,7,1') or can be input as a list of possible values for search\_type user\_specified (example: '4,5,6,7')|
  
- **rf\_max\_features**  
  
|||
|-|-|
|*Description:*|Max number of features to split random forest nodes. Hyperparameter searching requires 3 inputs: start, end, step when used with search\_type geometric or grid (example: '16,32,4') or can be input as a list of possible values for search\_type user\_specified (example: '16,20,24,28,32')|
|*Default:*|32|
  
---

<a name="Hybrid-model"></a>
## Hybrid model  

- **is\_ki**  
  
|||
|-|-|
|*Description:*|True/False flag for noting whether the dose-response activity is Ki or XC50, if it is **True**, the following **ki_convert_ratio** is also needed to convert Ki into IC50 and to single concentration activity.|
|*Default:*|False|
  
- **ki\_convert\_ratio**  
  
|||
|-|-|
|*Description:*|To convert Ki into IC50, a ratio is needed. It can be the ratio of \[S\]/Km for enzymatic inhibition assays, \[S\] is the concentration of substrate Km is the Michaelis constant. It can also be \[S\]/Kd for radioligand competitive binding, \[S\] is the concentration of the radioligand, Kd is its dissociation constant. The \[S\] and Kd/Km should have the same unit so that the ratio is unitless.|
|*Default:*|None|
  
- **loss\_func**  
  
|||
|-|-|
|*Description:*|The loss function used in the hybrid model training, currently support poisson and l2|
|*Default:*|poisson|
  
---

<a name="Splitting"></a>
## Splitting  

- **base\_splitter**  
  
|||
|-|-|
|*Description:*|Type of splitter to use for train/validation split if temporal split used for test set. May be random, scaffold, or ave\_min. The allowable choices are set in splitter.py|
|*Default:*|scaffold|
|*Type:*|str|
  
- **butina\_cutoff**  
  
|||
|-|-|
|*Description:*|cutoff Tanimoto similarity for clustering in Butina splitter. TODO: will be implemented when DeepChem updates their butina splitter. TODO rename to butina\_cutoff in v2|
|*Default:*|0.18|
|*Type:*|float|
  
- **cutoff\_date**  
  
|||
|-|-|
|*Description:*|Cutoff date for test set compounds in temporal splitter TODO: Needs some formatting guidelines|
|*Type:*|str|
  
- **date\_col**  
  
|||
|-|-|
|*Description:*|Column in dataset containing dates for temporal splitter |
|*Type:*|str|
  
- **num\_folds**  
  
|||
|-|-|
|*Description:*|Number of k-folds to use in k-fold cross validation|
|*Default:*|5|
|*Type:*|int|
  
- **previously\_split**  
  
|||
|-|-|
|*Description:*|Boolean flag for loading in previously split train, validation, and test csv files.|
|*Default:*|FALSE|
|*Type:*|bool|
  
- **split\_strategy**  
  
|||
|-|-|
|*Description:*|Choice of splitting type between "k\_fold\_cv" for k fold cross validation and "train\_valid\_test" for a normal train/valid/test split. If split\_test\_frac or split\_valid\_frac are not set, "train\_valid\_test" sets are split according to the model type default|
|*Default:*|train\_valid\_test|
|*Type:*|Choice|
  
- **split\_test\_frac**  
  
|||
|-|-|
|*Description:*|Fraction of data to put in held-out test set for train\_valid\_test split strategy. TODO: Behavior of split\_test\_frac is dependent on the DeepChem model\_wrapper.|
|*Default:*|0.1|
|*Type:*|float|
  
- **split\_uuid**  
  
|||
|-|-|
|*Description:*|UUID for csv file containing train, validation, and test split information|
  
- **split\_valid\_frac**  
  
|||
|-|-|
|*Description:*|Fraction of data to put in validation set for train\_valid\_test split strategy. TODO: Behavior of split\_valid\_frac is dependent on the DeepChem model\_wrapper.|
|*Default:*|0.1|
|*Type:*|float|
  
- **splitter**  
  
|||
|-|-|
|*Description:*|Type of splitter to use: index, random, scaffold, butina, ave\_min, temporal, fingerprint, multitaskscaffold, or stratified. Used to set the splitting.py subclass. Can be input as a comma separated list for hyperparameter search (e.g. 'scaffold','random')|
|*Default:*|scaffold|
|*Type:*|str|
  
- **sampling_method**  
  
|||
|-|-|
|*Description:*|The sampling method for addressing class imbalance in classification datasets. Options include 'undersampling' and 'SMOTE'.|
|*Default:*|None|
|*Type:*|str|

- **sampling_ratio**  
  
|||
|-|-|
|*Description:*|The desired ratio of the minority class to the majority class after sampling (e.g., if str, 'minority', 'not minority'; if float, '0.2', '1.0'). |
|*Default:*|auto|
|*Type:*|str|

- **sampling_k_neighbors**  
  
|||
|-|-|
|*Description:*|The number of nearest neighbors to consider when generating synthetic samples (e.g., 5, 7, 9). Specifically used for SMOTE sampling method.|
|*Default:*|5|
|*Type:*|int|

- **mtss\_num\_super\_scaffolds**  
  
|||
|-|-|
|*Description:*|This specifies the number of genes in a chromosome for the genetic algorithm. Scaffolds bins are often very small and only contain 1 compound. Scaffolds are therefore combined into super scaffolds to the number of genes and also reduce complexity and runtime.|
|*Default:*|40|
|*Type:*|int|

- **mtss\_num\_generations**  
  
|||
|-|-|
|*Description:*|The number of generations the genetic algorithm will run.|
|*Default:*|20|
|*Type:*|int|

- **mtss\_num\_pop**  
  
|||
|-|-|
|*Description:*|Size of population per generation in the genetic algorithm.|
|*Default:*|100|
|*Type:*|int|

- **mtss\_train\_test\_dist\_weight**  
  
|||
|-|-|
|*Description:*|How much weight to give the tanimoto distance between training and test partitions.|
|*Default:*|1.0|
|*Type:*|float|

- **mtss\_train\_valid\_dist\_weight**  
  
|||
|-|-|
|*Description:*|How much weight to give the tanimoto distance between training and valid partitions.|
|*Default:*|1.0|
|*Type:*|float|

- **mtss\_response\_distr\_weight**  
  
|||
|-|-|
|*Description:*|How much weight to give to matching the response value distributions between split subsets.|
|*Default:*|1.0|
|*Type:*|float|

- **mtss\_split\_fraction\_weight**  
  
|||
|-|-|
|*Description:*|How much weight to give adherence to requested subset franctions.|
|*Default:*|1.0|
|*Type:*|float|
  
---

<a name="Transformers"></a>
## Transformers  

- **feature\_transform\_type**  
  
|||
|-|-|
|*Description:*|type of transformation for the features|
|*Default:*|normalization|
|*Type:*|Choice|
  
- **response\_transform\_type**  
  
|||
|-|-|
|*Description:*|type of transformation for the response column (defaults to "normalization") TODO: Not currently implemented|
|*Default:*|normalization|
  
- **weight\_transform\_type**  
  
|||
|-|-|
|*Description:*|type of transformation for class weights in a classification model loss function. Use the "balancing" option to offset the effect of imbalanced datasets. Works with NN, random forest and XGBoost models. |
|*Default:*|None|
|*Type:*|Choice|
  
- **transformer\_bucket**  
  
|||
|-|-|
|*Description:*|Datastore bucket where the transformer is stored. Specific to LLNL datastore system.|
|*Default:*|gsk\_ml|
  
- **transformer\_key**  
  
|||
|-|-|
|*Description:*|Path to a saved transformer (stored as tuple, e.g. (transform\_features, transform\_response))|
|*Type:*|str|
  
- **transformer\_oid**  
  
|||
|-|-|
|*Description:*|Dataset oid of the transformer saved in the datastore. Specific to LLNL datastore system.|
  
- **transformers**  
  
|||
|-|-|
|*Description:*|Boolean switch for using transformation on regression output. Default is True|
|*Default:*|TRUE|
|*Type:*|Bool|
  
---

<a name="UMAP"></a>
## UMAP  

- **umap\_dim**  
  
|||
|-|-|
|*Description:*|Dimension of projected feature space, if UMAP transformation is requested. Can be input as a comma separated list for hyperparameter search (e.g. '2,6,10').|
|*Default:*|10|
  
- **umap\_metric**  
  
|||
|-|-|
|*Description:*|Distance metric used, if UMAP transformation is requested. Can be input as a comma separated list for hyperparameter search (e.g. 'euclidean','cityblock')|
|*Default:*|euclidean|
  
- **umap\_min\_dist**  
  
|||
|-|-|
|*Description:*|Minimum distance used in UMAP projection, if UMAP transformation is requested. Can be input as a comma separated list for hyperparameter search (e.g. '0.01,0.02,0.05')|
|*Default:*|0.05|
  
- **umap\_neighbors**  
  
|||
|-|-|
|*Description:*|Number of nearest neighbors used in UMAP projection, if UMAP transformation is requested. Can be input as a comma separated list for hyperparameter search (e.g. '10,20,30')|
|*Default:*|20|
  
- **umap\_targ\_wt**  
  
|||
|-|-|
|*Description:*|Weight given to training set response values in UMAP projection, if UMAP transformation is requested. Can be input as a comma separated list for hyperparameter search (e.g. '0.0,0.1,0.2')|
|*Default:*|0.0|
  
---

<a name="XGBoost"></a>
## XGBoost  

- **xgb\_colsample\_bytree**  
  
|||
|-|-|
|*Description:*|Subsample ratio of columns when constructing each tree. Can be input as a comma separated list for hyperparameter search (e.g. '0.8,0.9,1.0')|
|*Default:*|1.0|
  
- **xgb\_gamma**  
  
|||
|-|-|
|*Description:*|Minimum loss reduction required to make a further partition on a leaf node of the tree. Can be input as a comma separated list for hyperparameter search (e.g. '0.0,0.1,0.2')|
|*Default:*|0.0|
  
- **xgb\_alpha**  
  
|||
|-|-|
|*Description:*|L1 regularization term on weights. Increasing this value will make model more conservative. Can be input as a comma separated list for hyperparameter search (e.g. '0.0,0.1,0.2')|
|*Default:*|0.0|
  
- **xgb\_lambda**  
  
|||
|-|-|
|*Description:*|L2 regularization term on weights. Increasing this value will make model more conservative. Can be input as a comma separated list for hyperparameter search (e.g. '0.0,0.1,0.2')|
|*Default:*|1.0|
  
- **xgb\_learning\_rate**  
  
|||
|-|-|
|*Description:*|Boosting learning rate (xgboost's \"eta\"). Can be input as a comma separated list for hyperparameter search (e.g. '0.1,0.01,0.001')|
|*Default:*|0.1|
  
- **xgb\_max\_depth**  
  
|||
|-|-|
|*Description:*|Maximum tree depth for base learners. Can be input as a comma separated list for hyperparameter search (e.g. '4,5,6')|
|*Default:*|6|
  
- **xgb\_min\_child\_weight**  
  
|||
|-|-|
|*Description:*|Minimum sum of instance weights (hessian) needed in a child. Can be input as a comma separated list for hyperparameter search (e.g. '1.0,1.1,1.2')|
|*Default:*|1.0|
  
- **xgb\_n\_estimators**  
  
|||
|-|-|
|*Description:*|Number of estimators to use in xgboost models. Can be input as a comma separated list for hyperparameter search (e.g. '100,200,300')|
|*Default:*|100|
  
- **xgb\_subsample**  
  
|||
|-|-|
|*Description:*|Subsample ratio of the training instance. Can be input as a comma separated list for hyperparameter search (e.g. '0.8,0.9,1.0')|
|*Default:*|1.0|
  
---

<a name="Auto-DCModels"></a>
## Additional DeepChem Models and Featurizers
As of version 1.3 AMPL partially supports several DeepChem models. It is possible to train and predict
using these models, but they are not currently integrated with the hyperparameter search wrapper.

### Models
AMPL supports the following models:

- [AttentiveFPModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#attentivefpmodel)
- [GCNModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#gcnmodel)
- [GraphConvModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#graphconvmodel)
- [MPNNModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#mpnnmodel)
- [PytorchMPNNModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#id38)


These models can be selected by using the `model_type` paramter e.g. `"model_type":"AttentiveFPModel"`.
Parameters for each model can be passed in by prefixing the parameter with the name of the model.

```
    "comment": "Model",
    "comment": "----------------------------------------",
    "model_type": "AttentiveFPModel",
    "AttentiveFPModel_num_layers":"3",
    "AttentiveFPModel_learning_rate": "0.0007",
    "AttentiveFPModel_n_tasks": "1",
```

### Featurizers
AMPL supports the following DeepChem featurizers:
- [MolGraphConvFeaturizer](https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html#molgraphconvfeaturizer)
- [WeaveFeaturizer](https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html#weavefeaturizer)
- [ConvMolFeaturizer](https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html#convmolfeaturizer)

Each DeepChem model expects a specific featurizer. Model/Featurizer compatibility is listed in [this table](https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html#convmolfeaturizer).
Featurizers can be specified by setting the `featurizer` parameter. Featurizer parameters are passed
in the same way as model parameters.

```
    "comment": "Features",
    "comment": "----------------------------------------",
    "featurizer":"MolGraphConvFeaturizer",
    "MolGraphConvFeaturizer_use_edges":"True",
```

<a name="Model-Saving"></a>
# Model Saving  

- **collection\_name**  
  
|||
|-|-|
|*Description:*|MongoDB collection to save model results in. Specific to LLNL model tracker system.|
|*Default:*|model\_tracker|
  
- **data\_owner**  
  
|||
|-|-|
|*Description:*|Option for setting group permissions for created files. Options: ['username', 'data\_owner\_group', 'gsk', 'public']. Specific to LLNL model tracker system.|
|*Default:*|gsk|
  
- **data\_owner\_group**  
  
|||
|-|-|
|*Description:*|When data\_owner is set to data\_owner\_group, this is the option for custom group name of created files. Specific to LLNL model tracker system.|
|*Default:*|gskcraa|
  
- **model\_bucket**  
  
|||
|-|-|
|*Description:*|Bucket in the datastore for the model. Specific to LLNL model tracker system.|
|*Default:*|gsk\_ml|
|*Type:*|str|
  
- **model\_dataset\_oid**  
  
|||
|-|-|
|*Description:*|OID of the model dataset inserted into the datastore. Specific to LLNL model tracker system|
  
- **model\_filter**  
  
|||
|-|-|
|*Description:*|Path to the model filter configuration file. Is loaded and stored as a dictionary. Specific to LLNL model tracker system.|
  
- **model\_uuid**  
  
|||
|-|-|
|*Description:*|UUID generated after model creation (pythonic\_ID). Specific to LLNL model tracker system.|
|*Type:*|str|
  
- **output\_dir**  
  
|||
|-|-|
|*Description:*|File location where the model output will be saved. Defaults to <result\_dir>/ TODO: this parameter is redundant with result\_dir|
  
- **result\_dir**  
  
|||
|-|-|
|*Description:*|Parent of directory where result files will be written, defaults to '/usr/local/data'|
|*Default:*|/usr/local/data/|
  
---

<a name="Model-Metadata"></a>
# Model Metadata  

- **system**  
  
|||
|-|-|
|*Description:*|Computational system you are running on, LC or twintron-blue. LLNL system specific|
|*Default:*|twintron-blue|
|*Type:*|str|
  
---

<a name="Miscellaneous"></a>
# Miscellaneous  

- **config\_file**  
  
|||
|-|-|
|*Description:*|Full path to the optional configuration file. The configuration file is a set of parameters in .json file format. TODO: Does not send a warning if set concurrently with other parameters.|
  
- **num\_model\_tasks**  
  
|||
|-|-|
|*Description:*|DEPRECATED AND IGNORED. This argument is now infered from the response_cols. Number of tasks to run for. 1 means a singletask model, > 1 means a multitask model|
|*Default:*|1|
|*Type:*|int|
  
---

<a name="Hyperparameter-Optimization"></a>
# Hyperparameter Optimization  

- **dropout\_list**  
  
|||
|-|-|
|*Description:*|Comma-separated list of dropout rates for permutation of NN layers (e.g. '0.0,0.4,0.6'). Used within permutate\_NNlayer\_combo\_params to return combinations from layer\_nums, node\_nums, dropout\_list and max\_final\_layer\_size. dropout\_list is used to set the allowable permutations of dropouts. For hyperparameters only.|
  
- **hyperparam**  
  
|||
|-|-|
|*Description:*|Boolean flag to indicate whether we are running the hyperparameter search script|
|*Default:*|FALSE|
  
- **hyperparam\_uuid**  
  
|||
|-|-|
|*Description:*|UUID of hyperparam search run model was generated in. Not applicable for single-run jobs. Specific to LLNL model tracker system.|
  
- **layer\_nums**  
  
|||
|-|-|
|*Description:*|Comma-separated list of number of layers for permutation of NN layers. (e.g. '2,3,4'). Used within permutate\_NNlayer\_combo\_params to return combinations from layer\_nums, node\_nums, dropout\_list and max\_final\_layer\_size. layer\_nums is used to set the allowable lengths of layer\_sizes. For hyperparameters only.|
  
- **lc\_account**  
  
|||
|-|-|
|*Description:*|SLURM account to charge hyperparameter batch runs to. This will be replaced by the slurm\_account option. If lc\_account and slurm\_account are both set, slurm\_account will be used. If set to None then this parameter will not be used.|
|*Default:*|baasic|
  
- **max\_final\_layer\_size**  
  
|||
|-|-|
|*Description:*|The max number of nodes in the last layer within layer\_sizes and dropouts in hyperparameter search; max\_final\_layer\_size = min(node\_nums) if min(node\_nums) > max\_final\_layer\_size. (e.g. '16,32'\). Used within permutate\_NNlayer\_combo\_params to return combinations from layer\_nums, node\_nums, dropout\_list and max\_final\_layer\_size. |
|*Default:*|32|
  
- **node\_nums**  
  
|||
|-|-|
|*Description:*|Comma-separated list of number of nodes per layer for permutation of NN layers. (e.g. '4,8,16'). Used within permutate\_NNlayer\_combo\_params to return combinations from layer\_nums, node\_nums, dropout\_list and max\_final\_layer\_size. node\_num is used to set the node values within layer\_sizes. For hyperparameters only.|
  
- **max\_jobs**  
  
|||
|-|-|
|*Description:*|Max number of jobs to be in the queue at one time for an LC machine. Specific to LLNL system.|
|*Default:*|80|
|*Type:*|int|
  
- **nn\_size\_scale\_factor**  
  
|||
|-|-|
|*Description:*|Scaling factor for constraining network size based on number of parameters in the network for hyperparam search|
|*Default:*|1|
|*Type:*|float|
  
- **python\_path**  
  
|||
|-|-|
|*Description:*|Path to desired python version|
|*Default:*|This defaults to the Python instllation used to parse the JSON file. This is done by using sys.executable|
  
- **rerun**  
  
|||
|-|-|
|*Description:*| After parameter combos have been generated, `rerun=False` will check the model tracker to see if a model with a particular param combination has already been built. If it’s been built, do not create a new model or submit a slurm job. If `rerun=True`, the check will be skipped completely and a slurm job will be submitted regardless of whether a model has previously been built with these parameters. Specific to hyperparameter search.|
|*Default:*|TRUE|
|*Type:*|Bool|
  
- **script\_dir**  
  
|||
|-|-|
|*Description:*|Path where pipeline file you want to run hyperparam search from is located|
|*Default:*|.|
  
- **search\_type**  
  
|||
|-|-|
|*Description:*|Type of hyperparameter search to do. Options = [grid, random, geometric, hyperopt and user\_specified]|
|*Default:*|grid|
  
- **shortlist\_key**  
  
|||
|-|-|
|*Description:*|CSV file of assays of interest. Specific to LLNL model tracker system.|

- **slurm\_account**  
  
|||
|-|-|
|*Description:*|SLURM account to charge hyperparameter batch runs to. This will replace the lc\_account option. If lc\_account and slurm\_account are both set, slurm\_account will be used. If set to None then this parameter will not be used.|
|*Default:*|None|
  
- **slurm\_export**  
  
|||
|-|-|
|*Description:*|SLURM environment variables propagated for hyperparameter search batch jobs. If set to None then this parameter will not be used.|
|*Default:*|ALL|
  
- **slurm\_nodes**  
  
|||
|-|-|
|*Description:*|Number of nodes for hyperparameter search batch jobs. If set to None then this parameter will not be used.|
|*Default:*|1|
|*Type:*|int|
  
- **slurm\_options**  
  
|||
|-|-|
|*Description:*|Additional SLURM options for hyperparameter search batch jobs. Example: '--option1=value1 --option2=value2'. If set to None then this parameter will not be used.|
|*Default:*|None|
  
- **slurm\_partition**  
  
|||
|-|-|
|*Description:*|SLURM partition to run hyperparameter batch runs on. If set to None then this parameter will not be used.|
|*Default:*|pbatch|
  
- **slurm\_time\_limit**  
  
|||
|-|-|
|*Description:*|Time limit in minutes for hyperparameter search batch jobs.|
|*Default:*|1440|
|*Type:*|int|
  
- **split\_only**  
  
|||
|-|-|
|*Description:*|Boolean flag used with model\_pipeline.py to indicate splitting of the datasets when running the hyperparameter search|
|*Default:*|FALSE|
|*Type:*|bool|
  
- **use\_shortlist**  
  
|||
|-|-|
|*Description:*|Use a list of assays. Specific to LLNL model tracker system.|
|*Default:*|FALSE|
|*Type:*|Bool|
  
<a name="bayesian-optimization"></a>
## Bayesian Optimization  
### Search Domain Specifications
The following parameters are used to specify the search domains for certain model parameters in a Bayesian hyperparameter optimization. Each search domain parameter is
tied to a specific model parameter. Only a subset of model parameters may be optimized in this way, but more will be supported in future releases. See the hyperopt package documentation at https://github.com/hyperopt/hyperopt/wiki/FMin#2-defining-a-search-space to learn more about the search domain format.

- **lr**  
  
|||
|-|-|
|*Description:*|Search domain for NN model `learning_rate` parameter in Bayesian Optimization. The format is `scheme\|parameters`, e.g. `choice\|0.0001,0.0005,0.0002,0.001`. See https://github.com/ATOMScience-org/AMPL#hyperparameter-optimization|
|*Default:*|None|

- **dp**  
  
|||
|-|-|
|*Description:*|Search domain for NN model `dropouts` parameter in Bayesian Optimization. The format is `scheme\|num_layers\|parameters`, e.g. `uniform\|3\|0,0.4`, Note that the number of layers (number between two \|) can not be changed during optimization, if you want to try different number of layers, just run several optimizations.
|*Default:*|None|

- **ls**  
  
|||
|-|-|
|*Description:*|Search domain for NN model `layer_sizes` parameter in Bayesian Optimization. The format is `scheme\|num_layers\|parameters`, e.g. `uniformint\|3\|8,512`, Note that the number of layers (number between two \|) can not be changed during optimization, if you want to try different number of layers, just run several optimizations.
|*Default:*|None|

- **ls_ratio**  
  
|||
|-|-|
|*Description:*|Alternative method to set search domain for NN model `layer_sizes` parameter in Bayesian Optimization by specifying layer_size/previous_layer_size ratios. The format is `scheme\|ratios`, e.g. `uniform\|0.1,0.9`; the number of layers and starting layer sizes are taken from the `ls` parameter.
|*Default:*|None|

- **wdp**  
  
|||
|-|-|
|*Description:*|Search domain for NN model `weight_decay_penalty` parameter in Bayesian Optimization. The format is `scheme\|parameters`, e.g. `loguniform\|-6.908,-4.605`.
|*Default:*|None|

- **wdt**  
  
|||
|-|-|
|*Description:*|Search domain for NN model `weight_decay_penalty_type` parameter in Bayesian Optimization. The format is `scheme\|parameters`, e.g. `choice\|l1,l2`.
|*Default:*|None|

- **rfe**  
  
|||
|-|-|
|*Description:*|Search domain for RF model `rf_num_estimators` parameter in Bayesian Optimization. The format is `scheme\|parameters`, e.g. `uniformint\|8,512`.
|*Default:*|None|

- **rfd**  
  
|||
|-|-|
|*Description:*|Search domain for RF model `rf_max_depth` parameter in Bayesian Optimization. The format is `scheme\|parameters`, e.g. `uniformint\|8,512`.
|*Default:*|None|

- **rff**  
  
|||
|-|-|
|*Description:*|Search domain for RF model `rf_max_features` parameter in Bayesian Optimization. The format is `scheme\|parameters`, e.g. `uniformint\|8,200`.
|*Default:*|None|

- **xgbg**  
  
|||
|-|-|
|*Description:*|Search domain for XGBoost model `xgb_gamma` parameter in Bayesian Optimization. The format is `scheme\|parameters`, e.g. `loguniform\|-9.2,-4.6`.
|*Default:*|None|

- **xgba**  
  
|||
|-|-|
|*Description:*|Search domain for XGBoost model `xgb_alpha` parameter in Bayesian Optimization. The format is `scheme\|parameters`, e.g. `uniform\|0,0.4`.
|*Default:*|None|

- **xgbb**  
  
|||
|-|-|
|*Description:*|Search domain for XGBoost model `xgb_lambda` parameter in Bayesian Optimization. The format is `scheme\|parameters`, e.g. `uniform\|0,0.4`.
|*Default:*|None|

- **xgbl**  
  
|||
|-|-|
|*Description:*|Search domain for XGBoost model `xgb_learning_rate` parameter in Bayesian Optimization. The format is `scheme\|parameters`, e.g. `loguniform\|-4.6,-2.3`.
|*Default:*|None|

- **xgbd**  
  
|||
|-|-|
|*Description:*|Search domain for XGBoost model `xgb_max_depth` parameter in Bayesian Optimization. The format is `scheme\|parameters`, e.g. `uniformint\|3,10`.
|*Default:*|None|

- **xgbc**  
  
|||
|-|-|
|*Description:*|Search domain for XGBoost model `xgb_colsample_bytree` parameter in Bayesian Optimization. The format is `scheme\|parameters`, e.g. `uniform\|0.1,1.0`.
|*Default:*|None|

- **xgbs**  
  
|||
|-|-|
|*Description:*|Search domain for XGBoost model `xgb_subsample` parameter in Bayesian Optimization. The format is `scheme\|parameters`, e.g. `uniform\|0.1,1.0`.
|*Default:*|None|

- **xgbn**  
  
|||
|-|-|
|*Description:*|Search domain for XGBoost model `xgb_n_estimators` parameter in Bayesian Optimization. The format is `scheme\|parameters`, e.g. `uniformint\|200,1000`.
|*Default:*|None|

- **xgbw**  
  
|||
|-|-|
|*Description:*|Search domain for XGBoost model `xgb_min_child_weight` parameter in Bayesian Optimization. The format is `scheme\|parameters`, e.g. `uniform\|0.5,2.0`.
|*Default:*|None|

### Checkpointing parameters

- **hp_checkpoint_save**  
  
|||
|-|-|
|*Description:*|binary file to save a checkpoint of the HPO trial project, which can be use to continue the HPO search later.
|*Default:*|None|c

- **hp_checkpoint_load**  
  
|||
|-|-|
|*Description:*|binary file to load a checkpoint of a previous HPO trial project, to continue the HPO search.
|*Default:*|None|
