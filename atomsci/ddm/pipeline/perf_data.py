#!/usr/bin/env python

"""Contains class PerfData and its subclasses, which are objects for collecting and computing model performance metrics
and predictions
"""


import deepchem as dc
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, average_precision_score, precision_score, recall_score
from sklearn.metrics import accuracy_score, matthews_corrcoef, cohen_kappa_score, log_loss, balanced_accuracy_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from atomsci.ddm.pipeline import transformations as trans



# ******************************************************************************************************************************
def rms_error(y_real, y_pred):
    """Calculates the root mean squared error. Score function used for model selection.

    Args:
       y_real (np.array): Array of ground truth values

       y_pred (np.array): Array of predicted values

    Returns:
       (np.array): root mean squared error of the input

    """
    return np.sqrt(mean_squared_error(y_real, y_pred))

# ---------------------------------------------
def negative_predictive_value(y_real, y_pred):
    """Computes negative predictive value of a binary classification model: NPV = TN/(TN+FN).

    Args:
        y_real (np.array): Array of ground truth values

        y_pred (np.array): Array of predicted values

    Returns:
        (float): The negative predictive value

    """
    TN = sum((y_pred == 0) & (y_real == 0))
    FN = sum((y_pred == 0) & (y_real == 1))
    if TN + FN == 0:
        return 0.0
    else:
        return float(TN)/float(TN+FN)

# ******************************************************************************************************************************

# params.model_choice_score_type must be a key in one of the dictionaries below:
regr_score_func = dict(r2 = r2_score, mae = mean_absolute_error, rmse = rms_error)
classif_score_func = dict(roc_auc = roc_auc_score, precision = precision_score, ppv = precision_score, recall = recall_score,
                          npv = negative_predictive_value, cross_entropy = log_loss, accuracy = accuracy_score, bal_accuracy = balanced_accuracy_score,
                          avg_precision = average_precision_score, mcc = matthews_corrcoef, kappa = cohen_kappa_score)

# The following score types are loss functions, meaning the result must be sign flipped so we can maximize it in model selection
loss_funcs = {'mae', 'rmse', 'cross_entropy'}

# The following score types for classification require predicted class probabilities rather than class labels as input
uses_class_probs = {'roc_auc', 'avg_precision', 'cross_entropy'}

# The following classification score types have an 'average' parameter to control how multilabel scores are combined
has_average_param = {'roc_auc', 'avg_precision', 'precision', 'recall'}

# The following classification score types allow the value 'binary' for the 'average' parameter to make them report scores
# for class 1 only
binary_average_param = {'precision', 'recall'}

# The following classification score types only support binary classifiers
binary_class_only = {'npv'}

# ******************************************************************************************************************************
def create_perf_data(prediction_type, model_dataset, transformers, subset, **kwargs):
    """Factory function that creates the right kind of PerfData object for the given subset,
    prediction_type (classification or regression) and split strategy (k-fold or train/valid/test).

    Args:
        prediction_type (str): classification or regression.
        
        model_dataset (ModelDataset): Object representing the full dataset.
        
        transformers (list): A list of transformer objects.
        
        subset (str): Label in ['train', 'valid', 'test', 'full'], indicating the type of subset of dataset for tracking predictions
        
        **kwargs: Additional PerfData subclass arguments
        
    Returns:
        PerfData object
        
    Raises:
        ValueError: if split_strategy not in ['train_valid_test','k_fold_cv']
        ValueError: prediction_type not in ['regression','classification']
    """
    if subset == 'full':
        split_strategy = 'train_valid_test'
    else:
        split_strategy = model_dataset.params.split_strategy
    if prediction_type == 'regression':
        if subset == 'full' or split_strategy == 'train_valid_test':
            # Called simple because no need to track compound IDs across multiple training folds
            return SimpleRegressionPerfData(model_dataset, transformers, subset, **kwargs)
        elif split_strategy == 'k_fold_cv':
            return KFoldRegressionPerfData(model_dataset, transformers, subset, **kwargs)
        else:
            raise ValueError('Unknown split_strategy %s' % split_strategy)
    elif prediction_type == 'classification':
        if subset == 'full' or split_strategy == 'train_valid_test':
            return SimpleClassificationPerfData(model_dataset, transformers, subset, **kwargs)
        elif split_strategy == 'k_fold_cv':
            return KFoldClassificationPerfData(model_dataset, transformers, subset, **kwargs)
        else:
            raise ValueError('Unknown split_strategy %s' % split_strategy)
    elif prediction_type == "hybrid":
        return SimpleHybridPerfData(model_dataset, transformers, subset, **kwargs)
    else:
        raise ValueError('Unknown prediction type %s' % prediction_type)

# ****************************************************************************************
class PerfData(object):
    """Class with methods for accumulating prediction data over multiple cross-validation folds
    and computing performance metrics after all folds have been run. Abstract class with
    concrete subclasses for classification and regression models.
    """
    # ****************************************************************************************
    def __init__(self, model_dataset, subset):
        """Initialize any attributes that are common to all PerfData subclasses"""

    # ****************************************************************************************
    def accumulate_preds(self, predicted_vals, ids, pred_stds=None):
        """Raises:
            NotImplementedError: The method is implemented by subclasses
        """
        raise NotImplementedError

    # ****************************************************************************************
    def get_pred_values(self):
        """Raises:
            NotImplementedError: The method is implemented by subclasses
        """
        raise NotImplementedError

    # ****************************************************************************************
    def get_real_values(self, ids=None):
        """Raises:
            NotImplementedError: The method is implemented by subclasses
        """
        raise NotImplementedError

    # ****************************************************************************************
    def get_weights(self, ids=None):
        """Returns the dataset response weights as an (ncmpds, ntasks) array

        Raises:
            NotImplementedError: The method is implemented by subclasses
        """
        raise NotImplementedError


    # ****************************************************************************************
    def compute_perf_metrics(self, per_task=False):
        """Raises:
            NotImplementedError: The method is implemented by subclasses
        """
        raise NotImplementedError

    # ****************************************************************************************
    def get_prediction_results(self):
        """Raises:
            NotImplementedError: The method is implemented by subclasses
        """
        raise NotImplementedError

    # ****************************************************************************************
    def _reshape_preds(self, predicted_vals):
        """Raises:
            NotImplementedError: The method is implemented by subclasses
        """
        raise NotImplementedError


# ****************************************************************************************
class RegressionPerfData(PerfData):
    """Class with methods for accumulating regression model prediction data over multiple
    cross-validation folds and computing performance metrics after all folds have been run.
    Abstract class with concrete subclasses for different split strategies.

    Attributes:
        set in __init__
            num_tasks (int): Set to None, the number of tasks

            num_cmpds (int): Set to None, the number of compounds

    """
    # ****************************************************************************************
    # class RegressionPerfData
    def __init__(self, model_dataset, subset):
        """Initialize any attributes that are common to all RegressionPerfData subclasses.

        Side effects:
            num_tasks (int) is set as a RegressionPerfData attribute

            num_cmps (int) is set as a RegressionPerfData attribute

        """
        # The code below is to document the atributes that methods in this class expect the
        # subclasses to define. Subclasses don't actually call this superclass method.
        self.num_tasks = None
        self.num_cmpds = None
        self.perf_metrics = []
        self.model_score = None
        self.weights = None

    # ****************************************************************************************
    def accumulate_preds(self, predicted_vals, ids, pred_stds=None):
        """Raises:
            NotImplementedError: The method is implemented by subclasses
        """
        raise NotImplementedError

    # ****************************************************************************************
    def get_pred_values(self):
        """Raises:
            NotImplementedError: The method is implemented by subclasses
        """
        raise NotImplementedError

    # ****************************************************************************************
    def compute_perf_metrics(self, per_task=False):
        """Raises:
            NotImplementedError: The method is implemented by subclasses
        """
        raise NotImplementedError


    # ****************************************************************************************
    # class RegressionPerfData
    def model_choice_score(self, score_type='r2'):
        """Computes a score function based on the accumulated predicted values, to be used for selecting
        the best training epoch and other hyperparameters.

        Args:
            score_type (str): The name of the scoring metric to be used, e.g. 'r2',
                              'neg_mean_squared_error', 'neg_mean_absolute_error', etc.; see
                              https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
                              and sklearn.metrics.SCORERS.keys() for a complete list of options.
                              Larger values of the score function indicate better models.

        Returns:
            score (float): A score function value. For multitask models, this will be averaged
                           over tasks.

        """
        ids, pred_vals, stds = self.get_pred_values()
        real_vals = self.get_real_values(ids)
        weights = self.get_weights(ids)
        scores = []
        for i in range(self.num_tasks):
            nzrows = np.where(weights[:,i] != 0)[0]
            task_real_vals = np.squeeze(real_vals[nzrows,i])
            task_pred_vals = np.squeeze(pred_vals[nzrows,i])
            scores.append(regr_score_func[score_type](task_real_vals, task_pred_vals))

        self.model_score = float(np.mean(scores))
        if score_type in loss_funcs:
            self.model_score = -self.model_score
        return self.model_score


    # ****************************************************************************************
    # class RegressionPerfData
    def get_prediction_results(self):
        """Returns a dictionary of performance metrics for a regression model.
        The dictionary values should contain only primitive Python types, so that it can
        be easily JSONified.

        Args:
            per_task (bool): True if calculating per-task metrics, False otherwise.

        Returns:
            pred_results (dict): dictionary of performance metrics for a regression model.

        """
        pred_results = {}

        # Get the mean and SD of R^2 scores over folds. If only single fold training was done, the SD will be None.
        r2_means, r2_stds = self.compute_perf_metrics(per_task=True)
        pred_results['r2_score'] = float(np.mean(r2_means))
        if r2_stds is not None:
            pred_results['r2_std'] = float(np.sqrt(np.mean(r2_stds ** 2)))
        if self.num_tasks > 1:
            pred_results['task_r2_scores'] = r2_means.tolist()
            if r2_stds is not None:
                pred_results['task_r2_stds'] = r2_stds.tolist()

        # Compute some other performance metrics. We do these differently than R^2, in that we compute the
        # metrics from the average predicted values, rather than computing them separately for each fold
        # and then averaging the metrics. If people start asking for SDs of MAE and RMSE scores over folds,
        # we'll change the code to compute all metrics the same way.

        (ids, pred_vals, pred_stds) = self.get_pred_values()
        real_vals = self.get_real_values(ids)
        weights = self.get_weights(ids)
        mae_scores = []
        rms_scores = []
        response_means = []
        response_stds = []
        # Iterate over tasks, call score funcs directly on weight masked values
        for i in range(self.num_tasks):
            nzrows = np.where(weights[:,i] != 0)[0]
            task_real_vals = np.squeeze(real_vals[nzrows,i])
            task_pred_vals = np.squeeze(pred_vals[nzrows,i])
            mae_scores.append(regr_score_func['mae'](task_real_vals, task_pred_vals))
            rms_scores.append(regr_score_func['rmse'](task_real_vals, task_pred_vals))
            response_means.append(task_real_vals.mean().tolist())
            response_stds.append(task_real_vals.std().tolist())
        pred_results['mae_score'] = float(np.mean(mae_scores))
        if self.num_tasks > 1:
            pred_results['task_mae_scores'] = mae_scores
        pred_results['rms_score'] = float(np.mean(rms_scores))
        if self.num_tasks > 1:
            pred_results['task_rms_scores'] = rms_scores

        # Add model choice score if one was computed
        if self.model_score is not None:
            pred_results['model_choice_score'] = self.model_score

        pred_results['num_compounds'] = self.num_cmpds
        pred_results['mean_response_vals'] = response_means
        pred_results['std_response_vals'] = response_stds

        return pred_results

    # ****************************************************************************************
    # class RegressionPerfData
    def _reshape_preds(self, predicted_vals):
        """Reshape an array of regression model predictions to a standard (ncmpds, ntasks)
        format. Checks that the task dimension matches what we expect for the dataset.

        Args:
            predicted_vals (np.array): array of regression model predictions.

        Returns:
            predicted_vals (np.array): reshaped array

        Raises:
            ValueError: if the dimensions of the predicted value do not match the dimensions of num_tasks for
            RegressionPerfData

        """
        # For regression models, predicted_vals can be 1D, 2D or 3D array depending on the type of
        # underlying DeepChem model.
        dim = len(predicted_vals.shape)
        ncmpds = predicted_vals.shape[0]
        if dim == 1:
            # Single task model
            predicted_vals = predicted_vals.reshape((ncmpds,1))
            ntasks = 1
        else:
            ntasks = predicted_vals.shape[1]
        if ntasks != self.num_tasks:
            raise ValueError("Predicted value dimensions don't match num_tasks for RegressionPerfData")
        if dim == 3:
            # FCNet models generate predictions with an extra dimension, possibly for the number of
            # classes, which is always 1 for regression models.
            predicted_vals = predicted_vals.reshape((ncmpds,ntasks))
        return predicted_vals

    # ****************************************************************************************

# ****************************************************************************************
class HybridPerfData(PerfData):
    """Class with methods for accumulating regression model prediction data over multiple
    cross-validation folds and computing performance metrics after all folds have been run.
    Abstract class with concrete subclasses for different split strategies.

    Attributes:
        set in __init__
            num_tasks (int): Set to None, the number of tasks

            num_cmpds (int): Set to None, the number of compounds

    """
    # ****************************************************************************************
    # class HybridPerfData
    def __init__(self, model_dataset, subset):
        """Initialize any attributes that are common to all HybridPerfData subclasses.

        Side effects:
            num_tasks (int) is set as a HybridPerfData attribute

            num_cmps (int) is set as a HybridPerfData attribute

        """
        # The code below is to document the atributes that methods in this class expect the
        # subclasses to define. Subclasses don't actually call this superclass method.
        self.num_tasks = 2
        self.num_cmpds = None
        self.perf_metrics = []
        self.model_score = None
        self.weights = None

    # ****************************************************************************************
    def accumulate_preds(self, predicted_vals, ids, pred_stds=None):
        """Raises:
            NotImplementedError: The method is implemented by subclasses
        """
        raise NotImplementedError

    # ****************************************************************************************
    def get_pred_values(self):
        """Raises:
            NotImplementedError: The method is implemented by subclasses
        """
        raise NotImplementedError

    # ****************************************************************************************
    def compute_perf_metrics(self, per_task=False):
        """Raises:
            NotImplementedError: The method is implemented by subclasses
        """
        raise NotImplementedError


    # ****************************************************************************************
    # class HybridPerfData
    def model_choice_score(self, score_type='r2'):
        """Computes a score function based on the accumulated predicted values, to be used for selecting
        the best training epoch and other hyperparameters.

        Args:
            score_type (str): The name of the scoring metric to be used, e.g. 'r2', 'mae', 'rmse'

        Returns:
            score (float): A score function value. For multitask models, this will be averaged
                           over tasks.

        """
        ids, pred_vals, stds = self.get_pred_values()
        real_vals = self.get_real_values(ids)
        weights = self.get_weights(ids)
        scores = []
        
        pos_ki = np.where(np.isnan(real_vals[:, 1]))[0]
        pos_bind = np.where(~np.isnan(real_vals[:, 1]))[0]

        # score for pKi/IC50
        nzrows = np.where(weights[:, 0] != 0)[0]
        rowki = np.intersect1d(nzrows, pos_ki)
        rowbind = np.intersect1d(nzrows, pos_bind)
        ki_real_vals = np.squeeze(real_vals[rowki,0])
        ki_pred_vals = np.squeeze(pred_vals[rowki,0])
        bind_real_vals = np.squeeze(real_vals[rowbind,0])
        bind_pred_vals = np.squeeze(pred_vals[rowbind,0])
        if len(rowki) > 0:
            scores.append(regr_score_func[score_type](ki_real_vals, ki_pred_vals))
            if len(rowbind) > 0:
                scores.append(regr_score_func[score_type](bind_real_vals, bind_pred_vals))
            else:
                # if all values are dose response activities, use the r2_score above.
                scores.append(scores[0])
        elif len(rowbind) > 0:
            # all values are single concentration activities.
            scores.append(regr_score_func[score_type](bind_real_vals, bind_pred_vals))
            scores.append(scores[0])

        self.model_score = float(np.mean(scores))
        if score_type in loss_funcs:
            self.model_score = -self.model_score
        return self.model_score


    # ****************************************************************************************
    # class HybridPerfData
    def get_prediction_results(self):
        """Returns a dictionary of performance metrics for a regression model.
        The dictionary values should contain only primitive Python types, so that it can
        be easily JSONified.

        Args:
            per_task (bool): True if calculating per-task metrics, False otherwise.

        Returns:
            pred_results (dict): dictionary of performance metrics for a regression model.

        """
        pred_results = {}

        # Get the mean and SD of R^2 scores over folds. If only single fold training was done, the SD will be None.
        r2_means, r2_stds = self.compute_perf_metrics(per_task=True)
        pred_results['r2_score'] = float(np.mean(r2_means))
        if r2_stds is not None:
            pred_results['r2_std'] = float(np.sqrt(np.mean(r2_stds ** 2)))
        if self.num_tasks > 1:
            pred_results['task_r2_scores'] = r2_means.tolist()
            if r2_stds is not None:
                pred_results['task_r2_stds'] = r2_stds.tolist()

        # Compute some other performance metrics. We do these differently than R^2, in that we compute the
        # metrics from the average predicted values, rather than computing them separately for each fold
        # and then averaging the metrics. If people start asking for SDs of MAE and RMSE scores over folds,
        # we'll change the code to compute all metrics the same way.

        (ids, pred_vals, pred_stds) = self.get_pred_values()
        real_vals = self.get_real_values(ids)
        weights = self.get_weights(ids)
        mae_scores = []
        rms_scores = []
        response_means = []
        response_stds = []
        
        pos_ki = np.where(np.isnan(real_vals[:, 1]))[0]
        pos_bind = np.where(~np.isnan(real_vals[:, 1]))[0]

        # score for pKi/IC50
        nzrows = np.where(weights[:, 0] != 0)[0]
        rowki = np.intersect1d(nzrows, pos_ki)
        rowbind = np.intersect1d(nzrows, pos_bind)
        ki_real_vals = np.squeeze(real_vals[rowki,0])
        ki_pred_vals = np.squeeze(pred_vals[rowki,0])
        bind_real_vals = np.squeeze(real_vals[rowbind,0])
        bind_pred_vals = np.squeeze(pred_vals[rowbind,0])
        if len(rowki) > 0:
            mae_scores.append(regr_score_func['mae'](ki_real_vals, ki_pred_vals))
            rms_scores.append(regr_score_func['rmse'](ki_real_vals, ki_pred_vals))
            if len(rowbind) > 0:
                mae_scores.append(regr_score_func['mae'](bind_real_vals, bind_pred_vals))
                rms_scores.append(regr_score_func['rmse'](bind_real_vals, bind_pred_vals))
            else:
                # if all values are dose response activities, use the r2_score above.
                mae_scores.append(mae_scores[0])
                rms_scores.append(rms_scores[0])
        elif len(rowbind) > 0:
            # all values are single concentration activities.
            mae_scores.append(regr_score_func['mae'](bind_real_vals, bind_pred_vals))
            rms_scores.append(regr_score_func['rmse'](bind_real_vals, bind_pred_vals))
            mae_scores.append(mae_scores[0])
            rms_scores.append(rms_scores[0])

        response_means.append(ki_real_vals.mean().tolist())
        response_stds.append(ki_real_vals.std().tolist())
        response_means.append(bind_real_vals.mean().tolist())
        response_stds.append(bind_real_vals.std().tolist())

        pred_results['mae_score'] = float(np.mean(mae_scores))
        if self.num_tasks > 1:
            pred_results['task_mae_scores'] = mae_scores
        pred_results['rms_score'] = float(np.mean(rms_scores))
        if self.num_tasks > 1:
            pred_results['task_rms_scores'] = rms_scores

        # Add model choice score if one was computed
        if self.model_score is not None:
            pred_results['model_choice_score'] = self.model_score

        pred_results['num_compounds'] = self.num_cmpds
        pred_results['mean_response_vals'] = response_means
        pred_results['std_response_vals'] = response_stds

        return pred_results

    # ****************************************************************************************
    # class HybridPerfData
    def _reshape_preds(self, predicted_vals):
        """Reshape an array of regression model predictions to a standard (ncmpds, ntasks)
        format. Checks that the task dimension matches what we expect for the dataset.

        Args:
            predicted_vals (np.array): array of regression model predictions.

        Returns:
            predicted_vals (np.array): reshaped array

        Raises:
            ValueError: if the dimensions of the predicted value do not match the dimensions of num_tasks for
            RegressionPerfData

        """
        # hybrid model is highly specific, there is no need to reshape
        return predicted_vals

    # ****************************************************************************************

# ****************************************************************************************
class ClassificationPerfData(PerfData):
    """Class with methods for accumulating classification model prediction data over multiple
    cross-validation folds and computing performance metrics after all folds have been run.
    Abstract class with concrete subclasses for different split strategies.

    Attributes:
        set in __init__
            num_tasks (int): Set to None, the number of tasks

            num_cmpds (int): Set to None, the number of compounds

            num_classes (int): Set to None, the number of classes

    """
    # ****************************************************************************************
    # class ClassificationPerfData
    def __init__(self, model_dataset, subset):
        """Initialize any attributes that are common to all ClassificationPerfData subclasses

        Side effects:
            num_tasks (int) is set as a ClassificationPerfData attribute

            num_cmps (int) is set as a ClassificationPerfData attribute

            num_classes (int) is set as a ClassificationPerfData attribute

        """
        # TODO: Allow num_classes to vary between tasks in a multitask, multilabel model.
        # This would require making self.num_classes a list or array. Also, the _reshape_preds method
        # would have to change to accept and generate lists of (ncmpds, nclasses) arrays, rather than
        # the 3D (ncmpds, ntasks, nclasses) arrays generated by DeepChem predict() methods; and downstream
        # code would have to be modified to deal with these lists. Recommend we hold off dealing with this
        # until we have some multitask/label datasets and models where it will be necessary.

        # The code below is to document the atributes that methods in this class expect the
        # subclasses to define. Subclasses don't actually call this superclass method.
        self.num_tasks = None
        self.num_cmpds = None
        self.num_classes = None
        self.perf_metrics = []
        self.model_score = None
        self.weights = None

    # ****************************************************************************************
    def accumulate_preds(self, predicted_vals, ids, pred_stds=None):
        """Raises:
            NotImplementedError: The method is implemented by subclasses
        """
        raise NotImplementedError

    # ****************************************************************************************
    def get_pred_values(self):
        """Raises:
            NotImplementedError: The method is implemented by subclasses
        """
        raise NotImplementedError


    # ****************************************************************************************
    # class ClassificationPerfData
    def model_choice_score(self, score_type='roc_auc'):
        """Computes a score function based on the accumulated predicted values, to be used for selecting
        the best training epoch and other hyperparameters.

        Args:
            score_type (str): The name of the scoring metric to be used, e.g. 'roc_auc', 'precision',
                              'recall', 'f1'; see
                              https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
                              and sklearn.metrics.SCORERS.keys() for a complete list of options.
                              Larger values of the score function indicate better models.

        Returns:
            score (float): A score function value. For multitask models, this will be averaged
                           over tasks.

        """
        ids, pred_classes, class_probs, prob_stds = self.get_pred_values()
        real_vals = self.get_real_values()
        weights = self.get_weights()
        scores = []
            
        for i in range(self.num_tasks):
            nzrows = np.where(weights[:,i] != 0)[0]
            average_param = None
            if self.num_classes > 2:
                if score_type in binary_class_only:
                    raise ValueError("Model selection by %s score not allowed for multi-label classifiers." % score_type)
                if score_type in has_average_param:
                    average_param = 'macro'
                # If more than 2 classes, task_real_vals is indicator matrix (one-hot encoded). 
                task_real_vals = real_vals[nzrows,i,:]
                task_class_probs = class_probs[nzrows,i,:]
                task_real_classes = np.argmax(task_real_vals, axis=1)
                task_pred_classes = np.argmax(task_class_probs, axis=1)
            else:
                # sklearn metrics functions are expecting single array of 1s and 0s for task_real_vals
                # and task_class_probs for class 1 only
                task_real_vals = real_vals[nzrows,i]
                task_real_classes = task_real_vals
                task_class_probs = class_probs[nzrows,i,1]
                if score_type in binary_average_param:
                    average_param = 'binary'

            if score_type in uses_class_probs:
                task_pred_vars = task_class_probs
                task_real_vars = task_real_vals
            else:
                task_pred_vars = pred_classes[nzrows,i]
                task_real_vars = task_real_classes
            if average_param is None:
                scores.append(classif_score_func[score_type](task_real_vars, task_pred_vars))
            else:
                scores.append(classif_score_func[score_type](task_real_vars, task_pred_vars, average=average_param))

        self.model_score = float(np.mean(scores))
        if score_type in loss_funcs:
            self.model_score = -self.model_score
        return self.model_score

    # ****************************************************************************************
    # class ClassificationPerfData
    def get_prediction_results(self):
        """Returns a dictionary of performance metrics for a classification model.
        The dictionary values will contain only primitive Python types, so that it can
        be easily JSONified.

        Args:
            per_task (bool): True if calculating per-task metrics, False otherwise.

        Returns:
            pred_results (dict): dictionary of performance metrics for a classification model.

        """
        pred_results = {}
        (ids, pred_classes, class_probs, prob_stds) = self.get_pred_values()

        real_vals = self.get_real_values(ids)
        weights = self.get_weights(ids)
        if self.num_classes > 2:
            real_val_list = [real_vals[:,i,:] for i in range(self.num_tasks)]
            class_prob_list = [class_probs[:,i,:] for i in range(self.num_tasks)]
            real_classes = np.argmax(real_vals, axis=2)
        else:
            real_classes = real_vals
            real_val_list = [real_vals[:,i] for i in range(self.num_tasks)]
            class_prob_list = [class_probs[:,i,1] for i in range(self.num_tasks)]

        # Get the mean and SD of ROC AUC scores over folds. If only single fold training was done, the SD will be None.
        roc_auc_means, roc_auc_stds = self.compute_perf_metrics(per_task=True)
        pred_results['roc_auc_score'] = float(np.mean(roc_auc_means))
        if roc_auc_stds is not None:
            pred_results['roc_auc_std'] = float(np.sqrt(np.mean(roc_auc_stds ** 2)))
        if self.num_tasks > 1:
            pred_results['task_roc_auc_scores'] = roc_auc_means.tolist()
            if roc_auc_stds is not None:
                pred_results['task_roc_auc_stds'] = roc_auc_stds.tolist()

        # Compute some other performance metrics. We do these differently than ROC AUC, in that we compute the
        # metrics from the average predicted values, rather than computing them separately for each fold
        # and then averaging the metrics. If people start asking for SDs of the other metrics over folds,
        # we'll change the code to compute all metrics the same way.

        prc_aucs = []
        cross_entropies = []
        precisions = []
        recalls = []
        if self.num_classes == 2:
            npvs = []
        accuracies = []
        bal_accs = []
        kappas = []
        matthews_ccs = []
        confusion_matrices = []
        for i in range(self.num_tasks):
            nzrows = np.where(weights[:,i] != 0)[0]
            task_pred_classes = pred_classes[nzrows,i]
            task_real_classes = real_classes[nzrows,i]

            if self.num_classes > 2:
                # If more than 2 classes, task_real_vals is indicator matrix (one-hot encoded).
                task_real_vals = real_vals[nzrows,i,:]
                task_class_probs = class_probs[nzrows,i,:]
                prc_aucs.append(average_precision_score(task_real_vals, task_class_probs, average='macro'))
                precisions.append(float(precision_score(task_real_classes, task_pred_classes, average='macro')))
                recalls.append(float(recall_score(task_real_classes, task_pred_classes, average='macro')))
                # NPV is not supported for multilabel classifiers, skip it
            else:
                # sklearn metrics functions are expecting single array of 1s and 0s for task_real_vals
                # and task_class_probs for class 1 only
                task_real_vals = real_vals[nzrows,i]
                task_class_probs = class_probs[nzrows,i,1]
                prc_aucs.append(average_precision_score(task_real_vals, task_class_probs))
                precisions.append(float(precision_score(task_real_vals, task_pred_classes, average='binary')))
                recalls.append(float(recall_score(task_real_vals, task_pred_classes, average='binary')))
                npvs.append(negative_predictive_value(task_real_vals, task_pred_classes))

            cross_entropies.append(log_loss(task_real_vals, task_class_probs))
            accuracies.append(accuracy_score(task_real_classes, task_pred_classes))
            bal_accs.append(balanced_accuracy_score(task_real_classes, task_pred_classes))
            kappas.append(float(cohen_kappa_score(task_real_classes, task_pred_classes)))
            matthews_ccs.append(float(matthews_corrcoef(task_real_classes, task_pred_classes)))
            confusion_matrices.append(confusion_matrix(task_real_classes, task_pred_classes).tolist())

        pred_results['prc_auc_score'] = float(np.mean(prc_aucs))
        if self.num_tasks > 1:
            pred_results['task_prc_auc_scores'] = prc_aucs

        pred_results['cross_entropy'] = float(np.mean(cross_entropies))
        if self.num_tasks > 1:
            pred_results['task_cross_entropies'] = cross_entropies

        # Add model choice score if one was computed
        if self.model_score is not None:
            pred_results['model_choice_score'] = self.model_score

        pred_results['precision'] = float(np.mean(precisions))
        if self.num_tasks > 1:
            pred_results['task_precisions'] = precisions

        pred_results['recall_score'] = float(np.mean(recalls))
        if self.num_tasks > 1:
            pred_results['task_recalls'] = recalls

        if self.num_classes == 2:
            pred_results['npv'] = float(np.mean(npvs))
            if self.num_tasks > 1:
                pred_results['task_npvs'] = npvs

        pred_results['accuracy_score'] = float(np.mean(accuracies))
        if self.num_tasks > 1:
            pred_results['task_accuracies'] = accuracies

        pred_results['bal_accuracy'] = float(np.mean(bal_accs))
        if self.num_tasks > 1:
            pred_results['task_bal_accuracies'] = bal_accs

        pred_results['kappa'] = float(np.mean(kappas))
        if self.num_tasks > 1:
            pred_results['task_kappas'] = kappas

        pred_results['matthews_cc'] = float(np.mean(matthews_ccs))
        if self.num_tasks > 1:
            pred_results['task_matthews_ccs'] = matthews_ccs

        pred_results['confusion_matrix'] = confusion_matrices
        pred_results['num_compounds'] = self.num_cmpds

        return pred_results

    # ****************************************************************************************
    # class ClassificationPerfData
    def _reshape_preds(self, predicted_vals):
        """Reshape an array of classification model predictions to a standard (ncmpds, ntasks, nclasses)
        format. Checks that the task and class dimensions match what we expect for the dataset.

        Args:
            predicted_vals (np.array): array of classification model predictions

        Returns:
            predicted_vals (np.array): reshaped array of classification model predictions

        """
        # For classification models, predicted_vals can be 2D or 3D array depending on whether the
        # underlying DeepChem or sklearn model supports multitask datasets.
        dim = len(predicted_vals.shape)
        ncmpds = predicted_vals.shape[0]
        if dim == 2:
            # Single task model
            ntasks = 1
            nclasses = predicted_vals.shape[1]
            predicted_vals = predicted_vals.reshape((ncmpds, 1, nclasses))
        else:
            ntasks = predicted_vals.shape[1]
            nclasses = predicted_vals.shape[2]

        if nclasses != self.num_classes:
            raise ValueError("Predicted value dimensions doesn't match num_classes for ClassificationPerfData")
        if ntasks != self.num_tasks:
            raise ValueError("Predicted value dimensions doesn't match num_tasks for ClassificationPerfData")

        return predicted_vals


# ****************************************************************************************
class KFoldRegressionPerfData(RegressionPerfData):
    """Class with methods for accumulating regression model prediction data over multiple
    cross-validation folds and computing performance metrics after all folds have been run.

    Arguments:
        Set in __init__:
            subset (str): Label of the type of subset of dataset for tracking predictions

            num_cmps (int): The number of compounds in the dataset

            num_tasks (int): The number of tasks in the dataset

            pred-vals (dict): The dictionary of prediction results

            folds (int): Initialized at zero, flag for determining which k-fold is being assessed

            transformers (list of Transformer objects): from input arguments

            real_vals (dict): The dictionary containing the origin response column values

    """

    # ****************************************************************************************
    # class KFoldRegressionPerfData
    def __init__(self, model_dataset, transformers, subset, transformed=True):
        """# Initialize any attributes that are common to all KFoldRegressionPerfData subclasses
        Args:
            model_dataset (ModelDataset object): contains the dataset and related methods

            transformers (list of transformer objects): contains the list of transformers used to transform the dataset

            subset (str): Label in ['train', 'valid', 'test', 'full'], indicating the type of subset of dataset for
            tracking predictions

            transformed (bool): True if values to be passed to accumulate preds function are transformed values

        Side effects:
            Sets the following attributes of KFoldRegressionPerfData:
                subset (str): Label of the type of subset of dataset for tracking predictions

                num_cmps (int): The number of compounds in the dataset

                num_tasks (int): The number of tasks in the dataset

                pred_vals (dict): The dictionary of prediction results

                folds (int): Initialized at zero, flag for determining which k-fold is being assessed

                transformers (list of Transformer objects): from input arguments

                real_vals (dict): The dictionary containing the origin response column values

        """
        self.subset = subset
        if self.subset in ('train', 'valid', 'train_valid'):
            dataset = model_dataset.combined_training_data()
        elif self.subset == 'test':
            dataset = model_dataset.test_dset
        else:
            raise ValueError('Unknown dataset subset type "%s"' % self.subset)
        self.num_cmpds = dataset.y.shape[0]
        self.num_tasks = dataset.y.shape[1]
        self.pred_vals = dict([(id, np.empty((0, self.num_tasks), dtype=np.float32)) for id in dataset.ids])
        self.folds = 0
        self.perf_metrics = []
        self.model_score = None
        # Want predictions and real values to be in the same space, either transformed or untransformed
        if transformed:
            # Predictions passed to accumulate_preds() will be transformed
            self.real_vals, self.weights = model_dataset.get_subset_responses_and_weights(self.subset, [])
            self.transformers = transformers
        else:
            # If these were never transformed, transformers will be [], which is fine with undo_transforms
            self.real_vals, self.weights = model_dataset.get_subset_responses_and_weights(self.subset, transformers)
            self.transformers = []


    # ****************************************************************************************
    # class KFoldRegressionPerfData
    def accumulate_preds(self, predicted_vals, ids, pred_stds=None):
        """Add training, validation or test set predictions from the current fold to the data structure
        where we keep track of them.

        Args:
            predicted_vals (np.array): Array of the predicted values for the current dataset

            ids (np.array): An np.array of compound ids for the current dataset

            pred_stds (np.array): An array of the standard deviation in the predictions, not used in this method

        Returns:
            None

        Raises:
            ValueError: If Predicted value dimensions don't match num_tasks for RegressionPerfData

        Side effects:
            Overwrites the attribute pred_vals

            Increments folds by 1

        """
        # For regression models, predicted_vals can be 1D, 2D or 3D array depending on the type of
        # underlying DeepChem model. Reshape the array to (ncmpds, ntasks) regardless.
        dim = len(predicted_vals.shape)
        ncmpds = predicted_vals.shape[0]
        if dim == 1:
            # Single task model
            predicted_vals = predicted_vals.reshape((ncmpds,1))
            ntasks = 1
        else:
            ntasks = predicted_vals.shape[1]
        if ntasks != self.num_tasks:
            raise ValueError("Predicted value dimensions don't match num_tasks for RegressionPerfData")
        if dim == 3:
            # FCNet models generate predictions with an extra dimension, possibly for the number of
            # classes, which is always 1 for regression models.
            predicted_vals = predicted_vals.reshape((ncmpds,ntasks))

        for i, id in enumerate(ids):
            self.pred_vals[id] = np.concatenate([self.pred_vals[id], predicted_vals[i,:].reshape((1,-1))], axis=0)
        self.folds += 1

        pred_vals = dc.trans.undo_transforms(predicted_vals, self.transformers)

        real_vals = self.get_real_values(ids)
        weights = self.get_weights(ids)
        scores = []
        for i in range(self.num_tasks):
            nzrows = np.where(weights[:,i] != 0)[0]
            task_real_vals = np.squeeze(real_vals[nzrows,i])
            task_pred_vals = np.squeeze(pred_vals[nzrows,i])
            scores.append(regr_score_func['r2'](task_real_vals, task_pred_vals))
        self.perf_metrics.append(np.array(scores))
        return float(np.mean(scores))


    # ****************************************************************************************
    # class KFoldRegressionPerfData
    def get_pred_values(self):
        """Returns the predicted values accumulated over training, with any transformations undone.
        If self.subset is 'train' or 'test', the function will return averages over the training folds for each compound
        along with standard deviations when there are predictions from multiple folds. Otherwise, returns a
        single predicted value for each compound.

        Returns:
            ids (np.array): list of compound IDs

            vals (np.array): (ncmpds, ntasks) array of mean predicted values

            fold_stds (np.array): (ncmpds, ntasks) array of standard deviations over folds if applicable, and None
            otherwise.

        """
        ids = sorted(self.pred_vals.keys())
        if self.subset in ['train', 'test', 'train_valid']:
            rawvals = np.concatenate([self.pred_vals[id].mean(axis=0, keepdims=True).reshape((1,-1)) for id in ids])
            vals = dc.trans.undo_transforms(rawvals, self.transformers)
            if self.folds > 1:
                stds = dc.trans.undo_transforms(np.concatenate([self.pred_vals[id].std(axis=0, keepdims=True).reshape((1,-1))
                                       for id in ids]), self.transformers)
            else:
                stds = None
        else:
            rawvals = np.concatenate([self.pred_vals[id].reshape((1,-1)) for id in ids], axis=0)
            vals = dc.trans.undo_transforms(rawvals, self.transformers)
            stds = None
        return (ids, vals, stds)


    # ****************************************************************************************
    # class KFoldRegressionPerfData
    def get_real_values(self, ids=None):
        """Returns the real dataset response values, with any transformations undone, as an (ncmpds, ntasks) array
        in the same ID order as get_pred_values() (unless ids is specified).

        Args:
            ids (list of str): Optional list of compound IDs to return values for.

        Returns:
            np.array (ncmpds, ntasks) of the real dataset response values, with any transformations undone, in the same
            ID order as get_pred_values().

        """
        if ids is None:
            ids = sorted(self.pred_vals.keys())
        real_vals = np.concatenate([self.real_vals[id].reshape((1,-1)) for id in ids], axis=0)
        return dc.trans.undo_transforms(real_vals, self.transformers)


    # ****************************************************************************************
    # class KFoldRegressionPerfData
    def get_weights(self, ids=None):
        """Returns the dataset response weights, as an (ncmpds, ntasks) array
        in the same ID order as get_pred_values() (unless ids is specified).

        Args:
            ids (list of str): Optional list of compound IDs to return values for.

        Returns:
            np.array (ncmpds, ntasks) of the real dataset response weights, in the same
            ID order as get_pred_values().

        """
        if ids is None:
            ids = sorted(self.pred_vals.keys())
        return np.concatenate([self.weights[id].reshape((1,-1)) for id in ids], axis=0)



    # ****************************************************************************************
    # class KFoldRegressionPerfData
    def compute_perf_metrics(self, per_task=False):
        """Computes the R-squared metrics for each task based on the accumulated values, averaged over
        training folds, along with standard deviations of the scores. If per_task is False, the scores
        are averaged over tasks and the overall standard deviation is reported instead.

        Args:
            per_task (bool): True if calculating per-task metrics, False otherwise.

        Returns:
            A tuple (r2_mean, r2_std):

                r2_mean: A numpy array of mean R^2 scores for each task, averaged over folds, if per_task is True.
                         Otherwise, a float giving the R^2 score averaged over both folds and tasks.

                r2_std:  A numpy array of standard deviations over folds of R^2 values, if per_task is True.
                         Otherwise, a float giving the overall standard deviation.
        """
        r2_scores = np.stack(self.perf_metrics)
        if per_task:
            r2_mean = np.mean(r2_scores, axis=0)
            r2_std = np.std(r2_scores, axis=0)
        else:
            r2_mean = np.mean(r2_scores.flatten())
            r2_std = np.std(r2_scores.flatten())
        return (r2_mean, r2_std)


# ****************************************************************************************
class KFoldClassificationPerfData(ClassificationPerfData):
    """Class with methods for accumulating classification model performance data over multiple
    cross-validation folds and computing performance metrics after all folds have been run.

    Attributes:
        Set in __init__:
            subset (str): Label of the type of subset of dataset for tracking predictions
            num_cmps (int): The number of compounds in the dataset
            num_tasks (int): The number of tasks in the dataset
            pred-vals (dict): The dictionary of prediction results
            folds (int): Initialized at zero, flag for determining which k-fold is being assessed
            transformers (list of Transformer objects): from input arguments
            real_vals (dict): The dictionary containing the origin response column values
            class_names (np.array): Assumes the classes are of deepchem index type (e.g. 0,1,2,...)
            num_classes (int): The number of classes to predict on
    """

    # ****************************************************************************************
    # class KFoldClassificationPerfData
    def __init__(self, model_dataset, transformers, subset, predict_probs=True, transformed=True):
        """Initialize any attributes that are common to all KFoldClassificationPerfData subclasses

        Args:
           model_dataset (ModelDataset object): contains the dataset and related methods

           transformers (list of transformer objects): contains the list of transformers used to transform the dataset

           subset (str): Label in ['train', 'valid', 'test', 'full'], indicating the type of subset of dataset for
           tracking predictions

           predict_probs (bool): True if using classifier supports probabilistic predictions, False otherwise

           transformed (bool): True if values to be passed to accumulate preds function are transformed values

                Raises:
           ValueError if subset not in ['train','valid','test'], unsupported dataset subset

           NotImplementedError if predict_probs is not True, non-probabilistic classifiers are not supported yet

        Side effects:
           Sets the following attributes of KFoldClassificationPerfData:
               subset (str): Label of the type of subset of dataset for tracking predictions

               num_cmps (int): The number of compounds in the dataset

               num_tasks (int): The number of tasks in the dataset

               pred_vals (dict): The dictionary of prediction results

               folds (int): Initialized at zero, flag for determining which k-fold is being assessed

               transformers (list of Transformer objects): from input arguments

               real_vals (dict): The dictionary containing the origin response column values in one-hot encoding

               class_names (np.array): Assumes the classes are of deepchem index type (e.g. 0,1,2,...)

               num_classes (int): The number of classes to predict on
        """

        self.subset = subset
        if self.subset in ('train', 'valid', 'train_valid'):
            dataset = model_dataset.combined_training_data()
        elif self.subset == 'test':
            dataset = model_dataset.test_dset
        else:
            raise ValueError('Unknown dataset subset type "%s"' % self.subset)

        # All currently used classifiers generate class probabilities in their predict methods;
        # if in the future we implement a classification algorithm such as kNN that doesn't support
        # probabilistic predictions, the ModelWrapper for that classifier should pass predict_probs=False
        # when constructing this object. When that happens, modify the code in this class to support
        # this option.
        if not predict_probs:
            raise NotImplementedError("Need to add support for non-probabilistic classifiers")

        self.num_cmpds = dataset.y.shape[0]
        self.num_tasks = dataset.y.shape[1]
        self.num_classes = len(set(model_dataset.dataset.y.flatten()))
        self.pred_vals = dict([(id, np.empty((0, self.num_tasks, self.num_classes), dtype=np.float32)) for id in dataset.ids])

        real_vals, self.weights = model_dataset.get_subset_responses_and_weights(self.subset, [])
        self.real_classes = real_vals
        # Change real_vals to one-hot encoding
        if self.num_classes > 2:
            self.real_vals = dict([(id, 
                                   np.concatenate([dc.metrics.to_one_hot(np.array([class_labels[j]]), self.num_classes)
                                                   for j in range(self.num_tasks)], axis=0))
                                   for id, class_labels in real_vals.items()])
        else:
            self.real_vals = real_vals

        self.folds = 0
        self.perf_metrics = []
        self.model_score = None
        if transformed:
            # Predictions passed to accumulate_preds() will be transformed
            self.transformers = transformers
        else:
            self.transformers = []


    # ****************************************************************************************
    # class KFoldClassificationPerfData
    def accumulate_preds(self, predicted_vals, ids, pred_stds=None):
        """Add training, validation or test set predictions from the current fold to the data structure
        where we keep track of them.

        Args:
            predicted_vals (np.array): Array of the predicted values for the current dataset

            ids (np.array): An np.array of compound ids for the current dataset

            pred_stds (np.array): An array of the standard deviation in the predictions, not used in this method

        Returns:
            None

        Side effects:
            Overwrites the attribute pred_vals

            Increments folds by 1

        """
        class_probs = self._reshape_preds(predicted_vals)
        for i, id in enumerate(ids):
            self.pred_vals[id] = np.concatenate([self.pred_vals[id], class_probs[i,:,:].reshape((1,self.num_tasks,-1))], axis=0)
        self.folds += 1
        real_vals = self.get_real_values(ids)
        weights = self.get_weights(ids)
        # Break out different predictions for each task, with zero-weight compounds masked out, and compute per-task metrics
        scores = []
        for i in range(self.num_tasks):
            nzrows = np.where(weights[:,i] != 0)[0]
            if self.num_classes > 2:
                # If more than 2 classes, real_vals is indicator matrix (one-hot encoded). 
                task_real_vals = np.squeeze(real_vals[nzrows,i,:])
                task_class_probs = dc.trans.undo_transforms(
                                                            np.squeeze(class_probs[nzrows,i,:]),
                                                            self.transformers)
                scores.append(roc_auc_score(task_real_vals, task_class_probs, average='macro'))
            else:
                # For binary classifier, sklearn metrics functions are expecting single array of 1s and 0s for real_vals_list,
                # and class_probs for class 1 only.
                task_real_vals = np.squeeze(real_vals[nzrows,i])
                task_class_probs = dc.trans.undo_transforms(
                                                            np.squeeze(class_probs[nzrows,i,1]),
                                                            self.transformers)
                scores.append(roc_auc_score(task_real_vals, task_class_probs))
        self.perf_metrics.append(np.array(scores))
        return float(np.mean(scores))

    # ****************************************************************************************
    # class KFoldClassificationPerfData
    def get_pred_values(self):
        """Returns the predicted values accumulated over training, with any transformations undone.  If self.subset
        is 'train', 'train_valid' or 'test', the function will return the means and standard deviations of the class probabilities
        over the training folds for each compound, for each task.  Otherwise, returns a single set of predicted probabilites for
        each validation set compound. For all subsets, returns the compound IDs and the most probable classes for each task.

        Returns:
            ids (list): list of compound IDs.

            pred_classes (np.array): an (ncmpds, ntasks) array of predicted classes.

            class_probs (np.array): a (ncmpds, ntasks, nclasses) array of predicted probabilities for the classes, and

            prob_stds (np.array): a (ncmpds, ntasks, nclasses) array of standard errors over folds for the class
            probability estimates (only available for the 'train' and 'test' subsets; None otherwise).

        """
        ids = sorted(self.pred_vals.keys())
        if self.subset in ['train', 'test', 'train_valid']:
            #class_probs = np.concatenate([dc.trans.undo_transforms(self.pred_vals[id], self.transformers).mean(axis=0, keepdims=True)
            #                       for id in ids], axis=0)
            #prob_stds = np.concatenate([dc.trans.undo_transforms(self.pred_vals[id], self.transformers).std(axis=0, keepdims=True)
            #                       for id in ids], axis=0)
            class_probs = dc.trans.undo_transforms(np.concatenate([self.pred_vals[id].mean(axis=0, keepdims=True)
                                   for id in ids], axis=0), self.transformers)
            prob_stds = dc.trans.undo_transforms(np.concatenate([self.pred_vals[id].std(axis=0, keepdims=True)
                                   for id in ids], axis=0), self.transformers)
        else:
            class_probs = np.concatenate([dc.trans.undo_transforms(self.pred_vals[id], self.transformers) for id in ids], axis=0)
            prob_stds = None
        pred_classes = np.argmax(class_probs, axis=2)
        return (ids, pred_classes, class_probs, prob_stds)


    # ****************************************************************************************
    # class KFoldClassificationPerfData
    def get_real_values(self, ids=None):
        """Returns the real dataset response values as an (ncmpds, ntasks, nclasses) array of indicator bits
        (if nclasses > 2) or an (ncmpds, ntasks) array of binary classes (if nclasses == 2),
        with compound IDs in the same order as in the return from get_pred_values() (unless ids is specified).

        Args:
            ids (list of str): Optional list of compound IDs to return values for.

        Returns:
            np.array of shape (ncmpds, tasks, nclasses): of either indicator bits or a 2D array of binary classes

        """
        if ids is None:
            ids = sorted(self.pred_vals.keys())
        if self.num_classes > 2:
            return np.concatenate([self.real_vals[id].reshape((1,-1,self.num_classes)) for id in ids], axis=0)
        else:
            return np.concatenate([self.real_vals[id].reshape((1,-1)) for id in ids], axis=0)

    # ****************************************************************************************
    # class KFoldClassificationPerfData
    def get_weights(self, ids=None):
        """Returns the dataset response weights, as an (ncmpds, ntasks) array
        in the same ID order as get_pred_values() (unless ids is specified).

        Args:
            ids (list of str): Optional list of compound IDs to return values for.

        Returns:
            np.array (ncmpds, ntasks) of the real dataset response weights, in the same
            ID order as get_pred_values().

        """
        if ids is None:
            ids = sorted(self.pred_vals.keys())
        return np.concatenate([self.weights[id].reshape((1,-1)) for id in ids], axis=0)


    # ****************************************************************************************
    # class KFoldClassificationPerfData
    def compute_perf_metrics(self, per_task=False):
        """Computes the ROC AUC metrics for each task based on the accumulated values, averaged over
        training folds, along with standard deviations of the scores. If per_task is False, the scores
        are averaged over tasks and the overall standard deviation is reported instead.

        Args:
            per_task (bool): True if calculating per-task metrics, False otherwise.

        Returns:
            A tuple (roc_auc_mean, roc_auc_std):

                roc_auc_mean: A numpy array of mean ROC AUC scores for each task, averaged over folds, if per_task is
                True.
                              Otherwise, a float giving the ROC AUC score averaged over both folds and tasks.

                roc_auc_std:  A numpy array of standard deviations over folds of ROC AUC values, if per_task is True.
                              Otherwise, a float giving the overall standard deviation.
        """
        roc_auc_scores = np.stack(self.perf_metrics)
        if per_task:
            roc_auc_mean = np.mean(roc_auc_scores, axis=0)
            roc_auc_std = np.std(roc_auc_scores, axis=0)
        else:
            roc_auc_mean = np.mean(roc_auc_scores.flatten())
            roc_auc_std = np.std(roc_auc_scores.flatten())
        return (roc_auc_mean, roc_auc_std)


# ****************************************************************************************
class SimpleRegressionPerfData(RegressionPerfData):
    """Class with methods for accumulating regression model prediction data from training,
    validation or test sets and computing performance metrics.

    Attributes:
        Set in __init__:
            subset (str): Label of the type of subset of dataset for tracking predictions

            num_cmps (int): The number of compounds in the dataset

            num_tasks (int): The number of tasks in the dataset

            pred-vals (dict): The dictionary of prediction results

            folds (int): Initialized at zero, flag for determining which k-fold is being assessed

            transformers (list of Transformer objects): from input arguments

            real_vals (dict): The dictionary containing the origin response column values

    """

    # ****************************************************************************************
    # class SimpleRegressionPerfData
    def __init__(self, model_dataset, transformers, subset, transformed=True):
        """Initialize any attributes that are common to all SimpleRegressionPerfData subclasses

        Args:
           model_dataset (ModelDataset object): contains the dataset and related methods

           transformers (list of transformer objects): contains the list of transformers used to transform the dataset

           subset (str): Label in ['train', 'valid', 'test', 'full'], indicating the type of subset of dataset for
           tracking predictions

           transformed (bool): True if values to be passed to accumulate preds function are transformed values

                Raises:
           ValueError: if subset not in ['train','valid','test','full'], subset not supported

        Side effects:
           Sets the following attributes of SimpleRegressionPerfData:
               subset (str): Label of the type of subset of dataset for tracking predictions

               num_cmps (int): The number of compounds in the dataset

               num_tasks (int): The number of tasks in the dataset

               pred_vals (dict): The dictionary of prediction results

               transformers (list of Transformer objects): from input arguments

               real_vals (dict): The dictionary containing the origin response column values

        """
        self.subset = subset
        if subset == 'train':
            dataset = model_dataset.train_valid_dsets[0][0]
        elif subset == 'valid':
            dataset = model_dataset.train_valid_dsets[0][1]
        elif subset == 'test':
            dataset = model_dataset.test_dset
        elif subset == 'full':
            dataset = model_dataset.dataset
        else:
            raise ValueError('Unknown dataset subset type "%s"' % subset)
        self.num_cmpds = dataset.y.shape[0]
        self.num_tasks = dataset.y.shape[1]
        self.weights = dataset.w
        self.ids = dataset.ids
        self.pred_vals = None
        self.pred_stds = None
        self.perf_metrics = []
        self.model_score = None
        if transformed:
            # Predictions passed to accumulate_preds() will be transformed
            self.transformers = transformers
            self.real_vals = dataset.y
        else:
            self.real_vals = dc.trans.undo_transforms(dataset.y, transformers)
            self.transformers = []


    # ****************************************************************************************
    # class SimpleRegressionPerfData
    def accumulate_preds(self, predicted_vals, ids, pred_stds=None):
        """Add training, validation or test set predictions to the data structure
        where we keep track of them.

        Args:
            predicted_vals (np.array): Array of predicted values

            ids (list): List of the compound ids of the dataset

            pred_stds (np.array): Optional np.array of the prediction standard deviations

        Side effects:
            Reshapes the predicted values and the standard deviations (if they are given)

        """

        self.pred_vals = self._reshape_preds(predicted_vals)
        if pred_stds is not None:
            self.pred_stds = self._reshape_preds(pred_stds)
        pred_vals = dc.trans.undo_transforms(self.pred_vals, self.transformers)
        real_vals = self.get_real_values(ids)
        weights = self.get_weights(ids)
        scores = []
        for i in range(self.num_tasks):
            nzrows = np.where(weights[:,i] != 0)[0]
            task_real_vals = np.squeeze(real_vals[nzrows,i])
            task_pred_vals = np.squeeze(pred_vals[nzrows,i])
            scores.append(r2_score(task_real_vals, task_pred_vals))
        self.perf_metrics.append(np.array(scores))
        return float(np.mean(scores))


    # ****************************************************************************************
    # class SimpleRegressionPerfData
    def get_pred_values(self):
        """Returns the predicted values accumulated over training, with any transformations undone.  Returns
        a tuple (ids, values, stds), where ids is the list of compound IDs, values is a (ncmpds, ntasks) array
        of predictions, and stds is always None for this class.

        Returns:
            Tuple (ids, vals, stds)
                ids (list): Contains the dataset compound ids

                vals (np.array): Contains (ncmpds, ntasks) array of prediction

                stds (np.array): Contains (ncmpds, ntasks) array of prediction standard deviations
        """
        vals = dc.trans.undo_transforms(self.pred_vals, self.transformers)
        stds = None
        if self.pred_stds is not None:
            stds = self.pred_stds
            if len(self.transformers) == 1 and (isinstance(self.transformers[0], dc.trans.NormalizationTransformer) or isinstance(self.transformers[0],trans.NormalizationTransformerMissingData)):
                # Untransform the standard deviations, if we can. This is a bit of a hack, but it works for
                # NormalizationTransformer, since the standard deviations used to scale the data are
                # stored in the transformer object.
                    y_stds = self.transformers[0].y_stds.reshape((1,-1,1))
                    stds = stds / y_stds
        return (self.ids, vals, stds)


    # ****************************************************************************************
    # class SimpleRegressionPerfData
    def get_real_values(self, ids=None):
        """Returns the real dataset response values, with any transformations undone, as an (ncmpds, ntasks) array
        with compounds in the same ID order as in the return from get_pred_values().

        Args:
            ids: Ignored for this class

        Returns:
            np.array: Containing the real dataset response values with transformations undone.

        """
        return dc.trans.undo_transforms(self.real_vals, self.transformers)


    # ****************************************************************************************
    # class SimpleRegressionPerfData
    def get_weights(self, ids=None):
        """Returns the dataset response weights as an (ncmpds, ntasks) array

        Args:
            ids: Ignored for this class

        Returns:
            np.array: Containing the dataset response weights

        """
        return self.weights


    # ****************************************************************************************
    # class SimpleRegressionPerfData
    def compute_perf_metrics(self, per_task=False):
        """Returns the R-squared metrics for each task or averaged over tasks based on the accumulated values

        Args:
            per_task (bool): True if calculating per-task metrics, False otherwise.

        Returns:
            A tuple (r2_score, std):
                r2_score (np.array): An array of scores for each task, if per_task is True.
                Otherwise, it is a float containing the average R^2 score over tasks.

                std: Always None for this class.
        """
        r2_scores = self.perf_metrics[0]
        if per_task or self.num_tasks == 1:
            return (r2_scores, None)
        else:
            return (r2_scores.mean(), None)



# ****************************************************************************************
class SimpleClassificationPerfData(ClassificationPerfData):
    """Class with methods for collecting classification model prediction and performance data from single-fold
    training and prediction runs.

    Attributes:
        Set in __init__:
            subset (str): Label of the type of subset of dataset for tracking predictions

            num_cmps (int): The number of compounds in the dataset

            num_tasks (int): The number of tasks in the dataset

            pred-vals (dict): The dictionary of prediction results

            folds (int): Initialized at zero, flag for determining which k-fold is being assessed

            transformers (list of Transformer objects): from input arguments

            real_vals (dict): The dictionary containing the origin response column values

            class_names (np.array): Assumes the classes are of deepchem index type (e.g. 0,1,2,...)

            num_classes (int): The number of classes to predict on

    """

    # ****************************************************************************************
    # class SimpleClassificationPerfData
    def __init__(self, model_dataset, transformers, subset, predict_probs=True, transformed=True):
        """Initialize any attributes that are common to all SimpleClassificationPerfData subclasses

        Args:
           model_dataset (ModelDataset object): contains the dataset and related methods

           transformers (list of transformer objects): contains the list of transformers used to transform the dataset

           subset (str): Label in ['train', 'valid', 'test', 'full'], indicating the type of subset of dataset for
           tracking predictions

           predict_probs (bool): True if using classifier supports probabilistic predictions, False otherwise

           transformed (bool): True if values to be passed to accumulate preds function are transformed values

                Raises:
           ValueError: if subset not in ['train','valid','test','full'], subset not supported

           NotImplementedError: if predict_probs is not True, non-probabilistic functions are not supported yet

        Side effects:
           Sets the following attributes of SimpleClassificationPerfData:
               subset (str): Label of the type of subset of dataset for tracking predictions

               num_cmps (int): The number of compounds in the dataset

               num_tasks (int): The number of tasks in the dataset

               pred_vals (dict): The dictionary of prediction results

               transformers (list of Transformer objects): from input arguments

               real_vals (dict): The dictionary containing the origin response column values

               num_classes (int): The number of classes to predict on

        """

        self.subset = subset
        if subset == 'train':
            dataset = model_dataset.train_valid_dsets[0][0]
        elif subset == 'valid':
            dataset = model_dataset.train_valid_dsets[0][1]
        elif subset == 'test':
            dataset = model_dataset.test_dset
        elif subset == 'full':
            dataset = model_dataset.dataset
        else:
            raise ValueError('Unknown dataset subset type "%s"' % subset)

        # All currently used classifiers generate class probabilities in their predict methods;
        # if in the future we implement a classification algorithm such as kNN that doesn't support
        # probabilistic predictions, the ModelWrapper for that classifier should pass predict_probs=False
        # when constructing this object. When that happens, modify the code in this class to support
        # this option.
        if not predict_probs:
            raise NotImplementedError("Need to add support for non-probabilistic classifiers")

        self.num_cmpds = dataset.y.shape[0]
        if len(dataset.y.shape) > 1:
            self.num_tasks = dataset.y.shape[1]
        else:
            self.num_tasks = 1
        self.pred_vals = None
        self.pred_stds = None
        self.ids = dataset.ids
        self.perf_metrics = []
        self.model_score = None
        if transformed:
            # Predictions passed to accumulate_preds() will be transformed
            self.transformers = transformers
        else:
            self.transformers = []
        self.weights = dataset.w

        # TODO: Everything down to here is same as in SimpleRegressionPerfData.__init__.
        # TODO: Consider defining a SimplePerfData class with the common stuff in its __init__
        # TODO: method, and doing multiple inheritance so we can call it from here.

        # DeepChem does not currently support arbitary class names in classification datasets; 
        # enforce class indices (0, 1, 2, ...) here.
        self.class_indeces = list(set(model_dataset.dataset.y.flatten()))
        self.num_classes = len(self.class_indeces)
        self.real_classes = dataset.y
        # Convert true values to one-hot encoding
        if self.num_classes > 2:
            self.real_vals = np.concatenate([dc.metrics.to_one_hot(dataset.y[:,j], self.num_classes).reshape(-1,1,self.num_classes)
                                             for j in range(self.num_tasks)], axis=1)
        else:
            self.real_vals = dataset.y.reshape((-1,self.num_tasks))


    # ****************************************************************************************
    # class SimpleClassificationPerfData
    def accumulate_preds(self, predicted_vals, ids, pred_stds=None):
        """Add training, validation or test set predictions from the current dataset to the data structure
        where we keep track of them.

        Arguments:
            predicted_vals (np.array): Array of predicted values (class probabilities)

            ids (list): List of the compound ids of the dataset

            pred_stds (np.array): Optional np.array of the prediction standard deviations

        Side effects:
            Updates self.pred_vals and self.perf_metrics

        """
        class_probs = self.pred_vals = self._reshape_preds(predicted_vals)
        if pred_stds is not None:
            self.pred_stds = self._reshape_preds(pred_stds)
        real_vals = self.get_real_values(ids)
        weights = self.get_weights(ids)
        # Break out different predictions for each task, with zero-weight compounds masked out, and compute per-task metrics
        scores = []
        for i in range(self.num_tasks):
            nzrows = np.where(weights[:,i] != 0)[0]
            if self.num_classes > 2:
                # If more than 2 classes, real_vals is indicator matrix (one-hot encoded). 
                task_real_vals = np.squeeze(real_vals[nzrows,i,:])
                task_class_probs = dc.trans.undo_transforms(
                                                            np.squeeze(class_probs[nzrows,i,:]),
                                                            self.transformers)
                scores.append(roc_auc_score(task_real_vals, task_class_probs, average='macro'))
            else:
                # For binary classifier, sklearn metrics functions are expecting single array of 1s and 0s for real_vals_list,
                # and class_probs for class 1 only.
                task_real_vals = np.squeeze(real_vals[nzrows,i])
                task_class_probs = dc.trans.undo_transforms(
                                                            np.squeeze(class_probs[nzrows,i,1]),
                                                            self.transformers)
                scores.append(roc_auc_score(task_real_vals, task_class_probs))
        self.perf_metrics.append(np.array(scores))
        return float(np.mean(scores))

    # ****************************************************************************************
    # class SimpleClassificationPerfData
    def get_pred_values(self):
        """Returns the predicted values accumulated over training, with any transformations undone.
        If self.subset is 'train', the function will average class probabilities over the k-1 folds in which each
        compound was part of the training set, and return the most probable class. Otherwise, there should be a
        single set of predicted probabilites for each validation or test set compound. Returns a tuple (ids,
        pred_classes, class_probs, prob_stds), where ids is the list of compound IDs, pred_classes is an
        (ncmpds, ntasks) array of predicted classes, class_probs is a (ncmpds, ntasks, nclasses) array of predicted
        probabilities for the classes, and prob_stds is a (ncmpds, ntasks, nclasses) array of standard errors for the
        class probability estimates.

        Returns:
            Tuple (ids, pred_classes, class_probs, prob_stds)
                ids (list): Contains the dataset compound ids

                pred_classes (np.array): Contains (ncmpds, ntasks) array of prediction classes

                class_probs (np.array): Contains (ncmpds, ntasks, nclasses) array of predict class probabilities

                prob_stds (np.array): Contains (ncmpds, ntasks, nclasses) array of standard errors for the class
                probability estimates
        """
        class_probs = dc.trans.undo_transforms(self.pred_vals, self.transformers)
        pred_classes = np.argmax(class_probs, axis=2)
        prob_stds = self.pred_stds
        return (self.ids, pred_classes, class_probs, prob_stds)


    # ****************************************************************************************
    # class SimpleClassificationPerfData
    def get_real_values(self, ids=None):
        """Returns the real dataset response values as an (ncmpds, ntasks, nclasses) array of indicator bits.
        If nclasses == 2, the returned array has dimension (ncmpds, ntasks).

        Args:
            ids: Ignored for this class

        Returns:
            np.array of the response values of the real dataset as indicator bits

        """
        return self.real_vals


    # ****************************************************************************************
    # class SimpleClassificationPerfData
    def get_weights(self, ids=None):
        """Returns the dataset response weights

        Args:
            ids: Ignored for this class

        Returns:
            np.array: Containing the dataset response weights

        """
        return self.weights


    # ****************************************************************************************
    # class SimpleClassificationPerfData
    def compute_perf_metrics(self, per_task=False):
        """Returns the ROC_AUC metrics for each task based on the accumulated predictions. If
        per_task is False, returns the average ROC AUC over tasks.

        Args:
            per_task (bool): Whether to return individual ROC AUC scores for each task

        Returns:
            A tuple (roc_auc, std):
                roc_auc: A numpy array of ROC AUC scores, if per_task is True. Otherwise,
                         a float giving the mean ROC AUC score over tasks.

                std:     Placeholder for an array of standard deviations. Always None for this class.

        """
        roc_auc_scores = self.perf_metrics[0]
        if per_task or self.num_tasks == 1:
            return (roc_auc_scores, None)
        else:
            return (roc_auc_scores.mean(), None)


# ****************************************************************************************
class SimpleHybridPerfData(HybridPerfData):
    """Class with methods for accumulating hybrid model prediction data from training,
    validation or test sets and computing performance metrics.

    Attributes:
        Set in __init__:
            subset (str): Label of the type of subset of dataset for tracking predictions

            num_cmps (int): The number of compounds in the dataset

            num_tasks (int): The number of tasks in the dataset

            pred-vals (dict): The dictionary of prediction results

            folds (int): Initialized at zero, flag for determining which k-fold is being assessed

            transformers (list of Transformer objects): from input arguments

            real_vals (dict): The dictionary containing the origin response column values

    """

    # ****************************************************************************************
    # class SimpleHybridPerfData
    def __init__(self, model_dataset, transformers, subset, is_ki, ki_convert_ratio=None, transformed=True):
        """Initialize any attributes that are common to all SimpleRegressionPerfData subclasses

        Args:
           model_dataset (ModelDataset object): contains the dataset and related methods

           transformers (list of transformer objects): contains the list of transformers used to transform the dataset

           subset (str): Label in ['train', 'valid', 'test', 'full'], indicating the type of subset of dataset for
           tracking predictions

           is_ki: whether the dose-response activity is Ki or IC50, it will decide how to convert them into single
           concentration activities.

           ki_convert_ratio: If the given activity is pKi, a ratio to convert Ki into IC50 is needed. It can be the
           ratio of concentration and Kd of the radioligand in a competitive binding assay, or the concentration
           of the substrate and Michaelis constant (Km) of enzymatic inhibition assay.

           transformed (bool): True if values to be passed to accumulate preds function are transformed values

        Raises:
           ValueError: if subset not in ['train','valid','test','full'], subset not supported

        Side effects:
           Sets the following attributes of SimpleRegressionPerfData:
               subset (str): Label of the type of subset of dataset for tracking predictions

               num_cmps (int): The number of compounds in the dataset

               num_tasks (int): The number of tasks in the dataset

               pred_vals (dict): The dictionary of prediction results

               transformers (list of Transformer objects): from input arguments

               real_vals (dict): The dictionary containing the origin response column values

        """
        self.subset = subset
        if subset == 'train':
            dataset = model_dataset.train_valid_dsets[0][0]
        elif subset == 'valid':
            dataset = model_dataset.train_valid_dsets[0][1]
        elif subset == 'test':
            dataset = model_dataset.test_dset
        elif subset == 'full':
            dataset = model_dataset.dataset
        else:
            raise ValueError('Unknown dataset subset type "%s"' % subset)
        self.num_cmpds = dataset.y.shape[0]
        self.num_tasks = dataset.y.shape[1]
        self.weights = dataset.w
        self.ids = dataset.ids
        self.pred_vals = None
        self.pred_stds = None
        self.perf_metrics = []
        self.model_score = None
        self.is_ki = is_ki
        self.ki_convert_ratio = ki_convert_ratio
        if transformed:
            # Predictions passed to accumulate_preds() will be transformed
            self.transformers = transformers
            self.real_vals = dataset.y
        else:
            self.real_vals = transformers[0].untransform(dataset.y)
            self.transformers = []

    # ****************************************************************************************
    # class SimpleHybridPerfData
    def accumulate_preds(self, predicted_vals, ids, pred_stds=None):
        """Add training, validation or test set predictions to the data structure
        where we keep track of them.

        Args:
            predicted_vals (np.array): Array of predicted values

            ids (list): List of the compound ids of the dataset

            pred_stds (np.array): Optional np.array of the prediction standard deviations

        Side effects:
            Reshapes the predicted values and the standard deviations (if they are given)

        """

        self.pred_vals = self._reshape_preds(predicted_vals)
        if pred_stds is not None:
            self.pred_stds = self._reshape_preds(pred_stds)
        # pred_vals = self.transformers[0].untransform(self.pred_vals, isreal=False)
        pred_vals = self.pred_vals
        real_vals = self.get_real_values(ids)
        weights = self.get_weights(ids)
        scores = []
        pos_ki = np.where(np.isnan(real_vals[:, 1]))[0]
        pos_bind = np.where(~np.isnan(real_vals[:, 1]))[0]

        # score for pKi/IC50
        nzrows = np.where(weights[:, 0] != 0)[0]
        rowki = np.intersect1d(nzrows, pos_ki)
        rowbind = np.intersect1d(nzrows, pos_bind)
        ki_real_vals = np.squeeze(real_vals[rowki,0])
        ki_pred_vals = np.squeeze(pred_vals[rowki,0])
        bind_real_vals = np.squeeze(real_vals[rowbind,0])
        bind_pred_vals = np.squeeze(pred_vals[rowbind,0])
        if len(rowki) > 0:
            scores.append(r2_score(ki_real_vals, ki_pred_vals))
            if len(rowbind) > 0:
                scores.append(r2_score(bind_real_vals, bind_pred_vals))
            else:
                # if all values are dose response activities, use the r2_score above.
                scores.append(scores[0])
        elif len(rowbind) > 0:
            # all values are single concentration activities.
            scores.append(r2_score(bind_real_vals, bind_pred_vals))
            scores.append(scores[0])

        self.perf_metrics.append(np.array(scores))
        return float(np.mean(scores))

    # ****************************************************************************************
    # class SimpleHybridPerfData
    def _predict_binding(self, activity, conc):
        """Predict measurements of fractional binding/inhibition of target receptors by a compound with the given activity,
        in -Log scale, at the specified concentration in nM. If the given activity is pKi, a ratio to convert Ki into IC50
        is needed. It can be the ratio of concentration and Kd of the radioligand in a competitive binding assay, or the concentration
        of the substrate and Michaelis constant (Km) of enzymatic inhibition assay.
        """
        
        if self.is_ki:
            if self.ki_convert_ratio is None:
                raise Exception("Ki converting ratio is missing. Cannot convert Ki into IC50")
            Ki = 10**(9-activity)
            IC50 = Ki * (1 + self.ki_convert_ratio)
        else:
            IC50 = 10**(9-activity)
        pred_frac = 1.0/(1.0 + IC50/conc)
        
        return pred_frac

    # ****************************************************************************************
    # class SimpleHybridPerfData
    def get_pred_values(self):
        """Returns the predicted values accumulated over training, with any transformations undone.  Returns
        a tuple (ids, values, stds), where ids is the list of compound IDs, values is a (ncmpds, ntasks) array
        of predictions, and stds is always None for this class.

        Returns:
            Tuple (ids, vals, stds)
                ids (list): Contains the dataset compound ids

                vals (np.array): Contains (ncmpds, ntasks) array of prediction

                stds (np.array): Contains (ncmpds, ntasks) array of prediction standard deviations
        """
        vals = self.pred_vals
        # pos_bind = np.where(~np.isnan(self.real_vals[:,1]))[0]
        # vals[pos_bind, 0] = self._predict_binding(vals[pos_bind, 0], self.real_vals[pos_bind, 1])
        stds = None
        
        return (self.ids, vals, stds)


    # ****************************************************************************************
    # class SimpleHybridPerfData
    def get_real_values(self, ids=None):
        """Returns the real dataset response values, with any transformations undone, as an (ncmpds, ntasks) array
        with compounds in the same ID order as in the return from get_pred_values().

        Args:
            ids: Ignored for this class

        Returns:
            np.array: Containing the real dataset response values with transformations undone.

        """
        return self.transformers[0].untransform(self.real_vals)


    # ****************************************************************************************
    # class SimpleHybridPerfData
    def get_weights(self, ids=None):
        """Returns the dataset response weights as an (ncmpds, ntasks) array

        Args:
            ids: Ignored for this class

        Returns:
            np.array: Containing the dataset response weights

        """
        return self.weights


    # ****************************************************************************************
    # class SimpleHybridPerfData
    def compute_perf_metrics(self, per_task=False):
        """Returns the R-squared metrics for each task or averaged over tasks based on the accumulated values

        Args:
            per_task (bool): True if calculating per-task metrics, False otherwise.

        Returns:
            A tuple (r2_score, std):
                r2_score (np.array): An array of scores for each task, if per_task is True.
                Otherwise, it is a float containing the average R^2 score over tasks.

                std: Always None for this class.
        """
        r2_scores = self.perf_metrics[0]
        if per_task or self.num_tasks == 1:
            return (r2_scores, None)
        else:
            return (r2_scores.mean(), None)


# ****************************************************************************************
class EpochManager:
    """Manages lists of PerfDatas

        This class manages lists of PerfDatas as well as variables related to iteratively
        training a model over several epochs. This class sets several varaibles in a given
        ModelWrapper for the sake of backwards compatibility

    Attributes:
       Set in __init__:
           _subsets (dict): Must contain the keys 'train', 'valid', 'test'. The values
               are used as subsets when calling create_perf_data.

           _model_choice_score_type (str): Passed into PerfData.model_choice_score

           _log (logger): This is the from wrapper.log

           _should_stop (bool): True when training as satisfied stopping conditions. Either
               it has reached the max number of epochs or has exceeded early_stopping_patience

           wrapper (ModelWrapper): The model wrapper where this object is being used.

           _new_best_valid_score (function): This function takes no arguments and is called
               whenever a new best validation score is achieved.

    """

    # ****************************************************************************************
    # class EpochManager
    def __init__(self, wrapper,
            subsets={'train':'train',  'valid':'valid', 'test':'test'}, production=False, **kwargs):
        """Initialize EpochManager

        Args:
           wrapper (ModelWrapper): The ModelWrapper that's doing the training

           subsets (dict): Must contain the keys 'train', 'valid', 'test'. The values
               are used as subsets when calling create_perf_data.

           production (bool): True if this is running in production mode.

           kwargs (dict): Additional keyword args are passed to create_perf_data. The
               subset argument should not be passed.

        Side effects:
           Creates the following attributes in wrapper:
               best_epoch
               best_valid_score
               train_epoch_perfs
               valid_epoch_perfs
               test_epoch_perfs
               train_epoch_perf_stds
               valid_epoch_perf_stds
               test_epoch_perf_stds
               model_choice_scores
               early_stopping_min_improvement
               early_stopping_patience
               train_perf_data
               valid_perf_data
               test_perf_data
        """
        params = wrapper.params
        self.production = production
        self._subsets = subsets
        self._model_choice_score_type = params.model_choice_score_type
        self._log = wrapper.log
        self._should_stop = False
        self.wrapper = wrapper
        
        self._new_best_valid_score = lambda: False

        self.wrapper.best_epoch = 0
        self.wrapper.best_valid_score = None
        self.wrapper.train_epoch_perfs = np.zeros(params.max_epochs)
        self.wrapper.valid_epoch_perfs = np.zeros(params.max_epochs)
        self.wrapper.test_epoch_perfs = np.zeros(params.max_epochs)
        self.wrapper.train_epoch_perf_stds = np.zeros(params.max_epochs)
        self.wrapper.valid_epoch_perf_stds = np.zeros(params.max_epochs)
        self.wrapper.test_epoch_perf_stds = np.zeros(params.max_epochs)
        self.wrapper.model_choice_scores = np.zeros(params.max_epochs)
        self.wrapper.early_stopping_min_improvement = params.early_stopping_min_improvement
        self.wrapper.early_stopping_patience = params.early_stopping_patience

        self.wrapper.train_perf_data = []
        self.wrapper.valid_perf_data = []
        self.wrapper.test_perf_data = []

        for _ in range(params.max_epochs):
            self.wrapper.train_perf_data.append(
                create_perf_data(subset=self._subsets['train'], **kwargs))
            self.wrapper.valid_perf_data.append(
                create_perf_data(subset=self._subsets['valid'], **kwargs))
            self.wrapper.test_perf_data.append(
                create_perf_data(subset=self._subsets['test'], **kwargs))

    # ****************************************************************************************
    # class EpochManager
    def should_stop(self):
        """Returns True when the training loop should stop

        Returns:
            bool: True when the training loop should stop
        """
        return self._should_stop

    # ****************************************************************************************
    # class EpochManager
    def update_epoch(self, ei, train_dset=None, valid_dset=None, test_dset=None):
        """Update training state after an epoch

                This function updates train/valid/test_perf_data. Call this function once
                per epoch. Call self.should_stop() after calling this function to see if you should
                exit the training loop.

                Subsets with None arguments will be ignored

        Args:
           ei (int): The epoch index

           train_dset (dc.data.Dataset): The train dataset

           valid_dset (dc.data.Dataset): The valid dataset. Providing this argument updates
               best_valid_score and _should_stop

           test_dset (dc.data.Dataset): The test dataset

        Returns:
           list: A list of performance values for the provided datasets.

        Side effects:
           This function updates self._should_stop

        """
        train_perf = self.update(ei, 'train', train_dset)
        valid_perf = self.update(ei, 'valid', valid_dset)
        test_perf = self.update(ei, 'test', test_dset)

        return [p for p in [train_perf, valid_perf, test_perf] if p is not None]

    # ****************************************************************************************
    # class EpochManager
    def accumulate(self, ei, subset, dset):
        """Accumulate predictions

                Makes predictions, accumulate predictions and calculate the performance metric. Calls PerfData.accumulate_preds
                belonging to the epoch, subset, and given dataset.

        Args:
           ei (int): Epoch index

           subset (str): Which subset, should be train, valid, or test.

           dset (dc.data.Dataset): Calculates the performance for the given dset

        Returns:
           float: Performance metric for the given dset.
        """
        pred = self._make_pred(dset)
        perf = getattr(self.wrapper, f'{subset}_perf_data')[ei].accumulate_preds(pred, dset.ids)
        return perf

    # ****************************************************************************************
    # class EpochManager
    def compute(self, ei, subset):
        """Computes performance metrics

                This calls PerfData.compute_perf_metrics and saves the result in
                f'{subset}_epoch_perfs'

        Args:
           ei (int): Epoch index

           subset (str): Which subset to compute_perf_metrics. Should be train, valid, or test

        Returns:
           None

        """
        getattr(self.wrapper, f'{subset}_epoch_perfs')[ei], _ = \
            getattr(self.wrapper, f'{subset}_perf_data')[ei].compute_perf_metrics()

    # ****************************************************************************************
    # class EpochManager
    def update_valid(self, ei):
        """Checks validation score

                Checks validation performance of the given epoch index. Updates self._should_stop, checks
                on early stopping conditions, calls self._new_best_valid_score() when necessary.

        Args:
           ei (int): Epoch index

        Returns:
           None

        Side effects:
           Updates self._should_stop when it's time to exit the training loop.
        """
        valid_score = self.wrapper.valid_perf_data[ei].model_choice_score(self._model_choice_score_type)
        self.wrapper.model_choice_scores[ei] = valid_score
        if self.wrapper.best_valid_score is None or self.production:
            # If we're in production mode, every epoch is the new best epoch
            self._new_best_valid_score()
            self.wrapper.best_valid_score = valid_score
            self.wrapper.best_epoch = ei
            self._log.info(f"Total score for epoch {ei} is {valid_score:.3}")
        elif valid_score - self.wrapper.best_valid_score > self.wrapper.early_stopping_min_improvement:
            self._new_best_valid_score()
            self.wrapper.best_valid_score = valid_score
            self.wrapper.best_epoch = ei
            self._log.info(f"*** Total score for epoch {ei} is {valid_score:.3}, is new maximum")
        elif ei - self.wrapper.best_epoch > self.wrapper.early_stopping_patience:
            self._log.info(f"No improvement after {self.wrapper.early_stopping_patience} epochs, stopping training")
            self._should_stop = True

    # ****************************************************************************************
    # class EpochManager
    def update(self, ei, subset, dset=None):
        """Update training state

                Updates the training state for a given subset and epoch index with the given dataset.

        Args:
           ei (int): Epoch index.

           subset (str): Should be train, valid, test

           dset (dc.data.Dataset): Updates using this dset

        Returns:
           perf (float): the performance of the given dset.

        """
        if dset is None:
            return None

        perf = self.accumulate(ei, subset, dset)
        self.compute(ei, subset)

        if subset == 'valid':
            self.update_valid(ei)

        return perf

    # ****************************************************************************************
    # class EpochManager
    def set_make_pred(self, functional):
        """Sets the function used to make predictions

        Sets the function used to make predictions. This must be called before invoking
        self.update and self.accumulate

        Args:
           functional (function): This function takes one argument, a dc.data.Dataset, and
               returns an array of predictions for that dset. This function is called
               when updating the training state after a given epoch.

        Returns:
           None

        Side effects:
           Saves the functional as self._make_pred
        """
        self._make_pred = functional

    # ****************************************************************************************
    # class EpochManager
    def on_new_best_valid(self, functional):
        """Sets the function called when a new best validation score is achieved

                Saves the function called when there's a new best validation score.

        Args:
           functional (function): This function takes no arguments and returns nothing. This
               function is called when there's a new best validation score. This can be used
               to tell the ModelWrapper to save the model.

        Returns:
           None

        Side effects:
           Saves the _new_best_valid_score function.

        """
        self._new_best_valid_score = functional

# ****************************************************************************************
class EpochManagerKFold(EpochManager):
    """This class manages the training state when using KFold cross validation. This is
    necessary because this manager uses f'{subset}_epoch_perf_stds' unlike EpochManager

    """
    # ****************************************************************************************
    # class EpochManagerKFold
    def compute(self, ei, subset):
        """Calls PerfData.compute_perf_metrics()

        This differs from EpochManager.compute in that it saves the results into
        f'{subset}_epoch_perf_stds'

        Args:
           ei (int): Epoch index

           subset (str): Should be train, valid, test.

        Returns:
           None

        """
        getattr(self.wrapper, f'{subset}_epoch_perfs')[ei], getattr(self.wrapper, f'{subset}_epoch_perf_stds')[ei]= \
            getattr(self.wrapper, f'{subset}_perf_data')[ei].compute_perf_metrics()
