# Interpretability-Guided Human Feedback During Neural Network Training

## About
Implementation of the paper [_"Interpretability-Guided Human Feedback During Neural Network Training"_](#interpretability-guided-human-feedback-during-neural-network-training) by Pedro Serrano e Silva, Ricardo Cruz, ASM Shihavuddin and Tiago Gonçalves.

## Abstract
When a model makes a wrong prediction, a typical solution is to acquire more data related to the error -- this is an expensive process known as active learning. Our proposal combines active learning with interpretability so that the user is able to correct such mistakes while the model is being trained. At the end of each epoch, our training pipeline shows the user cases of mistakes and uses interpretability to allow the user to visualize which regions of the images are receiving the attention of the model. The user is then able to guide the training through a regularization term in the loss function. Overall, in low-data regimens, the proposed method returned lower loss values in the predictions made for all three datasets used: 0.61, 0.47, 0.36, when compared with fully automated training methods using the same amount of data: 0.63, 0.52, 0.41, respectively. Higher accuracy values were was also seen in two of the datasets: 81.14% and 92.58% over the 78.41% and 92.52% seen in fully automated methods. Using the method while training with 100% of the dataset did not yield any relevant results, as the performance was similar to automated training. During testing, it was noted that the method does help in comprehending the inner works of the model, however, it shows limitations in some situations that prevent it from being useful in all datasets.

## Usage
To run the HITL training you must run the train.py file and enter the training parameters. The parsing structure is as follows:

-dr -> data directory (str)

-md -> trained models directory (str)

-E -> number of training epochs (int)

-tf -> fraction of the training data to use (float)

-vf -> fraction of the validation data to use (float)

-td -> the title of the training iteration (str)

-sp -> the sampling process (low_entropy / high_entropy) (str)

-et -> the entropy sampling threshold (float)

-qu -> number of queries per epoch (int)

-ov -> if oversampling is to be used or not (bool)

-se -> the epoch on which the querying starts (int)

-ds -> the dataset to train on (APTOS19 / ISIC17 / NCI) (string)

An example command could be:

python train.py -dr '/home/up201605633/Desktop' -md 'results/ones_test' -E 10 -tf 0.1 -vf 1 -td 'example_train_10epochs' -sp 'low_entropy' -et 0.1 -qu 10 -ov True -se 5 -ds 'APTOS'

The code will save the best model and the training history and metrics. It will also generate two graphs that describe the evaluation metrics progression.
