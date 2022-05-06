import numpy as np
import os
from PIL import Image

# PyTorch Imports
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# My imports
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
from xAI_utils import takeThird
from xAI_utils import GenerateDeepLiftAtts
from choose_rects import GetOracleFeedback

HITL_LAMBDA = 1

def HITL_Loss(logits, Y, images, W, image_grads):
    # Aggregate along color channels and normalize to [-1, 1]
    #image_grads /= torch.max(torch.abs(image_grads))

    return torch.mean(W * (image_grads**2))

# Train model and sample the most useful images for decision making (entropy based sampling)
def active_train_model(model, train_loader, entropy_thresh, nr_queries, data_classes, EPOCHS, DEVICE, LOSS):
    
    # Hyper-parameters
    LEARNING_RATE = 1e-6
    OPTIMISER = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Initialise min_train and min_val loss trackers
    min_train_loss = np.inf

    # Initialise losses arrays
    train_losses = np.zeros((EPOCHS, ))

    # Initialise metrics arrays
    train_metrics = np.zeros((EPOCHS, 4))

    # Weights for human in the loop loss
    W = torch.zeros((len(train_loader.dataset), 224, 224), device=DEVICE)

    for epoch in range(EPOCHS):
        # Epoch 
        print(f"Epoch: {epoch+1}")
        
        # Training Loop
        print(f"Training Phase")
        
        # Initialise lists to compute scores
        y_train_true = list()
        y_train_pred = list()

        # Initialise list of high entropy predictions
        # and corresponding images
        high_entropy_pred = list()

        # Running train loss
        run_train_loss = 0.0


        # Put model in training mode
        model.train()

        # Iterate through dataloader
        for batch_idx, (images, images_og, labels, indices) in enumerate(train_loader):
            
            # move data, labels and model to DEVICE (GPU or CPU)
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            model = model.to(DEVICE)

            # Find the loss and update the model parameters accordingly
            # Clear the gradients of all optimized variables
            OPTIMISER.zero_grad()

            # Forward pass: compute predicted outputs by passing inputs to the model
            images.requires_grad = True
            logits = model(images)

            logits.sum().backward(retain_graph=True)
            #image_grads = torch.autograd.grad(logits.sum(), images, create_graph=True)[0]
            image_grads = images.grad.mean(axis=1)


            if(epoch >= 0):
                # Copy logits to cpu
                pred_logits = logits.cpu().detach().numpy()
                pred_logits = torch.FloatTensor(pred_logits)
                pred_probs = torch.softmax(pred_logits,1)

                # Iterate logits tensor 
                for idx in range(len(pred_probs)):
                    # calculate entropy for each single image logits in batch
                    pred_entropy = entropy(pred_probs[idx])

                    if(pred_entropy > entropy_thresh):
                        temp_image_info = [images_og[idx], labels[idx], pred_entropy, indices[idx]]
                        high_entropy_pred.append(temp_image_info)  



            # Compute the batch loss
            # Using CrossEntropy w/ Softmax
            loss = LOSS(logits, labels) #+ HITL_LAMBDA*HITL_Loss(logits, labels, images, W[indices], image_grads)

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            
            # Perform a single optimization step (parameter update)
            OPTIMISER.step()
            
            # Update batch losses
            run_train_loss += (loss.item() * images.size(0))

            # Concatenate lists
            y_train_true += list(labels.cpu().detach().numpy())
            
            # Using Softmax
            # Apply Softmax on Logits and get the argmax to get the predicted labels
            s_logits = torch.nn.Softmax(dim=1)(logits)
            s_logits = torch.argmax(s_logits, dim=1)
            y_train_pred += list(s_logits.cpu().detach().numpy())

        
        # Compute Average Train Loss
        avg_train_loss = run_train_loss/len(train_loader.dataset)

        # Compute Train Metrics
        train_acc = accuracy_score(y_true=y_train_true, y_pred=y_train_pred)
        # train_recall = recall_score(y_true=y_train_true, y_pred=y_train_pred, average="weighted")
        # train_precision = precision_score(y_true=y_train_true, y_pred=y_train_pred, average="weighted")
        # train_f1 = f1_score(y_true=y_train_true, y_pred=y_train_pred, average="weighted")


        # Print high entropy prediciton data points
        print(f"Number of high entropy predictions after {epoch+1} epochs: {len(high_entropy_pred)}")

        # # Visualize entropy distribution
        # df = pd.DataFrame(high_entropy_pred, columns = ['Image','Label','Entropy'])
        # plt.hist(df['Entropy'], color = 'blue', edgecolor = 'black',
        #  bins = int(160/10))
        # plt.savefig(f"/home/up201605633/Desktop/Results/DeepLift/AL_tests/entropy_dist_e{epoch+1}.png")

        # Get highest entropy prediction information
        high_entropy_pred.sort(key=takeThird, reverse=True)
        print(f"Highest entropy predictions after {epoch+1} epochs: ")

        # Print query entropies and perform Deep Lift on each data point
        #save_file_dir = "/home/up201605633/Desktop/Results/DeepLift/AL_tests/"
        for i in range(len(high_entropy_pred)):
            if(i < nr_queries):
                print(high_entropy_pred[i][2]) 
                query_image = high_entropy_pred[i][0]
                query_index = high_entropy_pred[i][3]
                deepLiftAtts = GenerateDeepLiftAtts(image=query_image, label=high_entropy_pred[i][1], model = model, data_classes=data_classes)

                # Aggregate along color channels and normalize to [-1, 1]
                deepLiftAtts = deepLiftAtts.sum(axis=np.argmax(np.asarray(deepLiftAtts.shape) == 3))
                deepLiftAtts /= np.max(np.abs(deepLiftAtts))
                deepLiftAtts = torch.tensor(deepLiftAtts)
                print(deepLiftAtts.shape)

                __,selectedRectangles = GetOracleFeedback(query_image, deepLiftAtts, rectSize=28, rectStride=28, nr_rects=5)
                print(selectedRectangles)

                # change the weights W=1 in the selected rectangles area
                print("index:", query_index)
                for rect in selectedRectangles:
                    W[query_index, rect[1]:rect[3], rect[0]:rect[2]] = 1

        # Print Statistics
        print(f"Train Loss: {avg_train_loss}\tTrain Accuracy: {train_acc}")

        # Append values to the arrays
        # Train Loss
        train_losses[epoch] = avg_train_loss
        
        # Train Metrics
        # Acc
        train_metrics[epoch, 0] = train_acc
        # Recall
        # train_metrics[epoch, 1] = train_recall
        # Precision
        # train_metrics[epoch, 2] = train_precision
        # F1-Score
        # train_metrics[epoch, 3] = train_f1

        # Update Variables
        # Min Training Loss
        if avg_train_loss < min_train_loss:
            print(f"Train loss decreased from {min_train_loss} to {avg_train_loss}.")
            min_train_loss = avg_train_loss


    # Finish statement
    print("Finished.")
    return train_losses,train_metrics

# Train model and iterate through the validation set, saving the best model
def train_model(model, model_name, train_loader, val_loader, history_dir, weights_dir, nr_classes, data_name ,EPOCHS, DEVICE, LOSS):
    
    # Mean and STD to Normalize the inputs into pretrained models
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # Input Data Dimensions
    img_nr_channels = 3
    img_height = 224
    img_width = 224

    # Hyper-parameters
    LEARNING_RATE = 1e-4
    OPTIMISER = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    BATCH_SIZE = 2

    # Train model and save best weights on validation set
    # Initialise min_train and min_val loss trackers
    min_train_loss = np.inf
    min_val_loss = np.inf

    # Initialise losses arrays
    train_losses = np.zeros((EPOCHS, ))
    val_losses = np.zeros_like(train_losses)

    # Initialise metrics arrays
    train_metrics = np.zeros((EPOCHS, 4))
    val_metrics = np.zeros_like(train_metrics)


    # Go through the number of Epochs
    for epoch in range(EPOCHS):
        # Epoch 
        print(f"Epoch: {epoch+1}")
        
        # Training Loop
        print(f"Training Phase")
        
        # Initialise lists to compute scores
        y_train_true = list()
        y_train_pred = list()


        # Running train loss
        run_train_loss = 0.0


        # Put model in training mode
        model.train()


        # Iterate through dataloader
        for batch_idx, (images, labels) in enumerate(train_loader):

            # Move data data anda model to GPU (or not)
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            model = model.to(DEVICE)


            # Find the loss and update the model parameters accordingly
            # Clear the gradients of all optimized variables
            OPTIMISER.zero_grad()


            # Forward pass: compute predicted outputs by passing inputs to the model
            logits = model(images)
            
            # Compute the batch loss
            # Using CrossEntropy w/ Softmax
            loss = LOSS(logits, labels)

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            
            # Perform a single optimization step (parameter update)
            OPTIMISER.step()
            
            # Update batch losses
            run_train_loss += (loss.item() * images.size(0))

            # Concatenate lists
            y_train_true += list(labels.cpu().detach().numpy())
            
            # Using Softmax
            # Apply Softmax on Logits and get the argmax to get the predicted labels
            s_logits = torch.nn.Softmax(dim=1)(logits)
            s_logits = torch.argmax(s_logits, dim=1)
            y_train_pred += list(s_logits.cpu().detach().numpy())
        

        # Compute Average Train Loss
        avg_train_loss = run_train_loss/len(train_loader.dataset)

        # Compute Train Metrics
        train_acc = accuracy_score(y_true=y_train_true, y_pred=y_train_pred)
        # train_recall = recall_score(y_true=y_train_true, y_pred=y_train_pred, average="weighted")
        # train_precision = precision_score(y_true=y_train_true, y_pred=y_train_pred, average="weighted")
        # train_f1 = f1_score(y_true=y_train_true, y_pred=y_train_pred, average="weighted")

        # Print Statistics
        print(f"Train Loss: {avg_train_loss}\tTrain Accuracy: {train_acc}")
        # print(f"Train Loss: {avg_train_loss}\tTrain Accuracy: {train_acc}\tTrain Recall: {train_recall}\tTrain Precision: {train_precision}\tTrain F1-Score: {train_f1}")


        # Append values to the arrays
        # Train Loss
        train_losses[epoch] = avg_train_loss
        # Save it to directory
        fname = os.path.join(history_dir, f"{model_name}_tr_losses.npy")
        np.save(file=fname, arr=train_losses, allow_pickle=True)


        # Train Metrics
        # Acc
        train_metrics[epoch, 0] = train_acc
        # Recall
        # train_metrics[epoch, 1] = train_recall
        # Precision
        # train_metrics[epoch, 2] = train_precision
        # F1-Score
        # train_metrics[epoch, 3] = train_f1
        # Save it to directory
        fname = os.path.join(history_dir, f"{model_name}_tr_metrics.npy")
        np.save(file=fname, arr=train_metrics, allow_pickle=True)


        # Update Variables
        # Min Training Loss
        if avg_train_loss < min_train_loss:
            print(f"Train loss decreased from {min_train_loss} to {avg_train_loss}.")
            min_train_loss = avg_train_loss


        # Validation Loop
        print("Validation Phase")


        # Initialise lists to compute scores
        y_val_true = list()
        y_val_pred = list()


        # Running train loss
        run_val_loss = 0.0


        # Put model in evaluation mode
        model.eval()

        # Deactivate gradients
        with torch.no_grad():

            # Iterate through dataloader
            for batch_idx, (images, labels) in enumerate(val_loader):

                # Move data data anda model to GPU (or not)
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                model = model.to(DEVICE)

                # Forward pass: compute predicted outputs by passing inputs to the model
                logits = model(images)
                
                # Compute the batch loss
                # Using CrossEntropy w/ Softmax
                loss = LOSS(logits, labels)
                
                # Update batch losses
                run_val_loss += (loss.item() * images.size(0))

                # Concatenate lists
                y_val_true += list(labels.cpu().detach().numpy())
                
                # Using Softmax Activation
                # Apply Softmax on Logits and get the argmax to get the predicted labels
                s_logits = torch.nn.Softmax(dim=1)(logits)
                s_logits = torch.argmax(s_logits, dim=1)
                y_val_pred += list(s_logits.cpu().detach().numpy())

            

            # Compute Average Train Loss
            avg_val_loss = run_val_loss/len(val_loader.dataset)

            # Compute Training Accuracy
            val_acc = accuracy_score(y_true=y_val_true, y_pred=y_val_pred)
            # val_recall = recall_score(y_true=y_val_true, y_pred=y_val_pred, average="weighted")
            # val_precision = precision_score(y_true=y_val_true, y_pred=y_val_pred, average="weighted")
            # val_f1 = f1_score(y_true=y_val_true, y_pred=y_val_pred, average="weighted")

            # Print Statistics
            print(f"Validation Loss: {avg_val_loss}\tValidation Accuracy: {val_acc}")
            # print(f"Validation Loss: {avg_val_loss}\tValidation Accuracy: {val_acc}\tValidation Recall: {val_recall}\tValidation Precision: {val_precision}\tValidation F1-Score: {val_f1}")

            # Append values to the arrays
            # Train Loss
            val_losses[epoch] = avg_val_loss
            # Save it to directory
            fname = os.path.join(history_dir, f"{model_name}_val_losses.npy")
            np.save(file=fname, arr=val_losses, allow_pickle=True)


            # Train Metrics
            # Acc
            val_metrics[epoch, 0] = val_acc
            # Recall
            # val_metrics[epoch, 1] = val_recall
            # Precision
            # val_metrics[epoch, 2] = val_precision
            # F1-Score
            # val_metrics[epoch, 3] = val_f1
            # Save it to directory
            fname = os.path.join(history_dir, f"{model_name}_val_metrics.npy")
            np.save(file=fname, arr=val_metrics, allow_pickle=True)

            # Update Variables
            # Min validation loss and save if validation loss decreases
            if avg_val_loss < min_val_loss:
                print(f"Validation loss decreased from {min_val_loss} to {avg_val_loss}.")
                min_val_loss = avg_val_loss

                print("Saving best model on validation...")

                # Save checkpoint
                model_path = os.path.join(weights_dir, f"{model_name}_{data_name}.pt")
                torch.save(model.state_dict(), model_path)

                print(f"Successfully saved at: {model_path}")


    # Finish statement
    print("Finished.")
    return val_losses,train_losses,val_metrics,train_metrics
    

def test_model(model, model_name, test_loader, nr_classes, LOSS, DEVICE):
    # Test Loop
    print("Test Phase")


    # Initialise lists to compute scores
    y_test_true = list()
    y_test_pred = list()


    # Running train loss
    run_test_loss = 0.0


    # Put model in evaluation mode
    model.eval()

    # Deactivate gradients
    with torch.no_grad():

        # Iterate through dataloader
        for batch_idx, (images, labels) in enumerate(test_loader):

            # Move data data anda model to GPU (or not)
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            model = model.to(DEVICE)

            # Forward pass: compute predicted outputs by passing inputs to the model
            logits = model(images)
            
            # Compute the batch loss
            # Using CrossEntropy w/ Softmax
            loss = LOSS(logits, labels)
            
            # Update batch losses
            run_test_loss += (loss.item() * images.size(0))

            # Concatenate lists
            y_test_true += list(labels.cpu().detach().numpy())
            
            # Using Softmax Activation
            # Apply Softmax on Logits and get the argmax to get the predicted labels
            s_logits = torch.nn.Softmax(dim=1)(logits)
            s_logits = torch.argmax(s_logits, dim=1)
            y_test_pred += list(s_logits.cpu().detach().numpy())

        

        # Compute Average Train Loss
        avg_test_loss = run_test_loss/len(test_loader.dataset)

        # Compute Training Accuracy
        test_acc = accuracy_score(y_true=y_test_true, y_pred=y_test_pred)
        # val_recall = recall_score(y_true=y_val_true, y_pred=y_val_pred, average="weighted")
        # val_precision = precision_score(y_true=y_val_true, y_pred=y_val_pred, average="weighted")
        # val_f1 = f1_score(y_true=y_val_true, y_pred=y_val_pred, average="weighted")

        # Print Statistics
        print(f"Test Loss: {avg_test_loss}\tTest Accuracy: {test_acc}")
        # print(f"Validation Loss: {avg_val_loss}\tValidation Accuracy: {val_acc}\tValidation Recall: {val_recall}\tValidation Precision: {val_precision}\tValidation F1-Score: {val_f1}")

        # Append values to the arrays
        # Train Loss
        test_loss = avg_test_loss

        # Train Metrics
        # Acc
        test_metrics = test_acc
        # Recall
        # val_metrics[epoch, 1] = val_recall
        # Precision
        # val_metrics[epoch, 2] = val_precision
        # F1-Score
        # val_metrics[epoch, 3] = val_f1

        # Update Variables
        # Min validation loss and save if validation loss decreases

    print("Finished")
    return test_loss, test_metrics



