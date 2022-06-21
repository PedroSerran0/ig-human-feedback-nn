import numpy as np
import os
from PIL import Image 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm

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
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from scipy.stats import entropy
from xAI_utils import takeThird
from xAI_utils import GenerateDeepLiftAtts
from choose_rects import GetOracleFeedback

HITL_LAMBDA = 1e6

def my_loss(Ypred, X, W):
    # it "works" with both retain_graph=True and create_graph=True, but I think
    # the latter is what we want
    grad = torch.autograd.grad(Ypred.sum(), X, retain_graph=True)[0]
    # grad has a shape: torch.Size([256, 3, 32, 32])
    return (W *(grad**2).mean(1)).mean()

# Train model and sample the most useful images for decision making (entropy based sampling)
def active_train_model(model, model_name, data_name, train_loader, val_loader, history_dir, weights_dir, entropy_thresh, nr_queries, start_epoch, data_classes, oversample, sampling_process, EPOCHS, DEVICE, LOSS, percentage=100):
    
    assert sampling_process in ['low_entropy', 'high_entropy']
    # Hyper-parameters
    LEARNING_RATE = 1e-4
    OPTIMISER = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Initialise min_train and min_val loss trackers
    min_train_loss = np.inf
    min_val_loss = np.inf

    # Initialise losses arrays
    train_losses = np.zeros((EPOCHS, ))
    val_losses = np.zeros_like(train_losses)

    # Initialise metrics arrays
    train_metrics = np.zeros((EPOCHS, 4))
    val_metrics = np.zeros_like(train_metrics)

    # Weights for human in the loop loss
    print(f"Length of train loader: {len(train_loader)}")
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
        informative_pred = list()

        # Running train loss
        run_train_loss = 0.0
        vanilla_run_train_loss = 0.0


        # Put model in training mode
        model.train()

        # Iterate through dataloader
        for batch_idx, (images, images_og, labels, indices) in enumerate(tqdm(train_loader)):
            # move data, labels and model to DEVICE (GPU or CPU)
            if W.sum() > 0 and oversample == True:
                print('W sum 1,2:', W.sum([1, 2]) > 0)
                print('len(W):', len(W))
                ix_annotated = torch.arange(0, len(W))[W.sum([1, 2]) > 0]
                # ix_annotated = [0, 4, 9, ...]
                print('ix_annotated:', ix_annotated)
                i = ix_annotated[torch.randint(0, len(ix_annotated), ())]
                print('adicionei a amostra', i)
                _image, _image_og, _label, _indice = train_loader.dataset[i]
                images = torch.cat((images, _image[None]))
                images_og = torch.cat((images_og, _image_og[None]))
                labels = torch.cat((labels, torch.tensor([_label])))
                indices = torch.cat((indices, torch.tensor([_indice])))

            images, labels, images_og = images.to(DEVICE), labels.to(DEVICE), images_og.to(DEVICE)
            model = model.to(DEVICE)

            # Find the loss and update the model parameters accordingly
            # Clear the gradients of all optimized variables
            OPTIMISER.zero_grad()

            # Forward pass: compute predicted outputs by passing inputs to the model
            images.requires_grad = True
            logits = model(images)
            loss = LOSS(logits, labels)

            vanilla_loss = loss.item()
            images_og.requires_grad = True
            logits = model(images_og)
            custom_loss = my_loss(logits, images_og, W[indices])*HITL_LAMBDA
            loss += custom_loss
            
            
#            if (epoch >= start_epoch):
#                print(f"Cross entropy loss: {vanilla_run_train_loss}")
#                print(f"Loss After AL: {loss} ")
#                print(f"Custom imposed loss: {custom_loss}")

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            
            # Perform a single optimization step (parameter update)
            OPTIMISER.step()
            
            # Initialise Active Learning
            if(epoch >= start_epoch):
                # Copy logits to cpu

                #with torch.no_grad():
                #    _logits = model(images_og)

                pred_logits = logits.cpu().detach().numpy()
                pred_logits = torch.FloatTensor(pred_logits)
                pred_probs = torch.softmax(pred_logits,1)
                _,pred = torch.max(pred_probs, 1)
                print('predictions:', pred_probs.shape, pred)
                
                if(sampling_process == 'high_entropy'):
                    # Iterate logits tensor 
                    for idx in range(len(pred_probs)):
                        # calculate entropy for each single image logits in batch
                        pred_entropy = entropy(pred_probs[idx])
                        if(pred_entropy > entropy_thresh):
                            temp_image_info = [images_og[idx], labels[idx], pred_entropy, indices[idx], idx]
                            informative_pred.append(temp_image_info)
                elif(sampling_process == 'low_entropy'):
                    for idx in range(len(pred_probs)):
                        pred_entropy = entropy(pred_probs[idx])
                        
                        if pred_entropy < entropy_thresh and pred[idx] != labels[idx].cpu():
                            temp_image_info = [images_og[idx], labels[idx], pred_entropy, indices[idx], idx, pred[idx]]
                            informative_pred.append(temp_image_info)
                            
                      
            
            # Update batch losses
            run_train_loss += (loss.item() * images.size(0))
            vanilla_run_train_loss += (vanilla_loss*images.size(0))

            # Concatenate lists
            y_train_true += list(labels.cpu().detach().numpy())
            
            # Using Softmax
            # Apply Softmax on Logits and get the argmax to get the predicted labels
            s_logits = torch.nn.Softmax(dim=1)(logits)
            s_logits = torch.argmax(s_logits, dim=1)
            y_train_pred += list(s_logits.cpu().detach().numpy())

        
        # Compute Average Train Loss
        avg_train_loss = run_train_loss/len(train_loader.dataset)
        vanilla_avg_train_loss = vanilla_run_train_loss/len(train_loader.dataset)

        # Compute Train Metrics
        train_acc = accuracy_score(y_true=y_train_true, y_pred=y_train_pred)
        train_recall = recall_score(y_true=y_train_true, y_pred=y_train_pred, average="weighted")
        train_precision = precision_score(y_true=y_train_true, y_pred=y_train_pred, average="weighted")
        train_f1 = f1_score(y_true=y_train_true, y_pred=y_train_pred, average="weighted")

        # Get highest entropy prediction information
        if(sampling_process == 'high_entropy' and len(informative_pred) > 0):
            informative_pred.sort(key=takeThird, reverse=True)
            print(f"Highest entropy predictions after {epoch+1} epochs: ")
        elif(sampling_process == 'low_entropy' and len(informative_pred) > 0):
            informative_pred.sort(key=takeThird, reverse=False)
            print(f"Lowest entropy predictions after {epoch+1} epochs: ")
        
        # Print query entropies and perform Deep Lift on each data point
        for i in range(len(informative_pred)):
            if(i < nr_queries):
                print(informative_pred[i][2]) 
                query_image = informative_pred[i][0]
                query_index = informative_pred[i][4]
                image_index = informative_pred[i][3]
                query_label = informative_pred[i][1]
                temp_pred = informative_pred[i][5]
                deepLiftAtts, query_pred = GenerateDeepLiftAtts(image=query_image, label=query_label, model = model, data_classes=data_classes, temp_pred=temp_pred)
                
                print("query_pred: ",query_pred,"query_label", query_label)
                # Aggregate along color channels and normalize to [-1, 1]
                deepLiftAtts = deepLiftAtts.sum(axis=np.argmax(np.asarray(deepLiftAtts.shape) == 3))
                deepLiftAtts /= np.max(np.abs(deepLiftAtts))
                deepLiftAtts = torch.tensor(deepLiftAtts)
                #print(deepLiftAtts.shape)

                __, selectedRectangles = GetOracleFeedback(image=query_image.detach().cpu().numpy(), label=query_label, idx=image_index, model_attributions=deepLiftAtts, pred=query_pred, rectSize=28, rectStride=28, nr_rects=10)
                print(selectedRectangles)

                # change the weights W=1 in the selected rectangles area
                print("index:", image_index)
                print(f"Length of rectangle vector: {len(W)}")
                for rect in selectedRectangles:
                    W[image_index, rect[1]:rect[3], rect[0]:rect[2]] = 1
            
        # Print high entropy prediciton data points
        print(f"Number of informative predictions after {epoch+1} epochs: {len(informative_pred)}")

        # # Visualize entropy distribution
        # df = pd.DataFrame(informative_pred, columns = ['Image','Label','Entropy'])
        # plt.hist(df['Entropy'], color = 'blue', edgecolor = 'black',
        #  bins = int(160/10))
        # plt.savefig(f"/home/up201605633/Desktop/Results/DeepLift/AL_tests/entropy_dist_e{epoch+1}.png")


        
        # Print Statistics
        print(f"Train Loss: {vanilla_avg_train_loss}\tTrain Accuracy: {train_acc}")
        
        # Append values to the arrays
        # Train Loss
        train_losses[epoch] = vanilla_avg_train_loss
        
        # Train Metrics
        # Acc
        train_metrics[epoch, 0] = train_acc
        # Recall
        train_metrics[epoch, 1] = train_recall
        # Precision
        train_metrics[epoch, 2] = train_precision
        # F1-Score
        train_metrics[epoch, 3] = train_f1

        # Update Variables
        # Min Training Loss
        if vanilla_avg_train_loss < min_train_loss:
            print(f"Train loss decreased from {min_train_loss} to {vanilla_avg_train_loss}.")
            min_train_loss = vanilla_avg_train_loss

        # DEBUG
        transf = transforms.ToPILImage()
        for j in range(len(W)):
            if W[j].sum() > 0:
                Xaug, X, labels, indices = train_loader.dataset[j]
                #print(len(X))
                #img1 = transf(X[j])
                #img1 = img1.save(f'/home/up201605633/Desktop/AL_debug/X-{epoch}-{j}.png')
                
                #img2 = transf(W[j])
                #plt.imshow(img2)
                #img2 = img2.save(f'/home/up201605633/Desktop/AL_debug/W-{epoch}-{j}.png')
                
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
            for batch_idx, (images, images_og, labels, indices) in enumerate(tqdm(val_loader)):

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
            val_recall = recall_score(y_true=y_val_true, y_pred=y_val_pred, average="weighted")
            val_precision = precision_score(y_true=y_val_true, y_pred=y_val_pred, average="weighted")
            val_f1 = f1_score(y_true=y_val_true, y_pred=y_val_pred, average="weighted")

            # Print Statistics
            print(f"Validation Loss: {avg_val_loss}\tValidation Accuracy: {val_acc}")
            # print(f"Validation Loss: {avg_val_loss}\tValidation Accuracy: {val_acc}\tValidation Recall: {val_recall}\tValidation Precision: {val_precision}\tValidation F1-Score: {val_f1}")

            # Append values to the arrays
            # Train Loss
            val_losses[epoch] = avg_val_loss
            # Save it to directory
            fname = os.path.join(history_dir, f"{model_name}_val_losses_{percentage}_{sampling_process}.npy")
            np.save(file=fname, arr=val_losses, allow_pickle=True)


            # Train Metrics
            # Acc
            val_metrics[epoch, 0] = val_acc
            # Recall
            val_metrics[epoch, 1] = val_recall
            # Precision
            val_metrics[epoch, 2] = val_precision
            # F1-Score
            val_metrics[epoch, 3] = val_f1
            # Save it to directory
            fname = os.path.join(history_dir, f"{model_name}_val_metrics_{percentage}_{sampling_process}.npy")
            np.save(file=fname, arr=val_metrics, allow_pickle=True)

            # Update Variables
            # Min validation loss and save if validation loss decreases
            if avg_val_loss < min_val_loss:
                print(f"Validation loss decreased from {min_val_loss} to {avg_val_loss}.")
                min_val_loss = avg_val_loss

                print("Saving best model on validation...")

                # Save checkpoint
                model_path = os.path.join(weights_dir, f"{model_name}_{percentage}p_{EPOCHS}e_{sampling_process}.pt")
                torch.save(model.state_dict(), model_path)

                print(f"Successfully saved at: {model_path}")
    

    # Finish statement
    print("Finished.")
    return val_losses,train_losses,val_metrics,train_metrics

# Train model and iterate through the validation set, saving the best model
def train_model(model, model_name, train_loader, val_loader, history_dir, weights_dir, nr_classes, data_name ,EPOCHS, DEVICE, LOSS, percentage = 100):
    
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
        for batch_idx, (images, images_og, labels, indices) in enumerate(tqdm(train_loader)):

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
        fname = os.path.join(history_dir, f"{model_name}_tr_losses_{percentage}.npy")
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
        fname = os.path.join(history_dir, f"{model_name}_tr_metrics_{percentage}.npy")
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
            for batch_idx, (images, images_og, labels, indices) in enumerate(tqdm(val_loader)):

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
            fname = os.path.join(history_dir, f"{model_name}_val_losses_{percentage}.npy")
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
                model_path = os.path.join(weights_dir, f"{model_name}_{percentage}p_{EPOCHS}e.pt")
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



