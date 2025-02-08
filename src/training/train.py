import os
import json
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  
from src.utils.visualization import visualize_keypoints
from src.utils.metrics import calculate_classification_accuracy, calculate_keypoint_accuracy, calculate_bbox_accuracy
from src.training.evaluate import evaluate_model


def train_model(train_loader, model, class_name_to_idx, num_epochs=10, log_dir="logs/train_logs", 
                checkpoint_dir="checkpoints", val_loader=None):

    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Initialize optimizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available

    idx_to_class_name = {idx: class_name for class_name, idx in class_name_to_idx.items()}  # Reverse the mapping

    model = model.to(device) # Move model to the same device as the data
    print(f"Using device: {device}")

    writer = SummaryWriter(log_dir=log_dir) # Initialize TensorBoard writer

    os.makedirs(checkpoint_dir, exist_ok=True) # Create checkpoint directory

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_classification_loss= 0.0
        running_keypoint_loss = 0.0
        running_bbox_loss = 0.0

        # Initialize accumulators for accuracy metrics at the epoch level
        total_classification_correct = 0
        total_classification_count = 0
        total_classification_TP = 0
        total_classification_FP = 0
        total_classification_FN = 0

        total_bbox_correct = 0
        total_bbox_count = 0
        total_keypoints_correct = 0
        total_keypoints_count = 0

        # Iterate over the training dataset
        for batch_idx, (images, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):

            # Move data to the same device as the model
            images = images.to(device)

            # Move targets to the same device as the model
            # List range from 0 to batch size
            new_targets = []
            for i in range(len(targets["bbox"])):  # Iterating over the batch size (64)
                new_targets.append({
                    "bbox": targets["bbox"][i].to(device),  # Bounding box for image i
                    "workout_label": targets["workout_label"][i].to(device),  # Class label for image i
                    "keypoints": targets["keypoints"][i].to(device),  # Keypoints for image i
                })

            optimizer.zero_grad()  # Zero the gradients before backward pass

            # Forward pass
            output = model(images) 

            # Losses
            losses = model.compute_losses(output, new_targets)
            loss_dict = {
                "classification_loss": losses[2],
                "bbox_loss": losses[0],
                "keypoint_loss": losses[1],
            }
            classification_loss = loss_dict["classification_loss"]
            keypoint_loss = loss_dict["keypoint_loss"]
            bbox_loss = loss_dict["bbox_loss"]

            total_loss = classification_loss + keypoint_loss + bbox_loss

            total_loss.backward()  # Backpropagate the loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()  # Update model weights

            running_classification_loss += classification_loss.item()
            running_keypoint_loss += keypoint_loss.item()
            running_bbox_loss += bbox_loss.item()
            running_loss += total_loss.item()  # Accumulate loss for averaging

            # Calculate overall accuracy for the epoch
            bbox, keypoints, workout_label = output

            # Calculate and accumulate accuracy metrics
            workout_label_targets = torch.stack([target['workout_label'] for target in new_targets]) 
            class_accuracy, batch_TP, batch_FP, batch_FN = calculate_classification_accuracy(workout_label, 
                                                                                              workout_label_targets, 
                                                                                              len(idx_to_class_name))
            total_classification_correct += class_accuracy * len(workout_label_targets)
            total_classification_count += len(workout_label_targets)
            total_classification_TP += batch_TP
            total_classification_FP += batch_FP
            total_classification_FN += batch_FN

            # Calculate bbox accuracy
            bbox_targets = torch.stack([target['bbox'] for target in new_targets]) 
            bbox_accuracy = calculate_bbox_accuracy(bbox, bbox_targets)
            total_bbox_correct += bbox_accuracy * len(bbox_targets)
            total_bbox_count += len(bbox_targets)

            # Calculate keypoint accuracy
            keypoints_targets = torch.stack([target['keypoints'] for target in new_targets]) 
            keypoints_accuracy = calculate_keypoint_accuracy(keypoints, keypoints_targets)
            total_keypoints_correct += keypoints_accuracy * len(keypoints_targets)
            total_keypoints_count += len(keypoints_targets) 

            # Log batch loss to TensorBoard
            writer.add_scalar("Batch_Loss/Classification", classification_loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar("Batch_Loss/Keypoint", keypoint_loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar("Batch_Loss/BBox", bbox_loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar("Batch_Loss/Total", total_loss.item(), epoch * len(train_loader) + batch_idx)

            # Visualize predictions and targets for each epoch at batch 1 for the first 5 images
            if batch_idx == 0:
                for i in range(4):

                    sample_image = images[i].cpu().detach().numpy() # Unfortunetely numpy doesn't work in CUDA
                    
                    # Visualize keypoints and bounding boxes
                    vis_image = visualize_keypoints(
                        sample_image, 
                        keypoints[i].cpu().detach().numpy(), 
                        keypoints_targets[i].cpu().numpy(), 
                        sample_image.shape[2], 
                        sample_image.shape[1], 
                        bbox[i].cpu().detach().numpy(), 
                        bbox_targets[i].cpu().detach().numpy()
                    )

                    # Log the visualization to TensorBoard
                    writer.add_image(f'Visualization/Image_{i}', vis_image, epoch)

                    # Prediction
                    log_probs = torch.nn.functional.log_softmax(workout_label[i], dim=0)
                    predicted_class_index = torch.argmax(log_probs, dim=0)
                    predicted_class_name = idx_to_class_name[predicted_class_index.item()]
                    predicted_prob = torch.exp(log_probs[predicted_class_index.item()]).item()

                    # Ground truth
                    true_class_index = torch.argmax(workout_label_targets[i], dim=0)
                    true_class_name = idx_to_class_name[true_class_index.item()]

                    log_entry = f"Predicted: {predicted_class_name} (Prob: {predicted_prob:.4f})\nTrue: {true_class_name}"
                    writer.add_text(f"Classification/Image_{i}", log_entry, epoch)

        # Compute epoch loss and log it
        epoch_classification_loss = running_classification_loss / len(train_loader)
        epoch_keypoint_loss = running_keypoint_loss / len(train_loader)
        epoch_bbox_loss = running_bbox_loss / len(train_loader)
        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar("Epoch_Loss/Classification", epoch_classification_loss, epoch)
        writer.add_scalar("Epoch_Loss/Keypoint", epoch_keypoint_loss, epoch)
        writer.add_scalar("Epoch_Loss/BBox", epoch_bbox_loss, epoch)
        writer.add_scalar("Epoch_Loss/Total", epoch_loss, epoch)

        # Compute epoch accuracies and log them
        epoch_classification_accuracy = total_classification_correct / total_classification_count
        epoch_bbox_accuracy = total_bbox_correct / total_bbox_count
        epoch_keypoints_accuracy = total_keypoints_correct / total_keypoints_count

        writer.add_scalar("Epoch_Accuracy/Classification_Accuracy", epoch_classification_accuracy, epoch)
        writer.add_scalar("Epoch_Accuracy/Keypoint_PCK", epoch_keypoints_accuracy, epoch)
        writer.add_scalar("Epoch_Accuracy/BBox_IoU", epoch_bbox_accuracy, epoch)


        epoch_classification_precision = total_classification_TP / (total_classification_TP + total_classification_FP
                                                                    ) if (total_classification_TP + total_classification_FP) > 0 else 0.0
        epoch_classification_recall = total_classification_TP / (total_classification_TP + total_classification_FN
                                                                 ) if (total_classification_TP + total_classification_FN) > 0 else 0.0

        writer.add_scalar("Epoch_Accuracy/Classification_Precision", epoch_classification_precision, epoch)
        writer.add_scalar("Epoch_Accuracy/Classification_Recall", epoch_classification_recall, epoch)

        # Evaluate the model
        if val_loader != None:
            evaluate_model(val_loader, model, class_name_to_idx, num_epoch=epoch)

        # Save model checkpoint at the end of the epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

        with open('idx_to_class_name.json', 'w') as f:
            json.dump(idx_to_class_name, f)

    writer.close()
