from Get_model_and_data import *
import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils as ut
import logging
import datetime
import shutil
import os
import json
from utils import *
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import StepLR



def initialize_train_val(
               batch_size,
               decoder,
               encoder,
               encoder_weight,
               train_image_dir,
               train_mask_dir,
               activation,
               data,
               resolution = 0):
    



    model, train_loader= get_train_val_data_and_model(
        encoder=encoder,
        encoder_weight=encoder_weight,
        decoder=decoder,
        batch_size=batch_size,
        train_image_dir="images/train/cropped_image",
        train_mask_dir= f"{train_mask_dir}/cropped_{data}",

        resolution=resolution,
        activation=activation
    )

    return model,train_loader

def initialize_model_info(data,
               batch_size,
               encoder,
               resolution = 0):
    model_info = {'encoder': encoder,"batch_size" : batch_size, 'resolution': resolution, "data":data}
    return model_info



from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, LambdaLR


def train_validate(epoch, lr, weight_decay, model, device, train_loader, valid_loader, encoder):
    os.makedirs("./best_models", exist_ok=True)
    loss = ut.losses.DiceLoss()
    metrics = [
        ut.metrics.IoU(threshold=0.5),
        ut.metrics.Accuracy(threshold=0.5),
        ut.metrics.Recall(threshold=0.5),
        ut.metrics.Fscore(threshold=0.5),
        ut.metrics.Precision(threshold=0.5)
    ]

    optimizer = torch.optim.AdamW([ 
        dict(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    ])

    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)

    train_epoch = ut.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = ut.train.ValidEpoch(
        model,
        loss=loss, 
        metrics=metrics, 
        device=device,
        verbose=True,
    )
    try:
        max_iou_score = 0
        for i in range(0, epoch+1):
            logging.info(f'Epoch: {i}')
            logging.info(f'Epoch: {i}, Learning Rate: {optimizer.param_groups[0]["lr"]}')

            # Update the learning rate scheduler
   
            plateau_scheduler.step(max_iou_score)

            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)

            if max_dice_loss > valid_logs['iou_score']:
                max_dice_loss = valid_logs['iou_score']
                torch.save(model.state_dict(), 'best_step_model.pth')
                print("Model is saved")

    except KeyboardInterrupt:
        print('Training interrupted.')


    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f"best_models/{current_time}_{encoder}_{lr}")
    locate = f"best_models/{current_time}_{encoder}_{lr}/model.pth"
    model_info = {'lr': lr,"epoch" : epoch, 'weight_decay': weight_decay}
    filename_model = './best_step_model.pth'
    shutil.copyfile(filename_model, locate)
    os.remove(filename_model)
    model.load_state_dict(torch.load(locate))
    print("Training completed.")
    return model,model_info,f"best_models/{current_time}_{encoder}_{lr}"


def test_model(cropped_res, stride, best_model, device, encoder, encoder_weight, model_info_train, model_info_initialize, resolution, data, locate):

    results = {}

    logging.info(f"Testing model: {encoder}_{datetime.datetime.now().strftime('%Y_%H-%M-%S')}")

    try:
        predict_and_save_folder(input_folder="images/test/cropped_image", output_maskfolder="images/test/pred_masks", output_prob_folder="images/test/pred_masks_probs", encoder=encoder, encoder_weight=encoder_weight, best_model=best_model, device=device, resolution=resolution)
        logging.info("Prediction and saving completed successfully.")

        merge_cropped_images(2752, 2752, cropped_res=cropped_res, stride=stride, input_dir="images/test/pred_masks", output_dir=f"images/test/merged_pred_masks_{data}")
        logging.info("Merging cropped images completed successfully.")

        merge_cropped_arrays(2752, 2752, cropped_res=cropped_res, stride=stride, input_dir="images/test/pred_masks_probs", output_dir=f"images/test/merged_pred_probs_masks_{data}")
        logging.info("Merging cropped arrays completed successfully.")

        plot_save_mismatches(f"images/test/merged_pred_masks_{data}", f"images/test/mask/{data}", save_dir=locate)
        logging.info("Plotting and saving mismatches completed successfully.")

        auc_pr_result = auc_pr_folder_calculation(pred_mask_dir=f"images/test/merged_pred_probs_masks_{data}", test_mask_dir=f"images/test/mask/{data}", stride=stride)
        logging.info("AUC-PR calculation completed successfully.")

        iou, accuracy, recall, fscore, precision = calculate_metrics(f"images/test/mask/{data}", f"images/test/merged_pred_masks_{data}")
        iou1, accuracy1, recall1, fscore1, precision1 = calculate_metrics(f"images/test/mask/cropped_{data}", f"images/test/pred_masks")

        logs = {"iou": iou, "accuracy": accuracy, "recall": recall, "fscore": fscore, "precision": precision}
        logs2 = {"iou1": iou1, "accuracy1": accuracy1, "recall1": recall1, "fscore1": fscore1, "precision1": precision1}

        model_info_train.update(model_info_initialize)
        results[f'{encoder}_{datetime.datetime.now().strftime("%Y_%H-%M-%S")}'] = {'model_info': model_info_train, 'logs': logs, "logs2": logs2, "auc_pr": auc_pr_result}
        json_file_path = "./best_models/results.json"
        with open(json_file_path, 'a') as json_file:
            json.dump(results, json_file, indent=5)
        logging.info("Results saved successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

        