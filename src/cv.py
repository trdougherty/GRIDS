import os
import sys

import torch

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.use_deterministic_algorithms(True)

import numpy as np
import copy
from tqdm import tqdm_notebook
import random

import wandb

# now I want to build training and validation masks
def build_masks(training_mask, n_val = 100, n_cv = 10):
    true_idx = np.where(training_mask)[0]
    validation_batches = []

    for i in range(n_cv):
        validation_batches.append(
            np.random.choice(true_idx, n_val, replace=False)
        )
    
    cv_masks = []
    for validation_idx in validation_batches:
        cv_training_mask = copy.deepcopy(training_mask)
        cv_training_mask[validation_idx] = False

        validation_mask = np.repeat([False], len(cv_training_mask))
        validation_mask[validation_idx] = True

        cv_masks.append(
            (cv_training_mask, validation_mask)
        )
        
    return cv_masks

def reset_model(model):
    for layer in model.children():
       if hasattr(layer, 'reset_parameters'):
           layer.reset_parameters()
    return model

def crossvalidation(
    model,
    anonymous_function,
    cvs,
    recorded_values,
    epochs = 100,
    lr = 0.15,
    loss_func = torch.nn.MSELoss(),
    custom_optimizer = torch.optim.Adam,
    modelname = None,
    config = None,
    log_model = True
):
    # graph learning system
    training_losses = []
    validation_losses = []
    
    optimizer = custom_optimizer(model.parameters(), lr = lr)
    last_run = False

    if log_model:
        wandb.init(project='dbr', config=config)

    for c,masks in enumerate(tqdm_notebook(cvs, desc="CV Loop")):
        last_run = c == len(cvs)-1
        if last_run and log_model:
            print(f"Logging model on wandb: {c}")
            wandb.watch(models=model, log='all', log_freq = 10)

        model = reset_model(model)
    #     print(f"Training fold: {c}")
        train_mask, val_mask = masks

        trainmask_ex = train_mask
        valmask_ex = val_mask

    #     print(f"Training idx sample: {np.random.choice(train_mask, 100).sum()}")
        fold_trainingloss = []
        fold_validationloss = []

        for epoch in tqdm_notebook(range(epochs), desc="Epoch", leave=False):
            model.train()

            rand = 1
            torch.manual_seed(rand)
            torch.cuda.manual_seed(rand)
            torch.cuda.manual_seed_all(rand)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(rand)
            random.seed(rand)

            loss = 0
            validation_loss = 0

            out = anonymous_function()
            #out = model(nycgraph.x_dict, nycgraph.edge_index_dict)
            out_ex = out.squeeze()

            training_predictions = out_ex[trainmask_ex].to(torch.float)
            training_values = recorded_values[trainmask_ex].to(torch.float)

            loss = loss_func(
                training_predictions,
                training_values
            ).to(torch.float)

            tloss = loss.detach().cpu().numpy()
            fold_trainingloss.append(tloss)
            if last_run and log_model:
                wandb.log({'training_loss': tloss})

    #         if epoch % 200 == 0:
    #             print(f"Loss at step: {epoch} is {loss}")
            model.eval()
            with torch.no_grad():
                valout = anonymous_function()
                valout_ex = valout.squeeze()
                validation_predictions = valout_ex[valmask_ex].to(torch.float)      
                validation_values = recorded_values[valmask_ex].to(torch.float)
                validation_loss = loss_func(
                    validation_predictions,
                    validation_values
                ).to(torch.float)

                vloss = validation_loss.detach().cpu().numpy()
                fold_validationloss.append(vloss)

                if last_run and log_model:
                    wandb.log({'validation_loss': vloss})

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
    
        training_losses.append(fold_trainingloss)
        validation_losses.append(fold_validationloss)

    if log_model:
        wandb.finish()
    return np.array(training_losses), np.array(validation_losses)