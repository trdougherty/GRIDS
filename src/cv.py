import os
import sys
import torch
import numpy as np
import copy
from tqdm import tqdm_notebook

# now I want to build training and validation masks
def build_masks(training_mask, n_val = 100, n_cv = 10):
    true_idx = np.where(training_mask)[0]
    validation_batches = []

    np.random.seed(999)
    for i in range(n_cv):
        validation_batches.append(
            np.random.choice(true_idx, n_val, replace=False)
        )
    
    cv_masks = []
    for validation_idx in validation_batches:
        training_mask = copy.deepcopy(training_mask)
        training_mask[validation_idx] = False

        validation_mask = np.repeat([False], len(training_mask))
        validation_mask[validation_idx] = True

        cv_masks.append(
            (training_mask, validation_mask)
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
    rebuilding_data,
    epochs = 100,
    lr = 0.15,
    loss_func = torch.nn.MSELoss(),
    custom_optimizer = torch.optim.Adam
):
    # graph learning system
    training_losses = []
    validation_losses = []
    
    rebuilding_idx = rebuilding_data['rebuild_idx'].cpu().numpy()
    optimizer = custom_optimizer(model.parameters(), lr = lr)
    for c,masks in enumerate(tqdm_notebook(cvs, desc="CV Loop")):
        model = reset_model(model)
    #     print(f"Training fold: {c}")
        train_mask, val_mask = masks

        trainmask_ex = train_mask[rebuilding_idx]
        valmask_ex = val_mask[rebuilding_idx]

        np.random.seed(1)
    #     print(f"Training idx sample: {np.random.choice(train_mask, 100).sum()}")
        fold_trainingloss = []
        fold_validationloss = []

        for _ in tqdm_notebook(range(epochs), desc="Epoch", leave=False):
            loss = 0
            validation_loss = 0

            optimizer.zero_grad()
            out = anonymous_function()
            #out = model(nycgraph.x_dict, nycgraph.edge_index_dict)
            out_ex = out.squeeze()[rebuilding_idx]

            training_predictions = out_ex[trainmask_ex].to(torch.float)
            training_values = rebuilding_data['recorded'][trainmask_ex].to(torch.float)

            validation_predictions = out_ex[valmask_ex].to(torch.float)      
            validation_values = rebuilding_data['recorded'][valmask_ex].to(torch.float)

            loss = loss_func(
                training_predictions,
                training_values
            ).to(torch.float)

            fold_trainingloss.append(loss.detach().cpu().numpy())

    #         if epoch % 200 == 0:
    #             print(f"Loss at step: {epoch} is {loss}")
            with torch.no_grad():
                validation_loss = loss_func(
                    validation_predictions,
                    validation_values
                ).to(torch.float)
                fold_validationloss.append(validation_loss.detach().cpu().numpy())

            loss.backward()
            optimizer.step()

        training_losses.append(fold_trainingloss)
        validation_losses.append(fold_validationloss)

    return np.array(training_losses), np.array(validation_losses)