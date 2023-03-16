import os, sys
import random
import torch
import numpy as np
import torch_geometric
from typing import List
import math
from tqdm import tqdm_notebook

def trainer(
        model,
        anonymous_function, 
        rebuilding_idx:List, 
        mask:List,
        valmask:List,
        recorded, 
        loss_func, 
        config,
        fulldata:bool = False
    ):
    """Serves as a basemodel for all subsequent training"""
    optimizer = config['custom_optimizer'](model.parameters())
    training_losses = []
    test_losses = []
    minimum_validationloss = math.inf
    state_dict = None

    np.random.seed(1)
    torch.manual_seed(1)
    random.seed(0)

    for epoch in tqdm_notebook(range(config['epochs']), desc="Epoch", leave=False):
        loss = 0
        validation_loss = 0

        out = anonymous_function()
        #out = model(nycgraph.x_dict, nycgraph.edge_index_dict)
        out_ex = out.squeeze()[rebuilding_idx]

        training_predictions = out_ex[mask[rebuilding_idx]].to(torch.float)
        training_values = recorded[mask[rebuilding_idx]].to(torch.float)

        loss = loss_func(
            training_predictions,
            training_values
        ).to(torch.float)

        tloss = loss.detach().cpu().numpy()
        training_losses.append(tloss)

        if fulldata == False:
            model.eval()
            with torch.no_grad():
                valout = anonymous_function()
                #out = model(nycgraph.x_dict, nycgraph.edge_index_dict)
                valout_ex = valout.squeeze()[rebuilding_idx]
                validation_predictions = valout_ex[valmask[rebuilding_idx]].to(torch.float)      
                validation_values = recorded[valmask[rebuilding_idx]].to(torch.float)
                validation_loss = float(loss_func(
                    validation_predictions,
                    validation_values
                ).to(torch.float).detach().cpu().numpy())

                if validation_loss < minimum_validationloss:
                    # print(f"Saving model at epoch: {epoch}.\tLoss = {validation_loss:0.2f}")
                    state_dict = model.state_dict()
                    minimum_validationloss = validation_loss

                test_losses.append(validation_loss)
            model.train()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if fulldata == True:
        state_dict = model.state_dict()

    return state_dict, training_losses, test_losses