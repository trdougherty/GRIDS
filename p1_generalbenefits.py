#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import os
import sys
import copy
import random
import json

# In[2]:


import matplotlib.pyplot as plt


# In[3]:


import numpy as np
import networkx as nx

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)


# In[4]:


import wandb


# In[5]:


device = "cuda:0"


# In[6]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[7]:


module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


# In[68]:


# In[8]:


from src.cv import crossvalidation, build_masks


# In[9]:


import torch_geometric.transforms as T
transform = T.Compose([T.ToUndirected(), T.AddSelfLoops()])


# In[78]:
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, action="store", dest="output", default="p1")
parser.add_argument("--note", type=str, action="store", dest="note", default=None)
args = parser.parse_args()

results_dir = os.path.join('results', args.output)
os.makedirs(results_dir, exist_ok=True)

config = {
    "custom_optimizer": torch.optim.Adam,
    "lr" : 0.01,
    'epochs': 3000,
    'linear_layers': 2,
    'hidden_channels': 8,
    'dropout': 0.5,
    'graph_outchannels': 8,
    'graph_layers': 1,
    'graph_heads': 8,
    'test-percent': 30,
    'building_buffer': 100,
    'neighbor_radius': 100,
    'n_cv': 3,
    'cv_size': 600
}

config_filedata = copy.deepcopy(config)
config_filedata['custom_optimizer'] = str(config_filedata['custom_optimizer'])

with open(os.path.join(results_dir, "config.txt"), "w") as outfile:
    json.dump(config_filedata, outfile)

# In[84]:


f = open(os.path.join(results_dir, 'config_settings.txt'), "w")
if args.note is not None:
    f.write(str("NOTE:\n"))
    f.write(str(args.note))
    f.write("\n\n")

f.write(str("Config:\n"))
f.write(str(config))
f.close()


# In[11]:


from src.graph_construction import graph
nycgraph, nyc_rebuild_info = graph(
    "new-york",
    neighbor_radius=config['neighbor_radius'],
    building_buffer=config['building_buffer'],
    test_percent=config['test-percent']
)

# nycgraph = transform(nycgraph)


# In[85]:


f = open(os.path.join(results_dir, 'config_settings.txt'), "a")
f.write(str("\n\n"))
f.write(str("NY Graph:\n"))
f.write(str(nycgraph))
f.close()


# In[13]:


nycgraph['footprint'].y[:10]


# In[14]:


config['input_shape'] = nycgraph['footprint'].x.shape[1]


# In[15]:


nyc_rebuild_info['training_mask'].shape


# In[ ]:


from src.graph_construction import graph
sfgraph, sf_rebuild_info = graph(
    "san-fransisco",
    neighbor_radius=config['neighbor_radius'],
    building_buffer=config['building_buffer'],
    test_percent=config['test-percent']
)


# In[86]:


f = open(os.path.join(results_dir, 'config_settings.txt'), "a")
f.write(str("\n\n"))
f.write(str("San Fransisco Graph:\n"))
f.write(str(sfgraph))
f.close()


# In[16]:


from src.graph_construction import graph
austingraph, austin_rebuild_info = graph(
    "austin",
    neighbor_radius=config['neighbor_radius'],
    building_buffer=config['building_buffer'],
    test_percent=config['test-percent']
)


# In[87]:


f = open(os.path.join(results_dir, 'config_settings.txt'), "a")
f.write(str("\n\n"))
f.write(str("Austin Graph:\n"))
f.write(str(austingraph))
f.close()


# In[19]:


nyc_rebuild_info['training_mask'].sum()


# In[20]:


(~nyc_rebuild_info['training_mask']).sum()


# In[21]:


from sklearn.linear_model import LinearRegression


# In[22]:


loss_func = torch.nn.MSELoss()
nyrebuild_idx = torch.tensor(nyc_rebuild_info['rebuild_idx'])


# In[23]:


nyc_mask = nyc_rebuild_info['training_mask'].cpu().numpy()


# In[79]:


np.random.seed(0)
nyc_cvs = build_masks(
    nyc_mask, 
    n_cv = config['n_cv'],
    n_val = config['cv_size']
)


# In[25]:


valmean = []
valstd = []

trainmean = []
trainstd = []

nyX = nycgraph['footprint'].x[nyrebuild_idx].detach().cpu().numpy()
nyY = nycgraph['footprint'].y.cpu()

cvtrain_errors = []
cv_errors = []
for crossfold in nyc_cvs:
    nytrain, nyval = crossfold

    nytrain = nytrain[nyrebuild_idx]
    nyval = nyval[nyrebuild_idx]

    reg = LinearRegression().fit(nyX[nytrain], nyY[nytrain])
    overfitpred = reg.predict(nyX[nytrain])
    fitpred = reg.predict(nyX[nyval])

    overfitloss = float(loss_func(torch.tensor(overfitpred), nyY[nytrain]))
    regloss = float(loss_func(torch.tensor(fitpred), nyY[nyval]))

    cvtrain_errors.append(overfitloss)
    cv_errors.append(regloss)

linear_regtrainerr = np.mean(cvtrain_errors)
linear_regerr = np.mean(cv_errors)

trainmean.append(linear_regtrainerr)
valmean.append(linear_regerr)

trainstd.append(np.std(cvtrain_errors))
valstd.append(np.std(cv_errors))


# In[26]:


fullreg = LinearRegression().fit(nyX, nyY)

f = open(os.path.join(results_dir, 'config_settings.txt'), "a")
f.write(str("\n\n"))
f.write(str("Mean Building Vals - NYC:\n"))
f.write(str(nyc_rebuild_info['footprints'].describe(include=[np.number])))
f.write(str("\n\n"))
f.write(str("Mean Pano Vals - NYC:\n"))
f.write(str(nyc_rebuild_info['node_data'].describe(include=[np.number])))
f.write(str("\n\n"))
f.write(str("Regression Coefficients - New York:\n"))
f.write(str(fullreg.coef_)+"\n")
f.write(str(fullreg.get_params()))
f.write(str("\n\n"))
f.write(str("Mean Building Vals - San Fransisco:\n"))
f.write(str(sf_rebuild_info['footprints'].describe(include=[np.number])))
f.write(str("\n\n"))
f.write(str("Mean Pano Vals - San Fransisco:\n"))
f.write(str(sf_rebuild_info['node_data'].describe(include=[np.number])))
f.write(str("\n\n"))
f.write(str("Mean Building Vals - Austin:\n"))
f.write(str(austin_rebuild_info['footprints'].describe(include=[np.number])))
f.write(str("\n\n"))
f.write(str("Mean Pano Vals - Austin:\n"))
f.write(str(austin_rebuild_info['node_data'].describe(include=[np.number])))
f.write(str("\n\n"))
f.close()


# In[27]:


# plt.plot(range(ntesting), trainmean, label="training")
# plt.fill_between(
#     range(ntesting), 
#     np.array(trainmean) - np.array(trainstd), 
#     np.array(trainmean) + np.array(trainstd),
#     alpha=0.2
# ) 

# plt.plot(range(ntesting), valmean, label="validation")
# plt.fill_between(
#     range(ntesting), 
#     np.array(valmean) - np.array(valstd), 
#     np.array(valmean) + np.array(valstd),
#     alpha=0.2
# )

# plt.ylim(0.3, 1.3)
# plt.legend()
# 


# In[28]:


from src.model import NullModel

# input_shape = nycgraph['footprint'].x.shape[1]
nullmodel = NullModel(
    layers = config['linear_layers'],
    input_shape = config['input_shape'],
    hidden_channels = config['hidden_channels']
).to(device)

default_nullmodelstate = copy.deepcopy(nullmodel.state_dict())

null_training_tensor, null_validation_tensor = crossvalidation(
    nullmodel,
    lambda: nullmodel(nycgraph['footprint'].x),
    nyc_cvs,
    nyrebuild_idx,
    nycgraph['footprint'].y,
    epochs = config['epochs'],
    custom_optimizer = config['custom_optimizer'],
    lr = config['lr'],
    config = config,
    log_model = False
)


# In[29]:


null_mean_tl = null_training_tensor.mean(axis=0)
null_mean_vl = null_validation_tensor.mean(axis=0)

plt.plot(null_mean_tl, label="Null Training")
plt.plot(null_mean_vl, label="Null Validation")

plt.legend()

plt.yscale("log")
plt.ylim((0,5))


# In[30]:


config


# In[31]:


nycgraph


# In[32]:


# from torch import nn
# import torch_geometric.transforms as T
# from torch_geometric.nn import GATConv, Linear, to_hetero, GATv2Conv
# from torch_geometric.nn.conv.hetero_conv import HeteroConv

# custom_graphconv = GATv2Conv

# class CustomGAT(torch.nn.Module):
#     def __init__(
#             self, 
#             hidden_channels:int, 
#             out_channels:int, 
#             layers:int,
#             linear_layers:int,
#             input_shape:int,
#             heads:int = 1,
#             dropout = 0.5
#         ):
#         super().__init__()
#         self.layers = layers
        
#         self.convs = torch.nn.ModuleList()
#         self.lins = torch.nn.ModuleList()

#         self.nullmodel = NullModel(
#             layers = linear_layers,
#             input_shape = input_shape,
#             hidden_channels=hidden_channels
#         )
        
#         self.convs = torch.nn.ModuleList()
#         for _ in range(layers):
#             conv = HeteroConv({
#                 ('pano', 'links', 'pano'): custom_graphconv(-1, hidden_channels, add_self_loops = False, heads=heads),
#                 ('footprint', 'contains', 'pano'): custom_graphconv((-1, -1), hidden_channels, add_self_loops = False, heads=heads),
#                 ('pano', 'rev_contains', 'footprint'): custom_graphconv((-1, -1), hidden_channels, add_self_loops = False, heads=heads),
#             }, aggr='sum')
#             self.convs.append(conv)

#         self.lin = Linear(hidden_channels, out_channels)
#         self.mlp = nn.Sequential(
#             nn.Dropout(p=0.8),
#             nn.ReLU(),
#             nn.Linear(out_channels * heads, hidden_channels),
#             nn.Dropout(p=0.5),
#             nn.ReLU(),
#             nn.Linear(hidden_channels, hidden_channels),
#             nn.Dropout(p=0.5),
#             nn.ReLU(),
#             nn.Linear(hidden_channels, 1)
#         )

#     def forward(self, x_dict, edge_index_dict):
#         for conv in self.convs:
#             x_dict = conv(x_dict, edge_index_dict)
#             x_dict = {key: x.relu() for key, x in x_dict.items()}
#         return self.mlp(x_dict['footprint'])
# #         return x['pano']

# # model = to_hetero(model, data.metadata(), aggr='sum').to(device)


# In[33]:


config


# In[34]:


from src.model import CustomGAT

model = CustomGAT(
    hidden_channels = config['hidden_channels'], 
    out_channels=config['graph_outchannels'],
    layers=config['graph_layers'],
    heads=config['graph_heads'],
    linear_layers = config['linear_layers'],
    input_shape = config['input_shape'],
    dropout = config['dropout']
).to(device)

default_modelstate = copy.deepcopy(model.state_dict())


# In[35]:


graph_training_tensor, graph_validation_tensor = crossvalidation(
    model,
    lambda: model(nycgraph.x_dict, nycgraph.edge_index_dict),
    nyc_cvs,
    nyrebuild_idx,
    nycgraph['footprint'].y,
    epochs = config['epochs'],
    custom_optimizer = config['custom_optimizer'],
    lr = config['lr'],
    config = config,
    log_model = False
)


# In[36]:


mean_tl = graph_training_tensor.mean(axis=0)
std_tl = graph_training_tensor.std(axis=0)

mean_vl = graph_validation_tensor.mean(axis=0)
std_vl = graph_validation_tensor.std(axis=0)

plt.plot(mean_tl, label="Graph Training")
plt.plot(mean_vl, label="Graph Validation")
plt.legend()

plt.yscale("log")
plt.ylim((0,2))


# In[37]:


nycfootprints = nyc_rebuild_info['footprints']
nycfootprints['logenergy'] = np.log(nycfootprints.energy)

# nycfootprints.explore('logenergy')


# In[88]:


domain = np.arange(0, len(mean_tl))
plt.figure(figsize=(6, 6), dpi=400)

plt.plot(domain, mean_tl, label="Training")
plt.plot(domain, mean_vl, label="Validation", color="lightblue")
# plt.fill_between(domain, mean_vl + std_vl, mean_vl - std_vl, alpha=0.1)

plt.plot(domain, null_mean_tl, label="Null Training", color="indianred", linestyle='dashed')
plt.plot(domain, null_mean_vl, label="Null Validation", color="salmon", linestyle='dashed')

plt.hlines(regloss, min(domain), max(domain), color='pink', label="linear regression")

plt.title(f"Loss Function with Training - lr: {config['lr']}")

plt.xlabel("Epoch")
plt.ylabel("Training Loss - MSE")
plt.legend()

plt.yscale("log")
plt.ylim((0,2.5))

plt.savefig(os.path.join(results_dir, 'model_comparison.png'), bbox_inches="tight")



# In[39]:


graph_improvement = (min(mean_vl) - min(null_mean_vl)) / min(null_mean_vl)
print("Improvement from Context: {:0.2f}%".format(100*graph_improvement))


# In[40]:


def relative_benefit(errors, threshold):
    count_pass = (errors <= threshold).sum()
    return float(100 * (count_pass / len(errors)))


# In[41]:


# now examining how this may generalize
model.eval()
nullmodel.eval()
with torch.no_grad():
    print("New York City")
    testidx = (~nyc_rebuild_info['training_mask']).detach().cpu().numpy()
    rebuilding_idx = np.array(nyc_rebuild_info['rebuild_idx'])

    linear_predictions = reg.predict(nycgraph['footprint'].x.cpu().numpy())[rebuilding_idx][testidx[rebuilding_idx]]
    null_predictions = nullmodel(nycgraph['footprint'].x).squeeze()[rebuilding_idx][testidx[rebuilding_idx]]
    estimates = model(nycgraph.x_dict, nycgraph.edge_index_dict).squeeze().detach()[rebuilding_idx][testidx[rebuilding_idx]]
    recorded = nycgraph['footprint'].y[testidx[rebuilding_idx]]

    linear_loss = loss_func(torch.tensor(linear_predictions).to(device), recorded)
    null_loss = loss_func(null_predictions, recorded)
    graph_loss = loss_func(estimates, recorded)
    
    print("Linear Loss:\t{:0.2f}".format(linear_loss))
    print("Null Loss:\t{:0.2f}".format(null_loss))
    print("Graph Loss:\t{:0.2f}".format(graph_loss))
    print("Improvement:\t{:0.2f}".format(100 * (graph_loss - null_loss)/null_loss))


# In[42]:


linmae = torch.abs(torch.exp(recorded) - torch.tensor(np.exp(linear_predictions)).to(device))
nullmae = torch.abs(torch.exp(recorded) - torch.exp(null_predictions))
graphmae = torch.abs(torch.exp(recorded) - torch.exp(estimates))


# In[43]:


threshold = 1000

print(f"Linear Benefit:\t\t{relative_benefit(linmae, threshold)}")
print(f"Null Benefit:\t\t{relative_benefit(nullmae, threshold)}")
print(f"Graph Benefit:\t\t{relative_benefit(graphmae, threshold)}")


# In[44]:


torch.exp(null_predictions)


# In[45]:


testidx = (~sf_rebuild_info['training_mask']).detach().cpu().numpy()
testidx


# In[46]:


rebuilding_idx = np.array(sf_rebuild_info['rebuild_idx'])


# In[47]:


# now examining how this may generalize
nullmodel.eval()
model.eval()
with torch.no_grad():
    print("San Fransisco")
    testidx = (~sf_rebuild_info['training_mask']).detach().cpu().numpy()
    rebuilding_idx = np.array(sf_rebuild_info['rebuild_idx'])
    null_predictions = nullmodel(sfgraph['footprint'].x).squeeze()[rebuilding_idx][testidx[rebuilding_idx]]
    
    linear_predictions = reg.predict(sfgraph['footprint'].x.cpu().numpy())[rebuilding_idx][testidx[rebuilding_idx]]
    estimates = model(sfgraph.x_dict, sfgraph.edge_index_dict).squeeze().detach()[rebuilding_idx][testidx[rebuilding_idx]]
    recorded = sfgraph['footprint'].y[testidx[rebuilding_idx]]

    linear_loss = loss_func(torch.tensor(linear_predictions).to(device), recorded)
    null_loss = loss_func(null_predictions, recorded)
    graph_loss = loss_func(estimates, recorded)
    print("Linear Loss:\t{:0.2f}".format(linear_loss))
    print("Null Loss:\t{:0.2f}".format(null_loss))
    print("Graph Loss:\t{:0.2f}".format(graph_loss))
    print("Improvement:\t{:0.2f}".format(100 * (graph_loss - null_loss)/null_loss))


# In[48]:


nullmodel.eval()
model.eval()
with torch.no_grad():
    print("Austin Texas")
    testidx = (~austin_rebuild_info['training_mask']).detach().cpu().numpy()
    rebuilding_idx = np.array(austin_rebuild_info['rebuild_idx'])
    null_predictions = nullmodel(austingraph['footprint'].x).squeeze()[rebuilding_idx][testidx[rebuilding_idx]]
    
    linear_predictions = reg.predict(austingraph['footprint'].x.cpu().numpy())[rebuilding_idx][testidx[rebuilding_idx]]
    estimates = model(austingraph.x_dict, austingraph.edge_index_dict).squeeze().detach()[rebuilding_idx][testidx[rebuilding_idx]]
    recorded = austingraph['footprint'].y[testidx[rebuilding_idx]]
    
    linear_loss = loss_func(torch.tensor(linear_predictions).to(device), recorded)
    null_loss = loss_func(null_predictions, recorded)
    graph_loss = loss_func(estimates, recorded)
    print("Linear Loss:\t{:0.2f}".format(linear_loss))
    print("Null Loss:\t{:0.2f}".format(null_loss))
    print("Graph Loss:\t{:0.2f}".format(graph_loss))
    print("Improvement:\t{:0.2f}".format(100 * (graph_loss - null_loss)/null_loss))


# In[49]:


optim = config['custom_optimizer']
optim(model.parameters())


# In[50]:


### I used the below cells to evaluate how many epochs to use for the final model


# In[51]:


# from src.trainer import trainer
# from src.cv import reset_model

# model.load_state_dict(default_modelstate)
# model.train()

# trainmask = nyc_rebuild_info['training_mask']
# testmask = (~nyc_rebuild_info['training_mask'])

# rebuild_idx = np.array(nyc_rebuild_info['rebuild_idx'])

# nystate_dict_origin, trainlosses, testlosses = trainer(
#     model.to(device),
#     lambda: model(nycgraph.x_dict, nycgraph.edge_index_dict),
#     rebuild_idx,
#     trainmask[rebuild_idx],
#     testmask[rebuild_idx],
#     recorded = nycgraph['footprint'].y,
#     loss_func = loss_func,
#     config = config
# )

# nystate_dict = copy.deepcopy(nystate_dict_origin)


# In[52]:


# from src.trainer import trainer
# from src.cv import reset_model

# nullmodel.load_state_dict(default_nullmodelstate)

# trainmask = nyc_rebuild_info['training_mask']
# testmask = (~nyc_rebuild_info['training_mask'])

# rebuild_idx = np.array(nyc_rebuild_info['rebuild_idx'])

# nystate_dict_origin, trainlosses, testlosses = trainer(
#     nullmodel.to(device),
#     lambda: nullmodel(nycgraph['footprint'].x),
#     rebuild_idx,
#     trainmask[rebuild_idx],
#     testmask[rebuild_idx],
#     recorded = nycgraph['footprint'].y,
#     loss_func = loss_func,
#     config = config
# )

# nynull_state_dict = copy.deepcopy(nystate_dict_origin)


# In[53]:


# plt.plot(trainlosses, label="training")
# plt.plot(testlosses, label="testing")

# plt.legend()
# plt.yscale("log")

# 


# In[54]:


# this is manually set based on the above graph^
from src.trainer import trainer

model.load_state_dict(default_modelstate)

trainmask = nyc_rebuild_info['training_mask']
testmask = (~nyc_rebuild_info['training_mask'])

trainmask = np.repeat([True], len(trainmask))
rebuild_idx = np.array(nyc_rebuild_info['rebuild_idx'])

nystate_dict_origin, trainlosses, testlosses = trainer(
    model.to(device),
    lambda: model(nycgraph.x_dict, nycgraph.edge_index_dict),
    rebuild_idx,
    trainmask[rebuild_idx],
    testmask[rebuild_idx],
    recorded = nycgraph['footprint'].y,
    loss_func = loss_func,
    config = config,
    fulldata = True
)

nystate_dict = copy.deepcopy(nystate_dict_origin)


# In[55]:


nullmodel.train()
nullmodel.load_state_dict(default_nullmodelstate)

trainmask = nyc_rebuild_info['training_mask']
testmask = (~nyc_rebuild_info['training_mask'])

trainmask = np.repeat([True], len(trainmask))
rebuild_idx = np.array(nyc_rebuild_info['rebuild_idx'])

nystate_dict_origin, trainlosses, testlosses = trainer(
    nullmodel.to(device),
    lambda: nullmodel(nycgraph['footprint'].x),
    rebuild_idx,
    trainmask[rebuild_idx],
    testmask[rebuild_idx],
    recorded = nycgraph['footprint'].y,
    loss_func = loss_func,
    config = config,
    fulldata = True
)

nystate_nulldict = copy.deepcopy(nystate_dict_origin)


# In[56]:


np.vstack((nyX, sfgraph['footprint'].x[:50].cpu().detach())).shape


# In[57]:


from tqdm import tqdm

# now exploring how the model might generalize to SF
sf_linearvalloss = []
sf_valloss = []
sf_nullvalloss = []

config['epochs'] = 1000

for n_true in tqdm(range(100), leave=True):
    model.load_state_dict(nystate_dict)
    nullmodel.load_state_dict(nystate_nulldict)
    
    model.train()
    nullmodel.train()

    sf_trainmask = sf_rebuild_info['training_mask']
    sf_testmask = (~sf_rebuild_info['training_mask'])

    sf_to_false = np.where(sf_trainmask.cpu().numpy())[0]
    sf_to_false

    np.random.seed(1)
    drip_idx = np.random.choice(sf_to_false, n_true, replace=False)

    # this now just drips in a bit of the sf data
    trainmask = np.repeat([False], len(sf_trainmask))
    trainmask[drip_idx] = True

    rebuild_idx = np.array(sf_rebuild_info['rebuild_idx'])
    
    # shapes
    # print(f"trainmask shape: {trainmask.shape}")
    
    ## building the linear model
    x_linear_addition = np.vstack((nyX, sfgraph['footprint'].x[rebuild_idx][trainmask[rebuild_idx]].cpu().detach()))
    
    # print(f"Y shape: {nyY.shape}.")
    # print(f"trainyshape: {sfgraph['footprint'].y[trainmask[rebuild_idx]].cpu().detach().shape}")
    y_linear_addition = np.concatenate((nyY, sfgraph['footprint'].y[trainmask[rebuild_idx]].cpu().detach()), axis=None)
    
    reg = LinearRegression().fit(x_linear_addition, y_linear_addition)
    preds = reg.predict(sfgraph['footprint'].x[rebuild_idx][testmask[rebuild_idx]].cpu().detach())
    existing_terms = sfgraph['footprint'].y[testmask[rebuild_idx]].cpu().detach()
    # print(f"Predictions: {preds}")
    # print(f"Existing: {existing_terms}")
    sf_linearvalloss.append(loss_func(torch.tensor(preds), existing_terms))

    sf_state_dict, sf_trainlosses, sf_testlosses = trainer(
        model.to(device),
        lambda: model(sfgraph.x_dict, sfgraph.edge_index_dict),
        rebuild_idx,
        trainmask,
        testmask,
        recorded = sfgraph['footprint'].y,
        loss_func = loss_func,
        config = config
    )
    
    sf_state_dict_null, _, sf_nulltestlosses = trainer(
        nullmodel.to(device),
        lambda: nullmodel(sfgraph['footprint'].x),
        rebuild_idx,
        trainmask,
        testmask,
        recorded = sfgraph['footprint'].y,
        loss_func = loss_func,
        config = config
    )
    sf_valloss.append(min(sf_testlosses))
    sf_nullvalloss.append(min(sf_nulltestlosses))


# In[73]:


# plt.plot(sf_trainlosses, label="training")
plt.figure(figsize=(6, 6), dpi=400)

plt.plot(sf_linearvalloss, label="Linear", color="coral")
plt.plot(sf_nullvalloss, label="Null", color="orange")
plt.plot(sf_valloss, label="Graph", color="lightblue")

plt.legend()
plt.yscale("log")

plt.title("Generalization from New York to San Fransisco")
plt.xlabel("# Buildings from San Fransisco")
plt.ylabel("Loss - RMSE")

plt.ylim((0,2.5))

plt.savefig(os.path.join(results_dir, 'sf_generalization.png'))



# In[59]:


# In[60]:


from tqdm import tqdm

# now exploring how the model might generalize to SF
austin_linearvalloss = []
austin_valloss = []
austin_nullvalloss = []

config['epochs'] = 1000

for n_true in tqdm(range(10), leave=True):
    model.load_state_dict(nystate_dict)
    nullmodel.load_state_dict(nystate_nulldict)

    austin_trainmask = austin_rebuild_info['training_mask']
    austin_testmask = (~austin_rebuild_info['training_mask'])

    austin_to_false = np.where(austin_trainmask.cpu().numpy())[0]
    austin_to_false

    np.random.seed(1)
    drip_idx = np.random.choice(austin_to_false, n_true, replace=False)

    # this now just drips in a bit of the sf data
    trainmask = np.repeat([False], len(austin_trainmask))
    trainmask[drip_idx] = True

    rebuild_idx = np.array(austin_rebuild_info['rebuild_idx'])
    
    # shapes
    # print(f"trainmask shape: {trainmask.shape}")
    
    ## building the linear model
    x_linear_addition = np.vstack((nyX, austingraph['footprint'].x[rebuild_idx][trainmask[rebuild_idx]].cpu().detach()))
    
    # print(f"Y shape: {nyY.shape}.")
    # print(f"trainyshape: {sfgraph['footprint'].y[trainmask[rebuild_idx]].cpu().detach().shape}")
    y_linear_addition = np.concatenate((nyY, austingraph['footprint'].y[trainmask[rebuild_idx]].cpu().detach()), axis=None)
    
    reg = LinearRegression().fit(x_linear_addition, y_linear_addition)
    preds = reg.predict(austingraph['footprint'].x[rebuild_idx][testmask[rebuild_idx]].cpu().detach())
    existing_terms = austingraph['footprint'].y[testmask[rebuild_idx]].cpu().detach()
    # print(f"Predictions: {preds}")
    # print(f"Existing: {existing_terms}")
    austin_linearvalloss.append(loss_func(torch.tensor(preds), existing_terms))

    austin_state_dict, austin_trainlosses, austin_testlosses = trainer(
        model.to(device),
        lambda: model(austingraph.x_dict, austingraph.edge_index_dict),
        rebuild_idx,
        trainmask,
        testmask,
        recorded = austingraph['footprint'].y,
        loss_func = loss_func,
        config = config
    )
    
    _, _, austin_nulltestlosses = trainer(
        nullmodel.to(device),
        lambda: nullmodel(austingraph['footprint'].x),
        rebuild_idx,
        trainmask,
        testmask,
        recorded = austingraph['footprint'].y,
        loss_func = loss_func,
        config = config
    )
    austin_valloss.append(min(austin_testlosses))
    austin_nullvalloss.append(min(austin_nulltestlosses))


# In[74]:


# plt.plot(sf_trainlosses, label="training")
plt.figure(figsize=(6, 6), dpi=400)

plt.plot(austin_linearvalloss, label="Linear", color="firebrick")
plt.plot(austin_nullvalloss, label="Null", color="lightsalmon")
plt.plot(austin_valloss, label="Graph", color="lightblue")

plt.legend()
plt.yscale("log")

plt.title("Generalization from New York to Austin")
plt.xlabel("# Buildings from Austin")
plt.ylabel("Loss - RMSE")

plt.ylim((0,2.5))
plt.savefig(os.path.join(results_dir, 'austin_generalization.png'))


#### now to save all the model configs
torch.save({
    'default_graph_model': default_modelstate,
    'nygraph_model': nystate_dict,
    'default_null_model': default_nullmodelstate,
    'ny_null_model': nystate_nulldict,
    'sf_extension': sf_state_dict,
    'sf_extension_null': sf_state_dict_null
}, os.path.join(results_dir, "state_dicts.tar"))

# In[62]:


# ookaaay now I want to see how the model might generalize


# In[63]:


# param_size = 0
# for param in model.parameters():
#     param_size += param.nelement() * param.element_size()
# buffer_size = 0
# for buffer in model.buffers():
#     buffer_size += buffer.nelement() * buffer.element_size()

# size_all_mb = (param_size + buffer_size) / 1024**2
# print('model size: {:.3f}MB'.format(size_all_mb))

