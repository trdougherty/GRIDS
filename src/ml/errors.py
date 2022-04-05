import torch

def cvstd(y_pred, y, p:int=1):
    ### CV(STD)
    return ( (torch.sum((y-torch.mean(y))**2) / (len(y)-1)) **0.5 ) / torch.mean(y)

def cvrmse(y_pred, y, p:int=1):
    ### Provides the CV(RMSE) loss, as defined by ASHRAE: https://upgreengrade.ir/admin_panel/assets/images/books/ASHRAE%20Guideline%2014-2014.pdf
    return ( (torch.sum((y-y_pred)**2) / (len(y)-p) )**0.5 ) / torch.mean(y)

def nmbe(y_pred, y, p:int=1):
    ### Provides the NMBE again, as defined by ASHRAE^
    return  torch.sum(y-y_pred) / ( (len(y)-p)*torch.mean(y) )

def error_suite(y_pred, y):
    cvstd_error = cvstd(y_pred, y)
    cvrmse_error = cvrmse(y_pred, y)
    nmbe_error = nmbe(y_pred, y)

    return {
        "CV(STD)": cvstd_error,
        "CV(RMSE)": cvrmse_error,
        "NMBE": nmbe_error
    }