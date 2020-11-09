import os
from markdown import markdown
import numpy as np

import torch
import torch.nn as nn


from .args import args



# ----- Random Seed Control -----

def fix_random_seed(seed=0):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True


# ----- Save Model -----

def save_model(model, pth='./src/models/trained_models/'):
    """ Saves a torch model in two ways: to be retrained and/or for validation only.
    """
    pth = os.path.join(pth, args.dataset)
    if not os.path.exists(pth):
        os.makedirs(pth)

    # model to be used ONLY for inference
    m = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(m.state_dict(), os.path.join(pth, args.model + '.pt'))
    print('\nModel saved!')
    return


# ----- Tensorboard Utils -----

def namespace2markdown(args):
    txt = '<table> <thead> <tr> <td> <strong> Hyperparameter </strong> </td> <td> <strong> Values </strong> </td> </tr> </thead>'
    txt += ' <tbody> '
    for name, var in vars(args).items():
        txt += '<tr> <td> <code>' + str(name) + ' </code> </td> ' + '<td> <code> ' + str(var) + ' </code> </td> ' + '<tr> '
    txt += '</tbody> </table>'
    return markdown(txt)


if __name__ == "__main__":
    pass
