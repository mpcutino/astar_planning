import torch
import numpy as np
from ast import literal_eval
from sklearn.model_selection import train_test_split

from ann.training.methods import standard
from ann.loaders.dataset import StandardDS
from ann.structures.dummy import DirectSNet
from ann.training.data_utils import normalize
from ann.loaders.function import load_standard_data

from data.load_data import load_csv


if __name__ == '__main__':
    # ====== PARAMS
    csv_path_ = "../data/data.csv"
    state_col_ = "difference_state"
    action_col_ = "action"
    use_cuda_ = torch.cuda.is_available()
    device_ = torch.device('cuda' if use_cuda_ else 'cpu')
    #
    # Additional Info when using cuda
    if device_.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')
    else:
        print("Not using cuda")

    tr_func = lambda x: np.array(literal_eval(x))
    df_ = load_csv(csv_path_, **{
        'converters': {"current_state": tr_func, "target_state": tr_func},
    })
    df_.dropna(inplace=True)
    df_[state_col_] = df_["target_state"] - df_["current_state"]
    normalized_dict = normalize(df_, state_col_, action_col_)

    X_train, X_test, y_train, y_test = train_test_split(normalized_dict['states'], normalized_dict['actions'],
                                                        shuffle=True, test_size=.3)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      shuffle=True, test_size=.2)
    loaders_dict = {
        'train': load_standard_data(
            StandardDS(X_train, y_train), use_cuda_, device_, batch_size=128
        ),
        'eval': load_standard_data(
            StandardDS(X_val, y_val), use_cuda_, device_, batch_size=128
        )
    }
    m = DirectSNet(6, len(normalized_dict['categories']))
    m.to(device_)
    # TODO!! Put all of this in one function. Do Feature Ingeneering (add more characteristics. For instance,
    #  less than half of the trajectory traveled)
    standard(10, m, loaders_dict['train'], torch.optim.RMSprop(m.parameters()), torch.nn.CrossEntropyLoss(),
             accuracy_function=lambda y_hat, y: (y_hat.argmax(dim=1) == y).sum().float(), print_every=1)
