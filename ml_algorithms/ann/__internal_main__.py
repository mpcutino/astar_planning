import torch
import numpy as np
from ast import literal_eval
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from ml_algorithms.ann.training.methods import standard
from ml_algorithms.ann.loaders.dataset import StandardDS
from ml_algorithms.ann.structures.dummy import DirectSNet
from ml_algorithms.ann.loaders.function import load_standard_data

from data.load_data import load_csv


if __name__ == '__main__':
    # ====== PARAMS
    train_csv_path_ = "../../data/landing_train.csv"
    test_csv_path_ = "../../data/landing_test.csv"

    action_col_ = "action_codes"
    continuous_var = ['u', 'v', 'theta', 'omega', 'x', 'z']
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
    train_df = load_csv(train_csv_path_)
    test_df = load_csv(test_csv_path_)

    X_test, y_test = test_df[continuous_var].values, test_df[action_col_].values
    X_train, X_val, y_train, y_val = train_test_split(train_df[continuous_var].values, train_df[action_col_].values,
                                                      shuffle=True, test_size=.2, stratify=train_df[action_col_],
                                                      random_state=40)
    st_scaler = StandardScaler()
    st_scaler.fit(train_df[continuous_var])
    X_train = st_scaler.transform(X_train)
    X_val = st_scaler.transform(X_val)
    X_test = st_scaler.transform(X_test)

    loaders_dict = {
        'train': load_standard_data(
            StandardDS(X_train, y_train), use_cuda_, device_, batch_size=128
        ),
        'eval': load_standard_data(
            StandardDS(X_val, y_val), use_cuda_, device_, batch_size=128
        )
    }
    m = DirectSNet(X_train.shape[1], len(train_df[action_col_].unique()))
    m.to(device_)
    # TODO!! Put all of this in one function. Do Feature Engineering (add more characteristics. For instance,
    #  less than half of the trajectory traveled)
    vc = train_df[action_col_].value_counts()
    loss_w = torch.FloatTensor((1 - vc/vc.sum())**5).to(device_)
    standard(10, m, loaders_dict['train'], torch.optim.RMSprop(m.parameters()), torch.nn.CrossEntropyLoss(weight=loss_w),
             accuracy_function=lambda y_hat, y: (y_hat.argmax(dim=1) == y).sum().float(), print_every=2)
