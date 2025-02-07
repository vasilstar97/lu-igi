import torch
from tqdm import tqdm
import pandas as pd
from .gcn import GCN, CLASSES, NUM_CLASSES, DEVICE
from sklearn.model_selection import train_test_split

def get_masks(y, train_size : float =0.9):

    # labels = y.numpy()

    # Разбиваем данные с учетом распределения классов
    train_indices, test_indices = train_test_split(
        range(len(y)),
        train_size=train_size,
        stratify=y,  # Это обеспечит сохранение пропорций классов
    )

    # Создаем маски для обучающих и тестовых данных
    train_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)

    train_mask[train_indices] = True
    test_mask[test_indices] = True

    return train_mask, test_mask

def get_stratification(y, train_mask, test_mask) -> pd.DataFrame:
    train_y = list(y[train_mask])
    test_y = list(y[test_mask])

    y_df = pd.DataFrame.from_dict({
        'train': [train_y.count(-1), *[train_y.count(i) for i,_ in enumerate(CLASSES)]],
        'test': [test_y.count(-1), *[test_y.count(i) for i,_ in enumerate(CLASSES)]],
    }, orient='columns')
    y_df.index = ['None', *CLASSES]
    return y_df


def train_model(data, mask, epochs : int = 1_000, lr : float =3e-4) -> GCN:
    num_node_features = data.x.shape[1]
    model = GCN(num_node_features, NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # lp=0.01, weight_decay=1e-4 #3e-4

    class_counts = torch.bincount(data.y[data.y != -1])
    class_weights = 1. / class_counts.float()

    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1).to(DEVICE)
    model.train()
    
    pbar = tqdm(range(epochs))
    for _ in pbar:
        out = model(data)
        optimizer.zero_grad()
        loss = loss_function(out[mask], data.y[mask])
        loss.backward()
        pbar.set_description(f'loss : {round(loss.item(),2)}')
        optimizer.step()

    return model