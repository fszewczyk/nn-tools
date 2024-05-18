import torch 

from nn_tools.shared.constants import *

def train(model: nn.Module,
          train_loader,
          n_epochs=10,
          lr=0.001,
          lr_decay=1.0,
          weight_decay=0.0,
          loss='L2',
          validation_loader=None,
          early_stop_length=None,
          verbose=False,
          print_every=1,
          additional_models=[]):

    stats = {
        'loss_iteration': [],
        'loss_epoch': [],
        'loss_validation': []
    }

    criterion = LOSS_FUNCTIONS[loss]()
    
    optimizers = [torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)]
    for m in additional_models:
        optimizers.append(torch.optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay))
    
    schedulers = []
    for o in optimizers:
        schedulers.append(torch.optim.lr_scheduler.ExponentialLR(o, gamma=lr_decay))

    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        for i, (img, true) in enumerate(train_loader):
            for o in optimizers:
                o.zero_grad()

            y_pred = model(img)

            loss = criterion(y_pred, true)
            loss.backward()

            for o in optimizers:
                o.step()

            train_loss += loss.item()
            stats['loss_iteration'].append(loss.item())
        
        for s in schedulers:
            s.step()

        train_loss = train_loss / len(train_loader)
        stats['loss_epoch'].append(train_loss)
        if verbose and epoch % print_every == 0:
            print(f"Epoch: {epoch} \t Loss: {train_loss}")

        if validation_loader is not None:
            validation_loss = 0.0
            for i, (img, true) in enumerate(validation_loader):
                y_pred = model(img)
                loss = criterion(y_pred, true)
                validation_loss += loss.item()
            validation_loss /= len(validation_loader)
            stats['loss_validation'].append(validation_loss)

            if verbose and epoch % print_every == 0:
                print(f"\tValidation loss: {validation_loss}")

            if early_stop_length is not None and len(stats['loss_validation']) > early_stop_length:
                is_improved = False
                past_result = stats['loss_validation'][-early_stop_length]
                for i in range(early_stop_length):
                    if stats['loss_validation'][-i] < past_result:
                        is_improved = True
                        break
                if not is_improved:
                    print("Early stopped")
                    return stats

    return stats