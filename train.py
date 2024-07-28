# import package
# model
import torch

# utils
import time
from tqdm import tqdm

# # 3. Training part
# function to get current lr
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

# function to calculate metric per mini-batch
def metric_batch(output, target):
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects

# function to calculate loss per mini-batch
def loss_batch(loss_func, output_list, target, opt=None):
    losses = [loss_func(output,target) for output in output_list]
    metric_bs = [metric_batch(output, target) for output in output_list]
    if opt is not None:
        opt.zero_grad()
        #backprop
        for loss in losses[:-1]:
            #ee losses need to keep graph
            loss.backward(retain_graph=True)
        #final loss, graph not required
        losses[-1].backward()
        opt.step()
    return losses, metric_bs

# function to calculate loss and metric per epoch
def loss_epoch(model, loss_func, dataset_dl, opt=None):
    device = next(model.parameters()).device
    running_loss = 0.0
    running_metric = [0.0] * model.exit_num
    len_data = len(dataset_dl.dataset)
    tqdm_state = f'batch_training' if(opt is not None) else f'batch_validation'
    for xb, yb in tqdm(dataset_dl, desc=tqdm_state, leave=False):
        xb = xb.to(device)
        yb = yb.to(device)
        output_list = model(xb)

        losses, metric_bs = loss_batch(loss_func, output_list, yb, opt)
        for i, _ in enumerate(losses):
            running_loss += losses[i].item()
        running_metric = [sum(i) for i in zip(running_metric,metric_bs)]

    loss = running_loss / len_data # float
    metric = [100*i/len_data for i in running_metric] # float list[exit_num]

    return loss, metric

# function to start training
def train_val(model, params):
    num_epochs=params['num_epochs']
    loss_func=params["loss_func"]
    opt=params["optimizer"]
    train_dl=params["train_dl"]
    val_dl=params["val_dl"]
    lr_scheduler=params["lr_scheduler"]
    path2weights=params["path2weights"]

    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}

    # # GPU out of memoty error
    # best_model_wts = copy.deepcopy(model.state_dict())

    best_loss = float('inf')

    start_time = time.time()

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs-1, current_lr))

        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, opt)
        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl)
        loss_history['val'].append(val_loss)
        metric_history['val'].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            #best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            print('saved best model weights!')
            print('Get best val_loss')

        lr_scheduler.step(val_loss)

        print(f'train loss: {train_loss:.6f}, val loss: {val_loss:.6f}, accuracy: {val_metric}, time: {(time.time()-start_time)/60:.4f} min')
        print('-'*10)

    #model.load_state_dict(best_model_wts)

    return model, loss_history, metric_history