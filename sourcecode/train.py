# import package
# model
import torch
from torch.utils.tensorboard import SummaryWriter

# utils
import time
from tqdm import tqdm
from multi_exit_ResNet import createFolder

# # 3. Training part
# function to get current lr
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

# function to calculate metric per mini-batch
def metric_batch(output, label):
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(label.view_as(pred)).sum().item()
    return corrects

# function to calculate loss per mini-batch
def loss_batch(loss_func, output_list, label, opt=None):
    losses = [loss_func(output,label)/len(output_list) for output in output_list]   # raw losses -> 굳이 각각 exit의 길이로 나눠줘야하나? 로스 크기만 달라지지 않나;;
    acc_s = [metric_batch(output, label) for output in output_list]
    
    if opt is not None:
        opt.zero_grad()
        #backprop
        for loss in losses[:-1]:
            #ee losses need to keep graph
            loss.backward(retain_graph=True)
        #final loss, graph not required
        losses[-1].backward()
        opt.step()
    
    losses = [loss.item() for loss in losses] #for out of cuda memory error
    
    return losses, acc_s

# function to calculate loss and metric per epoch
def loss_epoch(model, loss_func, dataset_dl, writer, epoch, opt=None):
    device = next(model.parameters()).device
    running_loss = [0.0] * model.exit_num
    running_metric = [0.0] * model.exit_num
    len_data = len(dataset_dl.dataset)
    TorV='train' if opt is not None else 'val'
    for xb, yb in tqdm(dataset_dl, desc=TorV, leave=False):
        xb = xb.to(device)
        yb = yb.to(device)
        output_list = model(xb)

        losses, acc_s = loss_batch(loss_func, output_list, yb, opt)

        running_loss = [sum(i) for i in zip(running_loss,losses)]
        running_metric = [sum(i) for i in zip(running_metric,acc_s)]
    
    running_loss=[i/len_data for i in running_loss]
    running_acc=[100*i/len_data for i in running_metric]
    
    # Tensorboard
    tmp_loss_dict = dict();tmp_acc_dict = dict()
    for idx in range(model.exit_num):
        tmp_loss_dict[f'exit{idx}'] = running_loss[idx];tmp_acc_dict[f'exit{idx}'] = running_acc[idx]
    writer.add_scalars(f'{TorV}/loss', tmp_loss_dict, epoch)
    writer.add_scalars(f'{TorV}/acc', tmp_acc_dict, epoch)
    
    losses_sum = sum(running_loss) # float
    writer.add_scalar(f'{TorV}/loss_total_sum', losses_sum, epoch)
    accs = running_acc # float list[exit_num]

    return losses_sum, accs

# function to start training
def train_val(model, params):   #TODO 모델 불러오기
    num_epochs=params['num_epochs']
    loss_func=params["loss_func"]
    opt=params["optimizer"]
    train_dl=params["train_dl"]
    val_dl=params["val_dl"]
    lr_scheduler=params["lr_scheduler"]
    load=params["load"]
    loaded_loss=params["loaded_loss"]
    
    best_loss = float('inf') if not load else loaded_loss
    start_time = time.time()
    
    # path to save the model weights
    current_time = time.strftime('%m_%d_%H%M%S', time.localtime())
    path=createFolder('./models/'+current_time)
    
    #writer=None
    writer = SummaryWriter()
    writer.add_graph(model, torch.rand(1,3,224,224).to(next(model.parameters()).device))
    
    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs-1, current_lr))

        model.train()
        train_loss, train_accs = loss_epoch(model, loss_func, train_dl, writer, epoch, opt)

        model.eval()
        with torch.no_grad():
            val_loss, val_accs = loss_epoch(model, loss_func, val_dl, writer, epoch, opt=None)

        if val_loss < best_loss:
            best_loss = val_loss
            #best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path+'/best_model.pth')
            print('saved best model weights!')
            print('Get best val_loss')

        lr_scheduler.step(val_loss)

        total_time=(time.time()-start_time)/60
        hours, minutes = divmod(total_time, 60)
        print(f'train_loss: {train_loss:.6f}, train_acc: {train_accs}')
        print(f'val_loss: {val_loss:.6f}, val_acc: {val_accs}, time: {int(hours)}h {int(minutes)}m')
        print('-'*10)
        writer.flush()
    
    torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': val_loss,
            }, path+'/chckpoint.pth')
    writer.close()
    
    return model