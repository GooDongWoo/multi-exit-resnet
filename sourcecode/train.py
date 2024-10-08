# import package
# model
import torch
from torch.utils.tensorboard import SummaryWriter

# utils
import time
import os
from tqdm import tqdm

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
def loss_batch(loss_func, output_list, label, elws, opt=None):
    losses = [loss_func(output,label)*elw for output,elw in zip(output_list,elws)]   # raw losses -> 굳이 각각 exit의 길이로 나눠줘야하나? 로스 크기만 달라지지 않나;;
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
        elws=model.getELW()

        losses, acc_s = loss_batch(loss_func, output_list, yb, elws, opt)

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
    isload=params["isload"]
    path_chckpnt=params["path_chckpnt"]
    resize=params["resize"]
    
    start_time = time.time()
    
    # path to save the model weights
    current_time = time.strftime('%m%d_%H%M%S', time.localtime())
    path=f'./models/{current_time}'
    os.makedirs(path, exist_ok=True)
    
    spec_txt=f'opt: {opt.__class__.__name__}\nlr: {opt.param_groups[0]["lr"]}\nbatch: {train_dl.batch_size}\nepoch: {num_epochs}\nisload: {isload}\npath_chckpnt: {path_chckpnt}\n'
    with open(f"{path}/spec.txt", "w") as file:
        file.write(spec_txt)
    
    best_loss = float('inf')
    old_epoch=0
    if(isload):
        chckpnt = torch.load(path_chckpnt,weights_only=True)
        model.load_state_dict(chckpnt['model_state_dict'])
        opt.load_state_dict(chckpnt['optimizer_state_dict'])
        old_epoch = chckpnt['epoch']
        best_loss = chckpnt['loss']
    
    #writer=None
    writer = SummaryWriter('./runs/'+current_time,)
    writer.add_graph(model, torch.rand(1,3,resize,resize).to(next(model.parameters()).device))
    
    for epoch in range(old_epoch,old_epoch+num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, old_epoch+num_epochs-1, current_lr))

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
    linedivisor='#'*10+'\n'
    result_txt=linedivisor+f'last_val_acc: {val_accs}\nlast_train_acc: {train_accs}\nlast_val_loss: {best_loss:.6f}\ntotal_time: {total_time:.2f}m\n'
    with open(f"{path}/spec.txt", "a") as file:
        file.write(result_txt)    
    
    return model