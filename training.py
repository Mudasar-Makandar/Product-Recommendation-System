import torch
import torch.nn as nn
import torch.optim as optim



def train_batch(net, opt, loss_fn,data_loader, batch_size, teacher_force=True):
    net.train().to(device)
    opt.zero_grad()
    data_iter = iter(dataloader)
    profile, product = data_iter.next()
    profile_input = profile.type(torch.FloatTensor).to(device)
    product = product.view(product.size(1),1,24)
    product_input = product[0:10,:,:].type(torch.FloatTensor).to(device)
    target = product[10:,:,:].type(torch.FloatTensor).to(device)
    
    total_loss=0
    for i in range(batch_size):
        outputs = net(profile_input, product_input, ground_truth=target if teacher_force else None)
        for index in range(len(outputs)):
            output=torch.Tensor(outputs[index]).to(device)
            loss = loss_fn(output.double(), target[index,:,:].view(24).double())
            loss.backward(retain_graph=True)
            total_loss+=loss.item()
            print(loss.item())
    opt.step()
    print(total_loss)    
    return total_loss
    


from torch.utils.data import DataLoader
dataloader = DataLoader(data_loader, batch_size=1)
loss_fn = nn.MSELoss()
opt = optim.Adam(net.parameters(), lr=0.01)
batch_size=1
train_batch(net, opt, loss_fn,dataloader, batch_size, teacher_force=True)




def train_setup(net, lr = 0.01, n_batches = 10000, batch_size = 1, momentum = 0.9, display_freq=5):
    
    net = net.to(device)
    criterion = nn.NLLLoss(ignore_index = -1)
    opt = optim.Adam(net.parameters(), lr=lr)
    #teacher_force_upto = n_batches//3
    
    loss_arr = np.zeros(n_batches + 1)
    
    for i in range(n_batches):
        loss_arr[i+1] = (loss_arr[i]*i + train_batch(net, opt, criterion,dataloader, batch_size, teacher_force = True ))/(i + 1)
        
        if i%display_freq == display_freq-1:
            clear_output(wait=True)
            
            print('Iteration', i, 'Loss', loss_arr[i])
            plt.figure()
            plt.plot(loss_arr[1:i], '-*')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.show()
            print('\n\n')
            
    torch.save(net, 'model.pt')
    return loss_arr


train_setup(net, lr=0.001, n_batches=2000, batch_size = 1, display_freq=10)
