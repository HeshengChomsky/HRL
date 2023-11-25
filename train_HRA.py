import torch
from torch.utils.tensorboard import SummaryWriter
from myutils.mydata_sampler import get_dataset,read_file,HData_Sampler
from algorithm.HRA import Diffusion_QL
device=torch.device('cpu')
import time
def train_hight_agent():
    dateset = read_file('trajectory_high.npz', typefile='high',Number_days=3,Number_stores=4000)
    writer=SummaryWriter('run/high')
    data_simple = HData_Sampler(dateset, device)
    state,action,next_sate,reward,done,gmv = data_simple.sample(256)
    action=torch.cat([action,gmv],dim=-1)
    state_dim=state.shape[1]
    action_dim=action.shape[1]
    agent=Diffusion_QL(state_dim=state_dim,action_dim=action_dim,max_action=1.,device=device,discount=0.98,tau=0.005)
    loss_metric = agent.train(data_simple,iterations=2000,batch_size=64,log_writer=writer)
    print(loss_metric['actor_loss'])
    print(loss_metric['bc_loss'])
    print(loss_metric['critic_loss'])
    agent.save_model("model/HRA")

if __name__ == '__main__':
    train_hight_agent()



