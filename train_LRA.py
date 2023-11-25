import torch
from myutils.mydata_sampler import get_dataset,read_file,LData_Sampler
from algorithm.HRA import Diffusion_QL
from torch.utils.tensorboard import SummaryWriter
from algorithm.LRA import POR
device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def train_low_agent():
    dateset = read_file('trajectory_low_', typefile='low')
    writer=SummaryWriter('run/low')
    data_simple = LData_Sampler(dateset, device)
    state, action, next_sate, reward, done, costs, gmvs, rois = data_simple.sample(64)
    state_dim=state.shape[-1]
    action_dim=action.shape[-1]
    agent=POR(state_dim=state_dim,action_dim=action_dim,max_action=1.)
    agent.train(data_simple,batch_size=64,numb_iter=80000,channel_number=3,log_writer=writer)

if __name__ == '__main__':
    train_low_agent()