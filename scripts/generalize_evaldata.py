# import a s3 package as Client
from Dataset_HF.HFOpenXDataset import HFOpenXDataset
from Dataset_HF.utils import dataset_name, dataset_weight
from torch.utils.data import DataLoader
import pickle
import os
import logging
logging.basicConfig(filename='/xxx/xxx/Anonymous/Robotics/RT-1-X/rt1-pytorch/output.log', level = logging.INFO, format='%(asctime)s : %(message)s')

client = Client()
url = 's3://Anonymous/open_x_embodiment/eval/'


dataset = HFOpenXDataset(dataset_name = dataset_name,
                         dataset_weight = dataset_weight,
                         rank = 0,
                         world_size = 1,
                         traj_per_episode = 4,
                         seed = 12345)
dataloader = DataLoader(dataset, batch_size = 4, num_workers = 2)

dataset_iter = iter(dataloader)

for i in range(625):

    batch = next(dataset_iter)
    path = os.path.join(url, 'traj_' + str(i) + '.pkl')
    client.put(path, pickle.dumps(batch))
    logging.info( str(i+1) + ' trajectories done!')

