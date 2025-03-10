import torch
import datetime
import os
from hydra.core.hydra_config import HydraConfig

def resume_or_load_checkpoint(cfg, network, optimizer, scheduler, loss_scaler=None):
    run_dir = HydraConfig.get().run.dir
    if 'ckpt_path' in cfg and cfg.ckpt_path =='auto':
        run_dir = run_dir.split('/')[0] + '/auto'
    tensorboard_output_path = os.path.join(HydraConfig.get().runtime.cwd, run_dir,)
    checkpoint_path = os.path.join(HydraConfig.get().runtime.cwd, run_dir, "checkpoints")
    print('checkpointpath:', checkpoint_path)
    tensorboard_path = os.path.join(tensorboard_output_path, "tensorboard")
    log_path = os.path.join(HydraConfig.get().runtime.cwd, run_dir, "output.log") 
    start_epoch = 0
    total_iter_num = 0
    # import ipdb;ipdb.set_trace()
    if "pretrained_path" in cfg and cfg.pretrained_path != "None":
        ckpt_path = cfg.pretrained_path
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, sorted(os.listdir(ckpt_path), key=lambda x: os.path.getmtime(os.path.join(ckpt_path, x)))[-1])
        if os.path.exists(ckpt_path):
            print('load ', ckpt_path)
            ckpt = torch.load(ckpt_path, 'cpu')
            if "load_qformer" in cfg and cfg.load_qformer == False:
                # import ipdb;ipdb.set_trace()
                state_dict_new = {k:v for k,v in ckpt["parameter"].items() if not "qformer.queries" in k}
                print(network.load_state_dict(state_dict_new, strict = False))
                # import ipdb;ipdb.set_trace()
                _, num_quries, _ = ckpt["parameter"]["image_tokenizer.qformer.queries"].shape
                network.image_tokenizer.qformer.queries.data[:, :num_quries].copy_(ckpt["parameter"]["image_tokenizer.qformer.queries"])
                # import ipdb;ipdb.set_trace()
            else:
                print(network.load_state_dict(ckpt["parameter"]))
            print("Load checkpoint successfully!!", flush = True)
    
    if "ckpt_path" in cfg and cfg.ckpt_path != "None" :
        ckpt_path = cfg.ckpt_path
        if cfg.ckpt_path == 'auto' and os.path.exists(checkpoint_path):
            if len(os.listdir(checkpoint_path)) == 0:
                return start_epoch,total_iter_num,checkpoint_path,tensorboard_path,log_path, run_dir
            ckpt_path = os.path.join(checkpoint_path, sorted(os.listdir(checkpoint_path), key=lambda x: os.path.getmtime(os.path.join(checkpoint_path, x)))[-1])
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, sorted(os.listdir(ckpt_path), key=lambda x: os.path.getmtime(os.path.join(ckpt_path, x)))[-1])
        if os.path.exists(ckpt_path):
            print('load ', cfg.ckpt_path)
            ckpt = torch.load(ckpt_path, 'cpu')
            print(network.load_state_dict(ckpt["parameter"]))
            if optimizer is not None and 'optimizer' in ckpt:
                print("Load optimizer!!!!!!!!!!", flush = True)
                print(optimizer.load_state_dict(ckpt["optimizer"]))
            if scheduler is not None and 'scheduler' in ckpt:
                print("Load scheduler!!!!!!!!!!!!", flush = True)
                print(scheduler.load_state_dict(ckpt["scheduler"]))

            if loss_scaler is not None and 'loss_scaler' in ckpt:
                print("Load loss scaler!!!", flush = True)
                print(loss_scaler.load_state_dict(ckpt['loss_scaler']))

            start_epoch = ckpt["epoch"]
            total_iter_num = ckpt["total_iter_num"]+1 
            
            run_dir = HydraConfig.get().run.dir
            if 'ckpt_path' in cfg and cfg.ckpt_path =='auto':
                run_dir = 'auto'
            
            if not ckpt_path.__contains__('/2024'): # this means we resume from original directory. Thus we not update the directory ot origin
                pass
            else:
                # use checkpoints run.dir
                run_dir = str(ckpt_path).replace(str(HydraConfig.get().runtime.cwd), '')
                if run_dir.startswith('/'):
                    run_dir = run_dir[1:]
                run_dir = run_dir.split('checkpoints')[0]
                tensorboard_output_path = os.path.join(HydraConfig.get().runtime.cwd, run_dir,)
                checkpoint_path = os.path.join(HydraConfig.get().runtime.cwd, run_dir, "checkpoints")
                tensorboard_path = os.path.join(tensorboard_output_path, "tensorboard")
                log_path = os.path.join(HydraConfig.get().runtime.cwd, run_dir, "output.log")
        
    print("ALL Resume Successfully!!!!", flush = True)
    return start_epoch,total_iter_num,checkpoint_path,tensorboard_path,log_path, run_dir


def resume_or_load_checkpoint1(cfg, network, optimizer, scheduler, run_dir, cwd):
    import torch

    if 'ckpt_path' in cfg and cfg.ckpt_path =='auto':
        run_dir = run_dir.split('/')[0] + '/auto'
    tensorboard_output_path = os.path.join(cwd, run_dir,)
    checkpoint_path = os.path.join(cwd, run_dir, "checkpoints")
    print('checkpointpath:', checkpoint_path)
    tensorboard_path = os.path.join(tensorboard_output_path, "tensorboard")
    log_path = os.path.join(cwd, run_dir, "output.log") 
    start_epoch = 0
    total_iter_num = 0
    if "pretrained_path" in cfg and cfg.pretrained_path != "None":
        ckpt_path = cfg.pretrained_path
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, sorted(os.listdir(ckpt_path), key=lambda x: os.path.getmtime(os.path.join(ckpt_path, x)))[-1])
        if os.path.exists(ckpt_path):
            print('load ', ckpt_path)
            ckpt = torch.load(ckpt_path, 'cpu')
            print(network.load_state_dict(ckpt["parameter"]))
    
    if "ckpt_path" in cfg and cfg.ckpt_path != "None":
        ckpt_path = cfg.ckpt_path
        if cfg.ckpt_path == 'auto' and os.path.exists(checkpoint_path):
            if len(os.listdir(checkpoint_path)) == 0:
                return start_epoch,total_iter_num,checkpoint_path,tensorboard_path,log_path, run_dir
            ckpt_path = os.path.join(checkpoint_path, sorted(os.listdir(checkpoint_path), key=lambda x: os.path.getmtime(os.path.join(checkpoint_path, x)))[-1])
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, sorted(os.listdir(ckpt_path), key=lambda x: os.path.getmtime(os.path.join(ckpt_path, x)))[-1])
        if os.path.exists(ckpt_path):
            print('load ', cfg.ckpt_path)
            ckpt = torch.load(ckpt_path, 'cpu')
            print(network.load_state_dict(ckpt["parameter"]))
            if optimizer is not None:
                optimizer.load_state_dict(ckpt["optimizer"])
            if scheduler is not None:
                scheduler.load_state_dict(ckpt["scheduler"])

            start_epoch = ckpt["epoch"]
            total_iter_num = ckpt["total_iter_num"]+1 
            

            if not ckpt_path.__contains__('/2024'): # this means we resume from original directory. Thus we not update the directory ot origin
                pass
            else:
                # use checkpoints run.dir
                run_dir = str(ckpt_path).replace(str(cwd), '')
                if run_dir.startswith('/'):
                    run_dir = run_dir[1:]
                run_dir = run_dir.split('checkpoints')[0]
                tensorboard_output_path = os.path.join(cwd, run_dir,)
                checkpoint_path = os.path.join(cwd, run_dir, "checkpoints")
                tensorboard_path = os.path.join(tensorboard_output_path, "tensorboard")
                log_path = os.path.join(cwd, run_dir, "output.log")
    return start_epoch,total_iter_num,checkpoint_path,tensorboard_path,log_path, run_dir


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.module.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.module.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.module.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}