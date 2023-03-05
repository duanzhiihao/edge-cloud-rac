from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, OrderedDict
import os
import json
import time
import math
import argparse
import torch
import torch.cuda.amp as amp
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision as tv
import timm.utils
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from models.registry import get_model


IMAGENET_DIR = Path('../../datasets/imagenet')


def get_trainloader(root_dir, img_size: int,
                    batch_size: int, workers: int, distributed=False):
    """ get training data loader

    Args:
        root_dir ([type]): [description]
        img_size (int): input image size
        batch_size (int): [description]
        workers (int): [description]
    """
    transform = [
        tv.transforms.RandomResizedCrop(img_size),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ]
    transform = tv.transforms.Compose(transform)

    trainset = tv.datasets.ImageFolder(root=root_dir, transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(trainset) if distributed else None
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=(sampler is None), num_workers=workers,
        pin_memory=True, drop_last=False, sampler=sampler
    )
    return trainloader

def get_valloader(split='val',
        img_size=224, crop_ratio=0.875, batch_size=1, workers=0
    ):
    root_dir = IMAGENET_DIR / split

    transform = tv.transforms.Compose([
        tv.transforms.Resize(round(img_size/crop_ratio)),
        tv.transforms.CenterCrop(img_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])

    dataset = tv.datasets.ImageFolder(root=root_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=True, drop_last=False
    )
    return dataloader

def increment_dir(dir_root='runs/', name='exp'):
    """ Increament directory name. E.g., exp_1, exp_2, exp_3, ...

    Args:
        dir_root (str, optional): root directory. Defaults to 'runs/'.
        name (str, optional): dir prefix. Defaults to 'exp'.
    """
    assert isinstance(dir_root, (str, Path))
    dir_root = Path(dir_root)
    n = 0
    while (dir_root / f'{name}_{n}').is_dir():
        n += 1
    name = f'{name}_{n}'
    return name


class SimpleTable(OrderedDict):
    def __init__(self, init_keys=[]):
        super().__init__()
        # initialization: assign None to initial keys
        for key in init_keys:
            if not isinstance(key, str):
                print(f'Progress bar logger key: {key} is not a string')
            self[key] = None
        self._str_lengths = {k: 8 for k,v in self.items()}

    def _update_length(self, key, length):
        old = self._str_lengths.get(key, 0)
        if length <= old:
            return old
        else:
            self._str_lengths[key] = length
            return length

    def update(self, border=False):
        """ Update the string lengths, and return header and body

        Returns:
            str: table header
            str: table body
        """
        header = []
        body = []
        for k,v in self.items():
            # convert any object to string
            key = self.obj_to_str(k)
            val = self.obj_to_str(v)
            # get str length
            str_len = max(len(key), len(val)) + 2
            str_len = self._update_length(k, str_len)
            # make header and body string
            keystr = f'{key:^{str_len}}|'
            valstr = f'{val:^{str_len}}|'
            header.append(keystr)
            body.append(valstr)
        header = ''.join(header)
        if border:
            header = print(header)
        body = ''.join(body)
        return header, body

    def get_header(self, border=False):
        header = []
        body = []
        for k in self.keys():
            key = self.obj_to_str(k)
            str_len = self._str_lengths[k]
            keystr = f'{key:^{str_len}}|'
            header.append(keystr)
        header = ''.join(header)
        if border:
            header = print(header)
        return header

    def get_body(self):
        body = []
        for k,v in self.items():
            val = self.obj_to_str(v)
            str_len = self._str_lengths[k]
            valstr = f'{val:^{str_len}}|'
            body.append(valstr)
        body = ''.join(body)
        return body

    @staticmethod
    def obj_to_str(obj, digits=4):
        if isinstance(obj, str):
            return obj
        elif isinstance(obj, float) or hasattr(obj, 'float'):
            obj = float(obj)
            return f'{obj:.{digits}g}'
        elif isinstance(obj, list):
            strings = [SimpleTable.obj_to_str(item, 3) for item in obj]
            return '[' + ', '.join(strings) + ']'
        elif isinstance(obj, tuple):
            strings = [SimpleTable.obj_to_str(item, 3) for item in obj]
            return '(' + ', '.join(strings) + ')'
        else:
            return str(obj)


def get_cosine_lrf(n, lrf_min, T):
    """ Cosine learning rate factor

    Args:
        n (int): current epoch. 0, 1, 2, ..., T
        lrf_min (float): final (should also be minimum) learning rate factor
        T (int): total number of epochs
    """
    assert 0 <= n <= T, f'n={n}, T={T}'
    lrf = lrf_min + 0.5 * (1 - lrf_min) * (1 + math.cos(n * math.pi / T))
    return lrf



@torch.no_grad()
def imcls_evaluate(model: torch.nn.Module, testloader):
    """ Image classification evaluation with a testloader.

    Args:
        model (torch.nn.Module): pytorch model
        testloader (torch.utils.data.Dataloader): test dataloader
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    nC = len(testloader.dataset.classes)

    # tp1_sum, tp5_sum, total_num = 0, 0, 0
    stats_avg_meter = defaultdict(timm.utils.AverageMeter)
    print(f'Evaluating {type(model)}, device={device}, dtype={dtype}')
    print(f'batch_size={testloader.batch_size}, num_workers={testloader.num_workers}')
    pbar = tqdm(testloader)
    for imgs, labels in pbar:
        # sanity check
        imgs: torch.FloatTensor
        assert (imgs.dim() == 4)
        nB = imgs.shape[0]
        labels: torch.LongTensor
        assert (labels.shape == (nB,)) and (labels.dtype == torch.int64)
        assert 0 <= labels.min() and labels.max() <= nC-1

        # forward pass, get prediction
        # _debug(imgs)
        imgs = imgs.to(device=device, dtype=dtype)

        assert hasattr(model, 'self_evaluate')
        labels = labels.to(device=device)
        stats = model.self_evaluate(imgs, labels) # workaround

        for k, v in stats.items():
            stats_avg_meter[k].update(float(v), n=nB)

        # logging
        msg = ''.join([f'{k}={v.avg:.4g}, ' for k,v in stats_avg_meter.items()])
        pbar.set_description(msg)
    pbar.close()

    # compute total statistics and return
    _random_key = list(stats_avg_meter.keys())[0]
    assert stats_avg_meter[_random_key].count == len(testloader.dataset)
    results = {k: v.avg for k,v in stats_avg_meter.items()}
    return results


def get_config():
    # ====== set the run settings ======
    parser = argparse.ArgumentParser()
    # wandb setting
    parser.add_argument('--wbproject',  type=str,  default='edge-cloud-rac')
    parser.add_argument('--wbgroup',    type=str,  default='default-group')
    parser.add_argument('--wbmode',     type=str,  default='disabled')
    # model setting
    parser.add_argument('--model',      type=str,  default='ours_n4')
    parser.add_argument('--model_args', type=str,  default='')
    # resume setting
    parser.add_argument('--resume',     type=str,  default='')
    parser.add_argument('--pretrain',   type=str,  default='')
    # training setting
    parser.add_argument('--train_size', type=int,  default=224)
    # evaluation setting
    parser.add_argument('--val_size',   type=int,  default=224)
    parser.add_argument('--val_crop_r', type=float,default=0.875)
    # optimization setting
    parser.add_argument('--batch_size', type=int,  default=384)
    parser.add_argument('--accum_num',  type=int,  default=1)
    parser.add_argument('--optimizer',  type=str,  default='sgd')
    parser.add_argument('--lr',         type=float,default=0.01)
    parser.add_argument('--lr_sched',   type=str,  default='cosine')
    parser.add_argument('--wdecay',     type=float,default=0.0001)
    # training policy setting
    parser.add_argument('--epochs',     type=int,  default=20)
    parser.add_argument('--amp',        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--ema',        action=argparse.BooleanOptionalAction, default=False)
    # miscellaneous training setting
    parser.add_argument('--eval_first', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--eval_per',   type=int,  default=2)
    # device setting
    parser.add_argument('--fixseed',    action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--workers',    type=int,  default=8)
    parser.add_argument('--ddp_find',   action=argparse.BooleanOptionalAction, default=False)
    cfg = parser.parse_args()

    # optimizer
    cfg.momentum = 0.9
    # EMA
    cfg.ema_warmup_epochs = max(round(cfg.epochs / 20), 1)
    return cfg


class TrainWrapper():
    def __init__(self) -> None:
        pass

    def set_device_(self):
        cfg = self.cfg

        local_rank = int(os.environ.get('LOCAL_RANK', -1))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        _count = torch.cuda.device_count()

        if world_size == 1: # standard mode
            assert local_rank == -1
            print(f'Visible devices={_count}, using idx 0:', torch.cuda.get_device_properties(0), '\n')
            device = torch.device('cuda', 0)
            is_main = True
            distributed = False
        else: # DDP mode
            assert local_rank >= 0
            assert torch.distributed.is_nccl_available()
            torch.distributed.init_process_group(backend="nccl")
            assert local_rank == torch.distributed.get_rank()
            assert world_size == torch.distributed.get_world_size()

            torch.distributed.barrier()
            device = torch.device('cuda', local_rank)
            is_main = (local_rank == 0)
            distributed = True

        if cfg.fixseed: # fix random seeds for reproducibility
            timm.utils.random_seed(2 + local_rank)
        torch.backends.cudnn.benchmark = True

        if is_main:
            print(f'Batch size on each dataloader (ie, GPU) = {cfg.batch_size}')
            print(f'Gradient accmulation: {cfg.accum_num} backwards() -> one step()')
            bs_effective = cfg.batch_size * world_size * cfg.accum_num
            msg = f'Effective batch size = {bs_effective}, learning rate = {cfg.lr}, ' + \
                  f'weight decay = {cfg.wdecay}'
            print(msg)
            lr_per_1024img = cfg.lr / bs_effective * 1024
            print(f'Learning rate per 1024 images = {lr_per_1024img}')
            wd_per_1024img = cfg.wdecay / bs_effective * 1024
            print(f'Weight decay per 1024 images = {wd_per_1024img}', '\n')
            cfg.bs_effective = bs_effective

        cfg.world_size   = world_size
        self.device      = device
        self.local_rank  = local_rank
        self.is_main     = is_main
        self.distributed = distributed

    def set_dataset_(self):
        cfg = self.cfg

        if self.is_main:
            print('Initializing Datasets and Dataloaders...')
        train_split = 'train'
        val_split = 'val'

        trainloader = get_trainloader(
            root_dir=IMAGENET_DIR/train_split,
            img_size=cfg.train_size,
            batch_size=cfg.batch_size, workers=cfg.workers, distributed=self.distributed
        )
        num_classes = len(trainloader.dataset.classes)

        if self.is_main: # test set
            print(f'Number of classes = {num_classes}')
            print(f'Training root: {trainloader.dataset.root}')
            print(f'First training data: {trainloader.dataset.samples[0]}')
            print('Training transform:', trainloader.dataset.transform)

            valloader = get_valloader(val_split,
                img_size=cfg.val_size, crop_ratio=cfg.val_crop_r,
                batch_size=cfg.batch_size//2, workers=cfg.workers//2
            )
            print(f'Val root: {valloader.dataset.root}')
            print(f'First val data: {valloader.dataset.samples[0]}', '\n')
        else:
            valloader = None

        self.trainloader = trainloader
        self.valloader  = valloader
        self.cfg.num_classes = num_classes

    def set_model_(self):
        cfg = self.cfg

        model_func = get_model(cfg.model)
        kwargs = dict(num_classes=cfg.num_classes)
        kwargs.update(eval(f'dict({cfg.model_args})'))
        model = model_func(**kwargs)
        if self.is_main:
            print(f'Using model {type(model)}, args: {kwargs}', '\n')
        if cfg.pretrain: # (partially or fully) initialize from pretrained weights
            raise NotImplementedError()
            load_partial(model, cfg.pretrain, verbose=self.is_main)

        self.model = model.to(self.device)

        if self.distributed: # DDP mode
            self.model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank,
                             find_unused_parameters=cfg.ddp_find)

    def set_optimizer_(self):
        cfg, model = self.cfg, self.model

        # different optimization setting for different layers
        pgen, pgb, pgw, pgo = [], [], [], []
        pg_info = defaultdict(list)
        for k, v in model.named_parameters():
            assert isinstance(k, str) and isinstance(v, torch.Tensor)
            if not v.requires_grad:
                continue
            if ('entropy' in k) or ('bottleneck' in k):
                pgen.append(v)
                pg_info['entropy'].append(f'{k:<80s} {v.shape}')
            elif ('.bn' in k) or ('.bias' in k): # batchnorm or bias
                pgb.append(v)
                pg_info['bn/bias'].append(f'{k:<80s} {v.shape}')
            elif '.weight' in k: # conv or linear weights
                pgw.append(v)
                pg_info['weights'].append(f'{k:<80s} {v.shape}')
            else: # other parameters
                pgo.append(v)
                pg_info['other'].append(f'{k:<80s} {v.shape}')
        parameters = [
            {'params': pgen, 'lr': cfg.lr, 'weight_decay': 0.0},
            {'params': pgb, 'lr': cfg.lr, 'weight_decay': 0.0},
            {'params': pgw, 'lr': cfg.lr, 'weight_decay': cfg.wdecay},
            {'params': pgo, 'lr': cfg.lr, 'weight_decay': 0.0}
        ]

        # optimizer
        if cfg.optimizer == 'sgd':
            optimizer = torch.optim.SGD(parameters, lr=cfg.lr, momentum=cfg.momentum)
        elif cfg.optimizer == 'adam':
            optimizer = torch.optim.Adam(parameters, lr=cfg.lr)
        elif cfg.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(parameters, lr=cfg.lr)
        else:
            raise ValueError(f'Unknown optimizer: {cfg.optimizer}')

        if cfg.pretrain:
            try:
                optimizer.load_state_dict(torch.load(cfg.pretrain)['optimizer'])
            except Exception as e:
                print('Failed loading optimizer. Error message:', e)

        if self.is_main:
            print('optimizer parameter groups:', *[f'[{k}: {len(pg)}]' for k, pg in pg_info.items()])
            self.pg_info_to_log = pg_info
            print()

        self.optimizer = optimizer
        self.scaler = amp.GradScaler(enabled=cfg.amp) # Automatic mixed precision

    def set_logging_dir_(self):
        cfg = self.cfg

        prev_loss = 1e8
        log_parent = Path(f'runs/{cfg.wbproject}')
        if cfg.resume: # resume
            assert not cfg.pretrain, '--resume not compatible with --pretrain'
            run_name = cfg.resume
            log_dir = log_parent / run_name
            assert log_dir.is_dir(), f'Try to resume from {log_dir} but it does not exist'
            ckpt_path = log_dir / 'last.pt'
            checkpoint = torch.load(ckpt_path)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scaler.load_state_dict(checkpoint['scaler'])
            start_epoch = checkpoint['epoch']
            prev_result = checkpoint.get('results', None)
            prev_loss = prev_result['loss'] if prev_result is not None else prev_loss
            if self.is_main:
                print(f'Resuming run {log_dir}. Loaded checkpoint from {ckpt_path}.',
                      f'Epoch={start_epoch}, results={prev_result}')
        else: # new experiment
            _base = f'{cfg.model}'
            run_name = increment_dir(dir_root=log_parent, name=_base)
            log_dir = log_parent / run_name # logging dir
            if self.is_main:
                os.makedirs(log_dir, exist_ok=False)
                print(str(self.model), file=open(log_dir / 'model.txt', 'w'))
                json.dump(cfg.__dict__, fp=open(log_dir / 'config.json', 'w'), indent=2)
                json.dump(self.pg_info_to_log, fp=open(log_dir / 'optimizer.json', 'w'), indent=2)
                print('Training config:\n', cfg, '\n')
            start_epoch = 0

        cfg.log_dir = str(log_dir)
        self._log_dir     = log_dir
        self._start_epoch = start_epoch
        self._best_loss   = prev_loss

    def set_wandb_(self):
        cfg = self.cfg

        # check if there is a previous run to resume
        wbid_path = self._log_dir / 'wandb_id.txt'
        if os.path.exists(wbid_path):
            run_ids = open(wbid_path, mode='r').read().strip().split('\n')
            rid = run_ids[-1]
        else:
            rid = None
        # initialize wandb
        import wandb
        run_name = f'{self._log_dir.stem} {cfg.model_args}'
        wbrun = wandb.init(project=cfg.wbproject, group=cfg.wbgroup, name=run_name,
                           config=cfg, dir='runs/', id=rid, resume='allow',
                           save_code=True, mode=cfg.wbmode)
        cfg = wbrun.config
        cfg.wandb_id = wbrun.id
        with open(wbid_path, mode='a') as f:
            print(wbrun.id, file=f)

        self.wbrun = wbrun
        self.cfg = cfg

    def set_ema_(self):
        cfg = self.cfg

        # Exponential moving average
        if cfg.ema:
            warmup = cfg.ema_warmup_epochs * len(self.trainloader)
            decay = 0.9999
            ema = timm.utils.ModelEmaV2(self.model, decay=decay)
            start_iter = self._start_epoch * len(self.trainloader)
            ema.updates = start_iter // cfg.accum_num # set EMA update number

            if cfg.resume:
                ckpt_path = self._log_dir / 'last_ema.pt'
                assert ckpt_path.is_file(), f'Cannot find EMA checkpoint: {ckpt_path}'
                ema.ema.load_state_dict(torch.load(ckpt_path)['model'])

            if self.is_main:
                print(f'Using EMA with warmup_epochs={cfg.ema_warmup_epochs},',
                      f'decay={decay}, past_updates={ema.updates}\n',
                      f'Loaded EMA from {ckpt_path}\n' if cfg.resume else '')
        else:
            ema = None

        self.ema = ema

    def main(self):
        # config
        self.cfg = get_config()

        # core
        self.set_device_()
        self.set_dataset_()
        self.set_model_()
        self.set_optimizer_()

        # logging
        self.set_logging_dir_()
        self.ema = None
        if self.is_main:
            self.set_wandb_()
            self.set_ema_()
            self.stats_table = SimpleTable(['Epoch', 'GPU_mem', 'lr'])

        cfg = self.cfg
        model = self.model

        # ======================== start training ========================
        for epoch in range(self._start_epoch, cfg.epochs):
            time.sleep(0.1)

            if self.distributed:
                self.trainloader.sampler.set_epoch(epoch)

            pbar = enumerate(self.trainloader)
            if self.is_main:
                if ((epoch != self._start_epoch) or cfg.eval_first) and (epoch % cfg.eval_per == 0):
                    self.evaluate(epoch, niter=epoch*len(self.trainloader))

                self.init_logging_()
                pbar = tqdm(pbar, total=len(self.trainloader))

            self.adjust_lr_(epoch)
            model.train()
            for bi, (imgs, labels) in pbar:
                niter = epoch * len(self.trainloader) + bi

                imgs = imgs.to(device=self.device)
                labels = labels.to(device=self.device)

                # forward
                with amp.autocast(enabled=cfg.amp):
                    stats = model(imgs, labels)
                    loss = stats['loss']
                    # loss should be averaged over batch (and if DDP, also gpus)
                    loss = loss / float(cfg.accum_num)
                self.scaler.scale(loss).backward()
                # gradient averaged between devices in DDP mode
                if niter % cfg.accum_num == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    if self.ema is not None:
                        self.ema.update(model)

                if self.is_main:
                    self.logging(pbar, epoch, bi, niter, imgs, stats)
            if self.is_main:
                pbar.close()

        if self.is_main:
            results = self.evaluate(epoch+1, niter)
            print('Training finished. results:', results)
        if self.distributed:
            torch.distributed.destroy_process_group()

    def adjust_lr_(self, epoch):
        cfg = self.cfg

        if cfg.lr_sched == 'cosine':
            lrf = get_cosine_lrf(epoch, 0.01, cfg.epochs-1)
        else:
            raise NotImplementedError()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cfg.lr * lrf

    def init_logging_(self):
        # initialize stats table and progress bar
        for k in self.stats_table.keys():
            self.stats_table[k] = 0.0
        self._pbar_header = self.stats_table.get_header()
        print('\n', self._pbar_header)
        time.sleep(0.1)

    @torch.no_grad()
    def logging(self, pbar, epoch, bi, niter, imgs, stats):
        cfg = self.cfg

        self.stats_table['Epoch'] = f'{epoch}/{cfg.epochs-1}'

        mem = torch.cuda.max_memory_allocated(self.device) / 1e9
        torch.cuda.reset_peak_memory_stats()
        self.stats_table['GPU_mem'] = f'{mem:.3g}G'

        cur_lr = self.optimizer.param_groups[0]['lr']
        self.stats_table['lr'] = cur_lr

        keys_to_log = []
        for k, v in stats.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                v = v.detach().cpu().item()
            assert isinstance(v, (float, int))
            prev = self.stats_table.get(k, 0.0)
            self.stats_table[k] = (prev * bi + v) / (bi + 1)
            keys_to_log.append(k)
        pbar_header, pbar_body = self.stats_table.update()
        if pbar_header != self._pbar_header:
            print(pbar_header)
            self._pbar_header = pbar_header
        pbar.set_description(pbar_body)

        # Weights & Biases logging
        if niter % 100 == 0:
            _num = min(16, imgs.shape[0])
            _log_dic = {
                'general/lr': cur_lr,
                'ema/n_updates': self.ema.updates if cfg.ema else 0,
                'ema/decay': self.ema.get_decay() if cfg.ema else 0
            }
            _log_dic.update(
                {'train/'+k: self.stats_table[k] for k in keys_to_log}
            )
            self.wbrun.log(_log_dic, step=niter)

    def evaluate(self, epoch, niter):
        assert self.is_main
        # Evaluation
        _log_dic = {'general/epoch': epoch}
        _eval_model = timm.utils.unwrap_model(self.model).eval()
        results = imcls_evaluate(_eval_model, testloader=self.valloader)
        _log_dic.update({'metric/plain_val_'+k: v for k,v in results.items()})
        # save last checkpoint
        checkpoint = {
            'model'     : _eval_model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'scaler'    : self.scaler.state_dict(),
            'epoch'     : epoch,
            'results'   : results,
        }
        torch.save(checkpoint, self._log_dir / 'last.pt')

        if self.cfg.ema:
            _ema = self.ema.ema.eval()
            results = imcls_evaluate(_ema, testloader=self.valloader)
            _log_dic.update({f'metric/ema_val_'+k: v for k,v in results.items()})
            # save last checkpoint of EMA
            checkpoint = {
                'model'     : _ema.state_dict(),
                'epoch'     : epoch,
                'results'   : results,
            }
            torch.save(checkpoint, self._log_dir / 'last_ema.pt')

        # wandb log
        self.wbrun.log(_log_dic, step=niter)
        # Log evaluation results to file
        msg = self.stats_table.get_body() + '||' + '%10.4g' % results['loss']
        with open(self._log_dir / 'results.txt', 'a') as f:
            f.write(msg + '\n')

        return results


def main():
    TrainWrapper().main()

if __name__ == '__main__':
    main()
