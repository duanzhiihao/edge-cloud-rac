# from mycv.utils.general import disable_multithreads
# disable_multithreads()
import os
from pathlib import Path
import argparse
import json
import time
from tqdm import tqdm
from collections import defaultdict
import torch
import torch.cuda.amp as amp
from torch.nn.parallel import DistributedDataParallel as DDP

from mycv.paths import IMAGENET_DIR
from mycv.utils.general import increment_dir, ANSI
from mycv.utils.pbar_utils import SimpleTable
import mycv.utils.torch_utils as mytu
import mycv.utils.lr_schedulers as lr_schedulers
from mycv.utils.image import save_tensor_images
from mycv.datasets.imcls import get_trainloader, imcls_evaluate
from mycv.datasets.imagenet import get_valloader

from models.registry import get_model


def get_config():
    # ====== set the run settings ======
    parser = argparse.ArgumentParser()
    # wandb setting
    parser.add_argument('--wbproject',  type=str,  default='mobile-cloud')
    parser.add_argument('--wbgroup',    type=str,  default='irvine')
    parser.add_argument('--wbmode',     type=str,  default='disabled')
    # model setting
    parser.add_argument('--model',      type=str,  default='baseline_vq8')
    parser.add_argument('--model_args', type=str,  default='')
    # resume setting
    parser.add_argument('--resume',     type=str,  default='')
    parser.add_argument('--pretrain',   type=str,  default='')
    # training setting
    parser.add_argument('--train_size', type=int,  default=224)
    parser.add_argument('--aug',        type=str,  default='baseline',
                        help='Examples: baseline, rand-re0.25, trivial-re0.1, finetune')
    # evaluation setting
    parser.add_argument('--val_size',   type=int,  default=224)
    parser.add_argument('--val_crop_r', type=float,default=0.875)
    # optimization setting
    parser.add_argument('--batch_size', type=int,  default=24)
    parser.add_argument('--accum_num',  type=int,  default=1)
    parser.add_argument('--optimizer',  type=str,  default='sgd')
    parser.add_argument('--lr',         type=float,default=0.01)
    parser.add_argument('--lr_sched',   type=str,  default='cosine')
    parser.add_argument('--wdecay',     type=float,default=0.0001)
    # training policy setting
    parser.add_argument('--epochs',     type=int,  default=100)
    parser.add_argument('--amp',        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--ema',        action=argparse.BooleanOptionalAction, default=True)
    # miscellaneous training setting
    parser.add_argument('--eval_first', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--eval_per',   type=int,  default=1)
    # device setting
    parser.add_argument('--fixseed',    action='store_true')
    parser.add_argument('--workers',    type=int,  default=0)
    parser.add_argument('--ddp_find',   action=argparse.BooleanOptionalAction, default=False)
    cfg = parser.parse_args()

    # model
    cfg.input_norm = 'imagenet'
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

            with mytu.torch_distributed_sequentially():
                print(f'local_rank={local_rank}, world_size={world_size}, total_visible={_count}')
                print(torch.cuda.get_device_properties(local_rank), '\n')
            torch.distributed.barrier()
            device = torch.device('cuda', local_rank)
            is_main = (local_rank == 0)
            distributed = True

        if cfg.fixseed: # fix random seeds for reproducibility
            mytu.set_random_seeds(2 + local_rank)
        torch.backends.cudnn.benchmark = True

        if is_main:
            print(f'Batch size on each dataloader (ie, GPU) = {cfg.batch_size}')
            print(f'Gradient accmulation: {cfg.accum_num} backwards() -> one step()')
            bs_effective = cfg.batch_size * world_size * cfg.accum_num
            msg = f'Effective batch size = {bs_effective}, learning rate = {cfg.lr}, ' + \
                  f'weight decay = {cfg.wdecay}'
            print(ANSI.udlstr(msg))
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
            print(ANSI.titlestr('Initializing Datasets and Dataloaders...'))
        train_split = 'train'
        val_split = 'val'

        with mytu.torch_distributed_zero_first(): # training set
            trainloader = get_trainloader(
                root_dir=IMAGENET_DIR/train_split,
                aug=cfg.aug, img_size=cfg.train_size, input_norm=cfg.input_norm,
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
                interp='bilinear', input_norm=cfg.input_norm,
                cache=True, batch_size=cfg.batch_size//2, workers=cfg.workers//2
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
            mytu.load_partial(model, cfg.pretrain, verbose=self.is_main)

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
            ema = mytu.ModelEMA(self.model, decay=decay, warmup=warmup)
            # TODO: from timm.utils.model_ema import ModelEmaV2
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
            if hasattr(mytu.de_parallel(self.model), 'set_epoch'):
                mytu.de_parallel(self.model).set_epoch(epoch, cfg.epochs, verbose=self.is_main)

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
            if self.distributed: # If DDP mode, synchronize model parameters on all gpus
                mytu.torch_distributed_check_equivalence(model, log_path=self._log_dir/'ddp.txt')
                mytu.torch_distributed_sync_buffers(model, ['running_mean', 'running_var', 'estimated_p'])
                mytu.torch_distributed_check_equivalence(model, log_path=self._log_dir/'ddp.txt')

        if self.is_main:
            results = self.evaluate(epoch+1, niter)
            print('Training finished. results:', results)
        if self.distributed:
            torch.distributed.destroy_process_group()

    def adjust_lr_(self, epoch):
        cfg = self.cfg

        if cfg.lr_sched == 'twostep':
            lrf = lr_schedulers.twostep(epoch, cfg.epochs)
        elif cfg.lr_sched == 'cosine':
            lrf = lr_schedulers.get_cosine_lrf(epoch, 0.01, cfg.epochs-1)
        elif cfg.lr_sched == 'warmup_cosine':
            assert cfg.lr_warmup > 0, f'cfg.lr_warmup = {cfg.lr_warmup}'
            lrf = lr_schedulers.get_warmup_cosine_lrf(epoch, 0.01, cfg.lr_warmup, cfg.epochs-1)
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
            save_tensor_images(imgs[:_num], save_path=self._log_dir / 'imgs.png',
                               how_normed=cfg.input_norm)
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
        _eval_model = mytu.de_parallel(self.model).eval()
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
        self._save_if_best(checkpoint)

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
            self._save_if_best(checkpoint)

        # wandb log
        self.wbrun.log(_log_dic, step=niter)
        # Log evaluation results to file
        msg = self.stats_table.get_body() + '||' + '%10.4g' % results['loss']
        with open(self._log_dir / 'results.txt', 'a') as f:
            f.write(msg + '\n')

        return results

    def _save_if_best(self, checkpoint):
        return
        assert self.is_main
        # save checkpoint if it is the best so far
        cur_loss = checkpoint['results']['loss']
        if cur_loss < self._best_loss:
            self._best_loss = cur_loss
            svpath = self._log_dir / 'best.pt'
            torch.save(checkpoint, svpath)
            print(f'Get best loss = {cur_loss}. Saved to {svpath}.')


def main():
    TrainWrapper().main()

if __name__ == '__main__':
    main()
