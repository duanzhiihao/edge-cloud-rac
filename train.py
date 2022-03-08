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
import torch.nn.functional as tnf
import torch.cuda.amp as amp
from torch.nn.parallel import DistributedDataParallel as DDP

from mycv.paths import IMAGENET_DIR
from mycv.utils.general import increment_dir, ANSI
import mycv.utils.torch_utils as mytu
import mycv.utils.lr_schedulers as lr_schedulers

from mycv.utils.image import save_tensor_images
from mycv.utils.coding import compute_bpp
from mycv.datasets.imcls import get_trainloader, imcls_evaluate
from mycv.datasets.imagenet import get_valloader


def get_config():
    # ====== set the run settings ======
    parser = argparse.ArgumentParser()
    # wandb setting
    parser.add_argument('--project',    type=str,  default='mobile-cloud')
    parser.add_argument('--group',      type=str,  default='irvine')
    parser.add_argument('--wbmode',     type=str,  default='disabled')
    # model setting
    parser.add_argument('--model',      type=str,  default='irvine-vqa')
    parser.add_argument('--cut_after',  type=str,  default='')
    parser.add_argument('--entropy',    type=str,  default='')
    parser.add_argument('--lmbda',      type=float,default=1.0)
    parser.add_argument('--en_only',    action='store_true')
    parser.add_argument('--detach',     type=int,  default=-1)
    parser.add_argument('--teacher',    type=str,  default='')
    parser.add_argument('--eval_teach', action='store_true')
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
    parser.add_argument('--batch_size', type=int,  default=64)
    parser.add_argument('--accum_num',  type=int,  default=1)
    parser.add_argument('--optimizer',  type=str,  default='sgd')
    parser.add_argument('--lr',         type=float,default=0.01)
    parser.add_argument('--lr_sched',   type=str,  default='cosine')
    parser.add_argument('--lr_warmup',  type=int,  default=None)
    parser.add_argument('--wdecay',     type=float,default=0.0001)
    # training policy setting
    parser.add_argument('--epochs',     type=int,  default=100)
    parser.add_argument('--amp',        type=bool, default=True)
    parser.add_argument('--ema',        type=bool, default=True)
    # miscellaneous training setting
    parser.add_argument('--skip_eval0', action='store_true')
    # device setting
    parser.add_argument('--fixseed',    action='store_true')
    parser.add_argument('--device',     type=int,  default=[0], nargs='+')
    parser.add_argument('--workers',    type=int,  default=0)
    cfg = parser.parse_args()

    # model
    cfg.input_norm = 'imagenet'
    # optimizer
    cfg.momentum = 0.9
    # EMA
    cfg.ema_warmup_epochs = max(round(cfg.epochs / 20), 1)
    # logging
    cfg.metric = 'rate-err' # metric to save best model

    # visible device setting
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.device)[1:-1]
    return cfg


class TrainWrapper():
    def __init__(self) -> None:
        pass

    def set_device_(self):
        cfg = self.cfg

        local_rank = int(os.environ.get('LOCAL_RANK', -1))
        world_size = int(os.environ.get('WORLD_SIZE', 1))

        if world_size == 1: # standard mode
            assert local_rank == -1
            if len(cfg.device) == 0: # single GPU mode
                pass
            elif len(cfg.device) > 1: # DP mode
                print(ANSI.warningstr(f'Will use DP mode on devices {cfg.device}...'))

            for i, _id in enumerate(cfg.device):
                print(f'Using device idx={i}, id={_id}:', torch.cuda.get_device_properties(i), '\n')
            device = torch.device('cuda', 0)
            is_main = True
            distributed = False
        else: # DDP mode
            assert local_rank >= 0
            msg = f'count {torch.cuda.device_count()}, devices {cfg.device} world size {world_size}'
            assert torch.cuda.device_count() == len(cfg.device) == world_size, f'{msg}'
            assert torch.distributed.is_nccl_available()
            torch.distributed.init_process_group(backend="nccl")
            assert local_rank == torch.distributed.get_rank()
            assert world_size == torch.distributed.get_world_size()

            with mytu.torch_distributed_sequentially():
                print(f'local_rank={local_rank}, world_size={world_size}, devices {cfg.device}')
                print(torch.cuda.get_device_properties(local_rank), '\n')
            torch.distributed.barrier()
            device = torch.device('cuda', local_rank)
            is_main = (local_rank == 0)
            distributed = True

        if cfg.fixseed: # fix random seeds for reproducibility
            mytu.set_random_seeds(2 + local_rank)
        torch.backends.cudnn.benchmark = True

        if is_main:
            print(f'Batch size on each dataloader = {cfg.batch_size}')
            bs_per_gpu = cfg.batch_size * world_size / len(cfg.device)
            print(f'Batch size on each GPU = {bs_per_gpu}')
            print(f'Gradient accmulation: {cfg.accum_num} backwards() -> one step()')
            bs_effective = cfg.batch_size * world_size * cfg.accum_num
            msg = f'Effective batch size = {bs_effective}, learning rate = {cfg.lr}, ' + \
                  f'weight decay = {cfg.wdecay}'
            print(ANSI.udlstr(msg))
            lr_per_1024img = cfg.lr / bs_effective * 1024
            print(f'Learning rate per 1024 images = {lr_per_1024img}')
            wd_per_1024img = cfg.wdecay / bs_effective * 1024
            print(f'Weight decay per 1024 images = {wd_per_1024img}', '\n')
            self.cfg.bs_per_gpu   = bs_per_gpu
            self.cfg.bs_effective = bs_effective

        self.cfg.world_size = world_size
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
        mname, num_classes = cfg.model, cfg.num_classes

        if mname == 'res50mc':
            from models.mobile import ResNet50MC
            model = ResNet50MC(cfg.cut_after, cfg.entropy)
        elif mname == 'vgg1mc':
            from models.mobile import VGG11MC
            model = VGG11MC(cfg.cut_after, cfg.entropy)
        elif mname == 'mobilev3mc':
            from models.mobile import MobileV3MC
            model = MobileV3MC(cfg.cut_after, cfg.entropy)
        elif mname == 'efb0':
            from timm.models.efficientnet import efficientnet_b0
            model = efficientnet_b0(drop_rate=0.2, drop_path_rate=0.2)
        elif mname == 'irvine':
            from models.irvine2022wacv import BottleneckResNetBackbone
            model = BottleneckResNetBackbone()
        elif mname == 'irvine-vqa':
            from models.irvine2022wacv import BottleneckResNetBackbone
            model = BottleneckResNetBackbone(zdim=64, bottleneck='vqa')
        else:
            raise ValueError()
        if self.is_main:
            print(f'Using model {type(model)}, {num_classes} classes', '\n')

        if cfg.pretrain: # (partially or fully) initialize from pretrained weights
            mytu.load_partial(model, cfg.pretrain, verbose=self.is_main)

        self.model = model.to(self.device)

        if cfg.teacher == 'res50tv':
            from models.teachers import ResNetTeacher
            teacher = ResNetTeacher(source='torchvision')
        elif cfg.teacher == 'res50timm':
            from models.teachers import ResNetTeacher
            teacher = ResNetTeacher(source='timm')
        elif cfg.teacher == '':
            from models.teachers import DummyTeacher
            teacher = DummyTeacher(feature_num=len(self.model.cache))
        else:
            raise ValueError()
        for p in teacher.parameters():
            p.requires_grad_(False)
        self.teacher = teacher.to(self.device)

        if self.is_main and cfg.eval_teach: # test set
            self.teacher.eval()
            print(f'Evaluating teacher model {type(self.teacher)}...')
            results = imcls_evaluate(self.teacher, testloader=self.valloader)
            print(results, '\n')
        self.teacher.train()

        if self.distributed: # DDP mode
            self.model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank,
                             find_unused_parameters=True)
        elif len(cfg.device) > 1: # DP mode
            print(f'DP mode on GPUs {cfg.device}', '\n')
            self.model = torch.nn.DataParallel(model, device_ids=cfg.device)

    def set_loss_(self):
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='mean')

    def set_optimizer_(self):
        cfg, model = self.cfg, self.model

        # different optimization setting for different layers
        pgen, pgb, pgw, pgo = [], [], [], []
        pg_info = {
            'bn/bias': [],
            'weights': [],
            'other': []
        }
        if cfg.en_only:
            for p in model.parameters():
                p.requires_grad_(False)
            for p in model.entropy_model.parameters():
                p.requires_grad_(True)
            _parameters = model.entropy_model.named_parameters()
        else:
            _parameters = model.named_parameters()
        for k, v in _parameters:
            assert isinstance(k, str) and isinstance(v, torch.Tensor)
            if 'entropy_model' in k:
                pgen.append(v)
            elif ('.bn' in k) or ('.bias' in k): # batchnorm or bias
                pgb.append(v)
                pg_info['bn/bias'].append((k, v.shape))
            elif '.weight' in k: # conv or linear weights
                pgw.append(v)
                pg_info['weights'].append((k, v.shape))
            else: # other parameters
                pgo.append(v)
                pg_info['other'].append((k, v.shape))
        parameters = [
            {'params': pgen, 'lr': cfg.lr, 'weight_decay': 0.0},
            {'params': pgb, 'lr': cfg.lr, 'weight_decay': 0.0},
            {'params': pgw, 'lr': cfg.lr, 'weight_decay': cfg.wdecay},
            {'params': pgo, 'lr': cfg.lr, 'weight_decay': 0.0}
        ]

        # optimizer
        if cfg.optimizer == 'sgd':
            optimizer = torch.optim.SGD(parameters, lr=cfg.lr, momentum=cfg.momentum)
        elif cfg.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(parameters, lr=cfg.lr)
        else:
            raise ValueError(f'Unknown optimizer: {cfg.optimizer}')

        if self.is_main:
            print('optimizer parameter groups [b,w,o]:', [len(pg['params']) for pg in parameters])
            print('pgo parameters:')
            for k, vshape in pg_info['other']:
                print(k, vshape)
            print()

        self.optimizer = optimizer
        self.scaler = amp.GradScaler(enabled=cfg.amp) # Automatic mixed precision

    def set_logging_dir_(self):
        cfg = self.cfg

        log_parent = Path(f'runs/{cfg.project}')
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
            results = checkpoint.get('results', defaultdict(float))
            if self.is_main:
                print(f'Resuming run {log_dir}. Loaded checkpoint from {ckpt_path}.',
                      f'Epoch={start_epoch}, results={results}')
        else: # new experiment
            _base = f'{cfg.model}'
            run_name = increment_dir(dir_root=log_parent, name=_base)
            log_dir = log_parent / run_name # logging dir
            if self.is_main:
                os.makedirs(log_dir, exist_ok=False)
                print(str(self.model), file=open(log_dir / 'model.txt', 'w'))
                json.dump(cfg.__dict__, fp=open(log_dir / 'config.json', 'w'), indent=2)
                print('Training config:\n', cfg, '\n')
            start_epoch = 0
            results = defaultdict(float)

        cfg.log_dir = str(log_dir)
        self._log_dir      = log_dir
        self._start_epoch  = start_epoch
        self._results      = results
        self._best_fitness = results[cfg.metric]

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
        run_name = f'{self._log_dir.stem}-{cfg.entropy}'
        wbrun = wandb.init(project=cfg.project, group=cfg.group, name=run_name,
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
        self.set_loss_()
        self.set_optimizer_()

        # logging
        self.set_logging_dir_()
        self.ema = None
        if self.is_main:
            self.set_wandb_()
            self.set_ema_()

        cfg = self.cfg
        model = self.model

        # ======================== start training ========================
        for epoch in range(self._start_epoch, cfg.epochs):
            time.sleep(0.1)

            if self.distributed:
                self.trainloader.sampler.set_epoch(epoch)

            pbar = enumerate(self.trainloader)
            if self.is_main:
                self.init_logging_()
                if not (epoch == self._start_epoch and cfg.skip_eval0):
                    # if cfg.skip_eval0, then skip the first evaluation
                    self.evaluate(epoch, niter=epoch*len(self.trainloader))

                print('\n' + self._pbar_title)
                pbar = tqdm(pbar, total=len(self.trainloader))

            self.adjust_lr_(epoch)
            if cfg.en_only:
                model.eval()
                model.entropy_model.train()
            else:
                model.train()
            if epoch <= cfg.detach:
                model.detach = True
            else:
                model.detach = False

            for bi, (imgs, labels) in pbar:
                niter = epoch * len(self.trainloader) + bi

                imgs = imgs.to(device=self.device)
                labels = labels.to(device=self.device)

                # teacher model
                with torch.no_grad():
                    tgt_logits = self.teacher(imgs)
                    teacher_features = self.teacher.cache

                # forward
                with amp.autocast(enabled=cfg.amp):
                    yhat, p_z, vq_loss = model(imgs)
                    assert yhat.shape == (imgs.shape[0], cfg.num_classes)
                    l_cls = self.loss_func(yhat, labels)
                    if p_z is not None:
                        bpp = -1.0 * torch.log2(p_z).mean(0).sum() / (imgs.shape[2]*imgs.shape[3])
                    else:
                        bpp = torch.zeros(1, device=self.device)

                    # teacher-student loss
                    student_features = mytu.de_parallel(model).cache
                    l_trs = []
                    assert len(student_features) == len(teacher_features)
                    for fake, real in zip(student_features, teacher_features):
                        if (fake is not None) and (real is not None):
                            assert fake.shape == real.shape, f'fake{fake.shape}, real{real.shape}'
                            _lt = tnf.mse_loss(fake, real, reduction='mean')
                            l_trs.append(_lt)
                        else:
                            l_trs.append(torch.zeros(1, device=self.device))

                    loss = l_cls + cfg.lmbda * bpp + sum(l_trs)
                    if loss is not None:
                        loss = loss + vq_loss
                    # loss is averaged over batch and gpus
                    loss = loss / float(cfg.accum_num)
                self.scaler.scale(loss).backward()
                # gradient averaged between devices in DDP mode
                if niter % cfg.accum_num == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    if self.ema is not None:
                        self.ema.update(model)
                        # if epoch < cfg.lr_warmup: # keep copying weights during warmup period
                        #     self.ema.updates = 0

                if self.is_main:
                    self.logging(pbar, epoch, bi, niter, imgs, labels, yhat, p_z, l_cls, bpp, loss, l_trs)
            if self.is_main:
                pbar.close()
            if self.distributed: # If DDP mode, synchronize model parameters on all gpus
                mytu.torch_distributed_check_equivalence(model, log_path=self._log_dir/'ddp.txt')
                mytu.torch_distributed_sync_buffers(model, ['running_mean', 'running_var', 'estimated_p'])
                mytu.torch_distributed_check_equivalence(model, log_path=self._log_dir/'ddp.txt')

        if self.is_main:
            self.evaluate(epoch, niter)
            print('Training finished. results:', self._results)
        if self.distributed:
            torch.distributed.destroy_process_group()

    def adjust_lr_(self, epoch):
        cfg = self.cfg

        if cfg.lr_sched == 'threestep':
            lrf = lr_schedulers.threestep(epoch, cfg.epochs)
        elif cfg.lr_sched == 'cosine':
            lrf = lr_schedulers.get_cosine_lrf(epoch, 1e-4, cfg.epochs-1)
        elif cfg.lr_sched == 'warmup_cosine':
            assert cfg.lr_warmup > 0, f'cfg.lr_warmup = {cfg.lr_warmup}'
            lrf = lr_schedulers.get_warmup_cosine_lrf(epoch, 1e-4, cfg.lr_warmup, cfg.epochs-1)
        else:
            raise NotImplementedError()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cfg.lr * lrf

    def init_logging_(self):
        l_trs = [f'trs_{i}' for i in range(len(mytu.de_parallel(self.model).cache))]
        self._epoch_stat_keys = ['bpp', 'bpdim', 'l_cls', *l_trs , 'loss', 'tr_acc %']
        self._epoch_stat_vals = torch.zeros(len(self._epoch_stat_keys))
        sn = 5 + len(self._epoch_stat_keys)
        self._pbar_title = ('{:^10s}|' * sn).format(
            'Epoch', 'GPU_mem', 'lr', *self._epoch_stat_keys, 'top1 %', 'top5 %',
        )
        self._pbar_title = ANSI.headerstr(self._pbar_title)
        self._msg = ''

    @torch.no_grad()
    def logging(self, pbar, epoch, bi, niter, imgs, labels, yhat, p_z, l_cls, bpp, loss, l_trs):
        cfg = self.cfg
        cur_lr = self.optimizer.param_groups[0]['lr']
        bpdim = - torch.log2(p_z.detach()).mean().cpu().item() if p_z is not None else 0
        acc = compute_acc(yhat.detach(), labels)
        l_trs = [t.item() for t in l_trs]
        stats = torch.Tensor(
            [bpp.item(), bpdim, l_cls.item(), *l_trs, loss.item(), acc]
        )
        self._epoch_stat_vals.mul_(bi).add_(stats).div_(bi+1)
        mem = torch.cuda.max_memory_allocated(self.device) / 1e9
        torch.cuda.reset_peak_memory_stats()
        sn = len(self._epoch_stat_keys) + 2
        msg = ('{:^10s}|' * 2 + '{:^10.4g}|' + '{:^10.4g}|' * sn).format(
            f'{epoch}/{cfg.epochs-1}', f'{mem:.3g}G', cur_lr,
            *self._epoch_stat_vals,
            100*self._results['top1'], 100*self._results['top5'],
        )
        pbar.set_description(msg)
        self._msg = msg

        # Weights & Biases logging
        if niter % 100 == 0:
            _num = min(16, imgs.shape[0])
            save_tensor_images(imgs[:_num], save_path=self._log_dir / 'imgs.png',
                               how_normed=cfg.input_norm)
            _log_dic = {
                'general/lr': cur_lr,
                'ema/n_updates': self.ema.updates if cfg.ema else 0,
                'ema0/decay': self.ema.get_decay() if cfg.ema else 0
            }
            _log_dic.update(
                {'train/'+k: v for k,v in zip(self._epoch_stat_keys, self._epoch_stat_vals)}
            )
            self.wbrun.log(_log_dic, step=niter)

    def evaluate(self, epoch, niter):
        assert self.is_main
        # Evaluation
        _log_dic = {'general/epoch': epoch}
        _eval_model = mytu.de_parallel(self.model)
        _eval_model.eval()
        _eval_model.init_testing()
        results = imcls_evaluate(_eval_model, testloader=self.valloader)
        num, bpp, bpd = _eval_model.testing_stats
        results.update({
            'bits_per_pixel': bpp,
            'bits_per_dim': bpd,
            'rate-err': bpp + 1 - results['top1']
        })
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
            _ema.init_testing()
            results = imcls_evaluate(_ema, testloader=self.valloader)
            num, bpp, bpd = _eval_model.testing_stats
            results.update({
                'bits_per_pixel': bpp,
                'bits_per_dim': bpd,
                'rate-err': bpp + 1 - results['top1']
            })
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
        _cur_fitness = results[self.cfg.metric]
        msg = self._msg + '||' + '%10.4g' * 1 % (_cur_fitness)
        with open(self._log_dir / 'results.txt', 'a') as f:
            f.write(msg + '\n')

        self._results = results

    def _save_if_best(self, checkpoint):
        assert self.is_main
        # save checkpoint if it is the best so far
        metric = self.cfg.metric
        fitness = checkpoint['results'][metric]
        if fitness < self._best_fitness:
            self._best_fitness = fitness
            svpath = self._log_dir / 'best.pt'
            torch.save(checkpoint, svpath)
            print(f'Get best {metric} = {fitness}. Saved to {svpath}.')


def compute_acc(p: torch.Tensor, labels: torch.LongTensor):
    assert not p.requires_grad and p.device == labels.device
    assert p.dim() == 2 and p.shape[0] == labels.shape[0]
    _, p_cls = torch.max(p, dim=1)
    tp = (p_cls == labels)
    acc = float(tp.sum()) / len(tp)
    assert 0 <= acc <= 1
    return acc * 100.0


def main():
    TrainWrapper().main()

if __name__ == '__main__':
    main()
