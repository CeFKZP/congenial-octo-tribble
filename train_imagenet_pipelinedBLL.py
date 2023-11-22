import random
import yaml
import wandb
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from ffcv.fields.basics import IntDecoder
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from fastargs.validation import And, OneOf
from fastargs import Param, Section
from fastargs.decorators import param
from fastargs import get_current_config
from argparse import ArgumentParser
from pathlib import Path
from typing import List
from uuid import uuid4
import json
import time
import os
from tqdm import tqdm
import numpy as np
import torchmetrics
# import BLL.model.loss as module_loss
# import BLL.model.model_zoo as module_arch
import loss as module_loss
import models as module_arch
from torchvision import models
from email.policy import default
import torch as ch

ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)
profile_epoch = 1

Section('model', 'model details').params(
    arch=Param(str, default='resnet18'),
    pretrained=Param(int, 'is pretrained? (1/0)', default=0),
    num_classes=Param(int, 'number of classes', default=1000),
    gradient_depth=Param(int, 'Gradient flow depth', default=1),
    num_splits=Param(int, 'Number of splits', default=1),
    dropout_p=Param(
        float, 'Dropout probability, set 0 to disable', default=0.0),
    no_detach=Param(int, 'Detach splits? (1/0)', default=0),
    enable_posterior_mixing=Param(int, 'Enable mixing? (1/0)', default=0),
    mixing_noise=Param(float, 'mixing noise probability', default=0.0),
    denoise=Param(float, 'input noise probability', default=0.01),
)

Section('resolution', 'resolution scheduling').params(
    min_res=Param(int, 'the minimum (starting) resolution', default=160),
    max_res=Param(int, 'the maximum (starting) resolution', default=160),
    end_ramp=Param(int, 'when to stop interpolating resolution', default=0),
    start_ramp=Param(int, 'when to start interpolating resolution', default=0)
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', required=True),
    val_dataset=Param(str, '.dat file to use for validation', required=True),
    test_dataset=Param(str, '.dat file to use for testing', required=True),
    num_workers=Param(int, 'The number of workers', required=True),
    in_memory=Param(
        int, 'does the dataset fit in memory? (1/0)', required=True)
)

Section('lr', 'lr scheduling').params(
    step_ratio=Param(float, 'learning rate step ratio', default=0.1),
    step_length=Param(int, 'learning rate step length', default=30),
    lr_schedule_type=Param(OneOf(['step', 'cyclic']), default='cyclic'),
    lr=Param(float, 'learning rate', default=0.5),
    lr_peak_epoch=Param(int, 'Epoch at which LR peaks', default=2),
)

Section('logging', 'how to log stuff').params(
    folder=Param(str, 'log location', required=True),
    log_level=Param(int, '0 if only at end 1 otherwise', default=1),
    wandb_config=Param(str, 'yml config file', default=''),
)

Section('validation', 'Validation parameters stuff').params(
    batch_size=Param(int, 'The batch size for validation', default=512),
    resolution=Param(int, 'final resized validation image size', default=224),
    lr_tta=Param(int, 'should do lr flipping/avging at test time', default=1)
)

Section('training', 'training hyper param stuff').params(
    seed=Param(int, 'seed', default=2335),
    eval_only=Param(int, 'eval only?', default=0),
    batch_size=Param(int, 'The batch size', default=512),
    optimizer=Param(And(str, OneOf(['sgd'])), 'The optimizer', default='sgd'),
    momentum=Param(float, 'SGD momentum', default=0.9),
    weight_decay=Param(float, 'weight decay', default=4e-5),
    epochs=Param(int, 'number of epochs', default=30),
    label_smoothing=Param(float, 'label smoothing parameter', default=0.1),
    distributed=Param(int, 'is distributed?', default=0),
    use_blurpool=Param(int, 'use blurpool?', default=0),
    bwd_lr=Param(float, 'backward SGD momentum', default=0.9),
    bwd_momentum=Param(float, 'backward SGD momentum', default=0.01),
)

Section('local_loss', 'local loss parameters').params(
    alpha=Param(float, 'forward loss scaler', default=1.0),
    beta=Param(float, 'corrrelation loss scaler', default=1.0),
    gamma=Param(float, 'similarity loss scaler', default=1.0),
    pred_loss_scaler=Param(float, 'prediction loss scaler', default=0.5),
    denoise_scaler=Param(float, 'denoising loss scaler', default=1.0),
)


Section('dist', 'distributed training options').params(
    world_size=Param(int, 'number gpus', default=1),
    address=Param(str, 'address', default='localhost'),
    port=Param(str, 'port', default='12353')
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256


@param('lr.lr')
@param('lr.step_ratio')
@param('lr.step_length')
@param('training.epochs')
def get_step_lr(epoch, lr, step_ratio, step_length, epochs):
    if epoch >= epochs:
        return 0

    num_steps = epoch // step_length
    return step_ratio**num_steps * lr


@param('lr.lr')
@param('training.epochs')
@param('lr.lr_peak_epoch')
def get_cyclic_lr(epoch, lr, epochs, lr_peak_epoch):
    xs = [0, lr_peak_epoch, epochs]
    ys = [1e-4 * lr, lr, 0]
    return np.interp([epoch], xs, ys)[0]


class BlurPoolConv2d(ch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = ch.tensor(
            [[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer('blur_filter', filt)

    def forward(self, x):
        blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
                           groups=self.conv.in_channels, bias=None)
        return self.conv.forward(blurred)


class ImageNetTrainer:
    @param('training.distributed')
    @param('model.num_splits')
    def __init__(self, gpu, distributed, num_splits):
        self.all_params = get_current_config()
        self.gpu = gpu * (num_splits+1)

        self.uid = str(uuid4())

        if distributed:
            self.setup_distributed()

        self.train_loader = self.create_train_loader()
        self.val_loader = self.create_val_loader()
        self.test_loader = self.create_test_loader()
        self.model, self.scaler, self.backward_scaler = self.create_model_and_scaler()
        self.create_optimizer()

        self.greedy_train_steps = 1
        self.create_local_losses()

        self.initialize_logger()

    @param('dist.address')
    @param('dist.port')
    @param('dist.world_size')
    @param('model.num_splits')
    def setup_distributed(self, address, port, world_size, num_splits):
        os.environ['MASTER_ADDR'] = address
        os.environ['MASTER_PORT'] = port
        world_size = world_size//(num_splits+1)
        rank = self.gpu // (num_splits+1)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        ch.cuda.set_device(self.gpu)

    def cleanup_distributed(self):
        dist.destroy_process_group()

    @param('lr.lr_schedule_type')
    def get_lr(self, epoch, lr_schedule_type):
        lr_schedules = {
            'cyclic': get_cyclic_lr,
            'step': get_step_lr
        }

        return lr_schedules[lr_schedule_type](epoch)

    # resolution tools
    @param('resolution.min_res')
    @param('resolution.max_res')
    @param('resolution.end_ramp')
    @param('resolution.start_ramp')
    def get_resolution(self, epoch, min_res, max_res, end_ramp, start_ramp):
        assert min_res <= max_res

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res

    @param('training.momentum')
    @param('training.optimizer')
    @param('training.weight_decay')
    @param('training.label_smoothing')
    @param('training.bwd_lr')
    @param('training.bwd_momentum')
    def create_optimizer(self, momentum, optimizer, weight_decay,
                         label_smoothing, bwd_lr, bwd_momentum):
        assert optimizer == 'sgd'

        # Only do weight decay on non-batchnorm parameters
        all_params = list(self.model.named_parameters())
        bn_params = [v for k, v in all_params if ('bn' in k)]
        bwd_params = [v for k, v in all_params if ('bwd' in k)]
        other_params = [v for k, v in all_params if not (
            ('bn' in k) or ('bwd' in k))]
        param_groups = [{
            'params': bn_params,
            'weight_decay': 0.
        }, {
            'params': other_params,
            'weight_decay': weight_decay
        }]

        self.optimizer = ch.optim.SGD(param_groups, lr=1, momentum=momentum)
        self.loss = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        param_groups = [{
            'params': bwd_params,
            'weight_decay': 0.01
        },]
        self.backward_optimizer = ch.optim.SGD(
            param_groups, lr=bwd_lr, momentum=bwd_momentum)

    @param('data.train_dataset')
    @param('data.num_workers')
    @param('training.batch_size')
    @param('training.distributed')
    @param('data.in_memory')
    def create_train_loader(self, train_dataset, num_workers, batch_size,
                            distributed, in_memory):
        this_device = f'cuda:{self.gpu}'
        train_path = Path(train_dataset)
        assert train_path.is_file()

        res = self.get_resolution(epoch=0)
        self.decoder = RandomResizedCropRGBImageDecoder((res, res))
        image_pipeline: List[Operation] = [
            self.decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device), non_blocking=True)
        ]

        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        loader = Loader(train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        os_cache=in_memory,
                        drop_last=True,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)

        return loader

    @param('data.val_dataset')
    @param('data.num_workers')
    @param('validation.batch_size')
    @param('validation.resolution')
    @param('training.distributed')
    def create_val_loader(self, val_dataset, num_workers, batch_size,
                          resolution, distributed):
        this_device = f'cuda:{self.gpu}'
        val_path = Path(val_dataset)
        assert val_path.is_file()
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(
            res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device),
                     non_blocking=True)
        ]

        loader = Loader(val_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=OrderOption.SEQUENTIAL,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)
        return loader

    @param('data.test_dataset')
    @param('data.num_workers')
    @param('validation.batch_size')
    @param('validation.resolution')
    @param('training.distributed')
    def create_test_loader(self, test_dataset, num_workers, batch_size,
                           resolution, distributed):
        this_device = f'cuda:{self.gpu}'
        val_path = Path(test_dataset)
        assert val_path.is_file()
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(
            res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device),
                     non_blocking=True)
        ]

        loader = Loader(test_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=OrderOption.SEQUENTIAL,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)
        return loader

    @param('training.epochs')
    @param('logging.log_level')
    def train(self, epochs, log_level):
        for epoch in range(epochs):
            res = self.get_resolution(epoch)
            self.decoder.output_size = (res, res)
            train_loss = self.train_loop(epoch)

            if log_level > 0:
                extra_dict = {
                    'train_loss': train_loss,
                    'epoch': epoch
                }

                self.eval_and_log(extra_dict)

        self.test_and_log({'epoch': epoch})
        if self.gpu == 0:
            ch.save(self.model.state_dict(),
                    self.log_folder / 'final_weights.pt')

    def eval_and_log(self, extra_dict={}):
        start_val = time.time()
        stats = self.val_loop()
        val_time = time.time() - start_val
        if self.gpu == 0:
            self.log(dict({
                'current_lr': self.optimizer.param_groups[0]['lr'],
                'top_1': stats['top_1'],
                'top_5': stats['top_5'],
                'val_time': val_time
            }, **extra_dict))

            if self.wandb_run is not None:
                self.wandb_run.log(dict({
                    'LR': self.optimizer.param_groups[0]['lr'],
                    'val_top_1': stats['top_1'],
                    'val_top_5': stats['top_5'],
                    'val_time': val_time
                }, **extra_dict))

        return stats

    def test_and_log(self, extra_dict={}):
        start_test = time.time()
        stats = self.test_loop()
        test_time = time.time() - start_test
        if self.gpu == 0:
            self.log(dict({
                'current_lr': self.optimizer.param_groups[0]['lr'],
                'test_top_1': stats['top_1'],
                'test_top_5': stats['top_5'],
                'test_time': test_time
            }, **extra_dict))
            if self.wandb_run is not None:
                self.wandb_run.log(dict({
                    'LR': self.optimizer.param_groups[0]['lr'],
                    'test_top_1': stats['top_1'],
                    'test_top_5': stats['top_5'],
                }, **extra_dict))

        return stats

    @param('local_loss.alpha')
    @param('local_loss.beta')
    @param('local_loss.gamma')
    @param('local_loss.pred_loss_scaler')
    @param('local_loss.denoise_scaler')
    def create_local_losses(self, alpha, beta, gamma, pred_loss_scaler, denoise_scaler):
        self.local_loss = getattr(module_loss, 'trainable_proto_vector_loss')
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.local_pred_loss = pred_loss_scaler
        self.denoise_loss = denoise_scaler

    @param('model.arch')
    @param('model.num_classes')
    @param('model.gradient_depth')
    @param('model.num_splits')
    @param('model.dropout_p')
    @param('model.no_detach')
    @param('model.enable_posterior_mixing')
    @param('model.mixing_noise')
    @param('model.denoise')
    @param('training.distributed')
    @param('training.use_blurpool')
    def create_model_and_scaler(self, arch, num_classes, gradient_depth, num_splits, dropout_p, no_detach,
                                enable_posterior_mixing, mixing_noise,  denoise, distributed, use_blurpool):
        scaler = GradScaler()
        backward_scaler = GradScaler()
        model_kwargs = {'num_classes': num_classes,
                        'gradient_depth': gradient_depth,
                        'num_splits': num_splits,
                        'dropout_p': dropout_p,
                        'no_detach': no_detach,
                        'enable_posterior_mixing': enable_posterior_mixing,
                        'mixing_noise': mixing_noise,
                        'denoise': denoise}
        model = getattr(module_arch, arch)(**model_kwargs)

        def apply_blurpool(mod: ch.nn.Module):
            for (name, child) in mod.named_children():
                if isinstance(child, ch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16):
                    setattr(mod, name, BlurPoolConv2d(child))
                else:
                    apply_blurpool(child)
        if use_blurpool:
            apply_blurpool(model)

        model = model.to(memory_format=ch.channels_last)
        model = model.to(self.gpu)

        data, _ = next(iter(self.train_loader))
        model.create_bwd_net(data.to(dtype=ch.float32, device=self.gpu))

        model.pipeline(self.gpu)

        if distributed:
            model = ch.nn.parallel.DistributedDataParallel(model)
            model.num_blocks = model.module.num_blocks
            model.bwd_net = model.module.bwd_net
            model.num_classes = model.module.num_classes
            model.bwd_pred_net = model.module.bwd_pred_net

        return model, scaler, backward_scaler

    @param('logging.log_level')
    def train_loop(self, epoch, log_level):
        model = self.model
        model.train()
        losses = []

        lr_start, lr_end = self.get_lr(epoch), self.get_lr(epoch + 1)
        iters = len(self.train_loader)
        lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

        if epoch == profile_epoch:
            prof = ch.profiler.profile(
                schedule=ch.profiler.schedule(
                    wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=ch.profiler.tensorboard_trace_handler(
                    './logs/resnet50_ModelPrallel_8gpu'),
                record_shapes=True,
                with_stack=False)
            prof.start()
        iterator = tqdm(self.train_loader, disable=self.gpu != 0)
        for ix, (images, target) in enumerate(iterator):
            # Training start
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lrs[ix]

            self.optimizer.zero_grad(set_to_none=True)
            with autocast():
                # Step 1: Backward pass
                target_onehot = F.one_hot(
                    target, num_classes=self.model.num_classes).to(ch.float16)
                bwd_activations = [net(target_onehot.to(f'cuda:{self.gpu + i}'))
                                   for i, net in enumerate(self.model.bwd_net)]

                # Step 2: Forward pass of the model
                forward_output, fwd_activations, denoise_activations = self.model(
                    images)

                # Step 3: Greedy Train backward network on prototype vectors derived from fwd_activations
                loss_proto = self._train_backward_net(fwd_activations, target)

            self.backward_scaler.scale(loss_proto.sum()).backward()
            self.backward_scaler.step(self.backward_optimizer)
            self.backward_scaler.update()

            with autocast():
                # Step 4: Train forward network (a.k.a. model) on the target and prototype vectors from backward network
                loss, forward_loss, loss_corr, loss_sim = self._train_forward_net(forward_output, fwd_activations,
                                                                                  bwd_activations, denoise_activations,
                                                                                  target)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # Training end

            # Logging start
            if log_level > 0:
                losses.append(forward_loss.detach())

                group_lrs = []
                for _, group in enumerate(self.optimizer.param_groups):
                    group_lrs.append(f'{group["lr"]:.3f}')

                names = ['ep', 'iter', 'shape', 'lrs']
                values = [epoch, ix, tuple(images.shape), group_lrs]
                if log_level > 1:
                    names += ['loss']
                    values += [f'{forward_loss.item():.3f}']

                msg = ', '.join(f'{n}={v}' for n, v in zip(names, values))
                iterator.set_description(msg)

                if self.wandb_run is not None:
                    losses_dict = {'epoch': epoch} | {'batch': ix} | \
                                  {f"local_losses/corr_loss-{ii}": loss.item() for ii, loss in
                                   enumerate(loss_corr)} | \
                                  {f"local_losses/sim_loss-{ii}": loss.item() for ii, loss in
                                   enumerate(loss_sim)} | \
                                  {f"local_losses/backward_loss-{ii}": loss.item() for ii, loss in
                                   enumerate(loss_proto)}

                    debug_dict = {'epoch': epoch} | {'batch': ix} | \
                        {'bwd_LR': self.backward_optimizer.param_groups[0]['lr']}

                    self.wandb_run.log(losses_dict)
                    self.wandb_run.log(debug_dict)
            # Logging end
            avg_train_loss = ch.mean(ch.Tensor(losses)).item()
            if epoch == profile_epoch:
                prof.step()
        if epoch == profile_epoch:
            prof.stop()
        return avg_train_loss

    def _train_forward_net(self, forward_output, fwd_activations, bwd_activations, denoise_activations, target):
        last_dev = f'cuda:{self.gpu+self.model.num_blocks-1}'
        forward_loss = self.loss(forward_output, target.to(last_dev))
        loss_corr, loss_sim, loss_pred, loss_denoise = (ch.zeros((self.model.num_blocks - 1,),
                                                                 device=last_dev
                                                                 ) for _ in range(4))

        for i, acts in enumerate(fwd_activations[1:-1]):
            loss_corr[i], loss_sim[i] = self.local_loss(
                acts, bwd_activations[i].detach())
            if self.local_pred_loss > 0:
                act_pred = self.model.bwd_pred_net[i](acts)
                loss_pred[i] = self.loss(
                    act_pred, target.to(f'cuda:{self.gpu + i}'))
            if self.denoise_loss > 0:
                loss_denoise[i] = F.mse_loss(
                    denoise_activations[i+1], acts.detach())
        loss = self.alpha * forward_loss + self.beta * sum(loss_corr) + self.gamma * sum(loss_sim) + \
            self.local_pred_loss * sum(loss_pred) + \
            self.denoise_loss * sum(loss_denoise)
        return loss, forward_loss, loss_corr, loss_sim

    def _train_backward_net(self, fwd_activations, target):
        for proto_loss_step in range(self.greedy_train_steps):
            self.backward_optimizer.zero_grad()  # (set_to_none=True)
            loss_proto = ch.zeros(
                (self.model.num_blocks - 1,), device=f'cuda:{self.gpu+self.model.num_blocks-1}')
            for i, acts in enumerate(fwd_activations[1:-1]):
                device = f'cuda:{self.gpu+i}'
                x_proto, labels = module_loss.get_proto_vectors(
                    acts.detach(), target.to(device), num_classes=self.model.num_classes)
                prototype = self.model.bwd_net[i](
                    ch.eye(x_proto.shape[0]).to(device))
                loss_proto[i] = 0.5 * \
                    F.mse_loss(prototype, x_proto, reduction='sum')
        return loss_proto

    @param('validation.lr_tta')
    def val_loop(self, lr_tta):
        model = self.model
        model.eval()

        with ch.no_grad():
            with autocast():
                for images, target in tqdm(self.val_loader, disable=self.gpu != 0):
                    output, *_ = self.model(images)
                    if lr_tta:
                        tta_output, * \
                            _ = self.model(ch.flip(images, dims=[3]))
                        output += tta_output

                    for k in ['top_1', 'top_5']:
                        self.val_meters[k](output.to(self.gpu), target)

                    loss_val = self.loss(output.to(self.gpu), target)
                    self.val_meters['loss'](loss_val)

        stats = {k: m.compute().item() for k, m in self.val_meters.items()}
        [meter.reset() for meter in self.val_meters.values()]
        return stats

    @param('validation.lr_tta')
    def test_loop(self, lr_tta):
        model = self.model
        model.eval()

        with ch.no_grad():
            with autocast():
                for images, target in tqdm(self.test_loader, disable=self.gpu != 0):
                    output, *_ = self.model(images)
                    if lr_tta:
                        tta_output, * \
                            _ = self.model(ch.flip(images, dims=[3]))
                        output += tta_output

                    for k in ['top_1', 'top_5']:
                        self.test_meters[k](output.to(self.gpu), target)

                    loss_test = self.loss(output.to(self.gpu), target)
                    self.test_meters['loss'](loss_test)

        stats = {k: m.compute().item() for k, m in self.test_meters.items()}
        [meter.reset() for meter in self.test_meters.values()]
        return stats

    @param('logging.folder')
    @param('model.num_classes')
    @param('logging.wandb_config')
    def initialize_logger(self, folder, num_classes, wandb_config):
        self.val_meters = {
            'top_1': torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(self.gpu),
            'top_5': torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5).to(self.gpu),
            'loss': MeanScalarMetric().to(self.gpu)
        }
        self.test_meters = {
            'top_1': torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(self.gpu),
            'top_5': torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5).to(self.gpu),
            'loss': MeanScalarMetric().to(self.gpu)
        }
        self.wandb_run = None

        if self.gpu == 0:
            folder = (Path(folder) / str(os.getenv('SLURM_JOB_ID')) /
                      str(self.uid)).absolute()
            folder.mkdir(parents=True)

            self.log_folder = folder
            self.start_time = time.time()

            print(f'=> Logging in {self.log_folder}')
            params = {
                '.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()
            }

            with open(folder / 'params.json', 'w+') as handle:
                json.dump(params, handle)
            if wandb_config != '':
                wandb_config = load_config(wandb_config)
                self.wandb_run = wandb.init(project=wandb_config['project'], entity=wandb_config['entity'], name=None,
                                            dir=wandb_config['dir'],
                                            allow_val_change=True)
                wandb.config.update(config_as_dict(
                    get_current_config()), allow_val_change=True)
                wandb.config.update({'SLURM job ID': os.getenv(
                    'SLURM_JOB_ID'),
                    'uid': self.uid}, allow_val_change=True)

    def log(self, content):
        print(f'=> Log: {content}')
        if self.gpu != 0:
            return
        cur_time = time.time()
        with open(self.log_folder / 'log', 'a+') as fd:
            fd.write(json.dumps({
                'timestamp': cur_time,
                'relative_time': cur_time - self.start_time,
                **content
            }) + '\n')
            fd.flush()

    @classmethod
    @param('training.distributed')
    @param('dist.world_size')
    @param('model.num_splits')
    def launch_from_args(cls, distributed, world_size, num_splits):
        if distributed:
            nprocs = world_size//(num_splits+1)
            assert nprocs > 0, f"Need atleast {num_splits+1} GPUs"
            print(
                f"Launching {nprocs} processes with {num_splits+1} GPUs each")
            ch.multiprocessing.spawn(
                cls._exec_wrapper, nprocs=nprocs, join=True)
        else:
            cls.exec(0)

    @classmethod
    def _exec_wrapper(cls, *args, **kwargs):
        make_config(quiet=True)
        cls.exec(*args, **kwargs)

    @classmethod
    @param('training.distributed')
    @param('training.eval_only')
    def exec(cls, gpu, distributed, eval_only):
        trainer = cls(gpu=gpu)
        if eval_only:
            trainer.test_and_log()
        else:
            trainer.train()

        if distributed:
            trainer.cleanup_distributed()

# Utils


class MeanScalarMetric(torchmetrics.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_state('sum', default=ch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('count', default=ch.tensor(0), dist_reduce_fx='sum')

    def update(self, sample: ch.Tensor):
        self.sum += sample.sum()
        self.count += sample.numel()

    def compute(self):
        return self.sum.float() / self.count

# Running


def load_config(filename: str) -> dict:
    """
    Load a configuration file as YAML.
    """
    with open(filename) as fh:
        config = yaml.safe_load(fh)
    return config


def config_as_dict(config):
    """ Stores config in a dict, like 'section.parameter' -> value """
    output = {}
    for path in config.entries.keys():
        try:
            value = config[path]
            if value is not None:
                output['.'.join(path)] = config[path]
        except:
            pass
    return output


def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()


@param('training.seed')
def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    ch.manual_seed(seed)
    ch.cuda.manual_seed(seed)


if __name__ == "__main__":
    seed_everything()
    make_config()
    ImageNetTrainer.launch_from_args()
