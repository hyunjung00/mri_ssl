import argparse
import os
from time import time

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
#from losses.loss import Loss
from optimizers.lr_scheduler import WarmupCosineSchedule
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from utils.crossmoda_data_utils import get_loader
from utils.ops import aug_rand, rot_rand
from timm.models.layers.helpers import to_3tuple
from lib.tools.visualization import patches3d_to_grid

#for mae
from lib.models.mae3d import MAE3D
from lib.networks.mae_vit import MAEViTEncoder, MAEViTDecoder

# for Forkedpdb
import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """
    PDB Subclass for debugging multi-processed code
    Suggested in: https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin



def main():

    def save_ckp(state, checkpoint_dir):
        torch.save(state, checkpoint_dir)

    def train(args, global_step, train_loader, val_best, scaler):
        #model.to(torch.double)
        model.train()
        loss_train = []

        for step, batch in enumerate(train_loader):
            t1 = time()
            x = batch["image"].cuda()
            with autocast(enabled=args.amp):
                loss = model(x, return_image=False)
                
            loss_train.append(loss.item())

            if args.amp:
                try:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                except RuntimeError:
                    ForkedPdb().set_trace()
            else:
                loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.lrdecay:
                scheduler.step()
            optimizer.zero_grad()
            if args.distributed:
                if dist.get_rank() == 0:
                    print("Step:{}/{}, Loss:{:.4f}, Time:{:.4f}".format(global_step, args.num_steps, loss, time() - t1))
            else:
                print("Step:{}/{}, Loss:{:.4f}, Time:{:.4f}".format(global_step, args.num_steps, loss, time() - t1))

            global_step += 1
            if args.distributed:
                val_cond = (dist.get_rank() == 0) and (global_step % args.eval_num == 0)
            else:
                val_cond = global_step % args.eval_num == 0

            if val_cond:
                val_loss, vis_grid_hw = validation(args, test_loader)
                writer.add_scalar("Validation/loss_recon", scalar_value=val_loss, global_step=global_step)
                writer.add_scalar("train/loss_total", scalar_value=np.mean(loss_train), global_step=global_step)

                #writer.add_image("Validation/x1_gt", vis_grid_hw, global_step, dataformats="HW")

                if val_loss< val_best:
                    val_best = val_loss
                    checkpoint = {
                        "global_step": global_step,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    save_ckp(checkpoint, logdir + "/model_bestValRMSE.pt")
                    print(
                        "Model was saved ! Best Val Loss: {:.4f}, Val Loss: {:.4f}".format(
                            val_best, val_loss
                        )
                    )
                else:
                    print(
                        "Model was not saved ! Best Val Loss: {:.4f} Val Loss: {:.4f}".format(
                            val_best, val_loss
                        )
                    )
        return global_step, loss, val_best

    def validation(args, test_loader):
        #model.to(torch.double)
        model.eval()
        loss_val = []
        with torch.no_grad():
            for step, batch in enumerate(test_loader):
                val_inputs = batch["image"].cuda()

                with autocast(enabled=args.amp):
                    loss, x, recon, masked_x = model(val_inputs, return_image=True)
                loss_val.append(loss.item())
                vis_tensor = torch.cat([x, masked_x, recon], dim=0)

                # visualize 
                grid_size = []
                input_size= (args.roi_x, args.roi_y, args.roi_z)
                for pa_size, in_size in zip(to_3tuple(args.patch_size), input_size):
                    grid_size.append(in_size // pa_size)
                vis_grid_hw = patches3d_to_grid(vis_tensor, patch_size=args.patch_size, grid_size=grid_size, in_chans=args.in_chans, hidden_axis='d')

        return  np.mean(loss_val), vis_grid_hw

    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--logdir", default="mae_example", type=str, help="directory to save the tensorboard logs")
    parser.add_argument("--epochs", default=100, type=int, help="number of training epochs")
    parser.add_argument("--num_steps", default=200000, type=int, help="number of training iterations")
    parser.add_argument("--eval_num", default=1000, type=int, help="evaluation frequency")
    parser.add_argument("--warmup_steps", default=500, type=int, help="warmup steps")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--feature_size", default=48, type=int, help="embedding size")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
    parser.add_argument("--a_min", default=0, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=2700, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
    parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
    parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")

    parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
    parser.add_argument("--sw_batch_size", default=2, type=int, help="number of sliding window batch size")
    parser.add_argument("--lr", default=1e-5, type=float, help="learning rate") # originally 4e-4
    parser.add_argument("--decay", default=0.1, type=float, help="decay rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--lrdecay", default=True, help="enable learning rate decay")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="maximum gradient norm")
    parser.add_argument("--opt", default="adamw", type=str, help="optimization algorithm")
    parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)
    parser.add_argument("--resume", default=None, type=str, help="resume training")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    parser.add_argument("--grad_clip", default=True, help="gradient clip")
    parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training") # default: "env://"
    parser.add_argument("--smartcache_dataset", action="store_true", help="use monai smartcache Dataset")
    parser.add_argument("--cache_dataset", action="store_true", help="use monai cache Dataset")
    parser.add_argument("--num_workers", default=12, help="number of workers ")
    parser.add_argument("--rank", default=0, help="rank for DDP")
    parser.add_argument("--fold", default=0, type=int, help="data fold")
    parser.add_argument("--resize_x", default=512, type=int, help="roi size in x direction")
    parser.add_argument("--resize_y", default=512, type=int, help="roi size in y direction")
    parser.add_argument("--resize_z", default=128, type=int, help="roi size in z direction")
    parser.add_argument("--roi_x", default=384, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=384, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=96,  type=int, help="roi size in z direction")

    ## arguments for MAE
    parser.add_argument("--patchembed", default='PatchEmbed3D', type=str)
    parser.add_argument("--pos_embed_type", default='sincos', type=str)
    parser.add_argument("--mask_ratio", default=0.75, type=int)
    
    parser.add_argument("--patch_size", default=16, type=int)
    parser.add_argument("--in_chans", default=1, type=int)
    parser.add_argument("--encoder_embed_dim", default=768, type=int)
    parser.add_argument("--encoder_depth", default=12, type=int)
    parser.add_argument("--encoder_num_heads", default=12, type=int)
    parser.add_argument("--decoder_embed_dim", default=384, type=int)
    parser.add_argument("--decoder_depth", default=8, type=int)
    parser.add_argument("--decoder_num_heads", default=12, type=int)
    parser.add_argument("--arch", default='vit_base', type=str)

    # parser.add_argument("--input_size_x", default=512, type=int)
    # parser.add_argument("--input_size_y", default=512, type=int)
    # parser.add_argument("--input_size_z", default=128, type=int)
    #parser.add_argument("--input_size", default=96, type=int)


    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

    logdir = "./runs/" + args.logdir
    args.amp = True
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
    args.distributed = True
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    args.device = "cuda:0"
    args.world_size = 1
    args.rank = 0

    if args.distributed:
        args.device = "cuda:%d" % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method=args.dist_url)
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print(
            "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (args.rank, args.world_size)
        )
    else:
        print("Training with a single process on 1 GPUs.")
    assert args.rank >= 0

    if args.rank == 0:
        os.makedirs(logdir, exist_ok=True)
        writer = SummaryWriter(logdir)
    else:
        writer = None

    model = MAE3D(
                encoder= MAEViTEncoder, 
                decoder= MAEViTDecoder, 
                args=args)
    model.cuda()

    if args.opt == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.decay)

    elif args.opt == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.decay)

    elif args.opt == "sgd":
        optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

    if args.resume:
        model_pth = args.resume
        model_dict = torch.load(model_pth)
        model.load_state_dict(model_dict["state_dict"])
        model.epoch = model_dict["epoch"]
        model.optimizer = model_dict["optimizer"]

    if args.lrdecay:
        if args.lr_schedule == "warmup_cosine":
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)

        elif args.lr_schedule == "poly":

            def lambdas(epoch):
                return (1 - float(epoch) / float(args.epochs)) ** 0.9

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)


    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[args.local_rank])
        
    train_loader, test_loader = get_loader(args)


    global_step = 0
    best_val = 1e8
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None
    while global_step < args.num_steps:
        global_step, loss, best_val = train(args, global_step, train_loader, best_val, scaler)
    checkpoint = {"epoch": args.epochs, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}

    if args.distributed:
        if dist.get_rank() == 0:
            torch.save(model.state_dict(), logdir + "final_model.pth")
        dist.destroy_process_group()
    else:
        torch.save(model.state_dict(), logdir + "final_model.pth")
    save_ckp(checkpoint, logdir + "/model_final_epoch.pt")


if __name__ == "__main__":
    main()

