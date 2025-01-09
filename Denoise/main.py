import torch.utils.data as Data
import Dataset
import argparse
from Runner import make_model,Runner
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import torch
import random
from torch.utils.data.distributed import DistributedSampler
import warnings

def setup(rank, world_size, port=12365):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, args, port):
    setup(rank, world_size, port)

    model, config = make_model(args)
    runner = Runner(model, args, config, rank)


    if rank == 0:
        print('Loading dataset ...\n')

    dataset_sidd_val = Dataset.Dataset(0, config["datasets"]["test"]["path"], aug_mode=False, train=False)
    sidd_val_sampler = DistributedSampler(dataset_sidd_val, num_replicas=world_size, rank=rank, shuffle=False)
    sidd_val_loader = Data.DataLoader(dataset=dataset_sidd_val, num_workers=4, batch_size=1, sampler= sidd_val_sampler)


    if args.mode == "train":

        patch_size = config["datasets"]["train"]["patch_size"]
        train_data = config["datasets"]["train"]["path"]
        aug_mode = config["datasets"]["train"]["aug_mode"]
        batch_size = config["datasets"]["train"]["batch_size"]


        dataset_train = Dataset.Dataset(win = patch_size,path = train_data,aug_mode = aug_mode,train=True)
        train_sampler = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank,shuffle=True)
        train_loader = Data.DataLoader(dataset=dataset_train, num_workers=4, batch_size=batch_size,sampler=train_sampler)

        if rank == 0:
            print("# of training samples: %d\n" % int(len(dataset_train)))

        runner.train(args,train_loader,sidd_val_loader)

    elif args.mode == "test":
        if args.path == None and rank == 0:
            raise ValueError("The folder where the test weights are located needs to be provided!")
        if len((args.cuda).split(',')) > 1 and rank == 0:
            warnings.warn("Using more than one graphics card for testing may cause deviations in the average indicators.")

        dataset_dnd_benchmark = Dataset.DND_benchmark_Dataset(config["datasets"]["submit"]["DND_path"])
        dnd_sampler = DistributedSampler(dataset_dnd_benchmark, num_replicas=world_size, rank=rank, shuffle=False)
        dnd_loader = Data.DataLoader(dataset=dataset_dnd_benchmark, num_workers=0, batch_size=1, sampler=dnd_sampler)

        dataset_sidd_benchmark = Dataset.SIDD_benchmark_Dataset(config["datasets"]["submit"]["SIDD_path"])
        block_len = dataset_sidd_benchmark.block_len()
        sidd_sampler = DistributedSampler(dataset_sidd_benchmark, num_replicas=world_size, rank=rank, shuffle=False)
        sidd_loader = Data.DataLoader(dataset=dataset_sidd_benchmark, num_workers=1, batch_size=1, sampler=sidd_sampler)
        # runner.val(sidd_val_loader,args.recode)
        runner.DND_benchmark_submit(dnd_loader)
        runner.SIDD_benchmark_submit(sidd_loader,block_len)

    else:
        if args.image == None and rank == 0:
            raise ValueError("The image path needs to be provided!")
        if args.path == None and rank == 0:
            raise ValueError("The folder where the test weights are located needs to be provided!")
        runner.single_test(args.image)


    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cuda", type=str, default="2,7", help="gpu to train")
    parser.add_argument("-r", "--recode", help="choose whether to recode", action="store_true")
    parser.add_argument("-p", "--path", type=str, default=None, help="pre weight path")
    parser.add_argument("-m", "--mode", type=str, default="train", choices=["train","test","single_test"], help="choose to train or test")
    parser.add_argument("-i", "--image", type=str, default=None, help="image path for single test")

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    WORLD_SIZE = torch.cuda.device_count()
    port = random.randint(10000, 20000)

    mp.spawn(main, args=(WORLD_SIZE, args, port), nprocs=WORLD_SIZE, join=True)




