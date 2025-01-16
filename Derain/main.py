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

    dataset_val_test100 = Dataset.Dataset(0, config["datasets"]["test"]["path_1"], aug_mode=False, train=False)
    dataset_val_rain100h = Dataset.Dataset(0, config["datasets"]["test"]["path_2"], aug_mode=False, train=False)
    dataset_val_rain100l = Dataset.Dataset(0, config["datasets"]["test"]["path_3"], aug_mode=False, train=False)
    dataset_val_test1200 = Dataset.Dataset(0, config["datasets"]["test"]["path_4"], aug_mode=False, train=False)
    dataset_val_test2800 = Dataset.Dataset(0, config["datasets"]["test"]["path_5"], aug_mode=False, train=False)


    val_test100_sampler = DistributedSampler(dataset_val_test100, num_replicas=world_size, rank=rank, shuffle=False)
    val_rain100h_sampler = DistributedSampler(dataset_val_rain100h, num_replicas=world_size, rank=rank, shuffle=False)
    val_rain100l_sampler = DistributedSampler(dataset_val_rain100l, num_replicas=world_size, rank=rank, shuffle=False)
    val_test1200_sampler = DistributedSampler(dataset_val_test1200, num_replicas=world_size, rank=rank, shuffle=False)
    val_test2800_sampler = DistributedSampler(dataset_val_test2800, num_replicas=world_size, rank=rank, shuffle=False)

    val_test100_loader = Data.DataLoader(dataset=dataset_val_test100, num_workers=4, batch_size=1,
                                         sampler= val_test100_sampler)
    val_rain100h_loader = Data.DataLoader(dataset=dataset_val_rain100h, num_workers=4, batch_size=1,
                                         sampler=val_rain100h_sampler)
    val_rain100l_loader = Data.DataLoader(dataset=dataset_val_rain100l, num_workers=4, batch_size=1,
                                         sampler=val_rain100l_sampler)
    val_test1200_loader = Data.DataLoader(dataset=dataset_val_test1200, num_workers=4, batch_size=1,
                                         sampler=val_test1200_sampler)
    val_test2800_loader = Data.DataLoader(dataset=dataset_val_test2800, num_workers=4, batch_size=1,
                                         sampler=val_test2800_sampler)


    if args.mode == "train":

        patch_size = config["datasets"]["train"]["patch_size"]
        train_data = config["datasets"]["train"]["path"]
        aug_mode = config["datasets"]["train"]["aug_mode"]
        batch_size = config["datasets"]["train"]["batch_size"]


        dataset_train = Dataset.Dataset(win = patch_size,path = train_data,aug_mode = aug_mode,train=True)
        train_sampler = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank,shuffle=True)
        train_loader = Data.DataLoader(dataset=dataset_train, num_workers=4, batch_size=batch_size,sampler=train_sampler)

        test_loader_dict = {"Test100": val_test100_loader,
                            "Rain100H": val_rain100h_loader,
                            "Rain100L": val_rain100l_loader,
                            "Test1200": val_test1200_loader,
                            "Test2800": val_test2800_loader}

        if rank == 0:
            print("# of training samples: %d\n" % int(len(dataset_train)))

        runner.train(args,train_loader,test_loader_dict)

    elif args.mode == "test":
        if args.path == None and rank == 0:
            raise ValueError("The folder where the test weights are located needs to be provided!")
        if len((args.cuda).split(',')) > 1 and rank == 0:
            warnings.warn("Using more than one graphics card for testing may cause deviations in the average indicators.")
        runner.test(val_test100_loader, "Test100", 256, args.recode)
        runner.test(val_rain100h_loader, "Rain100H", -1, args.recode)
        runner.test(val_rain100l_loader, "Rain100L", -1, args.recode)
        runner.test(val_test1200_loader, "Test1200", 256, args.recode)
        runner.test(val_test2800_loader, "Test2800", 256, args.recode)

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




