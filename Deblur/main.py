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

    dataset_gopro_test = Dataset.Dataset(0, config["datasets"]["test"]["path_1"], aug_mode=False, train=False)
    dataset_hide_test = Dataset.Dataset(0, config["datasets"]["test"]["path_2"], aug_mode=False, train=False)
    dataset_realblur_J_test = Dataset.Dataset(0, config["datasets"]["test"]["path_3"], aug_mode=False, train=False)
    dataset_realblur_R_test = Dataset.Dataset(0, config["datasets"]["test"]["path_4"], aug_mode=False, train=False)

    gopro_test_sampler = DistributedSampler(dataset_gopro_test, num_replicas=world_size, rank=rank, shuffle=False)
    hide_test_sampler = DistributedSampler(dataset_hide_test, num_replicas=world_size, rank=rank, shuffle=False)
    realblur_J_test_sampler = DistributedSampler(dataset_realblur_J_test, num_replicas=world_size, rank=rank, shuffle=False)
    realblur_R_test_sampler = DistributedSampler(dataset_realblur_R_test, num_replicas=world_size, rank=rank, shuffle=False)

    test_gopro_loader = Data.DataLoader(dataset=dataset_gopro_test, num_workers=4, batch_size=1, sampler=gopro_test_sampler)
    test_hide_loader = Data.DataLoader(dataset=dataset_hide_test, num_workers=4, batch_size=1, sampler=hide_test_sampler)
    test_realblur_J_loader = Data.DataLoader(dataset=dataset_realblur_J_test, num_workers=4, batch_size=1, sampler=realblur_J_test_sampler)
    test_realblur_R_loader = Data.DataLoader(dataset=dataset_realblur_R_test, num_workers=4, batch_size=1, sampler=realblur_R_test_sampler)


    if args.mode == "train":

        patch_size = config["datasets"]["train"]["patch_size"]
        train_data = config["datasets"]["train"]["path"]
        aug_mode = config["datasets"]["train"]["aug_mode"]
        batch_size = config["datasets"]["train"]["batch_size"]


        dataset_train = Dataset.Dataset(win = patch_size,path = train_data,aug_mode = aug_mode,train=True)
        train_sampler = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank,shuffle=True)
        train_loader = Data.DataLoader(dataset=dataset_train, num_workers=4, batch_size=batch_size,sampler=train_sampler)

        test_loader_dict = {"gopro": test_gopro_loader,
                            "hide": test_hide_loader,
                            "realblur_J": test_realblur_J_loader,
                            "realblur_R": test_realblur_R_loader}

        if rank == 0:
            print("# of training samples: %d\n" % int(len(dataset_train)))

        runner.train(args,train_loader,test_loader_dict)

    else:
        if args.path == None and rank == 0:
            raise ValueError("The folder where the test weights are located needs to be provided!")
        if len((args.cuda).split(',')) > 1 and rank == 0:
            warnings.warn("Using more than one graphics card for testing may cause deviations in the average indicators.")
        runner.test(test_gopro_loader, "Gopro", args.recode)
        runner.test(test_hide_loader, "HIDE", args.recode)
        runner.test(test_realblur_R_loader, "RealBlur-R", args.recode)
        runner.test(test_realblur_J_loader, "RealBlur-J", args.recode)

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cuda", type=str, default="2,7", help="gpu to train")
    parser.add_argument("-r", "--recode", help="choose whether to recode", action="store_true")
    parser.add_argument("-p", "--path", type=str, default=None, help="pre weight path")
    parser.add_argument("-m", "--mode", type=str, default="train", choices=["train","test"], help="choose to train or test")

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    WORLD_SIZE = torch.cuda.device_count()
    port = random.randint(10000, 20000)

    mp.spawn(main, args=(WORLD_SIZE, args, port), nprocs=WORLD_SIZE, join=True)




