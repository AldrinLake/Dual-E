import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager, DataManager2
from utils.toolkit import count_parameters
import os
import numpy as np
from utils.toolkit import count_parameters, save_results_to_excel, convert_time, get_device_name
import time


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["convnet_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            # logging.StreamHandler(sys.stdout),
        ],
    )
    
    device_name_list = get_device_name(args["device"])
    _set_random(args)
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    model = factory.get_model(args["model_name"], args)

    result_for_record = {
        'NCM': [],
        'NME': [],
        'CNN': [],
    }

    cnn_curve, nme_curve, ncm_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix, ncm_matrix = [], [], []
    start_time = time.time()
    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model.incremental_train(data_manager)
        cnn_accy, nme_accy, ncm_accy = model.eval_task()
        model.after_task()

        logging.info("CNN: {}".format(cnn_accy["grouped"]))
        logging.info("NME: {}".format(nme_accy["grouped"]))
        logging.info("NCM: {}".format(ncm_accy["grouped"]))

        cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
        cnn_keys_sorted = sorted(cnn_keys)
        cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys_sorted]
        cnn_matrix.append(cnn_values)

        nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
        nme_keys_sorted = sorted(nme_keys)
        nme_values = [nme_accy["grouped"][key] for key in nme_keys_sorted]
        nme_matrix.append(nme_values)

        ncm_keys = [key for key in ncm_accy["grouped"].keys() if '-' in key]
        ncm_keys_sorted = sorted(ncm_keys)
        ncm_values = [ncm_accy["grouped"][key] for key in ncm_keys_sorted]
        ncm_matrix.append(ncm_values)


        cnn_curve["top1"].append(cnn_accy["top1"])
        nme_curve["top1"].append(nme_accy["top1"])
        ncm_curve["top1"].append(ncm_accy["top1"])

        # cnn_curve["top5"].append(cnn_accy["top5"])
        # nme_curve["top5"].append(nme_accy["top5"])
        # ncm_curve["top5"].append(ncm_accy["top5"])

        logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
        logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
        logging.info("NCM top1 curve: {}".format(ncm_curve["top1"]))
        # logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
        # logging.info("NME top5 curve: {}".format(nme_curve["top5"]))
        # logging.info("NCM top5 curve: {}".format(ncm_curve["top5"]))

        logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
        logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"])/len(nme_curve["top1"])))
        logging.info("Average Accuracy (NCM): {}".format(sum(ncm_curve["top1"])/len(ncm_curve["top1"])))

    if len(ncm_matrix)>0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(ncm_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        np_acctable = np_acctable.T
        forgetting_ncm = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
        logging.info('Forgetting (NCM): {}'.format(forgetting_ncm))
    if len(cnn_matrix)>0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(cnn_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        np_acctable = np_acctable.T
        forgetting_cnn = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
        logging.info('Forgetting (CNN): {}'.format(forgetting_cnn))
    if len(nme_matrix)>0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(nme_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        np_acctable = np_acctable.T
        forgetting_nme = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
        logging.info('Forgetting (NME):{}'.format(forgetting_nme))

    result_for_record["NCM"].append((args['model_name']+args['version'], str(args), ncm_curve["top1"], "\n".join([str(item) for item in ncm_matrix]), forgetting_ncm))
    result_for_record["CNN"].append((args['model_name']+args['version'], str(args), cnn_curve["top1"], "\n".join([str(item) for item in cnn_matrix]), forgetting_cnn))
    result_for_record["NME"].append((args['model_name']+args['version'], str(args), nme_curve["top1"], "\n".join([str(item) for item in nme_matrix]), forgetting_nme))


    save_results_to_excel(
        dataset_name = args["dataset"],
        file_name = args["model_name"] + args["suffix_res_file"],
        incremental_num = str(args["init_cls"])+"_"+str(args["increment"]),
        results = result_for_record,
        runing_time= convert_time(time.time() - start_time),
        device=str(device_name_list), 
        note=str([args["note"]]), 
        seed= args["seed"]
    )


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(args):
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
