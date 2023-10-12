import logging
import numpy as np
from collections import OrderedDict
from collections import namedtuple
from itertools import product
import constants
import torch
from mymodels import TransICD, Transformer
from data.data import prepare_datasets, load_embedding_weights
from trainer import train
import random
import os
from predictors.predictor import Predictor
from torch.utils.data import DataLoader
from data.graph_generator import generate_graph
from models.TransGraphICD import TransGraphICDConfig, TransGraphICD


def get_hyper_params_combinations(args):
    params = OrderedDict(
        learning_rate=args.learning_rate,
        num_epoch=args.num_epoch
    )

    HyperParams = namedtuple('HyperParams', params.keys())
    hyper_params_list = []
    for v in product(*params.values()):
        hyper_params_list.append(HyperParams(*v))
    return hyper_params_list


def run(args, device):
    train_set, dev_set, test_set, train_labels, train_label_freq, input_indexer, mlb = prepare_datasets(data_setting="full",
                                                                                                   batch_size=args.batch_size,
                                                                                                   max_len=args.max_len)
    print("print dataset...................................................................")
    logging.info(f'Taining labels are: {train_labels}\n')
    embed_weights = load_embedding_weights()
    label_desc = None  # load_label_embedding(train_labels, input_indexer.index_of(constants.PAD_SYMBOL))
    model = None
    for hyper_params in get_hyper_params_combinations(args):
        if args.model == 'Transformer':
            print(f'train_set code count:{train_set.get_code_count()}')
            model = Transformer(embed_weights, args.embed_size, args.freeze_embed, args.max_len, args.num_trans_layers,
                                args.num_attn_heads, args.trans_forward_expansion, train_set.get_code_count(),
                                args.dropout_rate, device)
        elif args.model == 'TransICD':
            model = TransICD(embed_weights, args.embed_size, args.freeze_embed, args.max_len, args.num_trans_layers,
                             args.num_attn_heads, args.trans_forward_expansion, train_set.get_code_count(),
                             args.label_attn_expansion, args.dropout_rate, label_desc, device, train_label_freq)
        elif args.model == "TransGraphICD":
            config = TransGraphICDConfig(embed_weights, args.embed_size, args.freeze_embed, args.max_len,
                                         args.num_trans_layers,
                                         args.num_attn_heads, args.trans_forward_expansion, train_set.get_code_count(),
                                         args.label_attn_expansion, args.dropout_rate, device)
            print(config)
            model = TransGraphICD(config)
        else:
            raise ValueError("Unknown value for args.model. Pick Transformer or TransICD")

        if model:
            model.to(device)
            logging.info(f"Training with: {hyper_params}")
            train(model, train_set, dev_set, test_set, hyper_params, args.batch_size, device)


if __name__ == "__main__":
    args = constants.get_args()
    run_mode = args.mode
    use_cuda = torch.cuda.is_available()
    print("use_cuda : " + str(use_cuda))
    device = torch.device("cuda" if use_cuda else "cpu")
    print(run_mode)
    if run_mode == 'test':
        print("Running Test ....")
        model = torch.load(f'../results/model.pt')
        predictor = Predictor(model=model, device=device)
        train_set, dev_set, test_set, train_labels, train_label_freq, input_indexer = prepare_datasets(
            data_setting='50', batch_size=8, max_len=1500)
        dataLoader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=1)
        for batch in dataLoader:
            print(batch)
            output = predictor.predict(text=batch['text'])
            print(output)
            break
    elif run_mode == 'train':
        print(run_mode)
        if not os.path.exists('../results'):
            os.makedirs('../results')
        FORMAT = '%(asctime)-15s %(message)s'
        logging.basicConfig(filename='../results/app.log', filemode='w', format=FORMAT,
                            level=getattr(logging, args.log.upper()))
        logging.info(f'{args}\n')
        print(run_mode)
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        if use_cuda:
            torch.cuda.manual_seed_all(args.random_seed)
        run(args, device)
    elif run_mode == 'generate_graph':
        model = torch.load(f'../results/model.pt')
        generate_graph(model, device)
