"""
Adapted from https://github.com/TorchSpatiotemporal/tsl/blob/main/examples/prediction/run_traffic.py
"""

import os
import copy

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from tsl.datasets import MetrLA, PemsBay
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.nn.utils import casting
from tsl.predictors import Predictor
from tsl.utils import TslExperiment, ArgParser, parser_utils, numpy_metrics
from tsl.utils.parser_utils import str_to_bool
from tsl.utils.neptune_utils import TslNeptuneLogger

import tsl

from tsl.nn.metrics.metrics import MaskedMAE, MaskedMAPE, MaskedMSE

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np

import pathlib
import datetime
import yaml

from tsl.nn.models.stgn.dcrnn_model import DCRNNModel
from tsl.nn.models.stgn.stcn_model import STCNModel
from tsl.nn.models.stgn.graph_wavenet_model import GraphWaveNetModel
from tsl.nn.models.stgn.rnn2gcn_model import RNNEncGCNDecModel as Rnn2GcnModel
from tsl.nn.models.stgn.gated_gn_model import GatedGraphNetworkModel

from tsl.nn.models import RNNModel, TransformerModel, TCNModel, FCRNNModel

tsl.config.config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config', 'traffic')


def get_model_class(model_str):
    if model_str == 'dcrnn':
        model = DCRNNModel
    elif model_str == 'stcn':
        model = STCNModel
    elif model_str == 'gwnet':
        model = GraphWaveNetModel
    elif model_str == 'rnn':
        model = RNNModel
    elif model_str == 'fcrnn':
        model = FCRNNModel
    elif model_str == 'tcn':
        model = TCNModel
    elif model_str == 'transformer':
        model = TransformerModel
    elif model_str == 'rnn2gcn':
        model = Rnn2GcnModel
    elif model_str == 'gatedgn':
        model = GatedGraphNetworkModel
    elif model_str == 'gpvar':
        from graph_polynomial_var import GraphPolyVARFilter
        model = GraphPolyVARFilter
    else:
        raise NotImplementedError(f'Model "{model_str}" not available.')
    return model


def get_dataset(dataset_name):
    if dataset_name == 'la':
        dataset = MetrLA(impute_zeros=True)
    elif dataset_name == 'bay':
        dataset = PemsBay()
    elif dataset_name == 'gpolyvar':
        from graph_polynomial_var import GraphPolyVARDataset
        T = 30000
        communities = 5
        connectivity = "line"
        data_path = f"./data/gpvar-T{T}_{connectivity}-c{communities}"
        if os.path.isdir(data_path):
            dataset = GraphPolyVARDataset.load_dataset(path=data_path)
        else:
            dataset = GraphPolyVARDataset(coefs=torch.tensor([[5, 2], [-4, 6], [-1, 0]], dtype=torch.float32),
                                          sigma_noise=.4,
                                          communities=communities, connectivity=connectivity)
            dataset.generate_data(T=T)
            dataset.dump_dataset(path=data_path)
        dataset.G.plot(signal=dataset.numpy()[-200:, ..., 0], savefig=os.path.join(data_path, "graph-viz.pdf"))
    else:
        raise ValueError(f"Dataset {dataset_name} not available in this setting.")
    return dataset


def add_parser_arguments(parent):
    # Argument parser
    parser = ArgParser(strategy='random_search', parents=[parent], add_help=False)

    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument("--model-name", type=str, default='fcrnn')
    parser.add_argument("--dataset-name", type=str, default='la')
    parser.add_argument("--config", type=str, default='fcrnn.yaml')
    # training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2-reg', type=float, default=0.),
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--grad-clip-val', type=float, default=5.)
    parser.add_argument('--use-lr-schedule', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--val-len', type=float, default=0.1)
    parser.add_argument('--test-len', type=float, default=0.2)

    # logging
    parser.add_argument('--save-preds', action='store_true', default=False)
    parser.add_argument('--neptune-logger', action='store_true', default=False)
    parser.add_argument('--project-name', type=str, default="sandbox")
    parser.add_argument('--tags', type=str, default=tuple())

    parser.add_argument('--from-checkpoint', type=str, default=None)
    parser.add_argument('--resume-training', action='store_true', default=False)

    known_args, _ = parser.parse_known_args()
    model_cls = get_model_class(known_args.model_name)
    parser = model_cls.add_model_specific_args(parser)
    parser = SpatioTemporalDataset.add_argparse_args(parser)
    parser = SpatioTemporalDataModule.add_argparse_args(parser)
    return parser


def run_experiment(args):

    # Set configuration and seed
    _train_model = args.resume_training or (args.from_checkpoint is None)
    _resume_training = args.resume_training
    if args.from_checkpoint is not None:
        checkpoint = os.path.abspath(args.from_checkpoint)
        logging_path = os.path.abspath(os.path.join("/", "/".join(checkpoint.split("/")[:-1])))
        config_file = os.path.abspath(os.path.join(logging_path, "tsl_config.yaml"))
        result_file = os.path.abspath(os.path.join(logging_path, "results_.npy"))

        tsl.logger.info(f"Reading config_file: {config_file}")
        stored_args = tsl.Config.from_config_file(config_file)
        stored_args["from_checkpoint"] = args.from_checkpoint
        stored_args["neptune_logger"] = False
        args = stored_args
        for k, v in args.items():
            tsl.logger.info(f"{k:25s}: {v}")
    else:
        args = copy.deepcopy(args)
    if args.seed < 0:
        args.seed = np.random.randint(1e9)
    torch.set_num_threads(1)
    pl.seed_everything(args.seed)

    # create logdir
    exp_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.seed}"
    logdir = os.path.join(tsl.config.log_dir,
                          args.dataset_name,
                          args.model_name,
                          exp_name)

    if _train_model:
        pathlib.Path(logdir).mkdir(parents=True)

        from logging import FileHandler
        file_handler = FileHandler(os.path.join(logdir, "train_test.log"), "a")
        file_handler.setFormatter(tsl.logger.handlers[0].formatter)
        tsl.logger.addHandler(file_handler)
        # save config for logging
        with open(os.path.join(logdir, 'tsl_config.yaml'), 'w') as fp:
            yaml.dump(parser_utils.config_dict_from_args(args), fp, indent=4, sort_keys=True)

    tsl.logger.info(f'SEED: {args.seed}')

    model_cls = get_model_class(args.model_name)
    dataset = get_dataset(args.dataset_name)

    tsl.logger.info(args)

    ########################################
    # data module                          #
    ########################################

    # encode time of the day and use it as exogenous variable.
    if args.dataset_name in ["gpolyvar", "debug"]:
        exog_vars = {}
        adj = dataset.gso
    else:
        exog_vars = dataset.datetime_encoded('day').values
        exog_vars = {'global_u': exog_vars}
        adj = dataset.get_connectivity(method='distance', threshold=0.1,
                                       layout='edge_index')

    torch_dataset = SpatioTemporalDataset(*dataset.numpy(return_idx=True),
                                          connectivity=adj,
                                          mask=dataset.mask,
                                          horizon=args.horizon,
                                          window=args.window,
                                          stride=args.stride,
                                          exogenous=exog_vars)

    dm_conf = parser_utils.filter_args(args, SpatioTemporalDataModule, return_dict=True)
    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers={'data': StandardScaler(axis=(0, 1))},
        splitter=dataset.get_splitter(val_len=args.val_len,
                                      test_len=args.test_len),
        **dm_conf
    )
    dm.setup()

    if _train_model:
        ########################################
        # predictor                            #
        ########################################

        loss_fn = MaskedMAE(compute_on_step=True)

        metrics = {'mae': MaskedMAE(compute_on_step=False),
                   'mse': MaskedMSE(compute_on_step=False),
                   'mape': MaskedMAPE(compute_on_step=False),
                   }

        # setup predictor
        scheduler_class = CosineAnnealingLR if args.use_lr_schedule else None
        if _resume_training:
            tsl.logger.info(f"Loading checkpoint to resume training: {checkpoint}")
            predictor = Predictor.load_from_checkpoint(checkpoint_path=checkpoint)
        else:
            additional_model_hparams = dict(n_nodes=torch_dataset.n_nodes,
                                            input_size=torch_dataset.n_channels,
                                            output_size=torch_dataset.n_channels,
                                            horizon=torch_dataset.horizon,
                                            exog_size=torch_dataset.input_map.u.n_channels if len(
                                                exog_vars) else 0)

            model_kwargs = parser_utils.filter_args(
                args={**vars(args), **additional_model_hparams},
                target_cls=model_cls,
                return_dict=True)
            predictor = Predictor(
                model_class=model_cls,
                model_kwargs=model_kwargs,
                optim_class=torch.optim.Adam,
                optim_kwargs={'lr': args.lr,
                              'weight_decay': args.l2_reg},
                loss_fn=loss_fn,
                metrics=metrics,
                scheduler_class=scheduler_class,
                scheduler_kwargs={
                    'eta_min': 0.0001,
                    'T_max': args.epochs
                },
            )

        ########################################
        # logging options                      #
        ########################################

        # log number of parameters
        args.trainable_parameters = predictor.trainable_parameters

        # add tags
        tags = list(args.tags) + [args.model_name, args.dataset_name]

        if args.neptune_logger:
            logger = TslNeptuneLogger(api_key=tsl.config['neptune_token'],
                                      project_name=f"{tsl.config['neptune_username']}/{args.project_name}",
                                      experiment_name=exp_name,
                                      tags=tags,
                                      params=vars(args),
                                      offline_mode=False,
                                      upload_stdout=False)
            logger.log_text(log_name="sys/logdir", text=logdir)
        else:
            logger = TensorBoardLogger(
                save_dir=logdir,
                name=f'{exp_name}_{"_".join(tags)}',

            )

        ########################################
        # training                             #
        ########################################

        early_stop_callback = EarlyStopping(
            monitor='val_mae',
            patience=args.patience,
            mode='min'
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=logdir,
            save_top_k=1,
            monitor='val_mae',
            mode='min',
        )

        if _resume_training:
            import re
            done_epochs = re.findall(r"/epoch=(\d+)-step", checkpoint)
            assert len(done_epochs) == 1
            done_epochs = int(done_epochs[0])
        else:
            done_epochs = 0

        trainer = pl.Trainer(max_epochs=args.epochs - done_epochs,
                             default_root_dir=logdir,
                             logger=logger,
                             gpus=1 if torch.cuda.is_available() else None,
                             gradient_clip_val=args.grad_clip_val,
                             callbacks=[early_stop_callback, checkpoint_callback])

        trainer.fit(predictor, datamodule=dm)
    else:
        trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else None,)

    ########################################
    # testing                              #
    ########################################
    if not _train_model:
        tsl.logger.info(f"Reading checkpoint: {checkpoint}")
        predictor = Predictor.load_from_checkpoint(checkpoint_path=checkpoint)
    else:
        predictor.load_state_dict(
           torch.load(checkpoint_callback.best_model_path, lambda storage, loc: storage)['state_dict'])

        predictor.freeze()


    from graph_sign_test import masked_median, optimality_check
    multivariate=False

    def display_mae(ypr, ytr, mask):
        # Here the output should be (T, H, N, F)
        from einops import rearrange
        pattern_ = 't h n f -> t n (h f)'
        ypr = rearrange(ypr, pattern_)
        ytr = rearrange(ytr, pattern_)
        mask = rearrange(mask, pattern_).astype(bool)

        # Check mae
        r_ = ypr - ytr
        r_med_ = masked_median(r_, mask=mask)
        msg = []
        msg.append(f"original model MAE: {numpy_metrics.masked_mae(ypr, ytr, mask)}")
        msg.append(f"corrected model (median={r_med_.mean():.6f})" + \
                   f" MAE: {numpy_metrics.masked_mae(ypr - r_med_, ytr, mask)}")

        return ypr, ytr, mask, msg

    graph = dict(edge_index_spatial=adj[0], edge_weight_spatial=adj[1])

    if args.dataset_name == "gpolyvar":

        edge_index, edge_weight = dataset.gso
        import torch_geometric
        if torch_geometric.__version__ <= "2.0.3":
            dataset.filter.__explain__ = False
        y_pred, y_true = dataset.filter.predict(dataset.x,
                                                edge_index=edge_index, edge_weight=edge_weight)
        residuals_optmse = (y_pred - y_true)

        mae_score = lambda x: x.abs().mean()
        tsl.logger.info(f"MAE optimal: analytical {dataset.mae_optimal}, empirical {mae_score(dataset.sigma_noise * torch.randn(dataset.x.shape))}")
        tsl.logger.info(f"MAE (median baseline)    {mae_score(dataset.x - torch.median(dataset.x))}")
        tsl.logger.info(f"MAE (mse-optimal) {mae_score(residuals_optmse)}")

        signal_figure = os.path.abspath("./results/gpolyvar_viz.png")
        fig_signal = dataset.G.plot_temporal(signal=dataset.x, savefig=signal_figure)
        if args.neptune_logger:
            # logger.log_figure(fig_signal)
            logger.log_artifact(signal_figure)
        tsl.logger.info(f"Saved dataset visualization {signal_figure}")

        from graph_sign_test import optimality_check
        graph = dict(edge_index_spatial=adj[0], edge_weight_spatial=adj[1])
        msg = optimality_check(x=residuals_optmse, mask=None, **graph)
        for m in msg:
            tsl.logger.info(m)

    results = dict()

    tsl.logger.info("########################################")
    tsl.logger.info("Predict test loader")
    tsl.logger.info("########################################")

    output = trainer.predict(predictor, dataloaders=dm.test_dataloader(shuffle=False))
    output = casting.numpy(output)
    y_hat, y_true, mask = output['y_hat'], \
                          output['y'], \
                          output['mask']
    results.update(dict(test_mae=numpy_metrics.masked_mae(y_hat, y_true, mask),
                        test_rmse=numpy_metrics.masked_rmse(y_hat, y_true, mask),
                        test_mape=numpy_metrics.masked_mape(y_hat, y_true, mask)))
    # import numpy as np
    # print(f"Saving results to: {result_file}")
    # np.save(result_file, np.array((y_hat, y_true, mask, adj), dtype=object))
    y_hat, y_true, mask, output_messages = display_mae(y_hat, y_true, mask)

    #########################################################
    # Optimality
    #########################################################

    tsl.logger.info("########################################")
    tsl.logger.info("Test optimality")
    tsl.logger.info("########################################")
    res = y_hat - y_true

    tsl.logger.info(" --- all components ---------------------")
    msg_ = optimality_check(x=res, mask=mask, multivariate=multivariate, **graph)
    for m in msg_:
        tsl.logger.info(m)

    output_messages.append(" --- first component --------------------")
    output_messages += optimality_check(x=res[..., :1], mask=mask[..., :1], **graph)

    output_messages.append(" --- median subtracted --------------------")
    res_median = masked_median(res, mask=mask)
    output_messages += optimality_check(x=res[..., :1] - res_median[..., :1], mask=mask[..., :1], **graph)

    for m in output_messages:
        tsl.logger.info(m)


    tsl.logger.info("########################################")
    tsl.logger.info("Debug")
    tsl.logger.info("########################################")

    tsl.logger.info(" --- debug: random --------------------")
    res_ = np.random.randn(*res.shape)
    mask_ = mask
    msg_ = optimality_check(x=np.random.randn(*res_.shape), multivariate=multivariate, mask=mask_, **graph)
    for m in msg_:
        tsl.logger.info(m)

    tsl.logger.info(" --- debug: shuffle directions --------------------")
    from graph_sign_test import test_shuffle_dim
    res_ = res[..., :1]
    mask_ = mask[..., :1]
    test_shuffle_dim(x=res_, mask=mask_, multivariate=multivariate, **graph)


    if args.neptune_logger:
        logger.finalize('success')

    tsl.logger.info("########################################")
    tsl.logger.info("Output messages")
    tsl.logger.info("########################################")

    for m in output_messages:
        tsl.logger.info(m)

    return tsl.logger.info(results)

if __name__ == '__main__':
    parser = ArgParser(add_help=False)
    parser = add_parser_arguments(parser)
    exp = TslExperiment(run_fn=run_experiment, parser=parser)
    exp.run()

