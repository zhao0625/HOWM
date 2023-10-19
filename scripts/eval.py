import datetime
from collections import defaultdict

import numpy as np
import torch
import wandb
from pymongo import MongoClient
from sacred.observers import MongoObserver
from torch.utils import data

import utils.utils_dataset as utils
import utils.utils_func
from scripts.init import ex

# [mongo database]
mongo_url = 'mongodb://10.200.206.34:27017'
sacred_db_name = 'oorl_sacred'
ex.observers.append(MongoObserver(url=mongo_url, db_name=sacred_db_name))
stats_db_name = 'oorl_statistics'


@ex.capture
def eval_model(_log, _run, stats_collection, description, enable_wandb,
               model_train, model_eval, eval_steps,
               in_training=True, cmd_call=False, model_folder=None, model=None):
    """
    stats_collection: mongo collection name
    """

    # [db]
    client = MongoClient(mongo_url)
    db = client[stats_db_name]
    collection = db[stats_collection]
    _log.info('[preparing evaluation]')

    # [dataset]
    _log.debug('[loading eval dataset...]')
    # > Note: Training and evaluation must use new config data at the same time, but PathDataset doesn't change
    eval_dataset = utils.PathDataset(hdf5_file=model_eval['dataset'], path_length=max(eval_steps),
                                     action_mapping=model_eval['action_mapping'])
    eval_loader = data.DataLoader(eval_dataset, batch_size=model_eval['eval_batch_size'],
                                  shuffle=False, num_workers=4)

    # > Use training data for eval, TODO note for max step
    train_dataset = utils.PathDataset(hdf5_file=model_train['dataset'], path_length=100,  # TODO hardcode for now
                                      action_mapping=model_eval['action_mapping'])
    train_loader = data.DataLoader(train_dataset, batch_size=model_eval['eval_batch_size'],
                                   shuffle=False, num_workers=4)

    # > Use training data for eval - TODO using same length one
    seg_train_dataset = utils.SegmentedPathDataset(hdf5_file=model_train['dataset'],
                                                   segment_length=10,  # TODO hardcode!
                                                   action_mapping=model_eval['action_mapping'])
    seg_train_loader = data.DataLoader(seg_train_dataset, batch_size=model_eval['eval_batch_size'],
                                       shuffle=False, num_workers=4)

    # > Evaluation
    _log.debug('[start evaluation...]')
    res_dict = {}
    for _step in eval_steps:
        # >>> store in the existing format and also a table form (#steps & top-k)
        _res_step = eval_loop(model=model, num_steps=_step, loader=eval_loader)
        res_dict.update(_res_step)
    _log.warning('[evaluation results]: {}'.format(res_dict))

    # > Eval on training data - note for same #states
    res_train_dict = {}
    for _step in eval_steps:
        # >>> store in the existing format and also a table form (#steps & top-k)
        _res_step = eval_loop(model=model, num_steps=_step, loader=train_loader, prefix='train')
        res_train_dict.update(_res_step)
    _log.warning('[evaluation results on *training* data]: {}'.format(res_train_dict))

    # > Eval on segmented training data
    res_seg_train_dict = {}
    for _step in eval_steps:
        # >>> store in the existing format and also a table form (#steps & top-k)
        _res_step = eval_loop(model=model, num_steps=_step, loader=seg_train_loader, prefix='seg-train')
        res_seg_train_dict.update(_res_step)
    _log.warning('[evaluation results on *segmented training* data]: {}'.format(res_seg_train_dict))

    # > Compute gap between eval and training (prefix with 'train')
    # TODO now using segmented training data
    # TODO use different keys for gap results, since we will store into mongo
    res_gap_dict = {
        'gap-train-' + _key: (res_train_dict['train-' + _key] - res_dict[_key]) for _key in res_dict
    }
    _log.warning('[generalization gap between eval and training scenes:]: {}'.format(res_gap_dict))
    res_seg_gap_dict = {
        'gap-seg-train' + _key: (res_seg_train_dict['seg-train-' + _key] - res_dict[_key]) for _key in res_dict
    }
    _log.warning('[generalization gap between eval and training scenes:]: {}'.format(res_gap_dict))

    # > Add results and path to mongo - on both eval and training data
    if enable_wandb:
        if not in_training:
            wandb.log({
                'model_save_folder': model_folder if model_folder is not None else model_train['save_folder']
            }, commit=False)

        # >>> commit the last log call
        prefix = 'EvalInTraining/' if in_training else 'Eval/'
        wandb.log({prefix + _key: _value for _key, _value in res_dict.items()})

        # > log evaluation in training
        prefix_train = 'EvalInTraining-TrainingConfig/' if in_training else 'Eval/'
        wandb.log({prefix_train + _key: _value for _key, _value in res_train_dict.items()})
        wandb.log({prefix_train + _key: _value for _key, _value in res_seg_train_dict.items()})
        wandb.log({prefix_train + _key: _value for _key, _value in res_gap_dict.items()})
        wandb.log({prefix_train + _key: _value for _key, _value in res_seg_gap_dict.items()})

    # [add results]
    # >>> Mongo - only save final results (when not in training)
    if not in_training:
        res_dict['description'] = description
        res_dict['config'] = _run.config
        res_dict['model_save_folder'] = model_folder  # if from training
        res_dict['timestamp'] = datetime.datetime.utcnow()
        res_dict['cmd_call'] = cmd_call  # mark if this evaluation is from cmd and loading model

        # > TODO Add evaluation on training configuration
        res_dict.update(res_train_dict)
        res_dict.update(res_seg_train_dict)
        res_dict.update(res_gap_dict)
        res_dict.update(res_seg_gap_dict)

        result = collection.insert_one(res_dict)
        _run.info['result_id'] = result
        _log.warning(f'[evaluation results saved to mongo: {result.inserted_id}]')

    # >>> Sacred
    _run.info['model_save_folder'] = model_folder if model_folder is not None else model_train['save_folder']


@ex.capture
def eval_loop(_log, model_eval, num_steps, cuda, model, loader, prefix=None):
    """
    Args:
        num_steps: using input num_steps (not from model_eval)
    """
    # [load]
    device = torch.device('cuda' if cuda else 'cpu')

    # [evaluation]
    topk = [1, 5, 10]  # [1]
    # topk = [1]
    hits_at = defaultdict(int)
    num_samples = 0
    rr_sum = 0

    pred_states = []
    next_states = []

    result_dict = {}

    with torch.no_grad():

        for batch_idx, data_batch in enumerate(loader):
            data_batch = [[t.to(
                device) for t in tensor] for tensor in data_batch]
            observations, actions = data_batch

            if observations[0].size(0) != model_eval['eval_batch_size']:
                continue

            # [predict in the ground or latent space]
            pred_state, next_state = model.predict(
                observations=observations, actions=actions,
                num_steps=num_steps, space=model_eval['eval_space'],
                hard_bind=model_eval['hard_bind'], pseudo_inverse=model_eval['pseudo_inverse']
            )

            pred_states.append(pred_state.cpu())
            next_states.append(next_state.cpu())

        pred_state_cat = torch.cat(pred_states, dim=0)
        next_state_cat = torch.cat(next_states, dim=0)

        full_size = pred_state_cat.size(0)

        # Flatten object/feature dimensions
        next_state_flat = next_state_cat.view(full_size, -1)
        pred_state_flat = pred_state_cat.view(full_size, -1)

        _log.info(f'state sizes: {next_state_cat.size()}, {pred_state_flat.size()}, {pred_state_flat.dtype}')

        dist_matrix = utils.utils_func.pairwise_distance_matrix(
            next_state_flat, pred_state_flat)
        dist_matrix_diag = torch.diag(dist_matrix).unsqueeze(-1)
        dist_matrix_augmented = torch.cat(
            [dist_matrix_diag, dist_matrix], dim=1)

        # Workaround to get a stable sort in numpy.
        dist_np = dist_matrix_augmented.numpy()
        indices = []
        for row in dist_np:
            keys = (np.arange(len(row)), row)
            indices.append(np.lexsort(keys))
        indices = np.stack(indices, axis=0)
        indices = torch.from_numpy(indices).long()

        _log.info('Processed {} batches of size {}'.format(batch_idx + 1, model_eval['eval_batch_size']))

        labels = torch.zeros(
            indices.size(0), device=indices.device,
            dtype=torch.int64).unsqueeze(-1)

        num_samples += full_size
        _log.info('Size of current top-k evaluation batch: {}'.format(full_size))

        for k in topk:
            match = indices[:, :k] == labels
            num_matches = match.sum()
            hits_at[k] += num_matches.item()

        match = indices == labels
        _, ranks = match.max(1)

        reciprocal_ranks = torch.reciprocal(ranks.double() + 1)
        rr_sum += reciprocal_ranks.sum()

        pred_states = []
        next_states = []

    for k in topk:
        _log.warning('Hits @ {}: {}'.format(k, hits_at[k] / float(num_samples)))
        _key = 'step{}-hits-{}'.format(num_steps, k)
        if prefix is not None:
            _key = '-'.join([prefix, _key])
        result_dict[_key] = hits_at[k] / float(num_samples)

    _log.warning('MRR: {}'.format(rr_sum.item() / float(num_samples)))
    _key = 'step{}-mmr'.format(num_steps)
    if prefix is not None:
        _key = '-'.join([prefix, _key])
    result_dict[_key] = rr_sum.item() / float(num_samples)

    return result_dict
