"""
Code for the training loop including an epoch cycle function.
"""


import os
import time
from copy import deepcopy
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Literal

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from .losses import LossCalculator
from .optim import OptimScheduler
from .metrics import MetricsEvaluator
from .model import Transformer

@dataclass
class TrainConfig:
    epochs: int = 20
    device: str = None
    batch_size: int = 80
    grad_clip_max_norm: float = 1.0
    num_workers: int = 0
    # Checkpoints
    checkpoint_save_freq: int = 0
    model_save_path: str = None
    model_save_prefix: str = 'best_model'


def run_epoch_cycle(
        model: Transformer,
        data_loader: DataLoader,
        loss_calculator: LossCalculator,
        device: str,
        optim_scheduler: OptimScheduler = None,
        mode: Literal['train', 'test', 'val'] = 'train',
        verbose_freq: int = 20,
        grad_clip_max_norm: float = 1.0,
        max_tgt_seq_len: int = None,
        metrics_evaluator: MetricsEvaluator = None
    ) -> Dict[str, int|float]:
    """
    Run one epoch of training, validation or testing.

    Args for training (`mode` is `'train'`):
    - `optim_scheduler`
    - `grad_clip_max_norm (optional)`: Defaults to 1.0

    Args for validation and testing (`mode` is `'val'` or `'test'`):
    - `max_tgt_seq_len (optional)`
    - `metrics_evaluator (optional)`

    Returns:
        A dict that contains the final metrics for the epoch. The keys are the metric name and the values the metric values.
        - When `mode` is `'train'`, the dict only contains the loss.
        - When `mode` is `'val'` or `'test'`, the dict could contain additional metrics depending on `metrics_evaluator`.

        Example:
        {
            'loss': 0.1,
            'bleu': 0.3,
            ...
        }

    Regarding the batch dict, it takes these tensors from batch dict:
    - `src`: Tokenized input sequences for the encoder (source).
    - `tgt`: Tokenized input sequences fot the decoder (target).
    - `tgt_labels`: Target labels to compute the loss.
    - `src_mask`: Masks for src sequences.
    - `tgt_mask`: Masks for the tgt sequences.
    - `tgt_token_num`: Quantity of tokens in the tgt_labels sequences.
    """
    total_loss = 0.
    curr_loss = 0.
    last_loss = 0.
    total_tokens = 0
    batches_len = len(data_loader)

    result_metrics = {}

    if mode in ['val', 'test']:
        all_preds = []
        all_references = []

    if verbose_freq > 0:
        print(f'Mode: {mode}')

    for it, batch in enumerate(data_loader):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        tgt_labels = batch['tgt_labels'].to(device)
        src_mask = batch['src_mask'].to(device)
        tgt_mask = batch['tgt_mask'].to(device)
        tgt_token_num = batch['tgt_token_num'].to(device)
        batch_tgt_token_num = tgt_token_num.sum()

        if mode == 'train':
            model_out = model(src, tgt, src_mask, tgt_mask)
            loss, loss_node = loss_calculator(model_out=model_out, tgt=tgt_labels, norm=batch_tgt_token_num)
            
            curr_loss = loss.item()
            total_loss += curr_loss

            optim_scheduler.zero_grad()
            loss_node.backward()
            clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            optim_scheduler.step()

            del model_out
            del loss
            del loss_node

            if device == 'cuda':
                torch.cuda.empty_cache()
        else:
            with torch.no_grad():
                model_out = model(src, tgt, src_mask, tgt_mask)
                loss, _ = loss_calculator(model_out=model_out, tgt=tgt_labels, norm=batch_tgt_token_num)

                curr_loss = loss.item()
                total_loss += curr_loss
                
                preds = model.generate_sequences(src, src_mask, max_tgt_seq_len)
                
                all_preds.extend(preds.tolist())
                all_references.extend(tgt_labels.tolist())

                del model_out
                del loss
                del preds

        total_tokens += batch_tgt_token_num

        if verbose_freq > 0 and it % verbose_freq == 0:
            last_loss = curr_loss / batch_tgt_token_num
            batch_percentage = round(((it+1) / batches_len) * 100, 2)
            str_to_log = f'\tBatch {it+1} - {batch_percentage}% | Loss: {last_loss:.6f} | Accum. Tokens: {total_tokens}'

            if mode == 'train':
                curr_lr = optim_scheduler.get_learning_rate()
                str_to_log += f' | LR: {curr_lr:6.1e}'

            print(str_to_log)
            curr_loss = 0.
    
    if verbose_freq > 0:
        print('')
    
    result_metrics['loss'] = (total_loss / total_tokens).item()
    if metrics_evaluator is not None and mode in ['val', 'test']:
        result_metrics.update(metrics_evaluator(predictions=all_preds, references=all_references))

    return result_metrics


class Trainer:
    def __init__(self, config: TrainConfig, model: Transformer, train_dataset: Dataset, val_dataset: Dataset, loss_calculator: LossCalculator, optim_scheduler: OptimScheduler, metrics_evaluator: MetricsEvaluator) -> None:
        self.config = config
        self.device = config.device
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.loss_calculator = loss_calculator
        self.optim_scheduler = optim_scheduler
        self.metrics_evaluator = metrics_evaluator

        if config.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model.to(self.device)
        print(f'Trainer running on device: {self.device}')

        # Run variables
        self.total_time = 0.
        self.best_val_metrics = { 'loss': float('inf') }
        self.loss_history = { 'train': [], 'val' : [], 'best': [] }
        self.last_saved_model_filename = None
    
    def save_model(self, model_filename_prefix: str, model_state_dict) -> None:
        curr_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        model_filename = f'{model_filename_prefix}_{curr_timestamp}.pt'
        model_path = os.path.join(self.config.model_save_path, model_filename)
        torch.save(model_state_dict, model_path)
        self.last_saved_model_filename= model_filename
    
    def train(self, verbose: bool = True, verbose_batch_freq: int = 20) -> None:
        if not verbose:
            verbose_batch_freq = 0

        model, config = self.model, self.config

        train_loader = DataLoader(
            self.train_dataset,
            shuffle=False,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True
        )

        save_model = config.model_save_path is not None

        self.total_time = 0.
        self.best_val_metrics = { 'loss': float('inf') }
        best_model_state_dict = None
        self.loss_history = { 'train': [], 'val' : [], 'best': [] }
        self.val_metrics_history = {}

        for epoch in range(config.epochs):
            start_time = time.time()
            print(f'** Epoch {epoch+1}:')

            model.train()
            train_metrics = run_epoch_cycle(model, train_loader, self.loss_calculator, self.device, self.optim_scheduler, mode='train', verbose_freq=verbose_batch_freq, grad_clip_max_norm=config.grad_clip_max_norm)

            model.eval()
            val_metrics = run_epoch_cycle(model, val_loader, self.loss_calculator, self.device, mode='val', metrics_evaluator=self.metrics_evaluator, verbose_freq=verbose_batch_freq)

            elapsed_time = time.time() - start_time
            self.total_time += elapsed_time

            train_loss, val_loss = train_metrics['loss'], val_metrics['loss']

            if val_loss < self.best_val_metrics['loss']:
                self.best_val_metrics['loss'] = val_loss
                if save_model:
                    best_model_state_dict = deepcopy(model.state_dict())

            if verbose:
                print(f'-> Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Best Val Loss: {self.best_val_metrics['loss']:.6f} | Elapsed: {elapsed_time:.2f} sec.')
                
                if len(val_metrics) > 1:
                    print('Validation Metrics:')

                for metric, metric_value in val_metrics.items():
                    if metric == 'loss':
                        continue

                    # Update best val metrics
                    best_value_criteria = self.metrics_evaluator.get_metric(metric).best_value_criteria
                    if (
                        metric not in self.best_val_metrics or
                        MetricsEvaluator.metric_value_is_better(self.best_val_metrics[metric], metric_value, best_value_criteria=best_value_criteria)
                    ):
                        self.best_val_metrics[metric] = metric_value

                    print(f'\t- {metric}: {metric_value:.6f}')
                    # Save in history
                    self.val_metrics_history.setdefault(metric, []).append(metric_value)

                print('')
            
            self.loss_history['train'].append(train_loss)
            self.loss_history['val'].append(val_loss)
            self.loss_history['best'].append(self.best_val_metrics['loss'])

            # Save checkpoints
            if (
                save_model and
                config.checkpoint_save_freq > 0 and
                (epoch+1) % config.checkpoint_save_freq == 0 and
                best_model_state_dict is not None
            ):
                self.save_model(f'{config.model_save_prefix}_checkpoint', best_model_state_dict)

        # Save final best model
        if save_model and best_model_state_dict is not None:
            self.save_model(config.model_save_prefix, best_model_state_dict)
        
        if verbose:
            print(f'-> Training Completed - Total Time: {self.total_time:.2f} sec.')

            if len(self.best_val_metrics) > 1:
                print('Best Validation Metrics:')

            for metric, metric_value in self.best_val_metrics.items():
                print(f'\t- {metric}: {metric_value:.6f}')
