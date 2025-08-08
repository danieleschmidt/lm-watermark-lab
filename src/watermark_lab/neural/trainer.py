"""Production-grade neural training pipeline for watermark detection."""

import os
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict

# Optional imports with fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torch.utils.tensorboard import SummaryWriter
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import transformers
    from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..utils.logging import get_logger
from ..utils.exceptions import TrainingError, ModelError, ValidationError
from ..utils.metrics import MetricsCollector

logger = get_logger("neural.trainer")


@dataclass
class TrainingConfig:
    """Configuration for neural training."""
    
    # Model architecture
    model_type: str = "transformer"  # transformer, lstm, cnn
    backbone_model: str = "distilbert-base-uncased"
    hidden_size: int = 768
    num_classes: int = 2  # watermarked vs clean
    dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    
    # Data parameters
    max_sequence_length: int = 512
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # Training strategy
    early_stopping_patience: int = 3
    save_best_model: bool = True
    save_checkpoints: bool = True
    checkpoint_frequency: int = 5
    
    # Evaluation
    evaluation_frequency: int = 1
    compute_metrics_on_train: bool = False
    
    # Output
    output_dir: str = "neural_training"
    experiment_name: str = "watermark_detection"
    log_frequency: int = 100
    
    # Hardware
    device: str = "auto"  # auto, cpu, cuda
    mixed_precision: bool = True
    dataloader_num_workers: int = 4
    
    def __post_init__(self):
        """Validate and initialize configuration."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        if self.device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"


class WatermarkDataset(Dataset if TORCH_AVAILABLE else object):
    """Dataset for watermark detection training."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: Optional[Any] = None,
        max_length: int = 512
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available for dataset creation")
        
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if len(texts) != len(labels):
            raise ValueError("Texts and labels must have same length")
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        if self.tokenizer:
            # Use transformer tokenizer
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "labels": torch.tensor(label, dtype=torch.long)
            }
        else:
            # Simple character-level encoding fallback
            chars = [ord(c) % 256 for c in text[:self.max_length]]
            chars += [0] * (self.max_length - len(chars))  # Pad
            
            return {
                "input_ids": torch.tensor(chars, dtype=torch.long),
                "attention_mask": torch.ones(self.max_length, dtype=torch.long),
                "labels": torch.tensor(label, dtype=torch.long)
            }


class TransformerDetectionModel(nn.Module if TORCH_AVAILABLE else object):
    """Transformer-based watermark detection model."""
    
    def __init__(self, config: TrainingConfig):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available for model creation")
        
        super().__init__()
        self.config = config
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.backbone = AutoModel.from_pretrained(config.backbone_model)
                self.hidden_size = self.backbone.config.hidden_size
            except Exception as e:
                logger.warning(f"Failed to load transformer model: {e}. Using fallback.")
                self._create_fallback_model()
        else:
            self._create_fallback_model()
        
        # Classification head
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(self.hidden_size, config.num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _create_fallback_model(self):
        """Create fallback model without transformers."""
        self.hidden_size = self.config.hidden_size
        self.backbone = nn.Sequential(
            nn.Embedding(256, self.hidden_size),  # Character embeddings
            nn.LSTM(self.hidden_size, self.hidden_size // 2, batch_first=True, bidirectional=True),
        )
        self.use_lstm = True
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        if hasattr(self, 'use_lstm'):
            # LSTM fallback
            embeddings = self.backbone[0](input_ids)
            lstm_output, (hidden, _) = self.backbone[1](embeddings)
            # Use final hidden state
            pooled_output = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            # Transformer backbone
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state.mean(dim=1)
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)


class NeuralTrainer:
    """Advanced neural trainer for watermark detection models."""
    
    def __init__(self, config: TrainingConfig):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available for neural training")
        
        self.config = config
        self.logger = get_logger(f"trainer.{config.experiment_name}")
        
        # Setup device
        self.device = torch.device(config.device)
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = 0.0
        self.metrics_history = defaultdict(list)
        
        # Setup output directories
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.writer = None
        try:
            self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
        except Exception as e:
            self.logger.warning(f"TensorBoard logging not available: {e}")
        
        # Metrics collector
        self.metrics_collector = MetricsCollector()
    
    def initialize_model(self):
        """Initialize the detection model."""
        
        try:
            self.logger.info(f"Initializing {self.config.model_type} model")
            
            # Load tokenizer if using transformers
            if TRANSFORMERS_AVAILABLE and self.config.model_type == "transformer":
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.config.backbone_model)
                    self.logger.info(f"Loaded tokenizer: {self.config.backbone_model}")
                except Exception as e:
                    self.logger.warning(f"Failed to load tokenizer: {e}")
            
            # Create model
            if self.config.model_type == "transformer":
                self.model = TransformerDetectionModel(self.config)
            else:
                raise NotImplementedError(f"Model type not implemented: {self.config.model_type}")
            
            # Move to device
            self.model = self.model.to(self.device)
            
            # Initialize optimizer
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            self.logger.info(f"Model initialized with {self._count_parameters()} parameters")
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            raise ModelError(f"Failed to initialize model: {e}")
    
    def prepare_data(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data loaders."""
        
        try:
            # Create training dataset
            train_dataset = WatermarkDataset(
                texts=train_texts,
                labels=train_labels,
                tokenizer=self.tokenizer,
                max_length=self.config.max_sequence_length
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.dataloader_num_workers,
                pin_memory=self.device.type == "cuda"
            )
            
            # Create validation dataset
            val_loader = None
            if val_texts is not None and val_labels is not None:
                val_dataset = WatermarkDataset(
                    texts=val_texts,
                    labels=val_labels,
                    tokenizer=self.tokenizer,
                    max_length=self.config.max_sequence_length
                )
                
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=self.config.dataloader_num_workers,
                    pin_memory=self.device.type == "cuda"
                )
            
            self.logger.info(f"Prepared data: train={len(train_texts)}, val={len(val_texts) if val_texts else 0}")
            return train_loader, val_loader
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            raise TrainingError(f"Failed to prepare data: {e}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        """Execute the training loop."""
        
        if not self.model or not self.optimizer:
            raise TrainingError("Model not initialized. Call initialize_model() first.")
        
        try:
            self.logger.info("Starting training")
            start_time = time.time()
            
            # Setup learning rate scheduler
            num_training_steps = len(train_loader) * self.config.num_epochs
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=num_training_steps
            ) if TRANSFORMERS_AVAILABLE else None
            
            # Training loop
            for epoch in range(self.config.num_epochs):
                self.epoch = epoch
                self.logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
                
                # Training phase
                train_metrics = self._train_epoch(train_loader)
                
                # Validation phase
                val_metrics = {}
                if val_loader and (epoch + 1) % self.config.evaluation_frequency == 0:
                    val_metrics = self._validate_epoch(val_loader)
                
                # Combine metrics
                epoch_metrics = {**train_metrics, **val_metrics}
                self._log_epoch_metrics(epoch, epoch_metrics)
                
                # Save checkpoint
                if (epoch + 1) % self.config.checkpoint_frequency == 0:
                    self._save_checkpoint(epoch, epoch_metrics)
                
                # Early stopping check
                if val_metrics and self._should_early_stop(val_metrics):
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # Final model save
            final_metrics = self._finalize_training()
            
            training_time = time.time() - start_time
            self.logger.info(f"Training completed in {training_time:.2f}s")
            
            return {
                'final_metrics': final_metrics,
                'training_time': training_time,
                'total_epochs': self.epoch + 1,
                'model_path': str(self.output_dir / "final_model.pt")
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise TrainingError(f"Training failed: {e}")
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Execute one training epoch."""
        
        self.model.train()
        epoch_loss = 0.0
        epoch_metrics = defaultdict(list)
        
        for step, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            
            # Compute loss
            loss = self.criterion(logits, batch["labels"])
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            
            # Optimizer step
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            
            # Update metrics
            epoch_loss += loss.item()
            self.global_step += 1
            
            # Compute batch metrics
            if SKLEARN_AVAILABLE:
                predictions = torch.argmax(logits, dim=1)
                accuracy = accuracy_score(
                    batch["labels"].cpu().numpy(),
                    predictions.cpu().numpy()
                )
                epoch_metrics['accuracy'].append(accuracy)
            
            # Log step metrics
            if step % self.config.log_frequency == 0:
                self.logger.debug(f"Step {step}: loss={loss.item():.4f}")
                if self.writer:
                    self.writer.add_scalar("train/step_loss", loss.item(), self.global_step)
        
        # Aggregate epoch metrics
        avg_loss = epoch_loss / len(train_loader)
        metrics = {'train_loss': avg_loss}
        
        if epoch_metrics['accuracy']:
            metrics['train_accuracy'] = np.mean(epoch_metrics['accuracy'])
        
        return metrics
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Execute validation epoch."""
        
        self.model.eval()
        val_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                logits = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                
                # Compute loss
                loss = self.criterion(logits, batch["labels"])
                val_loss += loss.item()
                
                # Collect predictions
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
        
        # Compute metrics
        avg_loss = val_loss / len(val_loader)
        metrics = {'val_loss': avg_loss}
        
        if SKLEARN_AVAILABLE and all_predictions:
            accuracy = accuracy_score(all_labels, all_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='weighted', zero_division=0
            )
            
            metrics.update({
                'val_accuracy': accuracy,
                'val_precision': precision,
                'val_recall': recall,
                'val_f1': f1
            })
        
        return metrics
    
    def _log_epoch_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch metrics."""
        
        # Store in history
        for metric_name, value in metrics.items():
            self.metrics_history[metric_name].append(value)
        
        # Console logging
        metric_strs = [f"{k}={v:.4f}" for k, v in metrics.items()]
        self.logger.info(f"Epoch {epoch + 1} metrics: {', '.join(metric_strs)}")
        
        # TensorBoard logging
        if self.writer:
            for metric_name, value in metrics.items():
                self.writer.add_scalar(metric_name, value, epoch)
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        
        checkpoint_path = self.checkpoints_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': asdict(self.config)
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _should_early_stop(self, val_metrics: Dict[str, float]) -> bool:
        """Check if early stopping criteria are met."""
        
        # Use validation accuracy as the metric for early stopping
        current_metric = val_metrics.get('val_accuracy', 0.0)
        
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.patience_counter = 0
            return False
        else:
            self.patience_counter = getattr(self, 'patience_counter', 0) + 1
            return self.patience_counter >= self.config.early_stopping_patience
    
    def _finalize_training(self) -> Dict[str, Any]:
        """Finalize training and save final model."""
        
        # Save final model
        final_model_path = self.output_dir / "final_model.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': asdict(self.config),
            'metrics_history': dict(self.metrics_history),
            'tokenizer_name': self.config.backbone_model if self.tokenizer else None
        }, final_model_path)
        
        # Save training configuration
        config_path = self.output_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        # Save metrics history
        metrics_path = self.output_dir / "metrics_history.json"
        with open(metrics_path, 'w') as f:
            json.dump(dict(self.metrics_history), f, indent=2)
        
        # Close tensorboard writer
        if self.writer:
            self.writer.close()
        
        final_metrics = {
            'best_metric': self.best_metric,
            'total_steps': self.global_step,
            'final_epoch': self.epoch + 1
        }
        
        # Add final metrics from last epoch
        for metric_name, history in self.metrics_history.items():
            if history:
                final_metrics[f'final_{metric_name}'] = history[-1]
        
        self.logger.info("Training finalized successfully")
        return final_metrics
    
    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


# Convenience function for training
def train_watermark_detector(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: Optional[List[str]] = None,
    val_labels: Optional[List[int]] = None,
    config: Optional[TrainingConfig] = None,
    **kwargs
) -> Dict[str, Any]:
    """Train a watermark detection model with simple interface."""
    
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available for neural training")
    
    # Create config
    if config is None:
        config = TrainingConfig(**kwargs)
    
    # Initialize trainer
    trainer = NeuralTrainer(config)
    trainer.initialize_model()
    
    # Prepare data
    train_loader, val_loader = trainer.prepare_data(
        train_texts, train_labels, val_texts, val_labels
    )
    
    # Train model
    return trainer.train(train_loader, val_loader)


# Export main classes
__all__ = [
    "NeuralTrainer",
    "TrainingConfig",
    "WatermarkDataset",
    "TransformerDetectionModel",
    "train_watermark_detector"
]