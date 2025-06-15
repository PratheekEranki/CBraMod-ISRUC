import argparse
import random
import numpy as np
import torch

# --- Import only ISRUC dataset and model ---
from dataset import dataLoader as isruc_dataset
from finetune_trainer import Trainer
from models import model_for_isruc

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser(description='CBraMod fine-tuning on ISRUC')

    # Training params
    parser.add_argument('--seed', type=int, default=3407, help='Random seed')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device number')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer (AdamW or SGD)')
    parser.add_argument('--clip_value', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--classifier', type=str, default='avgpooling_patch_reps',
                        help='[all_patch_reps, all_patch_reps_twolayer, all_patch_reps_onelayer, avgpooling_patch_reps]')

    # Dataset settings
    parser.add_argument('--downstream_dataset', type=str, default='ISRUC', help='Dataset name')
    parser.add_argument('--datasets_dir', type=str, default='ISRUC/precessed_filter_35', help='Path to dataset directory')
    parser.add_argument('--num_of_classes', type=int, default=5, help='Number of output classes')
    parser.add_argument('--model_dir', type=str, default='checkpoints', help='Directory to save models')

    # Pretraining / Fine-tune settings
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing')
    parser.add_argument('--multi_lr', type=bool, default=True, help='Use different learning rates for model parts')
    parser.add_argument('--frozen', type=bool, default=False, help='Freeze pretrained backbone')
    parser.add_argument('--use_pretrained_weights', type=bool, default=True, help='Use pretrained CBraMod weights')
    parser.add_argument('--foundation_dir', type=str, default='pretrained_weights.pth',
                        help='Path to pretrained model weights')

    # Loader
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')

    params = parser.parse_args()
    print(params)

    # Setup
    setup_seed(params.seed)
    device = torch.device(f'cuda:{params.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f'🧠 Fine-tuning CBraMod on {params.downstream_dataset}...')

    # Dataset
    load_dataset = isruc_dataset.LoadDataset(params)
    data_loader = load_dataset.get_data_loader()

    # Model
    model = model_for_isruc.Model(params)
    trainer = Trainer(params, data_loader, model)
    trainer.train_for_multiclass()

    print("✅ Training complete!")

if __name__ == '__main__':
    main()
