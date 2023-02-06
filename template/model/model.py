import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import functional as FM
import pytorch_lightning as pl 
from modules.utils import load_config

class Model(pl.LightningModule):
    def __init__(self, model_args):
        super().__init__()
        
        self.input_feature = model_args['data']['dimension_reduction']['input_n_components']
        self.output_feature = model_args['data']['dimension_reduction']['output_n_components']
        
        self.criteria = model_args['train']['criteria']
        self.lr = model_args['train']['lr']
        self.encoder = nn.Sequential( 
            nn.Linear(self.input_feature, 128), 
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3),   
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12), 
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_feature),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y, barcode = batch
        y_hat = self(x)
        l1 = F.l1_loss(y_hat, y)
        l2 = F.mse_loss(y_hat, y)
        
        loss = l1 if self.criteria == 'l1' else l2 if self.criteria == 'l2' else ValueError
        
        return {'loss':loss, 'train_l1': l1, 'train_l2': l2} 

    def training_epoch_end(self, training_step_outputs):
        train_l1 = torch.hstack([output['train_l1'] for output in training_step_outputs]).mean()
        train_l2 = torch.hstack([output['train_l2'] for output in training_step_outputs]).mean()
        self.log('train_l1', train_l1)
        self.log('train_l2', train_l2)

    def validation_step(self, batch, batch_idx):
        x, y, barcode = batch
        y_hat = self(x)
        l1 = F.l1_loss(y_hat, y)
        l2 = F.mse_loss(y_hat, y)

        # mse = FM.mean_squared_error(logits, x)
        # mae = FM.mean_absolute_error(logits, x)
        # mape = FM.mean_absolute_percentage_error(logits, x)
        # metrics = {'val_loss': loss, 'val_mse': mse, 'val_mae': mae, 'val_mape': mape}
        metrics = {'val_l1': l1, 'val_l2': l2}
        
        self.log_dict(metrics)
        return {'pred':y_hat, 'label': x, 'val_l1': l1, 'val_l2': l2}

    def validation_epoch_end(self, valid_step_outputs):
        val_l1 = torch.hstack([output['val_l1'] for output in valid_step_outputs])
        val_l2 = torch.hstack([output['val_l2'] for output in valid_step_outputs])

    def test_step(self, batch, batch_idx):
        x, y, barcode = batch
        y_hat = self(x)
        
        l1 = F.l1_loss(y_hat, y)
        l2 = F.mse_loss(y_hat, y)
        # pred = torch.argmax(logits, dim = 1)
        # mse = FM.mean_squared_error(logits, x)
        # mae = FM.mean_absolute_error(logits, x)
        # mape = FM.mean_absolute_percentage_error(logits, x)
        # metrics = {'test_loss': loss, 'test_mse': mse, 'test_mae': mae, 'test_mape': mape}
        metrics = {'test_l1': l1, 'test_l2': l2}
        self.log_dict(metrics)
        return {'barcode':barcode, 'pred': y_hat, 'label': x, 'test_l1': l1, 'test_l2': l2}
        
    def test_epoch_end(self, test_step_outputs):
        test_l1 = torch.hstack([output['test_l1'] for output in test_step_outputs])
        test_l2 = torch.hstack([output['test_l2'] for output in test_step_outputs])
        print("test_l1: ", test_l1, "test_l2: ", test_l2)

    def predict_step(self, batch, batch_idx):
        x, y, barcode = batch
        y_hat = self(x)
        
        l1 = F.l1_loss(y_hat, y, reduction='none')
        l2 = F.mse_loss(y_hat, y, reduction='none')
        #pred = torch.argmax(logits, dim = 1)
        # mse = FM.mean_squared_error(logits, x)
        # mae = FM.mean_absolute_error(logits, x)
        # mape = FM.mean_absolute_percentage_error(logits, x)
        # metrics = {'test_loss': loss, 'test_mse': mse, 'test_mae': mae, 'test_mape': mape}
        return {'barcode':barcode, 'pred': y_hat, 'label': y, 'pred_l1': l1, 'pred_l2': l2}
        
    def test_epoch_end(self, test_step_outputs):
        #pred = torch.vstack([output['pred'] for output in test_step_outputs])
        #label = torch.hstack([output['label'] for output in test_step_outputs])
        test_l1 = torch.hstack([output['test_l1'] for output in test_step_outputs])
        test_l2 = torch.hstack([output['test_l2'] for output in test_step_outputs])
        print("test_l1: ", test_l1, "test_l2: ", test_l2)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)