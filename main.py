from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from core import LLM
from datacore.Foo.FooDatModule import FooDataModule as DataModule
import config

if __name__ == "__main__":
    args = config.get_args()
    print(args)

    model = LLM()
    
    training_checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='best',
        save_on_train_epoch_end=False,
        save_top_k=3,
    )
    
    
    wandb_logger = WandbLogger(project=f"LLM-Training-{args.strategy}",log_model="all")

    trainer = None
    
    if args.strategy == 'deepspeed':
        # deepspeed_stage_2_offload
        trainer = Trainer(accelerator="gpu",precision=16,callbacks=[training_checkpoint_callback],strategy="deepspeed_stage_2_offload",logger=wandb_logger)
    else:
        from lightning_colossalai import ColossalAIStrategy
        strategy = ColossalAIStrategy(placement_policy="auto") # cpu|cuda|auto
        trainer = Trainer(accelerator="gpu",precision=16,callbacks=[training_checkpoint_callback],strategy=strategy,logger=wandb_logger)
    
    datamodule = DataModule(batch_size=args.batch_size)
    trainer.fit(model, datamodule=datamodule)
    