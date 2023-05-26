from lightning.pytorch.callbacks import ModelCheckpoint
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
    
    trainer = None
    
    # deepspeed_stage_2_offload
    trainer = Trainer(
        accelerator="gpu",
        precision=16,
        max_steps=10,
        strategy=args.strategy,
        enable_checkpointing=False
    )
    
    datamodule = DataModule()
    trainer.fit(model, datamodule=datamodule)
    