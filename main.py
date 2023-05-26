from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer
from core import LLM
from datacore.Foo.FooDatModule import FooDataModule as DataModule
import config

if __name__ == "__main__":
    args = config.get_args()
    print(args)

    model = LLM()
    
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
    