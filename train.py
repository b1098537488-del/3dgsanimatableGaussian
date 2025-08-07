from animatableGaussian.model.nerf_model import NeRFModel
import hydra
import pytorch_lightning as pl


import importlib


@hydra.main(config_path="./confs", config_name="gala", version_base="1.1")
def main(opt):
    pl.seed_everything(0)

    # 支持动态加载模型类
    if hasattr(opt, 'model_class') and opt.model_class:
        # 解析模块和类名
        module_path, class_name = opt.model_class.rsplit('.', 1)
        module = importlib.import_module(module_path)
        ModelClass = getattr(module, class_name)
        model = ModelClass(opt)
    else:
        # 默认使用NeRFModel
        model = NeRFModel(opt)
    
    datamodule = hydra.utils.instantiate(opt.dataset)
    trainer = pl.Trainer(accelerator='gpu',
                         **opt.trainer_args)

    trainer.fit(model, datamodule=datamodule)
    trainer.save_checkpoint('model.ckpt')


if __name__ == "__main__":
    main()
