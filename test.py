from animatableGaussian.model.nerf_model import NeRFModel
import hydra
import pytorch_lightning as pl
import importlib


@hydra.main(config_path="./confs", config_name="gala", version_base="1.1")
def main(opt):
    pl.seed_everything(0)

    # 动态加载模型类
    if hasattr(opt, 'model_class') and opt.model_class:
        try:
            if '.' in opt.model_class:
                # 处理完整模块路径格式
                module_path, class_name = opt.model_class.rsplit('.', 1)
                module = importlib.import_module(module_path)
                model_class = getattr(module, class_name)
            else:
                # 处理简单类名格式（向后兼容）
                if opt.model_class == 'ImprovedNeRFModel':
                    from animatableGaussian.model.improved_nerf_model import ImprovedNeRFModel
                    model_class = ImprovedNeRFModel
                else:
                    print(f"Warning: Unknown model class {opt.model_class}, using default NeRFModel")
                    model_class = NeRFModel
        except (ImportError, AttributeError) as e:
            print(f"Warning: Failed to load model class {opt.model_class}: {e}, using default NeRFModel")
            model_class = NeRFModel
    else:
        model_class = NeRFModel

    # 加载检查点时忽略不匹配的键（如sobel_x, sobel_y等）
    try:
        model = model_class.load_from_checkpoint('model.ckpt')
    except RuntimeError as e:
        if "Unexpected key(s) in state_dict" in str(e):
            print(f"检测到状态字典不匹配，尝试非严格加载: {e}")
            model = model_class.load_from_checkpoint('model.ckpt', strict=False)
        else:
            raise e
    datamodule = hydra.utils.instantiate(opt.dataset, train=False)
    trainer = pl.Trainer(accelerator='gpu',
                         **opt.trainer_args)
    result = trainer.test(model, datamodule=datamodule)[0]


if __name__ == "__main__":
    main()
