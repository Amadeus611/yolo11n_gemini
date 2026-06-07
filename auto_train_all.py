import warnings

import torch  # type: ignore
from ultralytics import YOLO


warnings.filterwarnings("ignore", category=UserWarning, message=".*deterministic.*")


def main():
    """按 ge5 可信超参执行 YOLO11n 消融实验。"""
    experiments = [
        {
            "yaml": "ultralytics/cfg/models/11/yolo11.yaml",
            "name": "Exp01_Baseline",
            "use_dsnim": False,
        },
        {
            "yaml": "ultralytics/cfg/models/11/yolo11-gcam.yaml",
            "name": "Exp02_GCAM_Only",
            "use_dsnim": False,
        },
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11-lhde.yaml",
        #     "name": "Exp03_LHDE_Only",
        #     "use_dsnim": False,
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11-gcam-lhde.yaml",
        #     "name": "Exp04_GCAM_LHDE_NoDCRN",
        #     "use_dsnim": False,
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11-glam.yaml",
        #     "name": "Exp05_GL_SIM_Full",
        #     "use_dsnim": False,
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11-racnet.yaml",
        #     "name": "Exp06_RACNet_Only",
        #     "use_dsnim": False,
        # },
        {
            "yaml": "ultralytics/cfg/models/11/yolo11-glam.yaml",
            "name": "Exp07_GL_SIM_DSNIM",
            "use_dsnim": True,
        },
        {
            "yaml": "ultralytics/cfg/models/11/yolo11-full.yaml",
            "name": "Exp08_Full_DSNIM",
            "use_dsnim": True,
        },
        # {
        #     "yaml": "ultralytics/cfg/models/26/yolo26.yaml",
        #     "name": "Exp09_YOLO26n",
        #     "use_dsnim": False,
        # },
    ]

    for i, exp in enumerate(experiments):
        print(f"\n{'=' * 60}")
        print(f"  实验 {i + 1}/{len(experiments)}: {exp['name']}")
        print(f"  配置文件: {exp['yaml']}")
        print(f"  DS-NIM Loss: {exp['use_dsnim']}")
        print(f"{'=' * 60}\n")

        model = YOLO(exp["yaml"])

        common_kwargs = dict(
            # 数据集
            data="EVD4UAV.yaml",
            imgsz=640,
            batch=64,
            name=exp["name"],
            project="/home/ssssss/1yolo/Ablation_Results_ge6_fix",
            device=0,
            workers=8,
            val=True,
            plots=True,
            save=True,
            amp=True,
            cache=False,
            # 优化器
            optimizer="SGD",
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            cos_lr=True,
            warmup_epochs=5,
            # 损失权重
            box=7.5,
            cls=0.5,
            dfl=1.5,
            cls_pw=0.3,
            # DS-NIM
            use_dsnim=exp["use_dsnim"],
            # 训练策略
            epochs=300,
            patience=100,
            # 数据增强。检测框数据通常没有 segments，copy_paste 在当前实现下大概率不会生效。
            mosaic=1.0,
            close_mosaic=40,
            mixup=0.0,
            copy_paste=0.3,
            degrees=25.0,
            scale=0.3,
            translate=0.1,
            fliplr=0.5,
            erasing=0.1,
            hsv_h=0.015,
            hsv_s=0.5,
            hsv_v=0.4,
            # 正则化
            dropout=0.0,
        )

        for k in ("nwd_weight", "inner_mpdiou_weight", "inner_ratio"):
            if k in exp:
                common_kwargs[k] = exp[k]

        model.train(**common_kwargs)
        torch.cuda.empty_cache()

    print("\n所有消融实验已执行完毕。")


if __name__ == "__main__":
    main()
