import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*deterministic.*")
import torch  # type: ignore
from ultralytics import YOLO


def main():
    # =========================================================
    # GL-SIM + RAC-Net + DS-NIM 完整消融实验任务列表
    # ---------------------------------------------------------
    # Table 1: 核心对比 (Exp01-05) - GL-SIM 主创新消融
    # Table 2: 副创新 RAC-Net 消融 (Exp06-07)
    # Table 3: 损失函数 DS-NIM 消融 (Exp08-09)
    # Table 4: 全部启用 (Exp10)
    # Table 5: 强基线对比 (Exp11-12)
    # =========================================================
    experiments = [
        # =====================================================
        # Table 1: GL-SIM 主创新消融实验
        # =====================================================
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
        {
            "yaml": "ultralytics/cfg/models/11/yolo11-glam.yaml",
            "name": "Exp05_GL_SIM_Full",
            "use_dsnim": False,
        },

        # =====================================================
        # Table 2: RAC-Net 副创新消融
        # =====================================================
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11-racnet.yaml",
        #     "name": "Exp06_RACNet_Only",
        #     "use_dsnim": False,
        # },

        # # =====================================================
        # # Table 3: DS-NIM 损失函数消融
        # # =====================================================
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11-glam.yaml",
        #     "name": "Exp07_GL_SIM_DSNIM",
        #     "use_dsnim": True,
        # },

        # # =====================================================
        # # Table 4: 全部创新启用
        # # =====================================================
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11-full.yaml",
        #     "name": "Exp08_Full_DSNIM",
        #     "use_dsnim": True,
        # },

        # =====================================================
        # Table 5: 强基线对比
        # =====================================================
        # {
        #     "yaml": "ultralytics/cfg/models/26/yolo26s.yaml",
        #     "name": "Exp09_YOLO26s",
        #     "use_dsnim": False,
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/rt-detr/rtdetr-l.yaml",
        #     "name": "Exp10_RTDETR_l",
        #     "use_dsnim": False,
        # },
    ]

    # =========================================================
    # 循环执行实验
    # =========================================================
    for i, exp in enumerate(experiments):
        print(f"\n{'=' * 60}")
        print(f"  实验 {i + 1}/{len(experiments)}: {exp['name']}")
        print(f"  配置文件: {exp['yaml']}")
        print(f"  DS-NIM Loss: {exp['use_dsnim']}")
        print(f"{'=' * 60}\n")

        model = YOLO(exp["yaml"])

        common_kwargs = dict(
            # --- 数据集 ---
            data="EVD4UAV.yaml",
            imgsz=640,
            batch=64,
            name=exp["name"],
            project="/home/ssssss/1yolo/Ablation_Results",
            device=0,
            workers=8,
            val=True,
            plots=True,
            save=True,
            amp=True,
            cache=False,

            # --- 优化器 ---
            optimizer="SGD",
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            cos_lr=True,
            warmup_epochs=5,

            # --- 损失权重 ---
            box=7.5,
            cls=0.5,
            dfl=1.5,
            cls_pw=0.3,  # 类别平衡: 温和的逆频率加权 (bus ≈ 3.5x car)

            # --- DS-NIM ---
            use_dsnim=exp["use_dsnim"],

            # --- 训练策略 ---
            epochs=300,
            patience=70,

            # --- 数据增强 (航拍适配) ---
            mosaic=1.0,
            close_mosaic=40,
            mixup=0.0,
            copy_paste=0.0,
            degrees=25.0,
            scale=0.3,
            translate=0.1,
            fliplr=0.5,
            erasing=0.1,
            hsv_h=0.015,
            hsv_s=0.5,
            hsv_v=0.4,

            # --- 正则化 ---
            dropout=0.0,
        )

        # 透传 DS-NIM 内部参数
        for k in ("nwd_weight", "inner_mpdiou_weight", "inner_ratio"):
            if k in exp:
                common_kwargs[k] = exp[k]

        model.train(**common_kwargs)
        torch.cuda.empty_cache()

    print("\n  所有消融实验已全部执行完毕！")


if __name__ == "__main__":
    main()
