"""过采样少数类别（bus 和 truck）的训练图片。

扫描训练集标签，找到包含 bus(1) 或 truck(2) 的图片，
复制图片和标签文件以增加少数类别的实例数。
"""
import shutil
from pathlib import Path

LABEL_DIR = Path("/home/ssssss/1yolo/Dataset/EVD4UAV/labels/train")
IMAGE_DIR = Path("/home/ssssss/1yolo/Dataset/EVD4UAV/images/train")

# 目标: bus 从 725 -> ~2175 (3x), truck 从 1105 -> ~2210 (2x)
# 即 bus 复制 2 份, truck 复制 1 份
TARGET_MULTIPLIER = {1: 2, 2: 1}  # 类别 -> 额外复制份数

def main():
    # 统计包含各类别的图片
    class_images = {1: [], 2: []}

    for label_file in sorted(LABEL_DIR.glob("*.txt")):
        with open(label_file, "r") as f:
            classes_in_file = set()
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes_in_file.add(int(parts[0]))

        for cls_id in [1, 2]:
            if cls_id in classes_in_file:
                class_images[cls_id].append(label_file.stem)

    print(f"包含 bus(1) 的图片数: {len(class_images[1])}")
    print(f"包含 truck(2) 的图片数: {len(class_images[2])}")

    # 过采样
    created = 0
    for cls_id, multiplier in TARGET_MULTIPLIER.items():
        for stem in class_images[cls_id]:
            for copy_idx in range(1, multiplier + 1):
                suffix = f"_dup{copy_idx}"
                new_stem = stem + suffix

                # 复制图片
                for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                    src_img = IMAGE_DIR / f"{stem}{ext}"
                    if src_img.exists():
                        dst_img = IMAGE_DIR / f"{new_stem}{ext}"
                        if not dst_img.exists():
                            shutil.copy2(src_img, dst_img)
                            created += 1
                        break

                # 复制标签
                src_label = LABEL_DIR / f"{stem}.txt"
                dst_label = LABEL_DIR / f"{new_stem}.txt"
                if src_label.exists() and not dst_label.exists():
                    shutil.copy2(src_label, dst_label)

    print(f"新创建文件数: {created}")

    # 统计过采样后的类别分布
    counts = {0: 0, 1: 0, 2: 0}
    for label_file in LABEL_DIR.glob("*.txt"):
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls = int(parts[0])
                    if cls in counts:
                        counts[cls] += 1

    print(f"\n过采样后类别分布:")
    print(f"  car(0):   {counts[0]}")
    print(f"  bus(1):   {counts[1]} (原 725)")
    print(f"  truck(2): {counts[2]} (原 1105)")


if __name__ == "__main__":
    main()
