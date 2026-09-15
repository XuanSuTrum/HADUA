import numpy as np
import torch
from torch.utils import data as Data

def create_domain_loaders(test_id, BATCH_SIZE):
    EEG_FOLDER = "F:\\Emotion_datasets\\SEED\\multi-mode\\contact\\EEG"
    LABEL_FOLDER = "F:\\Emotion_datasets\\SEED\\multi-mode\\contact\\Label"
    EYE_FOLDER = "F:\\Emotion_datasets\\SEED\\multi-mode\\contact\\EYE"  # 眼动数据路径

    feature_list = []
    label_list = []
    eye_feature_list = []  # 存储眼动数据
    subject_ids = [i for i in range(1, 15) if i not in [6, 7]]  # 跳过 6 和 7，得到 [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]

    for subject_id in subject_ids:
        # 加载脑电特征（EEG）
        eeg_path = f"{EEG_FOLDER}\\{subject_id}.npy"
        features = np.load(eeg_path)

        # 加载标签
        label_path = f"{LABEL_FOLDER}\\{subject_id}.npy"  # 假设标签文件也以 .npy 格式存储
        labels = np.load(label_path)

        # 加载眼动数据
        eye_path = f"{EYE_FOLDER}\\{subject_id}.npy"
        eye_features = np.load(eye_path)  # 读取眼动数据

        feature_list.append(features)
        label_list.append(labels)
        eye_feature_list.append(eye_features)  # 存储眼动数据

    """
    为给定的 test_id 创建源域和目标域的 DataLoader
    test_id: 目标域被试的 ID（1到12，映射到原始的1,2,3,4,5,8,9,10,11,12,13,14）
    """
    # 调整 test_id 到实际的被试索引
    subject_ids = [i for i in range(1, 15) if i not in [6, 7]]  # [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]
    actual_subject_id = subject_ids[test_id - 1]  # 将 1-12 映射到实际的被试 ID

    # 目标域数据
    target_feature, target_label, target_eye_feature = feature_list[test_id - 1], label_list[test_id - 1], eye_feature_list[test_id - 1]

    # 移除目标域数据，剩余为源域数据
    feature_list_copy = feature_list.copy()
    label_list_copy = label_list.copy()
    eye_feature_list_copy = eye_feature_list.copy()
    del feature_list_copy[test_id - 1]
    del label_list_copy[test_id - 1]
    del eye_feature_list_copy[test_id - 1]

    # 合并源域数据
    source_data = np.vstack(feature_list_copy)
    source_label = np.vstack(label_list_copy)
    source_eye_data = np.vstack(eye_feature_list_copy)  # 合并源域的眼动数据

    # 打印目标域特征形状（用于调试）
    print('target_feature.shape', target_feature.shape)
    print('target_eye_feature.shape', target_eye_feature.shape)

    # 目标域的训练数据和测试数据（这里假设全部用于训练和测试）
    target_train_data, target_train_label, target_train_eye_data = target_feature, target_label, target_eye_feature
    target_test_data, target_test_label, target_test_eye_data = target_feature, target_label, target_eye_feature

    # 合并EEG数据和眼动数据（按axis=1拼接）
    source_data_combined = np.concatenate([source_data, source_eye_data], axis=1)
    target_train_data_combined = np.concatenate([target_train_data, target_train_eye_data], axis=1)
    target_test_data_combined = np.concatenate([target_test_data, target_test_eye_data], axis=1)

    # 转换为 PyTorch Tensor 并创建 DataLoader
    torch_dataset_source = Data.TensorDataset(
        torch.from_numpy(source_data_combined).float(),  # 确保数据类型为 float
        torch.from_numpy(source_label).long()  # 确保标签类型为 long
    )

    torch_dataset_target_train = Data.TensorDataset(
        torch.from_numpy(target_train_data_combined).float(),
        torch.from_numpy(target_train_label).long()
    )

    torch_dataset_target_test = Data.TensorDataset(
        torch.from_numpy(target_test_data_combined).float(),
        torch.from_numpy(target_test_label).long()
    )

    # 创建 DataLoader
    source_loader = Data.DataLoader(
        dataset=torch_dataset_source,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    target_train_loader = Data.DataLoader(
        dataset=torch_dataset_target_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    target_test_loader = Data.DataLoader(
        dataset=torch_dataset_target_test,
        batch_size=target_test_data_combined.shape[0],  # 目标测试集使用完整数据
        shuffle=False,  # 测试集通常不打乱
        num_workers=0,
        drop_last=False  # 测试集不丢弃最后一批
    )

    return source_loader, target_train_loader, target_test_loader

# if __name__ == "__main__":
#     # 依次让每个被试作为目标域（现在是12个被试）
#     for test_id in range(1, 13):  # 调整为12个被试
#         print(f"\nProcessing test_id: {test_id}")
#
#         # 获取数据加载器
#         source_loader, target_train_loader, target_test_loader = create_domain_loaders(test_id, 32)
#
#         # 打印源域训练数据的批次大小和形状
#         print("Source Loader - First Batch:")
#         for data, labels in source_loader:
#             print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
#             break  # 只查看第一个批次
#
#         # 打印目标域训练数据的批次大小和形状
#         print("Target Train Loader - First Batch:")
#         for data, labels in target_train_loader:
#             print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
#             break  # 只查看第一个批次
#
#         # 打印目标域测试数据的批次大小和形状
#         print("Target Test Loader - First Batch:")
#         for data, labels in target_test_loader:
#             print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
#             break  # 只查看第一个批次
#
#         print(f"Finished processing test_id: {test_id}")
#
#     print("\nAll test_id processing completed.")
