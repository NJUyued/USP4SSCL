# PorSche
## 执行命令
参见 `run_cifar.sh`。

命令样例
```
# CIFAR10
nohup python train_semi_der.py --flip_on_means --ckp_prefix R1_our_cifar10_0.6%_resnet32_etf512_semi_consistancy_bs64_buffer_size5120_ulb_aug_v2_der --gpu 0 --epochs 200 --epochs_new 200 --include_unlabel --update_proto --u_ratio 7 --use_conloss --kd_only_old --dim 512 --dataset cifar10 --num_classes 10 --nb_cl_fg 2 --nb_cl 2 --image_size 32 --unlabeled_num -1 --k_shot 30 --nb_protos 5 --base_lr 0.03 --new_lr 0.03 --train_batch_size 64 --test_batch_size 256 --model resnet32 --proto_dim 64 --warmup_epochs 10 --schedule cosine --unlabels_predict_mode cosine --p_cutoff 0.95 --u_iter 100 --buffer_size 5120 --use_ulb_aug --kd_mode logits --use_hard_labels --use_ulb_kd --ulb_kd_mode similarity --random_seed 0 2>&1 > R1_our_cifar10_0.6%_resnet32_etf512_semi_consistancy_bs64_buffer_size5120_ulb_aug_v2_der.log & echo $! >> der_v2_pids.txt

```

参数解释:

- `--epochs 100` 和 `--epochs_new 25`: 第一个任务和后续任务执行的epoch数


- `--base_lr 0.1` 和 `--new_lr 0.01`: 第一个任务和后续任务执行的学习率

- `--use_ulb_aug`: 使用所有类原型给没达到阈值的无标记数据打伪标签

- `--kd_mode logits`: 标记数据的蒸馏方式，可选 `['logits', 'feats', 'attention', 'logits_at']`

- `--unlabeled_num -1`: 无标记数据的数量，-1表示除了标记之外的其它全部数据作为无标记数据

- `--k_shot 25`: 标记数据的数量，每个类的

- `--percentage 0.05`: 标记数据的比例，目前只用于ImageNet100

- `--use_conloss`: 使用etf向量对齐损失

- `--dim`: 加入的投影头的输出维度

- `--model resnet18`: 训练使用的模型，可选 `['resnet32', 'resnet20', 'resnet18']`

- `--schedule cosine`: 训练时学习率变化策略，可选 `['step', 'Milestone', 'cosine']`

- `--unlabels_predict_mode cosine`: 类原型计算伪标签的策略，可选 `['sqeuclidean', 'cosine']`

其它可选参数:

- `--use_proto`: 使用类原型给无标记数据打伪标签

- `--update_proto`: 每个epoch使用标记数据更新类原型

- `--no_use_conloss_on_ulb`: 不对（达到p_cutoff阈值标准的）无标记数据计算etf的对齐损失

- `--proto_dim`: 模型本身的输出特征维度

- `--no_linear`: 使用非线性的投影头，带ReLu的两层MLP

- `--use_ulb_kd`: 使用无标记数据进行蒸馏

- `--ulb_kd_mode`: 无标记数据的蒸馏方式，可选`['logits', 'feats', 'attention', 'cosine', 'similarity']`

- `--use_hard_labels`: 无标记数据是否使用硬标签，目前只结合 `--use_proto` 和 `--proto_clissifier` 参数使用

- `--adapt_filled`: 是否在buffer未填满时对标记数据进行重复采样（待调试）

- `--use_metric_loss`: 是否使用对比学习损失

- `--use_mix_up`: 是否使用数据MixUp

## 主要修改文件
所有 *_der.py

## 实验结果
![实验结果图](./result/der1.png)

![实验结果图](./result/der2.png)

