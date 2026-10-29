非线性系统切换检测、动力学建模与抗扰预测控制
================================================

一、项目简介

本项目研究非线性系统发生未知切换后的检测、建模与控制流程，包含两个实例：
1. predator_prey：捕食—被捕食系统。
2. waterwheel：水轮系统。

主要方法包括 MG-CRLSTM 残差预测、GLR 切换检测、NGRC / BLI-NGRC
动力学建模，以及 MPC / 带扰动补偿的预测控制。

二、目录与文件用途

common/：两个系统共用的算法与工具。
  detection/residual_glr_detection.py：残差预测与 GLR 切换检测。
  detection/detector_training.py：检测器训练、模型权重加载和检测评估。
  modeling_control/ngrc_modeling.py：NGRC 模型训练及滚动预测。
  modeling_control/latent_integral_modeling.py：BLI-NGRC 建模。
  modeling_control/disturbance_rejection_mpc.py：观测器与抗扰 MPC。
  post_detection_data.py：检测报警后的建模与控制数据衔接。
  prediction_metrics.py：预测评价指标与多种子统计。
  control_experiment_utils.py：控制重复实验的噪声、指标与统计工具。

predator_prey/：捕食—被捕食系统的实验代码。
  pipeline.py：检测、建模和控制的完整流程。
  run_pipeline.py：完整实验入口。
  run_detection_trials.py：多个随机种子的检测实验。
  run_modeling_comparison.py：预测模型对比实验。
  run_control_trials.py：多个随机种子的控制实验。
  run_control_comparison.py：代表性控制器对比实验。

waterwheel/：水轮系统的模型、算法接口与实验代码。
  dynamics.py：水轮物理模型与轨迹仿真。
  detection.py：水轮检测器配置、训练和检测。
  modeling.py：水轮系统的 NGRC 建模接口。
  control.py：水轮系统的控制配置与仿真。
  pipeline.py：检测、建模和控制的完整流程。
  run_pipeline.py：完整实验入口。
  run_detection_trials.py：多个随机种子的检测实验。
  run_modeling_comparison.py：预测模型对比实验。
  run_control_trials.py：多个随机种子的控制实验。

outputs/：随代码保留的模型权重，以及程序运行后生成的实验结果。

__init__.py 是 Python 包初始化文件，请保留。
捕食—被捕食系统的基础动力学仍保留在共用算法实现中。

三、运行环境

整理验证使用 Windows 和 Python 3.12.10。
使用已安装所需依赖的 Python 环境；如需安装，可在该环境中执行：

  python -m pip install numpy scipy pandas matplotlib torch casadi

上述命令未限定版本，不等同于已经验证过的全新环境安装。

四、outputs 中保留的模型权重

目前随项目提供 3 个训练好的模型检查点：

  outputs/predator_prey/pipeline/checkpoints/epoch_0100.pt
    用于捕食系统完整流程和代表性控制器对比。

  outputs/predator_prey/detection_trials/checkpoints/epoch_0100.pt
    用于捕食系统检测实验，显式指定 --checkpoint-epoch 100 时加载。

  outputs/waterwheel/pipeline/detection/checkpoints/epoch_0090.pt
    用于水轮完整流程和检测实验，加载 epoch 90。

这些文件与代码运行有关。直接删除会使读取相应检查点的实验报文件不存在。
自定义 --checkpoint-dir，或修改水轮完整流程的 --output-dir，会改变权重
查找位置，需在相应目录提供兼容权重。

程序运行后还可能在 outputs 中生成图片、MATLAB 数据、训练记录和新权重。

