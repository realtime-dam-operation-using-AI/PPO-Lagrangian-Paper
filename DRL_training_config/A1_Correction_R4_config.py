import argparse
from easydict import EasyDict  # 导入EasyDict库，使字典可以通过点notation访问
from datetime import datetime
import pytz
import sys
import os
import torch.nn as nn  # 导入PyTorch神经网络模块
from gym.envs.registration import register
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)  # 过滤np.bool8弃用警告
warnings.filterwarnings("ignore", message=".*Overriding environment.*")  # 忽略Gym环境覆盖警告

# tensorboard --logdir=E:\GitHub\DI-engine\agent\reservoir_single_agent\result_paperOne
# python agent/reservoir_single_agent/visible/run_dashboard.py

# 获取当前文件的名称，包含后缀
full_filename = os.path.basename(__file__)
# 分割文件名称和扩展名，只保留文件名称部分
filename_without_ext = 'A1_Correction_R4_202511172246'  # 由实验脚本直接赋值
# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
unique_timestamp = datetime.now(pytz.timezone('Asia/Shanghai')).strftime("_%Y%m%d%H%M")
# 动态获取 atest 的路径（向上返回到 atest 文件夹）
atest_path = os.path.abspath(os.path.join(current_dir, "../../../"))
core_lib_path = os.path.abspath(os.path.join(current_dir, "../../../../"))
# 添加路径到系统路径
sys.path.append(atest_path)       # 添加 atest 文件夹路径
sys.path.append(core_lib_path)    # 添加 DI-engine 核心库路径

# 注册自定义策略以便在POLICY_REGISTRY可见
from agent.tool_function import ppo_lagrangian  # noqa: F401

register(
    id="ResevoirAgent-v0",
    entry_point="agent.reservoir_single_agent.envs.reservoir_env_gym:ResevoirAgentEnvGym", 
    max_episode_steps=100000,
    reward_threshold=2000,
)

CO_NUM = 5  # collector环境数量
EV_NUM = 1  # evaluator环境数量
MAX_EPISODE = 60 # 最大episode数量
SCHEDULE_EXPORT_INTERVAL = int(MAX_EPISODE/10) # 调度结果导出间隔
MAX_EPISODE_STEP = 365*4
HIDDEN_SIZE = 512  # 隐藏层大小
HIDDEN_NUM = 3  # 隐藏层数量
N_SAMPLE = int(MAX_EPISODE_STEP/2*CO_NUM)  # 每次训练收集的样本数
BATCH_SIZE = MAX_EPISODE_STEP  # 一次更新的样本批量大小
UPDATE_NUM = 10  # 每次更新的次数
LR = 0.001  # 学习率
POWER_REWARD_COEFF = 1.0 # 发电奖励系数
LEVEL_REWARD_COEFF = 1.0 # 水位约束系数
FLOOD_REWARD_COEFF = 1.0  # 洪水安全惩罚系数
ECOLOGICAL_REWARD_COEFF = 1.0 # 生态流量约束惩罚系数
POWERLIMIT_REWARD_COEFF = 1.0 # 发电保证约束惩罚系数
NAVIGATION_REWARD_COEFF = 1.0 # 航运约束惩罚系数
ACTION_REWARD_COEFF = 1.0 # 动作调整惩罚系数

# ===== PPO-Lagrangian 开关与参数 =====
USE_PPO_LAGRANGIAN = False   # 是否启用PPO-Lagrangian策略
LAGRANGIAN_THRESHOLD = 0.05   # 允许的约束期望（越小越严格）
LAGRANGIAN_LR = 0.01         # 拉格朗日乘子更新步长
LAGRANGIAN_INITIAL_PENALTY = 1.0  # 初始乘子
LAGRANGIAN_DUAL_CLIP = None       # 乘子上界（None表示不裁剪）
LAGRANGIAN_COST_NORMALIZER = 1.0  # 约束成本归一化因子
LAGRANGIAN_UPDATE_INTERVAL = 4    # 乘子更新频率（按minibatch）
LAGRANGIAN_USE_RELU = True        # True时仅对正向违约进行惩罚

# 实验配置参数
NORMALIZATION = True  # 是否进行归一化处理
CONSTRAINT_TYPES = 2  # 1：超限结束回合；2：不结束回合，纠正但不惩罚；3：不结束回合，纠正并惩罚
OBSERVATION_TYPES = 5  # 2：[水位, 入流, 未来来水, 约束, 日期sin日期cos, 水头高度, 航运流量约束, 生态流量约束]
# 观测类型对应的实际维度映射（类型5及以上由于日期sin/cos编码整体+1维）
OBSERVATION_DIM_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 7, 7: 8, 8: 9,}

ACTION_SPACE_TYPE='discrete' # 动作空间类型：连续continuous；discrete
ACTION_SHAPE = 1 if ACTION_SPACE_TYPE == 'continuous' else 100  # 动作空间维度

# 实验名称 - 使用相对路径以避免Windows路径问题

# ===== 对抗训练参数 =====
ADVERSARIAL_ENABLE = False           # 是否启用对抗训练
ADVERSARIAL_ATTACKER_TYPE = 'pgd'  # 攻击器类型: 'fgsm' 或 'pgd'
ADVERSARIAL_EPSILON = 0.05          # 扰动幅度
ADVERSARIAL_NORM_TYPE = 'linf'      # 范数类型: 'linf' 或 'l2'，仅在所有观测已严格归一到 [0,1] 且强调整体扰动时考虑 l2
ADVERSARIAL_ATTACK_FREQUENCY = 0.4   # 对抗训练频率，PGD 建议偏低取 0.4，FGSM 可到 0.5
ADVERSARIAL_LOSS_WEIGHT = 0.15       # 对抗损失权重，常规 0.1–0.2；PGD+freq≥0.4 先用 0.1–0.15，观测到奖励掉速小再升。目标占比: 让 adv_loss_weight * adversarial_loss ≈ 总损失的 10%–30%；>40% 下调，<5% 上调。
CURRICULUM_LEARNING = False          # 是否使用课程学习
INITIAL_EPSILON = 0.01               # 课程学习初始epsilon
TARGET_EPSILON = 0.05                # 课程学习目标epsilon

ENABLE_ACTION_MASK = False           # 是否启用动作掩码
MASK_RULE_TYPE = 'basic'            # 掩码规则类型：'basic'（基础规则）或 'enhanced'（增强规则含三峡规程）
ENABLE_NONLINEAR_MAPPING = False    # 是否启用非线性映射
NONLINEAR_MAPPING_TYPE = 'square'   # 非线性映射函数类型：'linear'/'square'/'sqrt'/'cubic'/'exp'/'log'
OBS_NOISE_ENABLE = False            # 是否开启观测扰动

# ===== 实验名称配置 =====
exp_name_base = filename_without_ext  # 实验名已包含时间戳
folder_name = os.path.basename(current_dir)
new_folder_name = folder_name.replace('test', 'result')
exp_name = os.path.join('agent', 'reservoir_single_agent', new_folder_name, exp_name_base)  # 相对于DI-engine根目录的路径
output_dir = os.path.abspath(os.path.join(current_dir, '..', '..', new_folder_name))
exp_full_path = os.path.join(output_dir, exp_name_base)  # 完整路径用于其他用途

# 与 Learner 日志显示频率一致的迭代间隔
TOTAL_STEPS = int((MAX_EPISODE_STEP*CO_NUM)/BATCH_SIZE*UPDATE_NUM*MAX_EPISODE)
LOG_SHOW_AFTER_ITER = int(TOTAL_STEPS/100)

ppo_rarl_config = dict( # 查看所有TensorBoard日志记录 (tensorboard --logdir=./)
    exp_name = exp_name,  # 实验名称，用于TensorBoard日志记录 (使用相对路径)
    env=dict(
        episode_num=MAX_EPISODE,
        collector_env_num=CO_NUM,
        evaluator_env_num=EV_NUM,
        env_id='ResevoirAgent-v0',  # 指定使用的环境ID
        act_scale=True if ACTION_SPACE_TYPE == 'continuous' else False,
        n_evaluator_episode=EV_NUM,
        stop_value=100000,
        continuous=True if ACTION_SPACE_TYPE == 'continuous' else False,
        action_space=ACTION_SPACE_TYPE,  # 动作空间类型：连续
        action_shape = ACTION_SHAPE,  # 动作空间维度
        max_episode_step = MAX_EPISODE_STEP,  # 每回合最大步数
        # 奖励权重配置
        reward_coeffs=dict(
            power_reward_coeff=POWER_REWARD_COEFF,       # 发电奖励系数
            level_reward_coeff=LEVEL_REWARD_COEFF,       # 水位约束系数
            flood_reward_coeff=FLOOD_REWARD_COEFF,       # 洪水安全惩罚系数（水位过高）
            ecological_reward_coeff=ECOLOGICAL_REWARD_COEFF,  # 生态流量约束惩罚系数（流量过低）
            powerlimit_reward_coeff=POWERLIMIT_REWARD_COEFF,  # 生态流量约束惩罚系数（流量过低）
            navigation_reward_coeff=NAVIGATION_REWARD_COEFF,   # 航运约束惩罚系数（流量过低）
            action_reward_coeff=ACTION_REWARD_COEFF,  # 动作调整惩罚系数.0 # 动作调整惩罚系数
        ),
        res_inflow_filename="ResInflowEnhan.xlsx",
        # 实验设置配置
        experimental_setup=dict(
            normalization=NORMALIZATION,      # 是否进行归一化处理
            constraint_types=CONSTRAINT_TYPES,  # 约束处理类型
            observation_types=OBSERVATION_TYPES,  # 观测类型
            enable_action_mask=ENABLE_ACTION_MASK,          # 是否启用动作掩码
            mask_rule_type=MASK_RULE_TYPE,                  # 掩码规则类型
            enable_nonlinear_mapping=ENABLE_NONLINEAR_MAPPING,    # 是否启用非线性映射
            nonlinear_mapping_type=NONLINEAR_MAPPING_TYPE,        # 非线性映射函数类型
            # ===== 观测扰动相关参数 =====
            obs_noise_enable=OBS_NOISE_ENABLE,           # 是否开启观测扰动
            obs_noise_inflow_ratio=0.1,      # 入流扰动比例（标准差=当前inflow*该比例）
            obs_noise_future_inflow_ratio=0.2, # 未来入流扰动比例（标准差=当前future_inflow*该比例，建议大于inflow）
        ),
        # 调度数据保存配置
        schedule_data_config=dict(
            collector_save_interval=SCHEDULE_EXPORT_INTERVAL,  # collector环境每10轮保存一次数据（可调整）
            evaluator_save_interval=1,   # evaluator环境每轮保存数据（固定为1，建议不修改）
            enable_data_export=True,     # 是否启用数据导出（True/False）
        ),
        # 环境端 TensorBoard 指标开关与目录
        tensorboard=dict(
            enable=True,
            log_dir=os.path.join(exp_name, 'log', 'env'),
            log_interval_envsteps=int((N_SAMPLE//CO_NUM) * LOG_SHOW_AFTER_ITER),
            only_episode_summary=True,
        ),
    ),
    policy=dict(
        cuda=True,            # 是否使用CUDA加速
        action_space=ACTION_SPACE_TYPE,  # 动作空间类型：连续continuous
        recompute_adv=True,    # 是否重新计算优势函数
        
        # 对抗训练配置
        adversarial=dict(
            enable=ADVERSARIAL_ENABLE,
            attacker_type=ADVERSARIAL_ATTACKER_TYPE,
            epsilon=ADVERSARIAL_EPSILON,
            norm_type=ADVERSARIAL_NORM_TYPE,
            attack_frequency=ADVERSARIAL_ATTACK_FREQUENCY,
            kl_type='forward',            # 'forward' | 'symmetric' | 'js'，symmetric/js 仅在梯度过不稳时再试
            kl_temperature=1.0,          # 仅离散动作生效的温度系数，需要更关注高概率动作时可降到 0.7–0.9
            kl_tame_threshold=2.0,      # KL驯化阈值；None 关闭驯化，建议开启 1.5–3.0（如 2.0）抑制 KL 尖峰，提升稳定性。
            attack_success_kl_threshold=0.05,  # 设 0.1–0.3（离散）用于统计；建议从 0.15 起，保证初期成功率在 30%–70%
            curriculum_learning=CURRICULUM_LEARNING,
            initial_epsilon=INITIAL_EPSILON,
            target_epsilon=TARGET_EPSILON,
            curriculum_steps=int(TOTAL_STEPS*0.2), # 课程学习步数，建议占总训练步数的20%
            pgd=dict(
                alpha=0.01, #取 epsilon/num_steps（±50% 微调）
                num_steps=5, #3–5 足够逼近最坏扰动
            ),
            adv_loss_weight=ADVERSARIAL_LOSS_WEIGHT,
            eval_attack=False, # 训练期保持 False；评估鲁棒性时临时置 True
        ),

        model=dict(
            # 观测空间维度
            obs_shape=OBSERVATION_DIM_MAP.get(OBSERVATION_TYPES, OBSERVATION_TYPES),
            action_shape = ACTION_SHAPE,  # 动作空间维度
            encoder_hidden_size_list=[HIDDEN_SIZE, HIDDEN_SIZE, 256],  # 编码器隐藏层大小
            action_space=ACTION_SPACE_TYPE,  # 动作空间类型：连续
            actor_head_layer_num=1,    # Actor网络头部额外层数
            critic_head_layer_num=1,   # Critic网络头部额外层数
            actor_head_hidden_size=256,   # 与 encoder 输出维度一致
            critic_head_hidden_size=256,  # 与 encoder 输出维度一致
        ),
        learn=dict(
            debug_print_entropy=False,
            learner=dict(
                log_policy=True,
                train_iterations=int((MAX_EPISODE_STEP*CO_NUM)/BATCH_SIZE*UPDATE_NUM*(MAX_EPISODE+1)), 
                hook=dict(
                    load_ckpt_before_run='',  # 运行前加载的检查点路径
                    save_ckpt_after_run=True,  # 运行结束后是否保存检查点
                    log_show_after_iter=LOG_SHOW_AFTER_ITER, # 每多少次迭代显示一次日志
                    save_ckpt_after_iter=int(TOTAL_STEPS/4), # 每多少次迭代保存一次检查点
                ),
            ),
            resume_training=False,
            epoch_per_collect=UPDATE_NUM,     # 每次收集数据后的训练epoch数
            batch_size=BATCH_SIZE,            # 批次大小，一次收集能够更新次数=n_sample/batch_size*epoch_per_collect
            learning_rate= LR,    # 学习率
            # lr_scheduler=dict(epoch_num=int(MAX_EPISODE), min_lr_lambda=0.1),
            value_weight=0.6,         # 价值n数损失权重
            entropy_weight=0.016,     # 熵正则化权重
            clip_ratio=0.2,           # PPO裁剪比例
            adv_norm=True,            # 是否对优势函数进行归一化
            value_norm=True,          # 是否对价值函数进行归一化
            ignore_done=False,         # 是否忽略done信号
        ),
        collect=dict(        
            n_sample=N_SAMPLE,            # 每次收集的样本数，小值更新的更频繁，使用最新的策略交互数据，下一次n_sample29220，batch_size4870
            unroll_len=1,             # 展开长度，一次收集的轨迹样本数=n_sample/unroll_len
            discount_factor=0.99,      # 折扣因子γ
            gae_lambda=0.95,           # GAE(Generalized Advantage Estimation)λ参数
            collector=dict(
                collect_print_freq=LOG_SHOW_AFTER_ITER,
            ),         
        ),
        eval=dict(
            evaluator=dict(
                eval_freq=LOG_SHOW_AFTER_ITER*10, 
                render_freq=-1, # 确保不进行渲染，渲染会极大降低速度
            ),
        ),
        lagrangian=dict(
            enable=USE_PPO_LAGRANGIAN,
            constraint_key='constraint_cost_normalized',
            constraint_threshold=LAGRANGIAN_THRESHOLD,
            learning_rate=LAGRANGIAN_LR,
            dual_clip=LAGRANGIAN_DUAL_CLIP,
            initial_penalty=LAGRANGIAN_INITIAL_PENALTY,
            update_interval=LAGRANGIAN_UPDATE_INTERVAL,
            cost_normalizer=LAGRANGIAN_COST_NORMALIZER,
            use_relu=LAGRANGIAN_USE_RELU,
            log_prefix='lagrangian',
        ),
    ),
)
ppo_rarl_config = EasyDict(ppo_rarl_config)  # 转换为EasyDict对象，支持点访问
main_config = ppo_rarl_config  # 设置为主配置

POLICY_TYPE = 'ppo_lagrangian' if USE_PPO_LAGRANGIAN else 'ppo'

# 环境创建配置
ppo_rarl_create_config = dict(
    env=dict(
        type='resevoiragent-v2',
        import_names=['agent.reservoir_single_agent.envs.reservoir_env_v2'], # 导入环境
        exp_name = exp_full_path,  # 使用完整路径用于环境配置
    ),
    env_manager=dict(type='base'),  # 环境管理器类型base | subprocess | async_subprocess
    policy=dict(type=POLICY_TYPE),
)
ppo_rarl_create_config = EasyDict(ppo_rarl_create_config)  # 转换为EasyDict对象
create_config = ppo_rarl_create_config  # 设置为创建配置

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO 训练入口")
    parser.add_argument("--seed", type=int, default=0, help="训练使用的随机种子")
    args = parser.parse_args()

    try:
        # 添加tool_function路径
        tool_function_path = os.path.abspath(os.path.join(current_dir, "../../../tool_function"))
        if tool_function_path not in sys.path:
            sys.path.append(tool_function_path)

        from ding.utils import POLICY_REGISTRY

        if USE_PPO_LAGRANGIAN:
            # 已在模块顶部导入，确保注册成功
            print("✓ PPO-Lagrangian策略已注册，开始训练")
            create_config.policy.type = 'ppo_lagrangian'
        elif ADVERSARIAL_ENABLE:
            from ppo_rarl_standalone import PPORARLPolicy  # noqa: F401
            if 'ppo_rarl' in POLICY_REGISTRY:
                print("✓ PPO-RARL策略已注册")
                create_config.policy.type = 'ppo_rarl'
            else:
                print("✗ PPO-RARL策略未注册，继续使用标准PPO")
                create_config.policy.type = POLICY_TYPE
        else:
            create_config.policy.type = POLICY_TYPE

        # 启动训练
        from agent.reservoir_single_agent.entry.resevoiragent_entry_onpolicy import resrevoiragent_pipeline_onpolicy
        resrevoiragent_pipeline_onpolicy([main_config, create_config], seed=args.seed)

    except Exception as e:
        print(f"✗ 训练启动失败: {e}")
        print("详细错误信息:")
        import traceback
        traceback.print_exc()