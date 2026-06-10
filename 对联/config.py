import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HAS_GPU = torch.cuda.is_available()
DATASET_ROOT = './couplet'
MAX_SEQ_LEN = 16
BATCH_SIZE = 64
EPOCHS = 25
D_MODEL = 256
NHEAD = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
LEARNING_RATE = 1e-4
LAMBDA_PZ = 0.35
BEAM_SIZE = 3

# ========== 训练加速配置 ==========
USE_MIXED_PRECISION = HAS_GPU       # 混合精度训练（仅GPU可用时）
GRADIENT_ACCUMULATION_STEPS = 2     # 梯度累积步数
NUM_WORKERS = 2                     # 数据加载进程数（Windows建议设为0或2）
USE_TORCH_COMPILE = False           # 禁用torch.compile（Windows兼容性问题）
VALIDATE_EVERY_EPOCH = False        # 是否每个epoch都验证（关闭可加速）
VALIDATE_EVERY_N_EPOCHS = 3         # 每N个epoch验证一次
PRINT_FREQ = 100                    # 每N个batch打印一次进度
PIN_MEMORY = HAS_GPU                # pin_memory（仅GPU可用时）
