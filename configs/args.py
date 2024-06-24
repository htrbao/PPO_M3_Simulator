from typing import List, Union

# Model configs
PI:List[int] = [256, 180, 180] # Có thể để mảng rỗng []
VF:List[int] = [161, 32] # Có thể để mảng rỗng []
MID_CHANNELS:int = 16
NUM_FIRST_CNN_LAYERS:int = 16
SHARE_FEATURES_EXTRACTOR:int = True
CHECKPOINTS:Union[None, str] = None

# Training configs
LR = 3e-5
N_STEPS = 32
BATCH_SIZE = 128
ENTROPY_COEFF = 0.00001
VF_COEF = 0.5

# Reward configs
GAMMA = 0.99

# Logging configs
WANDB = False
DEVICE = "cuda"
PREFIX_NAME = "test"