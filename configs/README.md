# Explanation of the configurable arguments

### Model configs
- `PI`: Size of each layer of the Linear layer used for the Actor Model
- `VF`: Size of each layer of the Linear layer used for the Critic Model
- `MID_CHANNELS`: Number of channels of the CNN (see example below)
- `NUM_FIRST_CNN_LAYERS`: Number of passes through the CNN (see example below)
- `SHARE_FEATURES_EXTRACTOR`: Whether sharing the feature extraction between Actor and Critic
- `CHECKPOINTS`: Load the old checkpoint to continue the training process if the `relative_path` to the old checkpoint is received.

![M3 CNN Architecture](configs/ARGS.png)

### Training configs
- `LR`: Learning rate
- `N_STEPS`: Rollout Length
- `BATCH_SIZE`: Batch size for each training process
- `ENTROPY_COEFF`: Entropy coefficient (Used in `loss` computing process)
- `VF_COEF`: Value coefficient (Used in `loss` computing process)

### Reward configs
- `GAMMA`: How important the points you get later are compared to the points you get right now.

### Logging configs
- `DEVICE`: `"cuda"` if using GPU else `"cpu"`
- `PREFIX_NAME`: Used in name for model

---
# Giải thích về các tham số cấu hình

### Cấu hình Mô hình
- `PI`: Kích thước của mỗi lớp trong mô hình Linear được sử dụng cho Actor
- `VF`: Kích thước của mỗi lớp trong mô hình Linear được sử dụng cho Critic
- `MID_CHANNELS`: Số kênh của CNN (xem ví dụ bên dưới)
- `NUM_FIRST_CNN_LAYERS`: Số lần thông qua CNN (xem ví dụ bên dưới)
- `SHARE_FEATURES_EXTRACTOR`: Có chia sẻ phần trích xuất đặc trưng giữa Actor và Critic hay không
- `CHECKPOINTS`: Tải Checkpoint cũ để tiếp tục quá trình huấn luyện nếu được nhận `relative_path`.

![Kiến trúc M3 CNN](configs/ARGS.png)

### Cấu hình Huấn luyện
- `LR`: Learning Rate
- `N_STEPS`: Độ dài Rollout
- `BATCH_SIZE`: Kích thước batch cho mỗi quá trình huấn luyện
- `ENTROPY_COEFF`: Hệ số loss entropy (Sử dụng trong quá trình tính toán loss)
- `VF_COEF`: Hệ số loss của Value Function (Sử dụng trong quá trình tính toán loss)

### Cấu hình Thưởng
- `GAMMA`: Tầm quan trọng của phần thưởng bạn nhận được khi tích lũy so với phần thưởng tức thời.

### Cấu hình Ghi log
- `DEVICE`: "cuda" nếu sử dụng GPU, ngược lại là "cpu"
- `PREFIX_NAME`: Sử dụng trong tên cho mô hình
