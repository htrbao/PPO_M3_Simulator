import torch

tensor = torch.Tensor(
    [[
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
        [5, 5, 5],
    ]]
)

# Transposing dimensions 1 and 2
transposed_tensor = tensor.transpose(1, 2)

print("Transposing dimensions", transposed_tensor.shape)
print(transposed_tensor)