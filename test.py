import torch
import torch.nn as nn

input = torch.randn(1, 1, 6, 6)
print(input)

output = nn.Dropout(0.25)(input)

print("\n" + 50 * "-" + "\n")
print(output)
