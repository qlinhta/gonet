## gonet
[IASD2023][DL] Training a network for the game Go

Question:
- Should use Global Average Pool?

Network should smaller than 100,000 parameters:
- Use Fewer Layers: Opt for a shallower network. Each layer adds parameters, so fewer layers will help stay within the limit.
- Limit the Number of Filters: In convolutional layers, reduce the number of filters. More filters mean more parameters.
- Smaller Kernel Sizes: Use smaller kernels in convolutional layers (e.g., 3x3 or even 2x2).
- Reduce Dense Layer Size: If using fully connected (dense) layers, especially in the policy and value heads, keep them small.
- Skip Residual Connections: Although beneficial for deeper networks, you might have to skip residual connections due to parameter constraints.
- Use Depthwise Separable Convolutions: These reduce the number of parameters compared to standard convolutions.
- Parameter Sharing: If applicable, use techniques that allow parameter sharing across different parts of the network.
- Pruning and Quantization: These techniques can reduce the number of parameters and the size of the network after initial training.
