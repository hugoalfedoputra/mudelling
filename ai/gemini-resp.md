Here is the verbose continuation for the "back-end" convolutional layers, following your exact formatting style.

As a quick note for your feed-forward: it is standard in deep learning frameworks (like PyTorch/TensorFlow) that convolutional layers also have a bias term added after the cross-correlation sum, i.e., $z_{\text{c2}} = S_{\text{c2}}(...) + b_{\text{c2}}$. I have included the partial derivative for the bias below, as it is mathematically required for updating the network, even if it was implicitly omitted in your feed-forward equation.

---

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c2}}(i_{\text{c2}}, j_{\text{c2}})}=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{c2}}(i_{\text{c2}}, j_{\text{c2}})} \times \frac{\delta O_{\text{c2}}(i_{\text{c2}}, j_{\text{c2}})}{\delta z_{\text{c2}}(i_{\text{c2}}, j_{\text{c2}})} \\
=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{c2}}(i_{\text{c2}}, j_{\text{c2}})} \times \mathbb{I}(z_{\text{c2}}(i_{\text{c2}}, j_{\text{c2}}) > 0)
$$

Note:

1. $i_{\text{c2}}$ and $j_{\text{c2}}$ are the row and column indices of the resulting feature map for the second convolution layer (c2).
2. $\frac{\delta L_{\text{BCE}}}{\delta O_{\text{c2}}}$ is the gradient map propagated from the flattening layer (calculated at the end of the previous section), reshaped back to a 6x2 matrix.
3. $\frac{\delta O_{\text{c2}}}{\delta z_{\text{c2}}}$ is the derivative of the ReLU activation function. It is defined as an indicator function $\mathbb{I}(\cdot)$ which outputs 1 if $z_{\text{c2}} > 0$ and 0 otherwise.

$$
\frac{\delta L_{\text{BCE}}}{\delta K_{\text{c2}}(m_{\text{c2}}, n_{\text{c2}})}=\sum_{i_{\text{c2}}}\sum_{j_{\text{c2}}}\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c2}}(i_{\text{c2}}, j_{\text{c2}})} \times \frac{\delta z_{\text{c2}}(i_{\text{c2}}, j_{\text{c2}})}{\delta K_{\text{c2}}(m_{\text{c2}}, n_{\text{c2}})}\right) \\
=\sum_{i_{\text{c2}}}\sum_{j_{\text{c2}}}\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c2}}(i_{\text{c2}}, j_{\text{c2}})} \times O_{\text{c1}}(i_{\text{c2}}+m_{\text{c2}}, j_{\text{c2}}+n_{\text{c2}})\right)
$$

Note:

1. This is the partial derivative with respect to the kernel weights of the second convolution layer ($K_{\text{c2}}$).
2. $m_{\text{c2}}$ and $n_{\text{c2}}$ are the row and column indices of the kernel.
3. Mathematically, the gradient of the kernel is the cross-correlation between the input to this layer ($O_{\text{c1}}$) and the gradient map of its pre-activation ($z_{\text{c2}}$).

$$
\frac{\delta L_{\text{BCE}}}{\delta b_{\text{c2}}}=\sum_{i_{\text{c2}}}\sum_{j_{\text{c2}}}\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c2}}(i_{\text{c2}}, j_{\text{c2}})} \times 1\right)
$$

Note:

1. $b_{\text{c2}}$ is the bias term for the second convolution layer. Its gradient is simply the sum of all elements in the error map $\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c2}}}$.

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{c1}}(i_{\text{c1}}, j_{\text{c1}})}=\sum_{m_{\text{c2}}}^{M_{\text{c2}}}\sum_{n_{\text{c2}}}^{N_{\text{c2}}}\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c2}}(i_{\text{c1}}-m_{\text{c2}}, j_{\text{c1}}-n_{\text{c2}})} \times \frac{\delta z_{\text{c2}}(...)}{\delta O_{\text{c1}}(i_{\text{c1}}, j_{\text{c1}})}\right) \\
=\sum_{m_{\text{c2}}}^{M_{\text{c2}}}\sum_{n_{\text{c2}}}^{N_{\text{c2}}}\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c2}}(i_{\text{c1}}-m_{\text{c2}}, j_{\text{c1}}-n_{\text{c2}})} \times K_{\text{c2}}(m_{\text{c2}}, n_{\text{c2}})\right)
$$

Note:

1. This calculates the error propagated backwards to the output of the first convolution layer ($O_{\text{c1}}$).
2. $i_{\text{c1}}$ and $j_{\text{c1}}$ are the row and column indices for the $O_{\text{c1}}$ matrix.
3. Mathematically, the error propagated to the input of a convolutional layer is the "full convolution" of the error map ($\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c2}}}$) with a 180-degree flipped kernel ($K_{\text{c2}}$). This is reflected in the subtraction of the indices $(i_{\text{c1}}-m_{\text{c2}}, j_{\text{c1}}-n_{\text{c2}})$. Zero-padding is implicitly applied for out-of-bounds indices.

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c1}}(i_{\text{c1}}, j_{\text{c1}})}=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{c1}}(i_{\text{c1}}, j_{\text{c1}})} \times \mathbb{I}(z_{\text{c1}}(i_{\text{c1}}, j_{\text{c1}}) > 0)
$$

Note:

1. Similar to Persamaan X.X (<REFERRING TO z_c2 EQ>), this applies the derivative of the ReLU activation function for the first convolution layer (c1).

$$
\frac{\delta L_{\text{BCE}}}{\delta K_{\text{c1}}(m_{\text{c1}}, n_{\text{c1}})}=\sum_{i_{\text{c1}}}\sum_{j_{\text{c1}}}\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c1}}(i_{\text{c1}}, j_{\text{c1}})} \times O_{\text{Concat}}(i_{\text{c1}}+m_{\text{c1}}, j_{\text{c1}}+n_{\text{c1}})\right)
$$

Note:

1. Similar to Persamaan X.X (<REFERRING TO K_c2 EQ>), this is the cross-correlation to find the gradient for the first layer's kernel weights ($K_{\text{c1}}$).
2. $O_{\text{Concat}}$ is the input to this layer.

$$
\frac{\delta L_{\text{BCE}}}{\delta b_{\text{c1}}}=\sum_{i_{\text{c1}}}\sum_{j_{\text{c1}}}\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c1}}(i_{\text{c1}}, j_{\text{c1}})} \times 1\right)
$$

Note:

1. This is the partial derivative for the bias term of the first convolution layer.

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{Concat}}(i_{\text{concat}}, j_{\text{concat}})}=\sum_{m_{\text{c1}}}^{M_{\text{c1}}}\sum_{n_{\text{c1}}}^{N_{\text{c1}}}\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c1}}(i_{\text{concat}}-m_{\text{c1}}, j_{\text{concat}}-n_{\text{c1}})} \times K_{\text{c1}}(m_{\text{c1}}, n_{\text{c1}})\right)
$$

Note:

1. Similar to Persamaan X.X (<REFERRING TO O_c1 EQ>), this propagates the error back to the concatenated matrix from the front-end ($O_{\text{Concat}}$).
2. $i_{\text{concat}}$ and $j_{\text{concat}}$ are the row and column indices of the concatenated matrix, which has a shape of 6x2.

---

Here is the revised section with the explicit 2-line chain rule expansions, explicit summation bounds, and the `"Concat"` to `"Cc"` subscript renaming applied exactly as you requested.

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c1}}(i_{\text{c1}}, j_{\text{c1}})}=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{c1}}(i_{\text{c1}}, j_{\text{c1}})} \times \frac{\delta O_{\text{c1}}(i_{\text{c1}}, j_{\text{c1}})}{\delta z_{\text{c1}}(i_{\text{c1}}, j_{\text{c1}})} \\
=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{c1}}(i_{\text{c1}}, j_{\text{c1}})} \times \mathbb{I}(z_{\text{c1}}(i_{\text{c1}}, j_{\text{c1}}) > 0)
$$

Note:

1. Similar to Persamaan X.X (\<REFERRING TO z_c2 EQ\>), this applies the derivative of the ReLU activation function for the first convolution layer (c1).

$$
\frac{\delta L_{\text{BCE}}}{\delta K_{\text{c1}}(m_{\text{c1}}, n_{\text{c1}})}=\sum_{i_{\text{c1}}=0}^{H_{\text{c1}}-1}\sum_{j_{\text{c1}}=0}^{W_{\text{c1}}-1}\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c1}}(i_{\text{c1}}, j_{\text{c1}})} \times \frac{\delta z_{\text{c1}}(i_{\text{c1}}, j_{\text{c1}})}{\delta K_{\text{c1}}(m_{\text{c1}}, n_{\text{c1}})}\right) \\
=\sum_{i_{\text{c1}}=0}^{H_{\text{c1}}-1}\sum_{j_{\text{c1}}=0}^{W_{\text{c1}}-1}\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c1}}(i_{\text{c1}}, j_{\text{c1}})} \times O_{\text{Cc}}(i_{\text{c1}}+m_{\text{c1}}, j_{\text{c1}}+n_{\text{c1}})\right)
$$

Note:

1. Similar to Persamaan X.X (\<REFERRING TO K*c2 EQ\>), this is the cross-correlation to find the gradient for the first layer's kernel weights ($K*{\text{c1}}$).
2. $O_{\text{Cc}}$ is the input to this layer.

$$
\frac{\delta L_{\text{BCE}}}{\delta b_{\text{c1}}}=\sum_{i_{\text{c1}}=0}^{H_{\text{c1}}-1}\sum_{j_{\text{c1}}=0}^{W_{\text{c1}}-1}\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c1}}(i_{\text{c1}}, j_{\text{c1}})} \times \frac{\delta z_{\text{c1}}(i_{\text{c1}}, j_{\text{c1}})}{\delta b_{\text{c1}}}\right) \\
=\sum_{i_{\text{c1}}=0}^{H_{\text{c1}}-1}\sum_{j_{\text{c1}}=0}^{W_{\text{c1}}-1}\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c1}}(i_{\text{c1}}, j_{\text{c1}})} \times 1\right)
$$

Note:

1. This is the partial derivative for the bias term of the first convolution layer.

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}}(i_{\text{Cc}}, j_{\text{Cc}})}=\sum_{m_{\text{c1}}=0}^{M_{\text{c1}}-1}\sum_{n_{\text{c1}}=0}^{N_{\text{c1}}-1}\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c1}}(i_{\text{Cc}}-m_{\text{c1}}, j_{\text{Cc}}-n_{\text{c1}})} \times \frac{\delta z_{\text{c1}}(i_{\text{Cc}}-m_{\text{c1}}, j_{\text{Cc}}-n_{\text{c1}})}{\delta O_{\text{Cc}}(i_{\text{Cc}}, j_{\text{Cc}})}\right) \\
=\sum_{m_{\text{c1}}=0}^{M_{\text{c1}}-1}\sum_{n_{\text{c1}}=0}^{N_{\text{c1}}-1}\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c1}}(i_{\text{Cc}}-m_{\text{c1}}, j_{\text{Cc}}-n_{\text{c1}})} \times K_{\text{c1}}(m_{\text{c1}}, n_{\text{c1}})\right)
$$

Note:

1. Similar to Persamaan X.X (\<REFERRING TO O*c1 EQ\>), this propagates the error back to the concatenated matrix from the front-end ($O*{\text{Cc}}$).
2. $i_{\text{Cc}}$ and $j_{\text{Cc}}$ are the row and column indices of the concatenated matrix, which has a shape of 6x2.

---

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CV}}(i_{\text{concat}}, 1)} = \frac{\delta L_{\text{BCE}}}{\delta O_{\text{Concat}}(i_{\text{concat}}, 1)}
$$

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CH}}(i_{\text{concat}}, 1)} = \frac{\delta L_{\text{BCE}}}{\delta O_{\text{Concat}}(i_{\text{concat}}, 2)}
$$

Note:

1. Because the feed-forward concatenation operation `Concat(.)` joined $O_{\text{CV}}$ and $O_{\text{CH}}$ column-wise (along the frequency dimension/width axis), the backpropagation simply splits the gradient matrix back into its original parts.
2. $\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CV}}}$ takes the first column ($j_{\text{concat}}=1$) of the error matrix, returning a 6x1 vector representing the gradient for the vertical filter's output.
3. $\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CH}}}$ takes the second column ($j_{\text{concat}}=2$) of the error matrix, returning a 6x1 vector representing the gradient for the horizontal filter's output.

---

Here is the verbose backpropagation for the front-end, completing your equations from the concatenation layer all the way back to the input spectrogram.

To ensure mathematical precision with the pooling operations you defined in your manualisation, I have introduced $A_{\text{CV}}$ (the output of the ReLU before max-pooling in the vertical branch) and $M_{\text{in}}$ (the output of the mean-pooling before the convolution in the horizontal branch).

---

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CV}}(i_{\text{CV}}, 0)}=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}}(i_{\text{CV}}, 0)} \times \frac{\delta O_{\text{Cc}}(i_{\text{CV}}, 0)}{\delta O_{\text{CV}}(i_{\text{CV}}, 0)} \\
=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}}(i_{\text{CV}}, 0)} \times 1
$$

Note:

1. $i_{\text{CV}}$ iterates from $0$ to $H_{\text{CV}}-1$.
2. Because $O_{\text{Cc}}$ concatenates $O_{\text{CV}}$ and $O_{\text{CH}}$ column-wise, $O_{\text{CV}}$ corresponds perfectly to the 0-th column (index 0) of the concatenated error matrix.

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CH}}(i_{\text{CH}}, 0)}=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}}(i_{\text{CH}}, 1)} \times \frac{\delta O_{\text{Cc}}(i_{\text{CH}}, 1)}{\delta O_{\text{CH}}(i_{\text{CH}}, 0)} \\
=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}}(i_{\text{CH}}, 1)} \times 1
$$

Note:

1. $i_{\text{CH}}$ iterates from $0$ to $H_{\text{CH}}-1$.
2. $O_{\text{CH}}$ corresponds to the 1-st column (index 1) of the concatenated error matrix.

### Backpropagation of the Vertical Filter Branch

$$
\frac{\delta L_{\text{BCE}}}{\delta A_{\text{CV}}(i_{\text{ACV}}, j_{\text{ACV}})}=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CV}}(i_{\text{ACV}}, 0)} \times \frac{\delta O_{\text{CV}}(i_{\text{ACV}}, 0)}{\delta A_{\text{CV}}(i_{\text{ACV}}, j_{\text{ACV}})} \\
= \begin{cases}
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CV}}(i_{\text{ACV}}, 0)} \times 1 & \text{if } j_{\text{ACV}} = \operatorname*{argmax}_{w} A_{\text{CV}}(i_{\text{ACV}}, w) \\
0 & \text{otherwise}
\end{cases}
$$

Note:

1. $A_{\text{CV}}$ is the intermediate matrix after the ReLU activation but before the max-pooling operation.
2. $i_{\text{ACV}}$ iterates from $0$ to $H_{\text{ACV}}-1$, and $j_{\text{ACV}}$ iterates from $0$ to $W_{\text{ACV}}-1$.
3. The max-pooling operation routes the gradient _only_ to the specific frequency bin (column index $w$) that contained the maximum value during the feed-forward pass. All other elements in that row receive a gradient of 0.

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})}=\frac{\delta L_{\text{BCE}}}{\delta A_{\text{CV}}(i_{\text{V}}, j_{\text{V}})} \times \frac{\delta A_{\text{CV}}(i_{\text{V}}, j_{\text{V}})}{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})} \\
=\frac{\delta L_{\text{BCE}}}{\delta A_{\text{CV}}(i_{\text{V}}, j_{\text{V}})} \times \mathbb{I}(z_{\text{V}}(i_{\text{V}}, j_{\text{V}}) > 0)
$$

Note:

1. This applies the derivative of the ReLU activation function for the vertical convolution layer.

$$
\frac{\delta L_{\text{BCE}}}{\delta K_{\text{V}}(m_{\text{V}}, n_{\text{V}})}=\sum_{i_{\text{V}}=0}^{H_{\text{V}}-1}\sum_{j_{\text{V}}=0}^{W_{\text{V}}-1}\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})} \times \frac{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})}{\delta K_{\text{V}}(m_{\text{V}}, n_{\text{V}})}\right) \\
=\sum_{i_{\text{V}}=0}^{H_{\text{V}}-1}\sum_{j_{\text{V}}=0}^{W_{\text{V}}-1}\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})} \times I(i_{\text{V}}+m_{\text{V}}, j_{\text{V}}+n_{\text{V}})\right)
$$

Note:

1. This is the cross-correlation to find the gradient for the vertical filter's kernel weights ($K_{\text{V}}$).
2. $I$ is the initial log-mel spectrogram input.

$$
\frac{\delta L_{\text{BCE}}}{\delta b_{\text{V}}}=\sum_{i_{\text{V}}=0}^{H_{\text{V}}-1}\sum_{j_{\text{V}}=0}^{W_{\text{V}}-1}\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})} \times \frac{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})}{\delta b_{\text{V}}}\right) \\
=\sum_{i_{\text{V}}=0}^{H_{\text{V}}-1}\sum_{j_{\text{V}}=0}^{W_{\text{V}}-1}\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})} \times 1\right)
$$

Note:

1. This is the partial derivative for the bias term of the vertical convolution layer.

### Backpropagation of the Horizontal Filter Branch

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{\text{H}}(i_{\text{H}}, 0)}=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CH}}(i_{\text{H}}, 0)} \times \frac{\delta O_{\text{CH}}(i_{\text{H}}, 0)}{\delta z_{\text{H}}(i_{\text{H}}, 0)} \\
=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CH}}(i_{\text{H}}, 0)} \times \mathbb{I}(z_{\text{H}}(i_{\text{H}}, 0) > 0)
$$

Note:

1. This applies the derivative of the ReLU activation function for the horizontal convolution layer.
2. Because the horizontal branch uses 1D convolution over the time axis, the second index is fixed to $0$ (representing a 1D column vector).

$$
\frac{\delta L_{\text{BCE}}}{\delta K_{\text{H}}(m_{\text{H}})}=\sum_{i_{\text{H}}=0}^{H_{\text{H}}-1}\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{H}}(i_{\text{H}}, 0)} \times \frac{\delta z_{\text{H}}(i_{\text{H}}, 0)}{\delta K_{\text{H}}(m_{\text{H}})}\right) \\
=\sum_{i_{\text{H}}=0}^{H_{\text{H}}-1}\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{H}}(i_{\text{H}}, 0)} \times M_{\text{in}}(i_{\text{H}}+m_{\text{H}}, 0)\right)
$$

Note:

1. $m_{\text{H}}$ iterates from $0$ to $M_{\text{H}}-1$. Because $K_{\text{H}}$ is a 1D kernel operating over the sequence length, it only requires one spatial iterator.
2. $M_{\text{in}}$ is the intermediate output vector from the mean-pooling layer that acts as the input to this convolution.

$$
\frac{\delta L_{\text{BCE}}}{\delta b_{\text{H}}}=\sum_{i_{\text{H}}=0}^{H_{\text{H}}-1}\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{H}}(i_{\text{H}}, 0)} \times \frac{\delta z_{\text{H}}(i_{\text{H}}, 0)}{\delta b_{\text{H}}}\right) \\
=\sum_{i_{\text{H}}=0}^{H_{\text{H}}-1}\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{H}}(i_{\text{H}}, 0)} \times 1\right)
$$

Note:

1. This is the partial derivative for the bias term of the horizontal convolution layer.

$$
\frac{\delta L_{\text{BCE}}}{\delta M_{\text{in}}(i_{\text{Min}}, 0)}=\sum_{m_{\text{H}}=0}^{M_{\text{H}}-1}\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{H}}(i_{\text{Min}}-m_{\text{H}}, 0)} \times \frac{\delta z_{\text{H}}(i_{\text{Min}}-m_{\text{H}}, 0)}{\delta M_{\text{in}}(i_{\text{Min}}, 0)}\right) \\
=\sum_{m_{\text{H}}=0}^{M_{\text{H}}-1}\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{H}}(i_{\text{Min}}-m_{\text{H}}, 0)} \times K_{\text{H}}(m_{\text{H}})\right)
$$

Note:

1. This propagates the error backward through the 1D horizontal convolution using a full convolution (subtraction of indices). Out-of-bounds indices evaluate to $0$.

$$
\frac{\delta L_{\text{BCE}}}{\delta I_{\text{H\_branch}}(i_{\text{Min}}, j_{\text{I}})}=\frac{\delta L_{\text{BCE}}}{\delta M_{\text{in}}(i_{\text{Min}}, 0)} \times \frac{\delta M_{\text{in}}(i_{\text{Min}}, 0)}{\delta I_{\text{H\_branch}}(i_{\text{Min}}, j_{\text{I}})} \\
=\frac{\delta L_{\text{BCE}}}{\delta M_{\text{in}}(i_{\text{Min}}, 0)} \times \frac{1}{W_{\text{I}}}
$$

Note:

1. This propagates the error backward through the mean-pooling layer to the original input matrix $I$.
2. $j_{\text{I}}$ iterates from $0$ to $W_{\text{I}}-1$ (the width of the frequency bin axis, which is 5). Mean-pooling averages the inputs, so the gradient is distributed equally across all $W_{\text{I}}$ elements in the pooling window.
3. _(Optional Note)_: If calculating the total gradient of the original input matrix $I$ is required, it is the sum of the gradients passed back from both the vertical branch and the horizontal branch: $\delta I_{\text{total}} = \delta I_{\text{V\_branch}} + \delta I_{\text{H\_branch}}$.
