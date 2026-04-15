This is a fantastic start! Your approach of writing the equations verbosely with explicit notes is excellent for an undergraduate thesis, as it leaves no room for ambiguity for your advisors. Your derivation of the $\frac{\delta L_{\text{BCE}}}{\delta z_O}$ chain rule is beautifully done and mathematically spot on!

Before we continue the backpropagation, I want to point out **two minor structural things in your Feed-Forward** that you should adjust so it perfectly matches your network architecture.

### 1. Feedback on Feed-Forward (Corrections to make)

**A. The "FC input layer" usually doesn't have an activation function.**
In your draft, you wrote:
$$O_{\text{FC}}=\sigma(z_{\text{FC}})$$
$$z_{\text{FC}}=O_{\text{c2}}$$
Usually, the output of the CNN ($O_{\text{c2}}$, shape 6x2) is simply "flattened" into a 1D array of 12 nodes to become the input of the Fully Connected (FC) layer. There are no weights, biases, or sigmoid activations applied during this flattening step. I recommend replacing those two equations with a simple flatten operation:
$$O_{\text{FC}} = \text{Flatten}(O_{\text{c2}})$$
_Note: Flatten(.) is an operation that reshapes the 2D matrix output of the second convolution layer ($O_{\text{c2}}$) of shape 6x2 into a 1D vector of shape 12x1._

**B. You missed the Pooling layers in the Front-End.**
Based on your manualisation document, the vertical branch has a Max-Pool _after_ the ReLU, and the horizontal branch has a Mean-Pool _before_ the convolution. You should update your feed-forward to reflect this:

- **Vertical:** $O_{\text{CV}} = \text{MaxPool}(\text{ReLU}(z_V))$
- **Horizontal:** $z_{\text{H}} = S_{\text{H}}(\text{MeanPool}(I), K_{\text{H}})$

---

### 2. Corrections to your Output Layer Backpropagation

Your formula for $\frac{\delta L_{\text{BCE}}}{\delta W_O}$ and $\frac{\delta L_{\text{BCE}}}{\delta b_O}$ currently uses $\sum_{n=1}^3$. However, a specific weight $W_{O_{(j,n)}}$ only connects one specific hidden node $j$ to one specific output node $n$. Therefore, you do **not** sum across all outputs for individual weights/biases.

Here is the corrected and verbose continuation matching your style:

$$
\frac{\delta L_{\text{BCE}}}{\delta W_{O_{(j,n)}}}=\frac{\delta L_{\text{BCE}}}{\delta z_{O_n}} \times \frac{\delta z_{O_n}}{\delta W_{O_{(j,n)}}} \\
=\left[\frac{\delta L_{\text{BCE}}}{\delta z_O}\right]_n \times \left[O_\text{FCH}\right]_j
$$

Note:

1. $W_{O_{(j,n)}}$ is the specific weight connecting the $j$-th node of the FCH layer to the $n$-th output node.
2. $\left[\frac{\delta L_{\text{BCE}}}{\delta z_O}\right]_n$ is the $n$-th element of the gradient calculated in Persamaan X.X (your $z_O$ equation).
3. $\left[O_\text{FCH}\right]_j$ is the $j$-th element of the hidden layer output (where $j$ ranges from 1 to 6).

$$
\frac{\delta L_{\text{BCE}}}{\delta b_{O_n}}=\frac{\delta L_{\text{BCE}}}{\delta z_{O_n}} \times \frac{\delta z_{O_n}}{\delta b_{O_n}} \\
=\left[\frac{\delta L_{\text{BCE}}}{\delta z_O}\right]_n \times 1
$$

Note:

1. $b_{O_n}$ is the specific bias for the $n$-th output node.
2. The derivative of $z_{O_n}$ with respect to its own bias is simply 1.

---

### 3. Backpropagation Through the Hidden Layer (FCH)

Now, we propagate the error backwards from the output nodes to the 6 hidden nodes ($O_{\text{FCH}}$), through the hidden sigmoid activation ($z_{\text{FCH}}$), and calculate the gradients for the hidden weights and biases.

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{FCH}_j}}=\sum_{n=1}^3 \left(\frac{\delta L_{\text{BCE}}}{\delta z_{O_n}} \times \frac{\delta z_{O_n}}{\delta O_{\text{FCH}_j}} \right) \\
=\sum_{n=1}^3 \left(\left[\frac{\delta L_{\text{BCE}}}{\delta z_O}\right]_n \times W_{O_{(j,n)}} \right)
$$

Note:

1. This calculates the error propagated to the $j$-th hidden node.
2. Unlike the weights, we **do** use $\sum_{n=1}^3$ here because the $j$-th hidden node connects to _all 3_ output nodes. We must sum the errors passed back from all 3 outputs.
3. $W_{O_{(j,n)}}$ is the weight connecting hidden node $j$ to output node $n$.

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{\text{FCH}_j}}=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{FCH}_j}} \times \frac{\delta O_{\text{FCH}_j}}{\delta z_{\text{FCH}_j}} \\
=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{FCH}_j}} \times \big(O_{\text{FCH}_j} \times (1 - O_{\text{FCH}_j})\big)
$$

Note:

1. This is the error at the hidden layer before the sigmoid activation function is applied.
2. $\frac{\delta O_{\text{FCH}_j}}{\delta z_{\text{FCH}_j}}$ is the derivative of the sigmoid activation function for the hidden layer, which resolves to $O_{\text{FCH}_j} \times (1 - O_{\text{FCH}_j})$.

$$
\frac{\delta L_{\text{BCE}}}{\delta W_{\text{FCH}_{(i,j)}}}=\frac{\delta L_{\text{BCE}}}{\delta z_{\text{FCH}_j}} \times \frac{\delta z_{\text{FCH}_j}}{\delta W_{\text{FCH}_{(i,j)}}} \\
=\frac{\delta L_{\text{BCE}}}{\delta z_{\text{FCH}_j}} \times \left[O_\text{FC}\right]_i
$$

Note:

1. $W_{\text{FCH}_{(i,j)}}$ is the weight connecting the $i$-th input node (from the 12-node flattened CNN output) to the $j$-th hidden node.
2. $\left[O_\text{FC}\right]_i$ is the $i$-th element of the flattened CNN output vector (where $i$ ranges from 1 to 12).

$$
\frac{\delta L_{\text{BCE}}}{\delta b_{\text{FCH}_j}}=\frac{\delta L_{\text{BCE}}}{\delta z_{\text{FCH}_j}} \times 1
$$

Note:

1. $b_{\text{FCH}_j}$ is the specific bias for the $j$-th hidden node.

---

### 4. Backpropagating to the "FC Input" (Flattened CNN Output)

Finally, we calculate the error that will be passed backward into the convolutional layers (the back-end).

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{FC}_i}}=\sum_{j=1}^6 \left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{FCH}_j}} \times \frac{\delta z_{\text{FCH}_j}}{\delta O_{\text{FC}_i}} \right) \\
=\sum_{j=1}^6 \left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{FCH}_j}} \times W_{\text{FCH}_{(i,j)}} \right)
$$

Note:

1. This calculates the error propagated to the $i$-th node of the flattened input layer (the 12 nodes).
2. Because the $i$-th input node contributes to _all 6_ hidden nodes, we sum the errors passed back from $j=1$ to $6$.

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{c2}}} = \text{Reshape}\left( \frac{\delta L_{\text{BCE}}}{\delta O_{\text{FC}}}, (6, 2) \right)
$$

Note:

1. Since the feed-forward flattening operation mathematically carries no weights or activations, the partial derivative with respect to the Conv2 output ($O_{\text{c2}}$) is simply the partial derivative of $O_{\text{FC}}$ reshaped back into its original 2D matrix shape of 6x2.
2. $\frac{\delta L_{\text{BCE}}}{\delta O_{\text{c2}}}$ will act as the starting gradient map for the convolutional back-end backpropagation.

---

This gets you perfectly to the boundary where the 12-node Fully Connected classifier ends and the CNN back-end begins! Let me know if you would like to continue verbosely writing out the back-propagation for the Conv2 and Conv1 layers next.
