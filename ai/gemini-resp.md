Before we define the backpropagation formulas, I must point out a slight mathematical inconsistency in how you wrote the feed-forward equations for the GRU.

In your updated text, you wrote the weight matrices _outside_ the activation functions, such as $W_z \sigma(...)$. However, if you look closely at **Cho et al. (2014) Equations 5, 6, and 8**, the weights are strictly **inside** the activation functions. Furthermore, since your $O_{\text{Cc}(t)}$ and $h_{t}$ are **row vectors** of shape 1x2, matrix multiplication rules dictate that they must multiply the weight matrices from the left (i.e., $xW$, not $Wx$).

To ensure the matrix shapes (1x2 and 2x2) correctly align during the backpropagation chain rule, I will use the mathematically correct standard for these feed-forward equations:

- $\tilde{h}_t = \tanh(O_{\text{Cc}(t)} W_h + (r_t \odot h_{t-1}) U_h + b_h)$
- $z_t = \sigma(O_{\text{Cc}(t)} W_z + h_{t-1} U_z + b_z)$
- $r_t = \sigma(O_{\text{Cc}(t)} W_r + h_{t-1} U_r + b_r)$

Here are the detailed, highly verbose backpropagation equations for the **CNN with GRU Back-End**. Backpropagation Through Time (BPTT) unfolds the network and calculates the gradients backwards from the last time-step ($t=6$) to the first ($t=1$).

### CNN with GRU Back-End

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{GRU}}} = \operatorname*{Reshape}\left( \frac{\delta L_{\text{BCE}}}{\delta O_{\text{FC}}}, (6, 2) \right)
$$

Note:

1. $O_{\text{GRU}}$ is the concatenated output of the GRU.
2. $\operatorname*{Reshape}(\cdot)$ reshapes the 12x1 gradient vector propagated from the Classifier back into a 6x2 matrix. Each row $t$ in this matrix represents the gradient flowing directly into the hidden state at that specific time-step, denoted as $\frac{\delta L_{\text{BCE}}}{\delta O_{\text{GRU}(t)}}$ with shape 1x2.

$$
\frac{\delta L_{\text{BCE}}}{\delta h_t} = \frac{\delta L_{\text{BCE}}}{\delta O_{\text{GRU}(t)}} + \frac{\delta L_{\text{BCE}}}{\delta h_t^{\text{next}}}
$$

Note:

1. $t \in \{6, 5, 4, 3, 2, 1\}$ because BPTT steps backwards.
2. $\frac{\delta L_{\text{BCE}}}{\delta h_t}$ (shape 1x2) is the total gradient accumulated at the hidden state $h_t$.
3. $\frac{\delta L_{\text{BCE}}}{\delta O_{\text{GRU}(t)}}$ (shape 1x2) is the direct upstream gradient from the Classifier for time-step $t$.
4. $\frac{\delta L_{\text{BCE}}}{\delta h_t^{\text{next}}}$ (shape 1x2) is the gradient passed backwards from the _subsequent_ time-step $t+1$. For the very last step ($t=6$), this value is an all-zero vector because there is no $t=7$.

$$
\frac{\delta L_{\text{BCE}}}{\delta \tilde{h}_t} = \frac{\delta L_{\text{BCE}}}{\delta h_t} \odot \frac{\delta h_t}{\delta \tilde{h}_t} \\
= \frac{\delta L_{\text{BCE}}}{\delta h_t} \odot (1 - z_t)
$$

Note:

1. $\frac{\delta L_{\text{BCE}}}{\delta \tilde{h}_t}$ (shape 1x2) is the gradient with respect to the candidate hidden state.
2. $\odot$ denotes element-wise multiplication.

$$
\frac{\delta L_{\text{BCE}}}{\delta z_t} = \frac{\delta L_{\text{BCE}}}{\delta h_t} \odot \frac{\delta h_t}{\delta z_t} \\
= \frac{\delta L_{\text{BCE}}}{\delta h_t} \odot (h_{t-1} - \tilde{h}_t)
$$

Note:

1. $\frac{\delta L_{\text{BCE}}}{\delta z_t}$ (shape 1x2) is the gradient with respect to the update gate.

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}_t}} = \frac{\delta L_{\text{BCE}}}{\delta \tilde{h}_t} \odot \frac{\delta \tilde{h}_t}{\delta z_{\tilde{h}_t}} \\
= \frac{\delta L_{\text{BCE}}}{\delta \tilde{h}_t} \odot (1 - \tilde{h}_t \odot \tilde{h}_t)
$$

Note:

1. $z_{\tilde{h}_t}$ denotes the linear pre-activation function before the $\tanh$ is applied: $z_{\tilde{h}_t} = O_{\text{Cc}(t)} W_h + (r_t \odot h_{t-1}) U_h + b_h$.
2. $\frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}_t}}$ (shape 1x2) is the gradient of the candidate hidden state before the $\tanh$ activation. $(1 - \tilde{h}_t \odot \tilde{h}_t)$ is the derivative of the $\tanh$ function.

$$
\frac{\delta L_{\text{BCE}}}{\delta r_t} = \left( \frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}_t}} \times \frac{\delta z_{\tilde{h}_t}}{\delta (r_t \odot h_{t-1})} \right) \odot \frac{\delta (r_t \odot h_{t-1})}{\delta r_t} \\
= \left( \frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}_t}} U_h^T \right) \odot h_{t-1}
$$

Note:

1. $\frac{\delta L_{\text{BCE}}}{\delta r_t}$ (shape 1x2) is the gradient with respect to the reset gate.
2. The term inside the parenthesis is a matrix multiplication between a 1x2 vector and a 2x2 transposed weight matrix, resulting in a 1x2 vector.

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{r_t}} = \frac{\delta L_{\text{BCE}}}{\delta r_t} \odot \frac{\delta r_t}{\delta z_{r_t}} \\
= \frac{\delta L_{\text{BCE}}}{\delta r_t} \odot (r_t \odot (1 - r_t))
$$

Note:

1. $z_{r_t}$ denotes the linear pre-activation before the sigmoid is applied: $z_{r_t} = O_{\text{Cc}(t)} W_r + h_{t-1} U_r + b_r$.
2. $\frac{\delta L_{\text{BCE}}}{\delta z_{r_t}}$ (shape 1x2) is the gradient of the reset gate before the sigmoid activation. $(r_t \odot (1 - r_t))$ is the derivative of the sigmoid function.

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{z_t}} = \frac{\delta L_{\text{BCE}}}{\delta z_t} \odot \frac{\delta z_t}{\delta z_{z_t}} \\
= \frac{\delta L_{\text{BCE}}}{\delta z_t} \odot (z_t \odot (1 - z_t))
$$

Note:

1. $z_{z_t}$ denotes the linear pre-activation before the sigmoid is applied: $z_{z_t} = O_{\text{Cc}(t)} W_z + h_{t-1} U_z + b_z$.
2. $\frac{\delta L_{\text{BCE}}}{\delta z_{z_t}}$ (shape 1x2) is the gradient of the update gate before the sigmoid activation.

_(The following weight and bias gradients are accumulated over all time-steps $t=1$ to $6$)_

$$
\frac{\delta L_{\text{BCE}}}{\delta W_h} = \sum_{t=1}^6 \left( \left[ \frac{\delta z_{\tilde{h}_t}}{\delta W_h} \right]^T \times \frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}_t}} \right) \\
= \sum_{t=1}^6 \left( O_{\text{Cc}(t)}^T \times \frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}_t}} \right)
$$

Note:

1. $\frac{\delta L_{\text{BCE}}}{\delta W_h}$ (shape 2x2) is the gradient for the candidate hidden state weight matrix $W_h$.
2. $O_{\text{Cc}(t)}^T$ (shape 2x1) is multiplied by the 1x2 pre-activation gradient to produce a 2x2 gradient matrix.

$$
\frac{\delta L_{\text{BCE}}}{\delta U_h} = \sum_{t=1}^6 \left( \left[ \frac{\delta z_{\tilde{h}_t}}{\delta U_h} \right]^T \times \frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}_t}} \right) \\
= \sum_{t=1}^6 \left( (r_t \odot h_{t-1})^T \times \frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}_t}} \right)
$$

Note:

1. $\frac{\delta L_{\text{BCE}}}{\delta U_h}$ (shape 2x2) is the gradient for the candidate hidden state recurrent weight matrix $U_h$.

$$
\frac{\delta L_{\text{BCE}}}{\delta b_h} = \sum_{t=1}^6 \left( \frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}_t}} \times \frac{\delta z_{\tilde{h}_t}}{\delta b_h} \right) \\
= \sum_{t=1}^6 \left( \frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}_t}} \times 1 \right)
$$

Note:

1. $\frac{\delta L_{\text{BCE}}}{\delta b_h}$ (shape 1x2) is the gradient for the candidate hidden state bias. Similar summation logic applies to the remaining weights and biases for the update and reset gates as follows:

$$
\frac{\delta L_{\text{BCE}}}{\delta W_z} = \sum_{t=1}^6 \left( O_{\text{Cc}(t)}^T \times \frac{\delta L_{\text{BCE}}}{\delta z_{z_t}} \right) ; \quad \frac{\delta L_{\text{BCE}}}{\delta U_z} = \sum_{t=1}^6 \left( h_{t-1}^T \times \frac{\delta L_{\text{BCE}}}{\delta z_{z_t}} \right) ; \quad \frac{\delta L_{\text{BCE}}}{\delta b_z} = \sum_{t=1}^6 \left( \frac{\delta L_{\text{BCE}}}{\delta z_{z_t}} \right)
$$

$$
\frac{\delta L_{\text{BCE}}}{\delta W_r} = \sum_{t=1}^6 \left( O_{\text{Cc}(t)}^T \times \frac{\delta L_{\text{BCE}}}{\delta z_{r_t}} \right) ; \quad \frac{\delta L_{\text{BCE}}}{\delta U_r} = \sum_{t=1}^6 \left( h_{t-1}^T \times \frac{\delta L_{\text{BCE}}}{\delta z_{r_t}} \right) ; \quad \frac{\delta L_{\text{BCE}}}{\delta b_r} = \sum_{t=1}^6 \left( \frac{\delta L_{\text{BCE}}}{\delta z_{r_t}} \right)
$$

_(End of weight gradient accumulation. Returning to BPTT sequence)_

$$
\frac{\delta L_{\text{BCE}}}{\delta h_{t-1}^{\text{next}}} = \left(\frac{\delta L_{\text{BCE}}}{\delta h_t} \odot \frac{\delta h_t}{\delta h_{t-1}}\right) + \left(\frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}_t}} \times \frac{\delta z_{\tilde{h}_t}}{\delta h_{t-1}}\right) + \left(\frac{\delta L_{\text{BCE}}}{\delta z_{z_t}} \times \frac{\delta z_{z_t}}{\delta h_{t-1}}\right) + \left(\frac{\delta L_{\text{BCE}}}{\delta z_{r_t}} \times \frac{\delta z_{r_t}}{\delta h_{t-1}}\right) \\
= \left( \frac{\delta L_{\text{BCE}}}{\delta h_t} \odot z_t \right) + \left( \left( \frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}_t}} U_h^T \right) \odot r_t \right) + \left( \frac{\delta L_{\text{BCE}}}{\delta z_{z_t}} U_z^T \right) + \left( \frac{\delta L_{\text{BCE}}}{\delta z_{r_t}} U_r^T \right)
$$

Note:

1. $\frac{\delta L_{\text{BCE}}}{\delta h_{t-1}^{\text{next}}}$ (shape 1x2) is the total gradient passed backward from time-step $t$ to the previous hidden state $h_{t-1}$.
2. Because $h_{t-1}$ is used in 4 different places during the forward pass (the direct $h_t$ equation, and the 3 pre-activations), the chain rule demands that we sum the gradients arriving from all 4 pathways.

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}(t)}} = \left( \frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}_t}} \times \frac{\delta z_{\tilde{h}_t}}{\delta O_{\text{Cc}(t)}} \right) + \left( \frac{\delta L_{\text{BCE}}}{\delta z_{z_t}} \times \frac{\delta z_{z_t}}{\delta O_{\text{Cc}(t)}} \right) + \left( \frac{\delta L_{\text{BCE}}}{\delta z_{r_t}} \times \frac{\delta z_{r_t}}{\delta O_{\text{Cc}(t)}} \right) \\
= \left( \frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}_t}} W_h^T \right) + \left( \frac{\delta L_{\text{BCE}}}{\delta z_{z_t}} W_z^T \right) + \left( \frac{\delta L_{\text{BCE}}}{\delta z_{r_t}} W_r^T \right)
$$

Note:

1. $\frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}(t)}}$ (shape 1x2) is the gradient propagated back to the CNN Front-End for a specific time step $t$.
2. The final gradient matrix $\frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}}}$ (shape 6x2) to be passed back to the Front-End Concatenation layer is constructed by vertically stacking these 1x2 row vectors from $t=1$ to $t=6$.

---

Here are the detailed, highly verbose backpropagation equations for the **CNN with Self-Attention Back-End**.

In the Transformer architecture, operations are processed as full matrices rather than iteratively through time-steps like in the GRU. We define the intermediate variables for the Position-wise Feed-Forward Network (FFN) and the Multi-Head Attention (MHA) to make the derivatives clear.

Let $d_{ff}$ be the hidden dimension of the FFN layer. Let $H_{\text{FFN}} = O_{\text{MHA}} W_1 + b_1$ be the pre-activation FFN matrix, and $A_{\text{FFN}} = \operatorname*{ReLU}(H_{\text{FFN}})$ be the post-activation matrix.

### CNN with Self-Attention Back-End

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{Attn}}} = \operatorname*{Reshape}\left( \frac{\delta L_{\text{BCE}}}{\delta O_{\text{FC}}}, (6, 2) \right)
$$

Note:

1. $O_{\text{Attn}}$ is the final output of the Self-Attention back-end.
2. $\operatorname*{Reshape}(\cdot)$ reshapes the 12x1 gradient vector propagated from the Classifier back into a 6x2 matrix (6 time-steps, 2 features).

_(Starting with the Position-wise Feed-Forward Network Block)_

$$
\frac{\delta L_{\text{BCE}}}{\delta W_2} = \frac{\delta L_{\text{BCE}}}{\delta O_{\text{Attn}}} \times \frac{\delta O_{\text{Attn}}}{\delta W_2} \\
= A_{\text{FFN}}^T \times \frac{\delta L_{\text{BCE}}}{\delta O_{\text{Attn}}}
$$

Note:

1. $\frac{\delta L_{\text{BCE}}}{\delta W_2}$ (shape $d_{ff}$x2) is the gradient for the second linear transformation weight matrix in the FFN.
2. $A_{\text{FFN}}^T$ (shape $d_{ff}$x6) is the transposed post-activation matrix.

$$
\frac{\delta L_{\text{BCE}}}{\delta b_2} = \sum_{\text{rows}} \left( \frac{\delta L_{\text{BCE}}}{\delta O_{\text{Attn}}} \times \frac{\delta O_{\text{Attn}}}{\delta b_2} \right)\\
= \sum_{\text{rows}} \left( \frac{\delta L_{\text{BCE}}}{\delta O_{\text{Attn}}} \right)
$$

Note:

1. $\frac{\delta L_{\text{BCE}}}{\delta b_2}$ (shape 1x2) is the gradient for the second bias vector. Since the bias is added to every row (time-step), its gradient is the sum of the incoming gradients across all 6 rows.

$$
\frac{\delta L_{\text{BCE}}}{\delta A_{\text{FFN}}} = \frac{\delta L_{\text{BCE}}}{\delta O_{\text{Attn}}} \times \frac{\delta O_{\text{Attn}}}{\delta A_{\text{FFN}}} \\
= \frac{\delta L_{\text{BCE}}}{\delta O_{\text{Attn}}} W_2^T
$$

Note:

1. $\frac{\delta L_{\text{BCE}}}{\delta A_{\text{FFN}}}$ (shape 6x$d_{ff}$) is the gradient passed back to the activated hidden layer of the FFN.
2. $W_2^T$ is the transposed weight matrix of shape 2x$d_{ff}$.

$$
\frac{\delta L_{\text{BCE}}}{\delta H_{\text{FFN}}} = \frac{\delta L_{\text{BCE}}}{\delta A_{\text{FFN}}} \odot \frac{\delta A_{\text{FFN}}}{\delta H_{\text{FFN}}} \\
= \frac{\delta L_{\text{BCE}}}{\delta A_{\text{FFN}}} \odot \mathbb{I}_{\mathbb{Z}^+}(H_{\text{FFN}})
$$

Note:

1. $\frac{\delta L_{\text{BCE}}}{\delta H_{\text{FFN}}}$ (shape 6x$d_{ff}$) is the gradient of the pre-activation hidden layer.
2. $\mathbb{I}_{\mathbb{Z}^+}(\cdot)$ is the indicator function for the derivative of ReLU, evaluating to 1 where $H_{\text{FFN}} > 0$ and 0 otherwise.

$$
\frac{\delta L_{\text{BCE}}}{\delta W_1} = \frac{\delta L_{\text{BCE}}}{\delta H_{\text{FFN}}} \times \frac{\delta H_{\text{FFN}}}{\delta W_1} \\
= O_{\text{MHA}}^T \times \frac{\delta L_{\text{BCE}}}{\delta H_{\text{FFN}}}
$$

Note:

1. $\frac{\delta L_{\text{BCE}}}{\delta W_1}$ (shape 2x$d_{ff}$) is the gradient for the first linear transformation weight matrix.

$$
\frac{\delta L_{\text{BCE}}}{\delta b_1} = \sum_{\text{rows}} \left( \frac{\delta L_{\text{BCE}}}{\delta H_{\text{FFN}}} \times \frac{\delta H_{\text{FFN}}}{\delta b_1} \right) \\
= \sum_{\text{rows}} \left( \frac{\delta L_{\text{BCE}}}{\delta H_{\text{FFN}}} \right)
$$

Note:

1. $\frac{\delta L_{\text{BCE}}}{\delta b_1}$ (shape 1x$d_{ff}$) is the gradient for the first bias vector.

_(Transitioning to the Multi-Head Attention Block)_

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{MHA}}} = \left(\frac{\delta L_{\text{BCE}}}{\delta O_{\text{Attn}}} \times \frac{\delta O_{\text{Attn}}}{\delta O_{\text{MHA}}}\right)_{\text{residual}} + \left(\frac{\delta L_{\text{BCE}}}{\delta H_{\text{FFN}}} \times \frac{\delta H_{\text{FFN}}}{\delta O_{\text{MHA}}}\right)_{\text{FFN}} \\
= \frac{\delta L_{\text{BCE}}}{\delta O_{\text{Attn}}} + \left( \frac{\delta L_{\text{BCE}}}{\delta H_{\text{FFN}}} W_1^T \right)
$$

Note:

1. $\frac{\delta L_{\text{BCE}}}{\delta O_{\text{MHA}}}$ (shape 6x2) is the total gradient accumulating at the output of the MHA block.
2. Because of the residual connection around the FFN ($O_{\text{Attn}} = O_{\text{MHA}} + \text{FFN}(O_{\text{MHA}})$), the chain rule demands that we sum the direct gradient bypassing the FFN and the gradient flowing _through_ the FFN.

Let $C_{\text{heads}} = \text{Concat}(\text{head}_1, \text{head}_2)$ be the 6x2 concatenated matrix from the two attention heads.

$$
\frac{\delta L_{\text{BCE}}}{\delta W^O} = \frac{\delta L_{\text{BCE}}}{\delta O_{\text{MHA}}} \times \frac{\delta O_{\text{MHA}}}{\delta W^O} \\
= C_{\text{heads}}^T \times \frac{\delta L_{\text{BCE}}}{\delta O_{\text{MHA}}}
$$

Note:

1. $\frac{\delta L_{\text{BCE}}}{\delta W^O}$ (shape 2x2) is the gradient for the final output projection weight matrix of the MHA block.

$$
\frac{\delta L_{\text{BCE}}}{\delta C_{\text{heads}}} = \frac{\delta L_{\text{BCE}}}{\delta O_{\text{MHA}}} \times \frac{\delta O_{\text{MHA}}}{\delta C_{\text{heads}}} \\
= \frac{\delta L_{\text{BCE}}}{\delta O_{\text{MHA}}} (W^O)^T
$$

Note:

1. $\frac{\delta L_{\text{BCE}}}{\delta C_{\text{heads}}}$ (shape 6x2) is the gradient flowing back into the concatenated heads.
2. This 6x2 matrix is split vertically into two 6x1 matrices: $\frac{\delta L_{\text{BCE}}}{\delta \text{head}_1}$ and $\frac{\delta L_{\text{BCE}}}{\delta \text{head}_2}$, which are routed to their respective attention head calculations.

_(Inside each Attention Head $i$ where $i \in \{1, 2\}$)_

Let $S_i = \frac{Q_i K_i^T}{\sqrt{d_k}}$ be the 6x6 scaled attention score matrix before softmax, and let $P_i = \text{softmax}(S_i)$ be the 6x6 attention probability matrix.

$$
\frac{\delta L_{\text{BCE}}}{\delta V_i} = \frac{\delta L_{\text{BCE}}}{\delta \text{head}_i} \times \frac{\delta \text{head}_i}{\delta V_i} \\
= P_i^T \times \frac{\delta L_{\text{BCE}}}{\delta \text{head}_i}
$$

Note:

1. $\frac{\delta L_{\text{BCE}}}{\delta V_i}$ (shape 6x1) is the gradient for the Value matrix of head $i$.

$$
\frac{\delta L_{\text{BCE}}}{\delta P_i} = \frac{\delta L_{\text{BCE}}}{\delta \text{head}_i} \times \frac{\delta \text{head}_i}{\delta P_i} \\
= \frac{\delta L_{\text{BCE}}}{\delta \text{head}_i} V_i^T
$$

Note:

1. $\frac{\delta L_{\text{BCE}}}{\delta P_i}$ (shape 6x6) is the gradient for the Softmax probabilities. $V_i^T$ has shape 1x6.

$$
\frac{\delta L_{\text{BCE}}}{\delta S_i} = \frac{\delta L_{\text{BCE}}}{\delta P_i} \times \frac{\delta P_i}{\delta S_i} \\
= P_i \odot \left( \frac{\delta L_{\text{BCE}}}{\delta P_i} - \sum_{\text{cols}} \left( \frac{\delta L_{\text{BCE}}}{\delta P_i} \odot P_i \right) \right)
$$

Note:

1. $\frac{\delta L_{\text{BCE}}}{\delta S_i}$ (shape 6x6) is the gradient for the pre-softmax attention scores.
2. Because softmax is applied row-wise, the Jacobian derivative requires us to subtract the sum of the elements across the columns (row sum) from the incoming gradient, element-wise multiplied by the softmax probabilities themselves.

$$
\frac{\delta L_{\text{BCE}}}{\delta Q_i} = \frac{\delta L_{\text{BCE}}}{\delta S_i} \times \frac{\delta S_i}{\delta Q_i} \\
= \frac{1}{\sqrt{d_k}} \left( \frac{\delta L_{\text{BCE}}}{\delta S_i} K_i \right)
$$

Note:

1. $\frac{\delta L_{\text{BCE}}}{\delta Q_i}$ (shape 6x1) is the gradient for the Query matrix of head $i$.
2. $d_k = 1$ based on the network assumptions.

$$
\frac{\delta L_{\text{BCE}}}{\delta K_i} = \frac{\delta L_{\text{BCE}}}{\delta S_i} \times \frac{\delta S_i}{\delta K_i} \\
= \frac{1}{\sqrt{d_k}} \left( \left(\frac{\delta L_{\text{BCE}}}{\delta S_i}\right)^T Q_i \right)
$$

Note:

1. $\frac{\delta L_{\text{BCE}}}{\delta K_i}$ (shape 6x1) is the gradient for the Key matrix of head $i$. Note the transpose on the incoming score gradient matrix.

$$
\frac{\delta L_{\text{BCE}}}{\delta W_i^Q} = O_{\text{Cc}}^T \times \frac{\delta L_{\text{BCE}}}{\delta Q_i} \quad ; \quad \frac{\delta L_{\text{BCE}}}{\delta W_i^K} = O_{\text{Cc}}^T \times \frac{\delta L_{\text{BCE}}}{\delta K_i} \quad ; \quad \frac{\delta L_{\text{BCE}}}{\delta W_i^V} = O_{\text{Cc}}^T \times \frac{\delta L_{\text{BCE}}}{\delta V_i}
$$

Note:

1. These are the gradients for the Query, Key, and Value linear projection weight matrices for head $i$. All have shape 2x1. $O_{\text{Cc}}^T$ has shape 2x6.

_(Final Gradient passed to the CNN Front-End)_

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}}} = \left( \frac{\delta L_{\text{BCE}}}{\delta O_{\text{MHA}}} \right)_{\text{residual}} + \sum_{i=1}^2 \left( \left[ \frac{\delta L_{\text{BCE}}}{\delta Q_i} \times \frac{\delta Q_i}{\delta O_{\text{Cc}}} \right] + \left[ \frac{\delta L_{\text{BCE}}}{\delta K_i} \times \frac{\delta K_i}{\delta O_{\text{Cc}}} \right] + \left[ \frac{\delta L_{\text{BCE}}}{\delta V_i} \times \frac{\delta V_i}{\delta O_{\text{Cc}}} \right] \right) \\
= \frac{\delta L_{\text{BCE}}}{\delta O_{\text{MHA}}} + \sum_{i=1}^2 \left( \frac{\delta L_{\text{BCE}}}{\delta Q_i} (W_i^Q)^T + \frac{\delta L_{\text{BCE}}}{\delta K_i} (W_i^K)^T + \frac{\delta L_{\text{BCE}}}{\delta V_i} (W_i^V)^T \right)
$$

Note:

1. $\frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}}}$ (shape 6x2) is the total accumulated gradient passed back to the Front-End concatenation layer.
2. Because $O_{\text{Cc}}$ branches out into the residual connection, and into the $Q_i, K_i, V_i$ matrices for _both_ heads, the chain rule dictates that the total gradient is the sum of the gradients propagated backward from all 7 pathways (1 residual + 3 matrices $\times$ 2 heads).
