# Identifikasi Permasalahan

This experiment aims to automate the labeling of emotion and theme on music. Music files or digital audio files can be used as input for such systems. However, discussion in chapter 2.2.1 have shown that converting digital audio into a spectrogram is more beneficial for deep neural networks. In signal processing, digital audio consists of discrete values at each time-step based on the sample rate. Said discrete values take into account the amplitude, frequency, and the offset of the sinusoid. Time in digital audio is based on the sample rate or at what discrete fraction of a second the signal is recorded. For an MP3 file, the usual sample rate is 44100 Hz, which means that there are 44100 discrete values in a second. These values can be represented as an array where the indices of such an array is the time-step.

Spectrograms are commonly used to represent digital audio in speech recognition and MER. In this case, the kind of spectrogram used is the log-mel spectrogram. Converting to a log-mel spectrogram requires calculating the STFT. STFT calculates DFT for short frames with length of some arbitrary fraction of the sample rate. STFT assumes that the signal does not or is not likely to repeat. DFT assumes the opposite, therefore it is not viable to be applied to digital audio. STFT takes the parameters: sample rate, frame length, hop length, and the number of mel bands. The output is a 2D matrix where each row represents the mel band and each column represents the time-frame. The data in each mel band at a time-frame is the magnitude in dB. The magnitudes are then scaled logarithmically.

The data provided in the MTG-Jamendo dataset consists of metadata and digital audio. Digital audio files and metadata are downloaded separately. Table X.X shows the first five rows of song metadata. The dataset consists of 55.609 audio files which has been preprocessed by the authors as discussed in chapter 3.2.2. The dataset to be used in this experiment is the "mood/theme" subset which only has 18.486 files. There is no identifying information for each audio file such that training does not exhibit the artist and album effects (bogdanov 2019 mtg jamendo). The training, validation, and testing splits for this subset has been set up by the authors of the dataset. The approximate split is 60% for training and 20% for validation and testing respectively. The split is random but it is ensured that no track appears in more than one set and no tracks in any set are from the same artist present in other sets, all labels are present in all three splits, and each label in each split is represented by at least 40 files and 10 artist in the training split and 20 files and 5 artists in the validation and testing split respectively. During training, validation, and inference, the data is split to 15 second chunks for the entire song duration.

Table X.X First five rows of song metadata (image in Google Docs)

The three back-ends of the experiment is the variable aspect which affects the PR-AUC and ROC-AUC score. Comparison of different back-ends is designed to be fair in terms of the model's parameter count. The baseline of the three is the CNN back-end as proposed by Pons et al (2018). It consists of 3 layers of convolution with one max-pooling inserted between the 2nd and 3rd layer where each layer has 64 filters. The other two back-ends share the same number of layers but without the max-pooling layer in between. The parameters for other back-ends are set to be similar with a maximum standard deviance of 5% with the CNN back-end similar to the setup by Shim dan Sung (2022).

# Perancangan Algoritme

The flow of the algorithm as shown in Gambar X.1 consists of three main steps: preprocessing, spectrogram calculation, and modeling. Preprocessing has already been discussed in chapter 3.2.4. After calculating STFT, the spectrogram output is then normalised using z-score normalisation. Without batching, all 15-second splits of the audio for each training file will be used because each model can accept variable time lengths. However, since batching is used in the implementation, the last split of the training audio is discarded because batching requires same time length for all inputs in the batch. The batch size is 32. This reflects past literature like Choi et al. (2016conv) and Pons et al. (2018) which uses the default batch size of 32 provided by TensorFlow. The simplified networks assumes no use of mini-batching and thus batch and layer normalisation. The exampled dataset for the modeling stage will consist of 5 rows of arbitrary data sampled from the subset which consists of 3 arbitrary labels: happy, sad, and tense. The example dataset is shown in Table X.X. Log-mel spectrogram amplitudes in each frequency bin were observed in preliminary data exploration to be around -80 and 10 (in dB). The example dataset reflects this range.

Gambar X.1 \<FLOWCHART 1\> (drawio:perancangan1)

Table X.X \<EXAMPLE DATASET\>

Gambar X.2 \<FLOWCHART PELATIHAN DETAIL\> (drawio:pelatihan1)

In helping during training, the MTG-Jamendo dataset provides an official split for training, validation, and testing. The split is 60%, 20%, and 20% respectively. Gambar X.1 and Gambar X.2 illustrate when the each of the dataset split is used. Notably, the validation set is used after one epoch of training has finished. Albeit this was meant for model experimentation in the literature that used this dataset, the validation set is to be used to determine model checkpoints and limit epochs. The training concludes when the maximum number of epochs has been reached. The best performing checkpoints within those models are loaded and tested. The results are then compared and analysed as per the research question. Similar publications by Pons et al. (2018) and Choi et al. (2016conv) which is the basis for this experiment uses a learning rate of 0.001 and 0.005 respectively. Similar research using CNN from Choi et al. (2016automatic) show diminishing returns after 40 epochs using the ADAM optimizer. Both observe the performance of the model per epoch and end training based on an arbitrary decision. For attention models, Sukhavasi and Adapa (2019) limits to 60 epochs for ADAM with a learning rate of 0.001 before implementing learning rate adjustments using other methods. Won et al. (2019toward) agrees with maximum number of epochs and set the learning rate to 0.0001. The example calculation and the implementation of this experiment will use the default ADAM learning rate of 0.001. The implementation will limit the epochs to 60.

# Front-End

Gambar X.X (drawio:cnn-fe)

<!-- // FLESH THIS SECTION OUT TO BE A COMPARISON TABLE AND NOT JUST BULLET POINTS AND NUMBERED LIST

- Number of filters: 2
- Input is split to 2 types: for the vertical and horizontal filters
- For the vertical filter type: max-pool AFTER convolution
- For the horizontal filter type: mean-pool BEFORE convolution
- Filter 1 (namely $F_V$) size: 5x3 (vertical)
- Filter 2 (namely $F_H$) size: 1x3 (horizontal) -->

Forward (vertical filter) assumption is as follows:

1. Input: 6x5 (HxW) where H is time and W is frequency. The filter size is 5x3 (vertical).
2. Number of channels: 1. This is because spectrogram consists of one scale of colors and not 3 like RGB image.
3. Convolution with the vertical filter with "same" padding.
4. Uses ReLU after convolution.
5. Max-pool the result.

Forward (horizontal filter) assumption is as follows:

1. Input: 6x5 (HxW) where H is frequency and W is time. The filter size is 1x3 (horizontal).
2. Number of channels: 1. This is because spectrogram consists of one scale of colors and not 3 like RGB image.
3. Mean-pool the spectrogram input such that the output dimension has a height of 1 and a width of time.
4. Convolution with horizonal filters with "same" padding.
5. Uses ReLU after convolution.

Gambar X.X visualises the front-end network assumption. Note that in the vertical filter feed-forward, H is time while W is frequency but in the horizontal filter feed-forward, H is frequency while W is time. This mimics time series data modeling where tabular representation has time (as a column) where each row denotes the values at that time. The values of each features are the columns or the width, assuming HxW representation. The rationale is also described in Pons et al. (2018) where the filters are to learn features across the time axis. Both filters are reducing the dimensionality of the frequency while keeping the time axis intact. This always results in something akin to time series data. The result of vertical and horizontal convolutions are then concatenated to become a tensor with the time axis intact and the frequency bin axis reduced in dimensionality. This concatenation is the reason of separating the feed-forward for the front-end based on the filter shape.

<!-- --- -->

<!-- // ADD VISUAL ILLUSTRATIONS FOR THE FEED-FORWARD OF VERTICAL AND HORIzONTAL FILTERS -->

## Feed-Forward of Vertical Filter Convolution

Given an input size of 6x5 with "same" padding, the output shape of the convolution is the same as the input shape. "Same" padding is also known as half padding. The amount of padding needed to achieve this is described in Domoulin (2018) where:

$$p=\left\lfloor{\frac{k}{2}}\right\rfloor$$

Note:

1. p is the amount of padding to be applied to each boundary
2. k is the size of the kernel assuming a square kernel

With a 2D rectangular kernel, the equation can be expanded to be:

$$p_H=\left\lfloor{\frac{k_H}{2}}\right\rfloor$$
$$p_W=\left\lfloor{\frac{k_W}{2}}\right\rfloor$$

With a kernel size of 5x3 with "same" padding, the amount of padding on the input matrix is as follows:

$$p_H=\left\lfloor{\frac{5}{2}}\right\rfloor=2$$
$$p_W=\left\lfloor{\frac{3}{2}}\right\rfloor=1$$

Visualised, the padded input matrix is shown in Gambar X.X

Gambar X.X \<PADDED MATRIX\> (drawio:paddedmatrix)

Note:

1. p_H is the amount of padding to be applied on the height boundaries (top and bottom) of the input matrix
2. k_H is the height of the kernel
3. p_W is the amount of padding to be applied on the width boundaries (left and right) of the input matrix
4. k_W is the width of the kernel

The max-pooling operation is not padded but has a kernel and stride size of 1x5 where 5 is the width of the frequency bin axis. Max-pooling takes a sliding window which strides to capture a portion of the matrix based on the kernel size. The operation finds the maximum value in that portion of the matrix. The operation is mathematically defined as follows:

$$
\text{MaxPool}(I)\\
=\max_{m=0,\dots,k_H-1}\max_{n=0,\dots,k_W-1}\text{in}(\text{stride}[0]\times h+m,\text{stride}[1] \times w+n)
$$

Note:

1. I is the input matrix (or vector) to be max-pooled
1. $\text{in}(\cdot)$ means the input matrix of max-pooling
1. stride is a tuple with elements on the 0th and 1st index which signifies how much the window slides over all locations in the input
1. h is the height of the input matrix and as the parameter of the output function
1. w is the width of the input matrix and as the parameter of the output function
1. m is the row index of the input matrix
1. n is the column index of the input matrix
1. k_H is the height of the kernel
1. k_W is the width of the kernel

The operation has output shape mathematically defined as follows:

$$H_{out}=\left\lfloor \frac{H_{in}+2*\text{padding}[0]-\text{dilation}[0]\times(\text{k}[0]-1)-1}{\text{stride}[0]}+1 \right\rfloor$$

Note:

1. H_out is the height of the output
2. H_in is the height of the input
3. padding[0] means the first index of the padding tuple or how much is the input padded on the top and bottom
4. dilation[0] means the first index of the dilation tuple or the rate of which the size of the kernel increases with which elements of the input are also skipped. Dilation value set to 1 is the same as not applying any dilation.
5. k[0] means the first index of the k tuple or the size of the kernel used as the window to calculate the max value at a given window
6. stride[0] means the first index of the stride tuple or how much the window slides over all locations in the input

$$W_{out}=\left\lfloor \frac{W_{in}+2*\text{padding}[1]-\text{dilation}[1]\times(\text{k}[1]-1)-1}{\text{stride}[1]}+1 \right\rfloor$$

Note:

1. W_out is the width of the output
2. W_in is the width of the input
3. padding[1] means the second index of the padding tuple
4. dilation[1] means the second index of the dilation tuple
5. k[1] means the second index of the k tuple
6. stride[1] means the second index of the stride tuple

Therefore, the shape of the output after the max-pooling operation is shown to be of shape 6x1 as calculated: (2 separate eqs)

$$H_{out}=\left\lfloor \frac{6+2*0-1\times(1-1)-1}{1}+1 \right\rfloor=6$$

$$W_{out}=\left\lfloor \frac{5+2*0-1\times(5-1)-1}{5}+1 \right\rfloor=1$$

Putting it all together, the final shape (including the channel axis) of the vertical filter feed-forward is 1x6x1 with dimensions: channel, time (height), and frequency (width).

## Feed-Forward of Horizontal Filter Convolution

Given an input shape of 6x5, the mean-pooling operation is not padded but has a kernel and stride size of 1x5 where 5 is the width of the frequency bin axis. In a similar fashion to max-pooling, mean-pooling takes the average of all values in the portion of the matrix. The operation has output shape is the same as defined in max-pooling. Therefore, the input of the convolution has a shape of 1x6. The operation is mathematically defined as follows:

$$
\text{MeanPool}(I)\\
=\frac{1}{k_H+k_W}\sum_{m=0}^{k_H-1}\sum_{n=0}^{k_W-1}\text{in}(\text{stride}[0]\times h+m,\text{stride}[1] \times w+n)
$$

Notes:

1. k_H is the height of the kernel
2. k_W is the width of the kernel
3. The other variables have similar notes to the max-pooling operation defined in Persamaan X.X (\<POINT TO THE MAXPOOL OP ABOVE\>).

The output of mean-pooling is the input of the convolution layer with horizontal filters. The convolution now becomes a 1D convolution where the frequency bins have been averaged per time-frame. However, to accomodate the input shape of PyTorch Conv1d class which are: channel and length, the frequency dimension is dropped while the length dimension becomes the time axis. This appropriates the time axis as the sequence's length dimension which consists of 6 values of the averaged frequeny bins. Moreover, this convolution has "same" padding, therefore the output size is the same as the input size, which is 1x6 (an array of 6 values). The operation has output shape mathematically defined as follows:

$$L_{out}=\left\lfloor \frac{L_{in}+2*\text{padding}-\text{dilation}\times(\text{kernel\_size}-1)-1}{\text{stride}}+1 \right\rfloor$$

Note:

1. L_out is the length of the sequence output
2. L_in is the length of the sequence input
3. padding means how much is the input padded on each ends of the sequence
4. dilation means the first index of the dilation tuple
5. kernel_size means the first index of the kernel_size tuple
6. stride means the first index of the stride tuple

Since the size of the kernel is defined to be two dimensional (1x3), the values used for kernel_size and stride is the larger among the two. This is because 1D convolution essentially convolves the data through its length rather of its height assuming the data is akin to an array. Therefore, the shape of the output after the convolution operation is shown to have the length of 6 as calculated:

$$L_{out}=\left\lfloor \frac{6+2*1-1\times(3-1)-1}{1}+1 \right\rfloor=6$$

Putting it all together, the final shape (including the channel axis) of the horizontal filter feed-forward is 1x6 with dimensions: channel and time (height)

## Concatenation of Vertical and Horizontal Filter Convolutions

The concatenation process takes the output of both the vertical and horizontal filter convolutions. Since temporal convolution uses 1D convolution, a new dimension can be added after the time dimension to replace the frequency dimension that was dropped before the convolution. Therefore, the input shapes of both the convolutions are 1x6x1, therefore the resulting concatenation will have shape 1x6x2 which keeps the length of the time axis intact. The order of concatenation (assuming the matrix is read left to right) is the vertical and then the horizontal filter convolution. The dimensions are: channel, time, and frequency.

# Back-End

## CNN

The CNN back-end consists of 1 convolution layer each with 1 filter of shape 3xW. W is the width of the concatenated features; from the results of as shown in CHAPTER X.Y.Z, it is 2. This back-end accepts the input shape 6x2 from the front-end. The height of the kernel is the time axis while the width is the concatenated feature axis. This convolution layer uses the "same" padding. The activation function for both layers is ReLU. The output of this back-end is the concatenated filters after ReLU.

Gambar X.X (drawio:cnn-be)

## CNN with GRU

The CNN with GRU back-end consists of 1 layer of uni-directional GRU with one hidden layer. This back-end accepts the input shape 6x2 from the front-end. This design is in line with Cho et al. (2014). The input tensor is adapted to be of shape LxH_in where L is the sequence length or the height of the time axis and H_in is the width of the concatenated features; from the results of as shown in CHAPTER X.Y.Z, it is 2. The output of this back-end is the final hidden state of GRU with shape LxH_out. H_out is set to 2. The output of this back-end returns the features from the last hidden state for each time-step. The shape of it is the same as the input.

Gambar X.X (drawio:cnn-gru-be)

## CNN with Self-Attention

The CNN with Self-Attention back-end consists of 1 layer of Self-Attention with 2 heads. Self-Attention is implemented as part of the encoder part of the Transformer. The transformer takes input an embedding vector. Given the output of the front-end is of shape 6x2, every time-step of it can be treated as 6 embedding vectors each with shape 1x2. In calculation, this is done simultaneously. The dimension of query, keys, and value is determined by Persamaan 2.12, where d_model equals 2 because the width of the concatenated feauters are 2 as shown in CHAPTER X.Y.Z and h (or head) equals 2. Therefore, d_k and d_v equals 1. Given there are two heads, this means that each head gets input of shape 6x1. After calculating Self-Attention, the result from each head is concatenaed to be 6x2 again to continue to the Feed-Forward Network (FFN) layer of the Transformer. The activation function in the feed-forward network portion of the architecture is ReLU. The output of this back-end.

Gambar X.X (drawio:cnn-attn-be)

# Classifier

The classifier consists of a FC layer which takes 12 nodes as its input. The back-end output will be concatenated to be of shape 12x1. The output becomes the input of the output layer with 3 nodes corresponding to the three arbitrary labels set for this calculation as per Tabel X.X (\<DUMMY DATASET\>). The activation function for the FC and output layer is sigmoid.

Gambar X.X (drawio:classifier)

<!-- This is basically Manualisasi -->

# Perumusan Feed-Forward dan Backpropagation

---

// ADD THIS IN TINJAUAN TEORI NOT HERE

In deep learning, convolution is usually implemented as cross-correlation. Convolution flips the kernel while cross-correlation does not (Goodfellow 2016). Cross-correlation (deep learning convolution) is mathematically defined as:

$$S(i,j)=\sum_{m=0}^{M-1}\sum_{n=0}^{N-1}{I(i+m,j+n)\times K(m,n)}$$

Note:

1. i and j are respectively the row and column indices of the input, both start at 0.
2. m and n are respectively the row and column indices of the kernel
3. M and N are respectively the height and width of the kernel
4. S(i,j) is the result of convolution at index i and j of the input
5. I(i+m,j+n) is respectively the row and column indices of the input
6. K(m,n) is respectively the row and column indices of the kernel

---

The simplified network to be used for the full numerical calculation is the CNN model. The definitions of feed-forward and backpropagation for the CNN with GRU model and the CNN with Self-Attention model is still provided. The CNN model consists of the CNN front-end, the CNN back-end, and the classifier. The loss function is defined in Persamaan 2.24 based on the prediction values. To ease understanding, Gambar X.X visualises each variable that are part of the CNN model. In the mathematical definitions in CHAPTER X.Y.Z (\<THE TWO SUBCHAPTERS FOR FF AND BPROP BELOW\>), the multiplication sign is used to denote scalar multiplication. If it is not present, then it denotes matrix multiplication.

## Perumusan Feed-Forward

The feed-forward equations are defined from the weighted BCE loss function towards the input. Some equation notes are incomplete because it has been noted in previous equations.

GAMBAR X.X \<VIS CNN TAPI ADA VARIABLE VARIABLE\> (drawio:cnn-ff-bprop)

### Classifier

$$y_n=\sigma(z_O)$$

Note:

1. y_n is the predicted output
2. $\sigma$ is the sigmoid activation function
3. z_O is the output of the output layer before applying the activation function as visualised in Gambar X.X

$$z_O=O_{\text{FC}}^T W_O + b_O$$

Note:

1. O_FC is the output of the "FC" layer
2. W_FC is the weights for the "FC" layer
3. b_FC is the biases for the "FC" layer

$$O_{\text{FC}} = \text{Flatten}(O_{\text{c}})$$

Note:

1. $O_{\text{c}}$ is the output of the second convolution layer in the back-end
2. Flatten(.) is a row-wise operation that reshapes the 2D matrix output of $O_{\text{c}}$ of shape 6x2 into a 1D vector of shape 12x1.

### CNN Back-End

$$O_{\text{c}}=\operatorname*{ReLU}(z_{\text{c}})$$

Note: $z_{\text{c}}$ is the output of the second convolution layer (denoted c) in the back-end before applying the activation function

The following re-purposes the mathematical definition of the convolution operation as defined in Persamaan 2.X (DI BAB 2) for use in this case:

$$
z_{\text{c}}=S_{\text{c}}(i_{\text{c}},j_{\text{c}}) + b_{c} \\
=\sum_{m_{\text{c}}=0}^{M_{\text{c}}-1}\sum_{n_{\text{c}}=0}^{N_{\text{c}}-1} \big(O_{\text{Cc}}(i_{\text{c}}+m_{\text{c}},j_{\text{c}}+n_{\text{c}})\times K_{\text{c}}(m_{\text{c}},n_{\text{c}}) \big) + b_{c}
$$

Note: similar to Persamaan 2.X; I defined in Persamaan 2.X is replaced with O_Cc which is the output of the first convolution layer in the back-end.

$$
O_{\text{Cc}}=\text{Concat}(O_{\text{CV}},O_{\text{CH}})
$$

Note:

1. Concat(.) is a column-wise concatenation function given two inputs O_CV and O_CH
2. O_CV is the output of the vertical convolution from the front-end; conversely, O_CH is the output of the horizontal convolution.

### CNN with GRU Back-End

The operations repurpose the mathematical definition of the GRU proposed by Cho et al. (2014):

$$
O_{\text{FC}} = \operatorname*{Flatten}(O^T_{\text{GRU}}) \\

O_{\text{GRU}}=\operatorname*{Concat}(h_1,h_2,\dots,h_L)
$$

Note:

1. $L=6$ because there are 6 time-steps in the data.
2. Concat(.) concatenates column-wise.
3. The output of the GRU layer is the hidden states at all $t$. Each is transposed (see Gambar X.X (drawio:cnn-gru-be)) then flattened. In this case, the output is 6x2 then flattened to 12x1.

$$h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t$$

Note:

1. $t \in \{1, 2, 3, 4, 5, 6\}$ because there size of the input matrix is 6x2 where 6 is the time axis. There are 6 time-steps.
2. $h_t$ is the final hidden state of the GRU at time-step $t$.
3. $\odot$ denotes element-wise multiplication (Hadamard product).
4. $h_0$ is initialized as an all-zero vector.

$$\tilde{h}_t = \tanh(W_h  O_{\text{Cc}(t)} + U_h (r_t \odot h_{t-1}) + b_h)$$

Note:

1. $\tilde{h}_t$ is the candidate hidden state at time-step $t$.
2. $O_{\text{Cc}(t)}$ is the $t$-th row of the front-end concatenation output $O_{\text{Cc}}$ representing the features at time-step $t$.
3. $W_h$ and $U_h$ are the weight matrices for the candidate hidden state, and $b_h$ is the bias. In Cho et al. (2014), this is denoted as just $W$.

$$z_t = \sigma(W_z O_{\text{Cc}(t)}+ U_z h_{t-1} + b_z)$$
$$r_t = \sigma(W_r O_{\text{Cc}(t)}+ U_r h_{t-1} + b_r)$$

Note:

1. $z_t$ is the update gate vector at time-step $t$.
2. $r_t$ is the reset gate vector at time-step $t$.
3. $W_z, U_z, W_r, U_r$ are the respective weight matrices for the update and reset gates, while $b_z, b_r$ are their respective biases.

### CNN with Self-Attention Back-End

The operations repurpose the mathematical definition of the Transformer encoder proposed by Vaswani et al. (2017):

$$O_{\text{FC}} = \text{Flatten}(O_{\text{Attn}})$$

Note: $O_{\text{Attn}}$ is the final 6x2 output matrix of the Self-Attention back-end.

The Position-Wise FFN layer as proposed in Vaswani et al. (2017) consist of two layers. The first layer and the second layer have different weights and biases. The first layer is applied ReLU as the activation function while the second does not get activated. As noted in the paper, the dimensionality of the input and output of the FFN is based on d_model, in this case it is 2. The inner layer of the FFN takes the parameter d_ff or the dimension of the feed-forward, in this case it is also set to be 2. Gambar X.X (drawio:cnn-attn-be) visualises the calculations being done inside of the FFN.

$$
O_{\text{Attn}} = O_{\text{MHA}} + \operatorname*{ReLU} (O_{\text{MHA}} W_1 + b_1) W_2 + b_2
$$

Note:

1. $O_{\text{Attn}}$ is the Position-Wise FFN with a residual connection to O_MHA or the output of the Multi-Head Attention (MHA) block.
2. $W_1, b_1$ are the weights and bias for the first linear transformation.
3. $W_2, b_2$ are the weights and bias for the second linear transformation.

$$O_{\text{MHA}} = O_{\text{Cc}} + \text{Concat}(\text{head}_1, \text{head}_2) W^O$$

Note:

1. $O_{\text{MHA}}$ is the output of the MHA block with a residual connection adding the original input $O_{\text{Cc}}$.
2. $\text{head}_1$ and $\text{head}_2$ are the outputs of the two individual attention heads.
3. $W^O$ is the output projection weight matrix.
4. $\text{Concat}(\cdot)$ concatenates the two heads along the feature dimension.

$$\text{head}_i = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i$$

Note:

1. This is the scaled dot product attention calculation for the $i$-th head, where $i \in \{1, 2\}$.
2. $d_k$ is the dimension of the keys (defined as 1 in the network assumptions).
3. The $\text{softmax}$ function is applied row-wise.

$$Q_i = O_{\text{Cc}} W_i^Q$$

$$K_i = O_{\text{Cc}} W_i^K$$

$$V_i = O_{\text{Cc}} W_i^V$$

Note:

1. $Q_i, K_i, V_i$ are the query, key, and value matrices for the $i$-th head.
2. $W_i^Q, W_i^K, W_i^V$ are the learned weight matrices for the $i$-th head.

### Vertical Filter CNN Front-End

Note:

1. Concat(.) is the concatenation operation which concatenates column-wise (keeps the length of the row the same)
2. O_CV is the output of the vertical convolution layer (denoted CV) in the front-end
3. O_CH is the output of the horizontal convolution layer (denoted CH) in the front-end

$$O_{\text{CV}} = \text{MaxPool}(\operatorname*{ReLU}(z_{\text{V}}))$$

$$
z_{\text{V}}=S_{\text{V}}(i_{\text{V}},j_{\text{V}}) + b_{\text{V}} \\
=\sum_{m_{\text{V}}=0}^{M_{\text{V}}-1}\sum_{n_{\text{V}}=0}^{N_{\text{V}}-1} \big(I(i_{\text{V}}+m_{\text{V}},j_{\text{V}}+n_{\text{V}})\times K_{\text{V}}(m_{\text{V}},n_{\text{V}}) \big) + b_{\text{V}}
$$

Note:

1. Both Persamaan X.X and Persamaan X.X (\<REFERRING TO THE 2 EQS ABOVE\>) is similar to Persamaan X.X (\<THE OG CONV EQ ABOVE THESE 2\>) just that it is for the vertical convolution layer in the front-end.
2. MaxPool(.) is defined in Persamaan X.X (\<THE MAXPOOL DEF ABOVE\>)
3. $I$ is the log-mel spectrogram input

### Horizontal Filter CNN Front-End

$$O_{\text{CH}}=\operatorname*{ReLU}(z_H)$$

$$
z_{\text{H}}=S_{\text{H}}(i_{\text{H}},j_{\text{H}}) + b_{\text{H}} \\
=\sum_{m_{\text{H}}=0}^{M_{\text{H}}-1}\sum_{n_{\text{H}}=0}^{N_{\text{H}}-1} \big(\text{MeanPool}(I)(i_{\text{H}}+m_{\text{H}},j_{\text{H}}+n_{\text{H}})\times K_{\text{H}}(m_{\text{H}},n_{\text{H}}) \big) + b_{\text{H}}
$$

Note:

1. MeanPool(.) is defined in Persamaan X.X (\<THE MEANPOOL DEF ABOVE\>)
2. Both Persamaan X.X and Persamaan X.X (\<REFERRING TO THE 2 EQS ABOVE\>) is similar to Persamaan X.X (\<THE OG CONV EQ ABOVE THESE 2\>) just that it is for the horizontal convolution layer in the front-end.

## Perumusan Backpropagation

With the definition of feed-forward equations in CHAPTER X.Y.Z (\<THE FF SUBCHAPTER DIRECTLY ABOVE THIS\>), the backpropagation equations can be defined as the partial derivatives with respect to each variable of each feed-forward equation. Some of the values in the definitions have been substituted based on the network assumption discussed in CHAPTER X.Y.Z (\<THE CHAPTER BEFORE THIS\>). Variable subscripts has been visualised in Gambar X.X (\<THE DETAILED NETWORK IMAGE ON THE PREV CHAPTER\>)

### Classifier

$$
\frac{\delta L_{\text{BCE}}}{\delta y_n}=-\frac{1}{N} \times \left(P_w \times \frac{t_n}{y_n} - N_w \times \frac{1-t_n}{1-y_n} \right) \\
=-\frac{1}{3} \times \left(P_w \times \frac{t_n}{y_n} - N_w \times \frac{1-t_n}{1-y_n} \right)
$$

$$P_w=\frac{2}{1+p_n}$$
$$N_w=\frac{2\times p_n}{1+p_n}$$

Note:

1. N is substituted with 3 because there are 3 classes in the network assumption.
2. P_w and N_w is defined as constants to make the definitions concise
3. The variables in Persamaan X.X, Persamaan X.X, and Persamaan X.X (\<THE 3 EQS ABOVE\>) is the same as defined in Persamaan 2.24

$$
\frac{\delta L_{\text{BCE}}}{\delta z_O}=\frac{\delta L_{\text{BCE}}}{\delta y_n} \times \frac{\delta y_n}{\delta z_O} \\
=\frac{\delta L_{\text{BCE}}}{\delta y_n}\times \sigma(z_O) \times (1-\sigma(z_O)) \\
=-\frac{1}{3} \times \left(P_w \times \frac{t_n}{y_n} - N_w \times \frac{1-t_n}{1-y_n} \right) \times \big(y_n \times (1-y_n) \big) \\
=-\frac{1}{3} \times \left(P_w \times t_n \times (1-y_n) - N_w \times (1-t_n) \times y_n \right)
$$

Note: The variables in Persamaan X.X (\<THE 1 EQS ABOVE\>) is the same as defined in Persamaan 2.24

$$
\frac{\delta L_{\text{BCE}}}{\delta W_{O(j,n)}}=\left[\frac{\delta L_{\text{BCE}}}{\delta y_n}\times \frac{\delta y_n}{\delta z_O}\right]_n \times\left[\frac{\delta z_O}{\delta W_{O(j,n)}}\right]_n \\
=\left[\frac{\delta L_{\text{BCE}}}{\delta z_O}\right]_n \times\left[\frac{\delta z_O}{\delta W_{O(j,n)}}\right]_n \\
=\left[\frac{\delta L_{\text{BCE}}}{\delta z_O}\right]_n \times\left[O_\text{FC}\right]_j
$$

Note:

1. n is the n-th class in the dataset as per Persamaan 2.24
2. $W_{O(j,n)}$ is the weight connecting the j-th node of the output-side of FC to the n-th output node (the "O" layer in Gambar X.X).
3. [.]\_n is the n-th gradient. Since there are 3 classes, there will be N calculations of weighted BCE losses as defined in Persamaan X.X (\<THE EQ ABOVE THIS ONE\>).
4. $z_{O(n)}$ is the n-th output after calculations with the weights in the output layer that have not been activated
5. $\left[O_\text{FC}\right]_j$ is the j-th node of the outputside of FC.

$$
\frac{\delta L_{\text{BCE}}}{\delta b_{O(n)}}=\left[\frac{\delta L_{\text{BCE}}}{\delta y_n}\times \frac{\delta y_n}{\delta z_O}\right]_n \times\left[\frac{\delta z_O}{\delta b_O}\right]_n \\
=\left[\frac{\delta L_{\text{BCE}}}{\delta z_O}\right]_n \times 1 \\
$$

Note: $b_{O(n)}$ is the bias for the n-th output ("O" layer) node

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{FC}(i)}}=\sum_{n=1}^3 \left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{O}(n)}} \times \frac{\delta z_{\text{O}(n)}}{\delta O_{\text{FC}(i)}} \right) \\
=\sum_{n=1}^3 \left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{O}(n)}} \times W_{\text{O}(i,n)} \right)
$$

Note:

1. $i=1,2,3,\dots,12$.
2. O_FC(i) is the i-th output of the "FC" layer which is the flattened layers from the back-end
3. The gradient propagated to the $i$-th node of the "FC" layer is the sum from each 3 nodes in the output layer or the "O" layer shown in Gambar X.X.

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{c}}} = \operatorname*{Reshape}\left( \frac{\delta L_{\text{BCE}}}{\delta O_{\text{FC}}}, (6, 2) \right)
$$

Note:

1. O_c is the output of the convolutional layer (the "c" layer) from the back-end
2. Reshape(.) is to reshape a vector to form an array with defined shape as per the second parameter of the function. In this case it is 6x2, which is written (6, 2) as to not be confused with scalar multiplication.

The concatenation function as defined in Persamaan X.X (\<SEE THE CONCAT FORMULA IN PREV SUBCHAPTER\>) means that the propagated gradient has to be reshaped so that it is the same shape of the feed-forward output of "c" layer.

### CNN Back-End

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c}}(i_{\text{c}}, j_{\text{c}})}=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{c}}(i_{\text{c}}, j_{\text{c}})} \times \left[\frac{\delta O_{\text{c}}}{\delta z_{\text{c}}} \right]_{i_{\text{c}}, j_{\text{c}}} \\
=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{c}}(i_{\text{c}}, j_{\text{c}})} \times \mathbb{I}_{\mathbb{Z}^+}(z_{\text{c}}(i_{\text{c}}, j_{\text{c}}))
$$

Note:

1. $\frac{\delta O_{\text{c}}}{\delta z_{\text{c}}}$ is the derivative of the ReLU activation function. $\mathbb{I}_{\mathbb{Z}^+}(\cdot)$ is the indicator function. It is further defined in Persamaan X.X (\<THE EQ BELOW\>)
2. $i_{\text{c}}$ and $j_{\text{c}}$ are the row and column indices of the resulting feature map for the second convolution layer (c).
3. $\frac{\delta L_{\text{BCE}}}{\delta O_{\text{c}}}$ is the gradient map propagated from the flattening layer (calculated at the end of CHAPTER X.Y.Z (\<PREV SUBCHAPTER\>)), reshaped back to a 6x2 matrix.

$$
\mathbb{I}_{\mathbb{Z}^+}(x) =
\begin{cases}
   1 &\text{jika } x \in {\mathbb{Z}^+}  \\
   0 &\text{jika } x \notin {\mathbb{Z}^+}
\end{cases} \\
=\begin{cases}
   1 &\text{jika } x \gt 0  \\
   0 &\text{jika } x \leq 0
\end{cases}
$$

Note: ${\mathbb{Z}^+}=\set{1,2,3,\dots}$

$$
\frac{\delta L_{\text{BCE}}}{\delta K_{\text{c}}(m_{\text{c}}, n_{\text{c}})}=\sum_{i_{\text{c}}=0}^5\sum_{j_{\text{c}}=0}^1\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c}}(i_{\text{c}}, j_{\text{c}})} \times \frac{\delta z_{\text{c}}(i_{\text{c}}, j_{\text{c}})}{\delta K_{\text{c}}(m_{\text{c}}, n_{\text{c}})}\right) \\
=\sum_{i_{\text{c}}=0}^5\sum_{j_{\text{c}}=0}^1\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c}}(i_{\text{c}}, j_{\text{c}})} \times O_{\text{Cc}}(i_{\text{c}}+m_{\text{c}}, j_{\text{c}}+n_{\text{c}})\right)
$$

Note:

1. $m_{\text{c}} = \set{0,1,2}$ and $n_{\text{c}} = \set{0,1}$ based on the summation upper bounds in Persamaan X.X and the network assumption that $M_{\text{c}} = 3$ and $N_{\text{c}} = 2$ discussed in CHAPTER X.Y.Z. (\<THE CNN BACKEND CHAPTER\>)
2. $m_{\text{c}}$ and $n_{\text{c}}$ are the row and column indices of the kernel.
3. $\frac{\delta L_{\text{BCE}}}{\delta K_{\text{c}}(m_{\text{c}}, n_{\text{c}})}$ means that the derivative updates the weight of the kernel at specific row and column indices.
4. $i_{\text{c}}$ and $j_{\text{c}}$ are the row and column indices of the input. The upper bounds are respectively 5 and 1 because the "c" layer has shape 6x2.
5. O_Cc is the output of the concatenation from the front-end.

$$
\frac{\delta L_{\text{BCE}}}{\delta b_{\text{c}}(m_{\text{c}}, n_{\text{c}})}
=\sum_{i_{\text{c}}=0}^5\sum_{j_{\text{c}}=0}^1 \left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c}}(i_{\text{c}}, j_{\text{c}})} \times \frac{\delta z_{\text{c}}(i_{\text{c}}, j_{\text{c}})}{\delta b_{\text{c}}(m_{\text{c}}, n_{\text{c}})} \right) \\
=\sum_{i_{\text{c}}=0}^5\sum_{j_{\text{c}}=0}^1 \left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c}}(i_{\text{c}}, j_{\text{c}})} \times 1\right)
$$

Note: $b_{\text{c}}$ is the bias term for the "c" layer.

When calculating derivatives for a layer that is not at the bottom of the network following Goodfellow (2016), the calculation of indices of the gradient to be propagated (from the "c" layer) has its sign flipped. Cross-correlation as per Persamaan 2.X denotes the indices of the input to increase (because of the addition), while the backpropagation of such decreases the index. Moreover, when faced with an invalid index (negative index or out of the bounds of shape 6x2), the value is 0. In this case, it is assumed that O_Cc is infinitely padded on all sides with zeros. To show this, the ReLU activation function in Persamaan X.X (\<FEEDFORWARD EQ O_Cc=ReLU(z_Cc)\>) can be ignored because the function does not change the shape nor affect gradient propagation. Therefore, it can be defined that:

$$
O_{\text{Cc}}(i_{\text{c}}+m_{\text{c}},j_{\text{c}}+n_{\text{c}}) = z_{\text{Cc}}(i_{\text{Cc}},j_{\text{Cc}})
$$

Note:

1. $O_{\text{Cc}}$ is from Persamaan X.X (\<THE BACKPROP BEFORE THIS\>)
2. $z_{\text{Cc}}$ is from Persamaan X.X (\<FROM FEEDFORWARD\>)

By matching the indices from each parameter of $O_{\text{Cc}}$ and $z_{\text{Cc}}$, it can be defined that:

$$
i_{\text{c}}+m_{\text{c}}=i_{\text{Cc}} \ ; \quad j_{\text{c}}+n_{\text{c}}=j_{\text{Cc}} \\
i_{\text{c}}=i_{\text{Cc}}-m_{\text{c}} \ ; \quad j_2=j_{\text{Cc}}-n_{\text{c}}
$$

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}}(i_{\text{Cc}}, j_{\text{Cc}})}=\sum_{m_{\text{c}}=0}^2\sum_{n_{\text{c}}=0}^1\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c}}(i_{\text{c}},j_{\text{c}})} \times \frac{\delta z_{\text{c}}(i_{\text{c}},j_{\text{c}})}{\delta O_{\text{Cc}}(i_{\text{Cc}}, j_{\text{Cc}})}\right) \\
=\sum_{m_{\text{c}}=0}^2\sum_{n_{\text{c}}=0}^1\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c}}(i_{\text{Cc}}-m_{\text{c}}, j_{\text{Cc}}-n_{\text{c}})} \times K_{\text{c}}(m_{\text{c}}, n_{\text{c}})\right)
$$

Note:

1. $i_{\text{Cc}} \in \set{0,1,2,3,4,5};\  j_{\text{Cc}} \in \set{0,1}$
2. $i_{\text{Cc}}$ and $j_{\text{Cc}}$ are the row and column indices for the concatenated $O_{\text{Cc}}$ matrix.
3. The upper bounds for the summations is discussed in Persamaan X.X (\<THE BACKPROP BEFORE THIS\>)

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CV}}} = \frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}}(i_{\text{Cc}},j_{\text{Cc}})} = \frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}}(i_{\text{Cc}},0)}
$$

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CH}}} = \frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}}(i_{\text{Cc}},j_{\text{Cc}})} = \frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}}(i_{\text{Cc}},1)}
$$

Note:

1. $i_{\text{Cc}}=\set{0,1,2,3,4,5};\ j_{\text{Cc}}=\set{0,1}$
2. The feed-forward concatenation operation Concat(.) defined in Persamaan X.X (\<FF CONCAT ABOVE\>) joined $O_{\text{CV}}$ and $O_{\text{CH}}$ column-wise (along the frequency dimension/width axis), the backpropagation splits the gradient matrix back into its respective 6x1 shape.
3. i_Cc is the i-th row of the concatenated output from the two front-end convolutions
4. j_Cc is the j-th column of the concatenated output from the two front-end convolutions. This is either 0 or 1 where the 0th index is the error matrix representing the gradient for the vertical filter's output and the 1st index is for the horizontal filter's output.
5. $i_{\text{MPCV}}$ is the i-th row of the max-pooling output (with $j = 0$) from the front-end vertical filter convolution layer or the "CV" layer

### CNN with GRU Back-End

Gambar X.X (drawio:cnn-gru-be-bprop)

Due to the differences of how the results from GRU is concatenated and flatted from the CNN back-end, the reshape process is defined as:

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{GRU}}} = \operatorname*{Reshape}\left(\frac{\delta L_{\text{BCE}}}{\delta O_{\text{FC}}}, (6, 2) \right)
$$

The parameters being derived are visualised in Gambar X.X. The last hidden state or h_t is affected by the hidden states before it. Therefore, the backpropagated gradient at a time-step is the sum of the all gradients after it or 0 if it is the last time-step. This creates a recursive sum when calculating the weights and biases of the recurrent network (Zhang 2023). The gradient recursion is defined as:

$$
\frac{\delta L_{\text{BCE}}}{\delta h_t} = \left[\frac{\delta L_{\text{BCE}}}{\delta O_{\text{GRU}}} \right]_t + \frac{\delta O_{\text{GRU}}}{\delta h_{t+1}}\frac{\delta h_{t+1}}{\delta h_t}
$$

Note:

1. $t\in\set{1,2,3,4,5,6}$ because the length of the sequence is 6.
2. If $t=6$ (the last time-step), the gradient from the future $\frac{\delta L_{\text{BCE}}}{\delta h_{t+1}} = 0$. The backpropagation iterates backwards from $t=6$ to $t=1$.

To ease defining gradients, the feed-forward equations is redefined to be:

$$
\tilde{h}_t=\operatorname*{tanh}(z_{\tilde{h}(t)}) \\
z_t=\sigma(z_{z(t)}) \\
r_t=\sigma(z_{r(t)})
$$

Note: all equations are defined in Persamaan X.X sampai dengan X.X (\<THE FEEDFORWARD GRU EQS ABOVE\>)

$$
\frac{\delta L_{\text{BCE}}}{\delta \tilde{h}_t} = \frac{\delta L_{\text{BCE}}}{\delta h_t} \odot \frac{\delta h_t}{\delta \tilde{h}_t}
= \frac{\delta L_{\text{BCE}}}{\delta h_t} \odot (1 - z_t)
$$

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}(t)}} = \frac{\delta L_{\text{BCE}}}{\delta \tilde{h}_t} \odot \frac{\delta \tilde{h}_t}{\delta z_{\tilde{h}(t)}}
= \frac{\delta L_{\text{BCE}}}{\delta \tilde{h}_t} \odot (1 - \tilde{h}_t \odot \tilde{h}_t)
$$

$$
\frac{\delta L_{\text{BCE}}}{\delta z_t} = \frac{\delta L_{\text{BCE}}}{\delta h_t} \odot \frac{\delta h_t}{\delta z_t}
= \frac{\delta L_{\text{BCE}}}{\delta h_t} \odot (h_{t-1} - \tilde{h}_t)
$$

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{z(t)}} = \frac{\delta L_{\text{BCE}}}{\delta z_t} \odot \frac{\delta z_t}{\delta z_{z(t)}}
= \frac{\delta L_{\text{BCE}}}{\delta z_t} \odot (z_t \odot (1 - z_t))
$$

$$
\frac{\delta L_{\text{BCE}}}{\delta r_t} = \left( \frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}(t)}} \times \frac{\delta z_{\tilde{h}(t)}}{\delta (r_t \odot h_{t-1})} \right) \odot \frac{\delta (r_t \odot h_{t-1})}{\delta r_t} \\
= \left( \frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}(t)}} U_h \right) \odot h_{t-1}
$$

Note: the operation $U_h (r_t \odot h_{t-1})$ defined in Persamaan X.X (\<FEEDFORWARD EQ ABOVE\>) can not be distributed. Therefore, it has to be partially derived with respect to it first.

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{r(t)}} = \frac{\delta L_{\text{BCE}}}{\delta r_t} \odot \frac{\delta r_t}{\delta z_{r(t)}}
= \frac{\delta L_{\text{BCE}}}{\delta r_t} \odot (r_t \odot (1 - r_t))
$$

As noted before, the summation of each time-step is recursive. The base case is known to be 0 when the case is $t+1 \notin t$. The recursive case is defined as the summation of partial derivatives from all equations defined in Persamaan X.X sampai dengan X.X (\<THE FEEDFORWARD GRU EQS ABOVE\>) where $h_{t-1}$ occurs. It is defined to be:

$$
\frac{\delta L_{\text{BCE}}}{\delta h_{t-1}} = \left(\frac{\delta L_{\text{BCE}}}{\delta h_t} \odot \frac{\delta h_t}{\delta h_{t+1}}\right) + \left(\frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}(t)}} \times \frac{\delta z_{\tilde{h}(t)}}{\delta h_{t+1}}\right) + \left(\frac{\delta L_{\text{BCE}}}{\delta z_{z(t)}} \times \frac{\delta z_{z(t)}}{\delta h_{t+1}}\right) + \left(\frac{\delta L_{\text{BCE}}}{\delta z_{r(t)}} \times \frac{\delta z_{r(t)}}{\delta h_{t+1}}\right) \\
= \left( \frac{\delta L_{\text{BCE}}}{\delta h_t} \odot z_t \right) + \left( \left( U_h^T \frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}(t)}}  \right) \odot r_t \right) + \left( U_z^T \frac{\delta L_{\text{BCE}}}{\delta z_{z(t)}}  \right) + \left( U_r^T \frac{\delta L_{\text{BCE}}}{\delta z_{r(t)}}  \right)
$$

In the definitions at Persamaan X.X to X.X (\<FEEDFORWARD EQS ABOVE\>), the parameters that are shared across recurrent layers are: $W_h$, $U_h$, $b_h$, $W_z$, $U_z$, $b_z$, $W_r$, $U_r$, and $b_r$. The gradients for the parameters of the candidate hidden state is defined to be:

$$
\frac{\delta L_{\text{BCE}}}{\delta W_h} = \sum_{t=1}^6 \left(\frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}(t)}} \times \frac{\delta z_{\tilde{h}(t)}}{\delta W_h} \right)
= \sum_{t=1}^6 \left( \frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}(t)}} O_{\text{Cc}(t)}^T\right)
$$

$$
\frac{\delta L_{\text{BCE}}}{\delta U_h} = \sum_{t=1}^6 \left( \frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}(t)}} \times \frac{\delta z_{\tilde{h}(t)}}{\delta U_h} \right)
= \sum_{t=1}^6 \left(\frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}(t)}} (r_t \odot h_{t-1})^T \right)
$$

$$
\frac{\delta L_{\text{BCE}}}{\delta b_h} = \sum_{t=1}^6 \left( \frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}(t)}} \times \frac{\delta z_{\tilde{h}(t)}}{\delta b_h} \right)
= \sum_{t=1}^6 \left( \frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}(t)}} \times 1 \right)
$$

Similarly, the gradients for the update and reset gate is respectively defined to be:

$$
\frac{\delta L_{\text{BCE}}}{\delta W_z} = \sum_{t=1}^6 \left(  \frac{\delta L_{\text{BCE}}}{\delta z_{z(t)}} O_{\text{Cc}(t)}^T \right)
$$

$$
\quad \frac{\delta L_{\text{BCE}}}{\delta U_z} = \sum_{t=1}^6 \left(\frac{\delta L_{\text{BCE}}}{\delta z_{z(t)}} h_{t-1}^T \right)
$$

$$
\quad \frac{\delta L_{\text{BCE}}}{\delta b_z} = \sum_{t=1}^6 \left( \frac{\delta L_{\text{BCE}}}{\delta z_{z(t)}} \right)
$$

$$
\frac{\delta L_{\text{BCE}}}{\delta W_r} = \sum_{t=1}^6 \left(  \frac{\delta L_{\text{BCE}}}{\delta z_{r(t)}} O_{\text{Cc}(t)}^T \right)
$$

$$
\quad \frac{\delta L_{\text{BCE}}}{\delta U_r} = \sum_{t=1}^6 \left(\frac{\delta L_{\text{BCE}}}{\delta z_{r(t)}} h_{t-1}^T \right)
$$

$$
\quad \frac{\delta L_{\text{BCE}}}{\delta b_r} = \sum_{t=1}^6 \left( \frac{\delta L_{\text{BCE}}}{\delta z_{r(t)}} \right)
$$

Similarly, the parameter O_Cc(t) occurs multiple times as defined in Persamaan X.X to X.X (\<FEEDFORWARD EQS ABOVE\>). The derivation of this parameter is used to propagate gradients backwards to the front-end layer. It is defined to be:

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}(t)}} = \left( \frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}(t)}} \times \frac{\delta z_{\tilde{h}(t)}}{\delta O_{\text{Cc}(t)}} \right) + \left( \frac{\delta L_{\text{BCE}}}{\delta z_{z(t)}} \times \frac{\delta z_{z(t)}}{\delta O_{\text{Cc}(t)}} \right) + \left( \frac{\delta L_{\text{BCE}}}{\delta z_{r(t)}} \times \frac{\delta z_{r(t)}}{\delta O_{\text{Cc}(t)}} \right) \\
= \left( W_h^T \frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}(t)}}  \right) + \left( W_z^T \frac{\delta L_{\text{BCE}}}{\delta z_{z(t)}}  \right) + \left( W_r^T \frac{\delta L_{\text{BCE}}}{\delta z_{r(t)}}  \right)
$$

Note: this operation is done for all $t \in \set{1,2,3,4,5,6}$. Notice that it is not summed unlike for the weights and biases of GRU.

The result of Persamaan X.X (\<THE EQ DIRECTLY ABOVE THIS\>) is concatenated column-wise. In this case, it would result in a matrix of shape 6x2, the same shape with the feed-forward. This matrix is then backpropagated to the front-end the same as is defined in Persamaan X.X (\<THE LAST EQ OF CNN BACKEND BACKPROP SUBCHAPTER WITH O_CV AND O_CH\>).

### CNN with Self-Attention Back-End

Gambar X.X (drawio:cnn-gru-be-bprop)

For the case of the attention back-end, the reshape process is defined as:

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{FFN}(2)}} = \operatorname*{Reshape}\left( \frac{\delta L_{\text{BCE}}}{\delta O_{\text{FC}}}, (6, 2) \right)
$$

The flow of backpropagation is visualised in Gambar X.X (drawio:cnn-attn-be-bprop). The operations for each step of backpropagation is defined as follows:

$$
z_{\text{FFN}(1)}=O_{\text{MHA}} W_1+b_1
$$

$$
O_{\text{FFN}(1)}=\operatorname*{ReLU}(z_{\text{FFN}(1)})
$$

<!-- w2, b2 -->

$$
\frac{\delta L_{\text{BCE}}}{\delta W_2} =\frac{\delta L_{\text{BCE}}}{\delta O_{\text{FFN}(2)}} \times \frac{\delta O_{\text{FFN}(2)}}{\delta W_2}
=O_{\text{FFN}(1)}^T \frac{\delta L_{\text{BCE}}}{\delta O_{\text{FFN}(2)}}
$$

$$
\frac{\delta L_{\text{BCE}}}{\delta b_2} = \sum_{i=0}^5\left[\frac{\delta L_{\text{BCE}}}{\delta O_{\text{FFN}(2)}} \right]^T \frac{\delta O_{\text{FFN}(2)}}{\delta b_2}
=\sum_{i=0}^5\left[\frac{\delta L_{\text{BCE}}}{\delta O_{\text{FFN}(2)}} \right]_{i,j}^T \times 1
$$

Note: $j=\set{0,1}$ because $d_{\text{model}} = 2$

<!-- offn1 -->

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{FFN}(1)}} = \frac{\delta L_{\text{BCE}}}{\delta O_{\text{FFN}(2)}} \times \frac{\delta O_{\text{FFN}(1)}}{\delta O_{\text{FFN}(1)}}
=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{FFN}(2)}} W_2^T
$$

<!-- zffn1 -->

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{\text{FFN}(1)}} = \frac{\delta L_{\text{BCE}}}{\delta O_{\text{FFN}(1)}} \times \frac{\delta O_{\text{FFN}(1)}}{\delta z_{\text{FFN}(1)}}
=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{FFN}(1)}} \odot \mathbb{I}_{\mathbb{Z}^+}(z_{\text{FFN}(1)})
$$

<!-- w1, b1 -->

$$
\frac{\delta L_{\text{BCE}}}{\delta W_1} = \frac{\delta L_{\text{BCE}}}{\delta z_{\text{FFN}(1)}} \times \frac{\delta z_{\text{FFN}(1)}}{\delta W_1}
=O_{\text{MHA}}^T \frac{\delta L_{\text{BCE}}}{\delta z_{\text{FFN}(1)}}
$$

$$
\frac{\delta L_{\text{BCE}}}{\delta b_1} = \sum_{i=0}^5 \left[\frac{\delta L_{\text{BCE}}}{\delta z_{\text{FFN}(1)}} \right]_{i,j}^T \frac{\delta z_{\text{FFN}(1)}}{\delta b_1}
=\sum_{i=0}^5 \left[\frac{\delta L_{\text{BCE}}}{\delta z_{\text{FFN}(1)}} \right]_{i,j}^T \times 1
$$

Note: $j=\set{0,1}$ because $d_{ff} = 2$

<!-- omha -->

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{MHA}}} = \frac{\delta L_{\text{BCE}}}{\delta O_{\text{FFN}(2)}} + \frac{\delta L_{\text{BCE}}}{\delta z_{\text{FFN}(1)}} \times \frac{\delta z_{\text{FFN}(1)}}{\delta O_{\text{MHA}}} \\
=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{FFN}(2)}} + \left( \frac{\delta L_{\text{BCE}}}{\delta z_{\text{FFN}(1)}} W_1^T \right)
$$

Note: first term is residual

$$
C_h = \operatorname*{Concat}(\text{head}_1, \text{head}_2)
$$

<!-- w^o, ch -->

$$
\frac{\delta L_{\text{BCE}}}{\delta W^O} = \frac{\delta L_{\text{BCE}}}{\delta O_{\text{MHA}}} \times \frac{\delta O_{\text{MHA}}}{\delta W^O}
=C_h^T \frac{\delta L_{\text{BCE}}}{\delta O_{\text{MHA}}}
$$

$$
\frac{\delta L_{\text{BCE}}}{\delta C_h} = \frac{\delta L_{\text{BCE}}}{\delta O_{\text{MHA}}} \times \frac{\delta O_{\text{MHA}}}{\delta C_h}
=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{MHA}}} (W^O)^T
$$

<!-- definitions -->

$$
z_{\text{Attn}}=\frac{QK^T}{\sqrt{d_k}}
$$

$$
s_{\text{Attn}}=\operatorname*{softmax}(z_{\text{Attn}})
$$

$$
C_{h(i)}=s_{\text{Attn}}V
$$

Note: $i=\set{1,2}$ because there are 2 heads as discussed in the network assumption.

<!-- v_i -->

$$
\frac{\delta L_{\text{BCE}}}{\delta V_i} = \frac{\delta L_{\text{BCE}}}{\delta C_{h(i)}} \times \left[\frac{\delta C_{h}}{\delta V} \right]_i
=s_{\text{Attn}}^T \frac{\delta L_{\text{BCE}}}{\delta C_{h(i)}}
$$

<!-- s_attn(i) -->

$$
\frac{\delta L_{\text{BCE}}}{\delta s_{\text{Attn}(i)}} = \frac{\delta L_{\text{BCE}}}{\delta C_{h(i)}} \times \left[ \frac{\delta C_{h}}{\delta s_{\text{Attn}}} \right]_i
=\frac{\delta L_{\text{BCE}}}{\delta C_{h(i)}} V_i^T
$$

<!-- z_attn(i) -->

$$
\delta_{mn}=
\begin{cases}
   1 &\text{jika } m=n \\
   0 &\text{sebaliknya}
\end{cases}
$$

Note: $\delta_{mn}$ is the Kronecker delta function which is used in the derivation of the softmax activation function.

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{\text{Attn}(i)}} = \frac{\delta L_{\text{BCE}}}{\delta s_{\text{Attn}(i)}} \times \left[ \frac{\delta s_{\text{Attn}}}{\delta z_{\text{Attn}}} \right]_i \\
\frac{\delta L_{\text{BCE}}}{\delta z_{\text{Attn}(i, m, n)}}=\sum_{k=0}^{L-1} \left( \frac{\delta L_{\text{BCE}}}{\delta s_{\text{Attn}(i, m, k)}} s_{\text{Attn}(i, m, k)} (\delta_{np} - s_{\text{Attn}(i, m, n)}) \right)
$$

Note:

1. $m=\set{0,1,\dots,L-1}$ is the row index (query sequence time-step).
1. $n=\set{0,1,\dots,L-1}$ is the column index (key sequence time-step) for which the gradient is being calculated.
1. $k=\set{0,1,\dots,L-1}$ is the summation iterator across the columns of the softmax output.
1. L is the input sequence length (time axis). In this case, it is 6.
1. The equation is evaluated for all $m$ and for all $n$. The resulting shape of the derivative is LxL, in this case it is 6x6.

<!-- k_i, q_i -->

$$
\frac{\delta L_{\text{BCE}}}{\delta K_i} = \frac{\delta L_{\text{BCE}}}{\delta z_{\text{Attn}(i)}} \times \left[ \frac{\delta z_{\text{Attn}}}{\delta K} \right]_i
=\frac{1}{\sqrt{d_k}} \left( \left[ \frac{\delta L_{\text{BCE}}}{\delta z_{\text{Attn}(i)}} \right]^T Q_i \right)
$$

$$
\frac{\delta L_{\text{BCE}}}{\delta Q_i} = \frac{\delta L_{\text{BCE}}}{\delta z_{\text{Attn}(i)}} \times \left[ \frac{\delta z_{\text{Attn}}}{\delta Q} \right]_i
=\frac{1}{\sqrt{d_k}} \left( \frac{\delta L_{\text{BCE}}}{\delta z_{\text{Attn}(i)}} K_i \right)
$$

$$
\frac{\delta L_{\text{BCE}}}{\delta W_i^Q} = \frac{\delta L_{\text{BCE}}}{\delta Q_i} \times \left[ \frac{\delta Q}{\delta W^Q} \right]_i
=O_{\text{Cc}}^T \frac{\delta L_{\text{BCE}}}{\delta Q_i}
$$

In a similar manner:

$$
\frac{\delta L_{\text{BCE}}}{\delta W_i^K} = O_{\text{Cc}}^T \frac{\delta L_{\text{BCE}}}{\delta K_i}
$$

$$
\frac{\delta L_{\text{BCE}}}{\delta W_i^V} = O_{\text{Cc}}^T \frac{\delta L_{\text{BCE}}}{\delta V_i}
$$

Note: these are the gradients for the Query, Key, and Value linear projection weight matrices for head $i$. All have shape 2x1. $O_{\text{Cc}}^T$ has shape 2x6.

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}}} = \frac{\delta L_{\text{BCE}}}{\delta O_{\text{MHA}}} \\
+ \sum_{i=1}^2 \left( \left[ \frac{\delta L_{\text{BCE}}}{\delta Q_i} \times \frac{\delta Q_i}{\delta O_{\text{Cc}}} \right] + \left[ \frac{\delta L_{\text{BCE}}}{\delta K_i} \times \frac{\delta K_i}{\delta O_{\text{Cc}}} \right] + \left[ \frac{\delta L_{\text{BCE}}}{\delta V_i} \times \frac{\delta V_i}{\delta O_{\text{Cc}}} \right] \right) \\
= \frac{\delta L_{\text{BCE}}}{\delta O_{\text{MHA}}} + \sum_{i=1}^2 \left( \frac{\delta L_{\text{BCE}}}{\delta Q_i} (W_i^Q)^T + \frac{\delta L_{\text{BCE}}}{\delta K_i} (W_i^K)^T + \frac{\delta L_{\text{BCE}}}{\delta V_i} (W_i^V)^T \right)
$$

Note:

1. $\frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}}}$ (shape 6x2) is the total accumulated gradient passed back to the Front-End concatenation layer.
2. Because $O_{\text{Cc}}$ branches out into the residual connection, and into the $Q_i, K_i, V_i$ matrices for both heads, the gradient to be backpropagated is the sum of the gradients propagated backward from all 7 pathways, see Gambar X.X (drawio:cnn-attn-be-bprop).

### Vertical Filter CNN Front-End

$$
\frac{\delta L_{\text{BCE}}}{\delta {MP}_{\text{CV}}(i_{\text{MPCV}}, j_{\text{MPCV}})}=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CV}}(i_{\text{MPCV}}, 0)} \times \frac{\delta O_{\text{CV}}(i_{\text{MPCV}}, 0)}{\delta {MP}_{\text{CV}}(i_{\text{MPCV}}, j_{\text{MPCV}})} \\
= \begin{cases}
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CV}}(i_{\text{MPCV}}, 0)} \times 1 & \text{jika } j_{\text{MPCV}} = \operatorname*{argmax}(\operatorname*{ReLU}(z_{\text{V}})) \\
0 & \text{sebaliknya}
\end{cases}
$$

Note:

1. $i_{\text{MPCV}}=i_{\text{Cc}}$
2. $j_{\text{MPCV}}=\set{0,1,2,3,4}$ because the width of the output of the vertical filter convolution is 5.
3. j_MPCV is the j-th column of the output of the vertical filter convolution.
4. argmax(.) is a function that returns the index where the value is at the maximum in a given input. In this case, the input is a vector or matrix.

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})}=\frac{\delta L_{\text{BCE}}}{\delta {MP}_{\text{CV}}(i_{\text{V}}, j_{\text{V}})} \times \frac{\delta {MP}_{\text{CV}}(i_{\text{V}}, j_{\text{V}})}{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})} \\
=\frac{\delta L_{\text{BCE}}}{\delta {MP}_{\text{CV}}(i_{\text{V}}, j_{\text{V}})} \times \mathbb{I}_{\mathbb{Z}^+}(z_{\text{V}}(i_{\text{V}}, j_{\text{V}}))
$$

Note:

1. $i_{\text{V}}=i_{\text{MPCV}}$
1. $j_{\text{V}}=j_{\text{MPCV}}$

$$
\frac{\delta L_{\text{BCE}}}{\delta K_{\text{V}}(m_{\text{V}}, n_{\text{V}})}=\sum_{i_{\text{V}}=0}^4\sum_{j_{\text{V}}=0}^2\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})} \times \frac{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})}{\delta K_{\text{V}}(m_{\text{V}}, n_{\text{V}})}\right) \\
=\sum_{i_{\text{V}}=0}^4\sum_{j_{\text{V}}=0}^2\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})} \times I(i_{\text{V}}+m_{\text{V}}, j_{\text{V}}+n_{\text{V}})\right)
$$

Note:

1. This is similar to Persamaan X.X (\<REFERRING TO CNN BACKEND KERNEL BACKPROP ABOVE\>). The shape of the error matrix (because of "same" padding during feed-forward) means that the upper bounds of summation are respectively 4 and 2.
2. $I$ is the initial log-mel spectrogram input.

$$
\frac{\delta L_{\text{BCE}}}{\delta b_{\text{V}}}=\sum_{i_{\text{V}}=0}^4\sum_{j_{\text{V}}=0}^2\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})} \times \frac{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})}{\delta b_{\text{V}}}\right) \\
=\sum_{i_{\text{V}}=0}^4\sum_{j_{\text{V}}=0}^2\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})} \times 1\right)
$$

Note: this is similar to Persamaan X.X (\<REFERRING TO THE CNN BACKEND BIAS BACKPROP ABOVE\>)

### Horizontal Filter CNN Front-End

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{\text{H}}(i_{\text{H}})}=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CH}}(i_{\text{H}})} \times \left[\frac{\delta O_{\text{CH}}}{\delta z_{\text{H}}} \right]_{i_{\text{H}}} \\
=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CH}}(i_{\text{H}})} \times \mathbb{I}_{\mathbb{Z}^+}(z_{\text{H}}(i_{\text{H}}))
$$

Note:

1. $i_{\text{H}}=i_{\text{Cc}}$
2. There is only 1 index parameter because the convolution is 1D.

$$
\frac{\delta L_{\text{BCE}}}{\delta K_{\text{H}}(m_{\text{H}})}=\sum_{i_{\text{H}}=0}^5\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{H}}(i_{\text{H}})} \times \frac{\delta z_{\text{H}}(i_{\text{H}})}{\delta K_{\text{H}}(m_{\text{H}})}\right) \\
=\sum_{i_{\text{H}}=0}^5\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{H}}(i_{\text{H}})} \times M_{\text{in}}(i_{\text{H}}+m_{\text{H}})\right)
$$

Note:

1. $i_{\text{H}}\in \set{0,1,2,3,4,5}$ because the height of the output vector from mean-pooling is 6.
2. $m_{\text{H}}=\set{0,1,2}$ because the horizontal kernel width is 3.
3. $M_{\text{in}}$ is the intermediate output vector from mean-pooling that acts as the input to the horizontal filter convolution.

$$
\frac{\delta L_{\text{BCE}}}{\delta b_{\text{H}}}=\sum_{i_{\text{H}}=0}^5\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{H}}(i_{\text{H}})} \times \frac{\delta z_{\text{H}}(i_{\text{H}})}{\delta b_{\text{H}}}\right) \\
=\sum_{i_{\text{H}}=0}^5\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{H}}(i_{\text{H}})} \times 1\right)
$$

Note: this is similar to Persamaan X.X (\<REFERRING TO THE CNN BACKEND BIAS BACKPROP ABOVE\>)

# Penghitungan Rancangan Model
