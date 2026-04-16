# Manualisation

The simplified network to be used for an example calculation is the CNN model, the CNN with GRU model, and the CNN with Self-Attention model. The flow of this algorithm as shown in Gambar X.1 consists of three main steps: preprocessing, spectrogram calculation, and modeling. Preprocessing has already been discussed in chapter 3.2.4. After calculating STFT, the spectrogram output is then normalised using z-score normalisation. Batching is will be used in implementation with a size of 32. This reflects past literature like Choi et al. (2016conv) and Pons et al. (2018) which uses the default batch size of 32 provided by TensorFlow. The simplified networks assumes no use of mini-batching and thus batch and layer normalisation. The exampled dataset for the modeling stage will consist of 5 rows of arbitrary data sampled from the subset which consists of 3 arbitrary labels: happy, sad, and tense. The example dataset is shown in Table X.X. Log-mel spectrogram amplitudes in each frequency bin were observed in preliminary data exploration to be around -80 and 10 (in dB). The example dataset will reflect this range.

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

Given an input size of 6x5 with "same" padding, the output shape of the convolution is the same as the input shape. "Same" padding is also known as half padding. The amount of padding needed to achieve this is described in Domoulin (2016) where:

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

The CNN back-end consists of 2 layers each with 1 filter of shape 3xW. W is the width of the concatenated features; from the results of as shown in CHAPTER X.Y.Z, it is 2. This back-end accepts the input shape 6x2 from the front-end. The height of the kernel is the time axis while the width is the concatenated feature axis. In the implementation by Pons et al. (2018), the first convolution layer uses "valid" padding (no padding) while the second convolution layer uses "same" padding. Using no padding would result in the output shape of the first convolution to be smaller. Using the same calculations done in CHAPTER X.Y.Z, its shape is 4x1. To keep its shape and ease further calculations, both convolution layers will use "same" padding. The activation function for both layers is ReLU. The output of this back-end is the concatenated filters after ReLU.

Gambar X.X (drawio:cnn-be)

## CNN with GRU

The CNN with GRU back-end consists of two layers of uni-directional GRU each with one hidden layer. This back-end accepts the input shape 6x2 from the front-end. This design is in line with Cho et al. (2014). When implemented using PyTorch, the number of hidden layers can be directly set to be 2 in one GRU layer instead. The input tensor is adapted to be of shape LxH_in where L is the sequence length or the height of the time axis and H_in is the width of the concatenated features; from the results of as shown in CHAPTER X.Y.Z, it is 2. The output of this back-end is the final hidden state of GRU with shape LxH_out. H_out is set to 2. The output of this back-end returns the features from the last hidden state for each time-step. The shape of it is the same as the input.

Gambar X.X (drawio:cnn-gru-be)

## CNN with Self-Attention

The CNN with Self-Attention back-end consists of two layers of Self-Attention each with 2 heads. Self-Attention is implemented as part of the encoder part of the Transformer. The transformer takes input an embedding vector. Given the output of the front-end is of shape 6x2, every time-step of it can be treated as 6 embedding vectors each with shape 1x2. In calculation, this is done simultaneously. The dimension of query, keys, and value is determined by Persamaan 2.12, where d_model equals 2 because the width of the concatenated feauters are 2 as shown in CHAPTER X.Y.Z and h (or head) equals 2. Therefore, d_k and d_v equals 1. Given there are two heads, this means that each head gets input of shape 6x1. After calculating the Self-Attention, the result from each head is concatenaed to be 6x2 again. The activation function in the feed-forward network portion of the architecture is ReLU. The output of this back-end.

Gambar X.X (drawio:cnn-attn-be)

# Classifier

The classifier consists of a FC layer which takes 12 nodes as its input with 6 hidden nodes. The back-end output will be concatenated again to be of shape 12x1. The activation function for the FC and output layer is sigmoid. There are 3 output nodes corresponding to the three arbitrary labels set for this calculation.

Gambar X.X (drawio:classifier)

<!-- # Log-Mel Spectrogram Calculation -->

<!-- # Feed-forward -->

# Perumusan Feed-Forward dan Backpropagation Model CNN

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

The model subject to have its backpropagation equations defined is only the CNN model, which consists of the CNN front-end, the CNN back-end, and the classifier. Given a prediction, the loss function is defined in Persamaan 2.24. To ease understanding, Gambar X.X visualises each variable that are part of the CNN model. In the mathematical definitions in CHAPTER X.Y.Z (\<THE TWO SUBCHAPTERS FOR FF AND BPROP BELOW\>), the multiplication sign is used to denote scalar multiplication. If it is not present, then it denotes matrix multiplication.

## Perumusan Feed-Forward

The feed-forward equations are defined from the weighted BCE loss function towards the input. Some equation notes are incomplete because it has been noted in previous equations.

GAMBAR X.X \<VIS CNN TAPI ADA VARIABLE VARIABLE\> (drawio:cnn-ff-bprop)

### Classifier

$$y_n=\sigma(z_O)$$

Note:

1. y_n is the predicted output
2. $\sigma$ is the sigmoid activation function
3. z_O is the output of the output layer before applying the activation function as visualised in Gambar X.X

$$z_O=O_{\text{FCH}} W_O + b_O$$

Note:

1. O_FCH is the output of the "FCH" layer
2. W_FCH is the weights for the "FCH" layer
3. b_FCH is the biases for the "FCH" layer

$$O_{\text{FCH}}=\sigma(z_{\text{FCH}})$$

Note: z_FCH is the output of the "FCH" layer before applying the activation function

$$z_{\text{FCH}}=O_{\text{FC}}W_{\text{FCH}}+b_{\text{FCH}}$$

<!-- Note: this is similar to Persamaan X.X just that it is for the FC layer -->

<!-- $$O_{\text{FC}}=\sigma ( z_{\text{FC}} )$$

$$z_{\text{FC}}=O_{\text{c2}}$$ -->

<!-- Note: Both Persamaan X.X and Persamaan X.X (\<REFERRING TO THE 2 EQS ABOVE\>) is similar to Persamaan X.X just that it is for the FC layer -->

$$O_{\text{FC}} = \text{Flatten}(O_{\text{c2}})$$

Note:

1. $O_{\text{c2}}$ is the output of the second convolution layer in the back-end
2. Flatten(.) is an operation that reshapes the 2D matrix output of $O_{\text{c2}}$ of shape 6x2 into a 1D vector of shape 12x1.

### CNN Back-End

$$O_{\text{c2}}=\operatorname*{ReLU}(z_{\text{c2}})$$

Note: $z_{\text{c2}}$ is the output of the second convolution layer (denoted c2) in the back-end before applying the activation function

The following re-purposes the mathematical definition of the convolution operation as defined in Persamaan 2.X (DI BAB 2) for use in this case:

$$
z_{\text{c2}}=S_{\text{c2}}(i_{\text{c2}},j_{\text{c2}}) + b_{c2} \\
=\sum_{m_{\text{c2}}=0}^{M_{\text{c2}}-1}\sum_{n_{\text{c2}}=0}^{N_{\text{c2}}-1} \big(O_{\text{c1}}(i_{\text{c2}}+m_{\text{c2}},j_{\text{c2}}+n_{\text{c2}})\times K_{\text{c2}}(m_{\text{c2}},n_{\text{c2}}) \big) + b_{c2}
$$

Note: similar to Persamaan 2.X; I defined in Persamaan 2.X is replaced with O_c1 which is the output of the first convolution layer in the back-end

$$O_{\text{c1}}=\operatorname*{ReLU}(z_{\text{c1}})$$

$$
z_{\text{c1}}=S_{\text{c1}}(i_{\text{c1}},j_{\text{c1}}) + b_{c1}\\
=\sum_{m_{\text{c1}}=0}^{M_{\text{c1}}-1}\sum_{n_{\text{c1}}=0}^{N_{\text{c1}}-1} \big(O_{\text{Cc}}(i_{\text{c1}}+m_{\text{c1}},j_{\text{c1}}+n_{\text{c1}})\times K_{\text{c1}}(m_{\text{c1}},n_{\text{c1}}) \big) + b_{c1}
$$

Note:

1. Both Persamaan X.X and Persamaan X.X (\<REFERRING TO THE 2 EQS ABOVE\>) is similar to Persamaan X.X (\<THE OG CONV EQ ABOVE THESE 2\>) just that it is for the first convolution layer in the back-end.
2. $O_{\text{Cc}}$ is the output of the concatenation layer from the front-end

$$O_{\text{Cc}}=\text{Concat}(O_{\text{CV}},O_{\text{CH}})$$

### CNN Front-End

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
=\left[\frac{\delta L_{\text{BCE}}}{\delta z_O}\right]_n \times\left[O_\text{FCH}\right]_j
$$

Note:

1. n is the n-th class in the dataset as per Persamaan 2.24
2. $W_{O(j,n)}$ is the weight connecting the j-th node of the output-side of FCH to the n-th output node (the "O" layer in Gambar X.X).
3. [.]\_n is the n-th gradient. Since there are 3 classes, there will be N calculations of weighted BCE losses as defined in Persamaan X.X (\<THE EQ ABOVE THIS ONE\>).
4. $z_{O(n)}$ is the n-th output after calculations with the weights in the output layer that have not been activated
5. $\left[O_\text{FCH}\right]_j$ is the j-th node of the outputside of FCH.

$$
\frac{\delta L_{\text{BCE}}}{\delta b_{O(n)}}=\left[\frac{\delta L_{\text{BCE}}}{\delta y_n}\times \frac{\delta y_n}{\delta z_O}\right]_n \times\left[\frac{\delta z_O}{\delta b_O}\right]_n \\
=\left[\frac{\delta L_{\text{BCE}}}{\delta z_O}\right]_n \times 1 \\
$$

Note: $b_{O(n)}$ is the bias for the n-th output ("O" layer) node

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{FCH}(j)}}=\sum_{n=1}^3 \left( \left[\frac{\delta L_{\text{BCE}}}{\delta z_O} \right]_n \times \frac{\delta z_{O(n)}}{\delta O_{\text{FCH}(j)}} \right) \\
=\sum_{n=1}^3 \left( \left[\frac{\delta L_{\text{BCE}}}{\delta z_O} \right]_n \times W_{O(j,n)} \right) \\
$$

Note:

1. $j = 1,2,3,\dots,6$ because there are 6 nodes in the "FCH" layer as discussed in the network assumption discussed in CHAPTER X.Y.Z (\<THE CHAPTER BEFORE THIS\>)
2. O_FCH(j) is the j-th output node of the "FCH" layer
3. This operation involves a sum from the gradients of each "O" layer output because the j-th output node of the "FCH" layer is connected to all three of the "O" layer output nodes.

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{\text{FCH}(j)}}=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{FCH}(j)}} \times \frac{\delta O_{\text{FCH}(j)}}{\delta z_{\text{FCH}(j)}} \\
=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{FCH}(j)}} \times \left(O_{\text{FCH}(j)} \times (1 - O_{\text{FCH}(j)})\right) \\
=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{FCH}(j)}} \times \left(\sigma(z_{\text{FCH}(j)}) \times (1 - \sigma(z_{\text{FCH}(j)}))\right)
$$

Note: z_FCH(j) is the j-th output from the calculations with weights and biases from the FH layer that has not been activated

$$
\frac{\delta L_{\text{BCE}}}{\delta W_{\text{FCH}(i,j)}}=\frac{\delta L_{\text{BCE}}}{\delta z_{\text{FCH}(j)}} \times \frac{\delta z_{\text{FCH}(j)}}{\delta W_{\text{FCH}(i,j)}} \\
=\frac{\delta L_{\text{BCE}}}{\delta z_{\text{FCH}(j)}} \times \left[O_\text{FC}\right]_i
$$

Note:

1. $i=1,2,3,\dots,12; j = 1,2,3,\dots,6$
2. $W_{\text{FCH}(i,j)}$ is the weight connecting the $i$-th input node (from the 12-node flattened CNN front-end output vector) to the $j$-th hidden node.
3. $\left[O_\text{FC}\right]_i$ is the $i$-th element of the flattened CNN front-end output vector.

$$
\frac{\delta L_{\text{BCE}}}{\delta b_{\text{FCH}(j)}}=\frac{\delta L_{\text{BCE}}}{\delta z_{\text{FCH}(j)}} \times \frac{\delta z_{\text{FCH}(j)}}{\delta b_{\text{FCH}(j)}}\\
=\frac{\delta L_{\text{BCE}}}{\delta z_{\text{FCH}(j)}} \times 1
$$

Note: b_FCH(j) is the j-th bias of the "FCH" layer.

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{FC}(i)}}=\sum_{j=1}^6 \left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{FCH}(j)}} \times \frac{\delta z_{\text{FCH}(j)}}{\delta O_{\text{FC}(i)}} \right) \\
=\sum_{j=1}^6 \left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{FCH}(j)}} \times W_{\text{FCH}(i,j)} \right)
$$

Note:

1. $i=1,2,3,\dots,12$.
2. O_FC(i) is the i-th output of the "FC" layer which is the flattened layers from the back-end
3. The gradient propagated to the $i$-th node of the "FC" layer is the sum from each 6 nodes in the "FCH" layer.

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{c2}}} = \text{Reshape}\left( \frac{\delta L_{\text{BCE}}}{\delta O_{\text{FC}}}, (6, 2) \right)
$$

Note:

1. O_c2 is the output of the second convolutional layer (the "c2" layer) in the back-end
2. Reshape(.) is to reshape a vector to form an array with defined shape as per the second parameter of the function. In this case it is 6x2, which is written (6, 2) as to not be confused with scalar multiplication.

The concatenation function as defined in Persamaan X.X (\<SEE THE CONCAT FORMULA IN PREV SUBCHAPTER\>) means that the propagated gradient has to be reshaped so that it is the same shape of the feed-forward output of "c2" layer.

### CNN Back-End

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c2}}(i_{\text{c2}}, j_{\text{c2}})}=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{c2}}(i_{\text{c2}}, j_{\text{c2}})} \times \left[\frac{\delta O_{\text{c2}}}{\delta z_{\text{c2}}} \right]_{i_{\text{c2}}, j_{\text{c2}}} \\
=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{c2}}(i_{\text{c2}}, j_{\text{c2}})} \times \mathbb{I}_{\mathbb{Z}^+}(z_{\text{c2}}(i_{\text{c2}}, j_{\text{c2}}))
$$

Note:

1. $\frac{\delta O_{\text{c2}}}{\delta z_{\text{c2}}}$ is the derivative of the ReLU activation function. $\mathbb{I}_{\mathbb{Z}^+}(\cdot)$ is the indicator function. It is further defined in Persamaan X.X (\<THE EQ BELOW\>)
2. $i_{\text{c2}}$ and $j_{\text{c2}}$ are the row and column indices of the resulting feature map for the second convolution layer (c2).
3. $\frac{\delta L_{\text{BCE}}}{\delta O_{\text{c2}}}$ is the gradient map propagated from the flattening layer (calculated at the end of CHAPTER X.Y.Z (\<PREV SUBCHAPTER\>)), reshaped back to a 6x2 matrix.

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
\frac{\delta L_{\text{BCE}}}{\delta K_{\text{c2}}(m_{\text{c2}}, n_{\text{c2}})}=\sum_{i_{\text{c2}}=0}^5\sum_{j_{\text{c2}}=0}^1\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c2}}(i_{\text{c2}}, j_{\text{c2}})} \times \frac{\delta z_{\text{c2}}(i_{\text{c2}}, j_{\text{c2}})}{\delta K_{\text{c2}}(m_{\text{c2}}, n_{\text{c2}})}\right) \\
=\sum_{i_{\text{c2}}=0}^5\sum_{j_{\text{c2}}=0}^1\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c2}}(i_{\text{c2}}, j_{\text{c2}})} \times O_{\text{c1}}(i_{\text{c2}}+m_{\text{c2}}, j_{\text{c2}}+n_{\text{c2}})\right)
$$

Note:

1. $m_{\text{c2}} = \set{0,1,2}$ and $n_{\text{c2}} = \set{0,1}$ based on the summation upper bounds in Persamaan X.X and the network assumption that $M_{\text{c2}} = 3$ and $N_{\text{c2}} = 2$ discussed in CHAPTER X.Y.Z. (\<THE CNN BACKEND CHAPTER\>)
2. $m_{\text{c2}}$ and $n_{\text{c2}}$ are the row and column indices of the kernel.
3. $\frac{\delta L_{\text{BCE}}}{\delta K_{\text{c2}}(m_{\text{c2}}, n_{\text{c2}})}$ means that the derivative updates the weight of the kernel at specific row and column indices.
4. $i_{\text{c2}}$ and $j_{\text{c2}}$ are the row and column indices of the input. The upper bounds are respectively 5 and 1 because the "c2" layer has shape 6x2.
5. O_c1 is the output of the first convolution layer in the back-end ("c1" layer).

$$
\frac{\delta L_{\text{BCE}}}{\delta b_{\text{c2}}(m_{\text{c2}}, n_{\text{c2}})}
=\sum_{i_{\text{c2}}=0}^5\sum_{j_{\text{c2}}=0}^1 \left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c2}}(i_{\text{c2}}, j_{\text{c2}})} \times \frac{\delta z_{\text{c2}}(i_{\text{c2}}, j_{\text{c2}})}{\delta b_{\text{c2}}(m_{\text{c2}}, n_{\text{c2}})} \right) \\
=\sum_{i_{\text{c2}}=0}^5\sum_{j_{\text{c2}}=0}^1 \left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c2}}(i_{\text{c2}}, j_{\text{c2}})} \times 1\right)
$$

Note: $b_{\text{c2}}$ is the bias term for the "c2" layer.

When calculating derivatives for a layer that is not at the bottom of the network following Goodfellow (2016), the calculation of indices of the gradient to be propagated (from the "c2" layer) has its sign flipped. Cross-correlation as per Persamaan 2.X denotes the indices of the input to increase (because of the addition), while the backpropagation of such decreases the index. Moreover, when faced with an invalid index (negative index or out of the bounds of shape 6x2), the value is 0. In this case, it is assumed that O_c1 is infinitely padded on all sides with zeros. To show this, the ReLU activation function in Persamaan X.X (\<FEEDFORWARD EQ O_c1=ReLU(z_c1)\>) can be ignored because the function does not change the shape nor affect gradient propagation. Therefore, it can be defined that:

$$
O_{\text{c1}}(i_{\text{c2}}+m_{\text{c2}},j_{\text{c2}}+n_{\text{c2}}) = z_{\text{c1}}(i_{\text{c1}},j_{\text{c1}})
$$

Note:

1. $O_{\text{c1}}$ is from Persamaan X.X (\<THE BACKPROP BEFORE THIS\>)
2. $z_{\text{c1}}$ is from Persamaan X.X (\<FROM FEEDFORWARD\>)

By matching the indices from each parameter of $O_{\text{c1}}$ and $z_{\text{c1}}$, it can be defined that:

$$
i_{\text{c2}}+m_{\text{c2}}=i_{\text{c1}} \ ; \quad j_{\text{c2}}+n_{\text{c2}}=j_{\text{c1}} \\
i_{\text{c2}}=i_{\text{c1}}-m_{\text{c2}} \ ; \quad j_2=j_{\text{c1}}-n_{\text{c2}}
$$

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{c1}}(i_{\text{c1}}, j_{\text{c1}})}=\sum_{m_{\text{c2}}=0}^2\sum_{n_{\text{c2}}=0}^1\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c2}}(i_{\text{c2}},j_{\text{c2}})} \times \frac{\delta z_{\text{c2}}(i_{\text{c2}},j_{\text{c2}})}{\delta O_{\text{c1}}(i_{\text{c1}}, j_{\text{c1}})}\right) \\
=\sum_{m_{\text{c2}}=0}^2\sum_{n_{\text{c2}}=0}^1\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c2}}(i_{\text{c1}}-m_{\text{c2}}, j_{\text{c1}}-n_{\text{c2}})} \times K_{\text{c2}}(m_{\text{c2}}, n_{\text{c2}})\right)
$$

Note:

1. $i_{\text{c1}} \in \set{0,1,2,3,4,5};\  j_{\text{c1}} \in \set{0,1}$
2. $i_{\text{c1}}$ and $j_{\text{c1}}$ are the row and column indices for the $O_{\text{c1}}$ matrix.
3. The upper bounds for the summations is discussed in Persamaan X.X (\<THE BACKPROP BEFORE THIS\>)

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c1}}(i_{\text{c1}}, j_{\text{c1}})}=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{c1}}(i_{\text{c1}}, j_{\text{c1}})} \times \left[\frac{\delta O_{\text{c1}}}{\delta z_{\text{c1}}} \right]_{i_{\text{c1}}, j_{\text{c1}}} \\
=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{c1}}(i_{\text{c1}}, j_{\text{c1}})} \times \mathbb{I}_{\mathbb{Z}^+}(z_{\text{c1}}(i_{\text{c1}}, j_{\text{c1}}) )
$$

Note: this is imilar to Persamaan X.X (\<REFERRING TO z_c2 EQ\>), but for the first convolution layer ("c1" layer).

$$
\frac{\delta L_{\text{BCE}}}{\delta K_{\text{c1}}(m_{\text{c1}}, n_{\text{c1}})}=\sum_{i_{\text{c1}}=0}^5\sum_{j_{\text{c1}}=0}^1\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c1}}(i_{\text{c1}}, j_{\text{c1}})} \times \frac{\delta z_{\text{c1}}(i_{\text{c1}}, j_{\text{c1}})}{\delta K_{\text{c1}}(m_{\text{c1}}, n_{\text{c1}})}\right) \\
=\sum_{i_{\text{c1}}=0}^5\sum_{j_{\text{c1}}=0}^1\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c1}}(i_{\text{c1}}, j_{\text{c1}})} \times O_{\text{Cc}}(i_{\text{c1}}+m_{\text{c1}}, j_{\text{c1}}+n_{\text{c1}})\right)
$$

Note:

1. This is similar to Persamaan X.X (\<REFERRING TO CONV2 KERNEL BACKPROP ABOVE\>). The shape of the error matrix (because of "same" padding during feed-forward) means that the upper bounds of summation are respectively 5 and 1.
2. $O_{\text{Cc}}$ is the output of the front-end concatenation layer or the "Cc" layer.

$$
\frac{\delta L_{\text{BCE}}}{\delta b_{\text{c1}}}=\sum_{i_{\text{c1}}=0}^5\sum_{j_{\text{c1}}=0}^1\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c1}}(i_{\text{c1}}, j_{\text{c1}})} \times \frac{\delta z_{\text{c1}}(i_{\text{c1}}, j_{\text{c1}})}{\delta b_{\text{c1}}}\right) \\
=\sum_{i_{\text{c1}}=0}^5\sum_{j_{\text{c1}}=0}^1\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c1}}(i_{\text{c1}}, j_{\text{c1}})} \times 1\right)
$$

Note: this is similar to Persamaan X.X (\<REFERRING TO THE CONV2 BIAS BACKPROP ABOVE\>)

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}}(i_{\text{Cc}}, j_{\text{Cc}})}=\sum_{m_{\text{c1}}=0}^2\sum_{n_{\text{c1}}=0}^1\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c1}}(i_{\text{Cc}}-m_{\text{c1}}, j_{\text{Cc}}-n_{\text{c1}})} \times \frac{\delta z_{\text{c1}}(i_{\text{Cc}}-m_{\text{c1}}, j_{\text{Cc}}-n_{\text{c1}})}{\delta O_{\text{Cc}}(i_{\text{Cc}}, j_{\text{Cc}})}\right) \\
=\sum_{m_{\text{c1}}=0}^2\sum_{n_{\text{c1}}=0}^1\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c1}}(i_{\text{Cc}}-m_{\text{c1}}, j_{\text{Cc}}-n_{\text{c1}})} \times K_{\text{c1}}(m_{\text{c1}}, n_{\text{c1}})\right)
$$

Note: this is similar to Persamaan X.X (\<REFERRING TO THE BACKPROP W.R.T. O_c1 ABOVE\>)

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

1. This is similar to Persamaan X.X (\<REFERRING TO CONV2 KERNEL BACKPROP ABOVE\>). The shape of the error matrix (because of "same" padding during feed-forward) means that the upper bounds of summation are respectively 4 and 2.
2. $I$ is the initial log-mel spectrogram input.

$$
\frac{\delta L_{\text{BCE}}}{\delta b_{\text{V}}}=\sum_{i_{\text{V}}=0}^4\sum_{j_{\text{V}}=0}^2\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})} \times \frac{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})}{\delta b_{\text{V}}}\right) \\
=\sum_{i_{\text{V}}=0}^4\sum_{j_{\text{V}}=0}^2\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})} \times 1\right)
$$

Note: this is similar to Persamaan X.X (\<REFERRING TO THE CONV2 BIAS BACKPROP ABOVE\>)

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

Note: this is similar to Persamaan X.X (\<REFERRING TO THE CONV2 BIAS BACKPROP ABOVE\>)

# Penghitungan Rancangan Model
