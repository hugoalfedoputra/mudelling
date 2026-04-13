# Simplified network calculations

The network to be used for an exemplary calculation is the CNN model, the CNN with GRU model, and the CNN with Self-Attention model. The flow of this algorithm consists of three main steps: preprocessing, spectrogram conversion, and modeling. Preprocessing has already been discussed in chapter 3.2.4. The input used for spectrogram conversion is different than the data used as the model input. In calculating STFT, one frame will be used with a length of 32 time-frames. There are techniques to avoid problems in calculating STFT naively, one example is spectral leakage, but the following calculation does not take those into account. After calculating STFT, the spectrogram output is then normalised using Z-score normalisation. This networks assumes no use of mini-batching and thus batch and layer normalisation. The exampled dataset for the modeling stage will consist of 5 rows of arbitrary data sampled from the subset which consists of 3 arbitrary labels: happy, sad, and tense. The example dataset is shown in Table X.X

Table X.X \<EXAMPLE DATASET\>

// ADD A FLOWCHART TO DESCRIBE THE FLOW OF THE NETWORK ABOVE

// FLOWCHART ABOUT LOOPING 3 TIMES BECAUSE THERE ARE 3 BACK-END VARIATIONS

# Front-end

// FLESH THIS SECTION OUT TO BE A COMPARISON TABLE AND NOT JUST BULLET POINTS AND NUMBERED LIST

- Number of filters: 2
- Input is split to 2 types: for the vertical and horizontal filters
- For the vertical filter type: max-pool AFTER convolution
- For the horizontal filter type: mean-pool BEFORE convolution
- Filter 1 (namely $F_V$) size: 5x3 (vertical)
- Filter 2 (namely $F_H$) size: 1x3 (horizontal)

Forward (vertical filter):

1. Input: 7x5 (HxW) where H is time and W is frequency. This mimics time series data modeling where tabular representation has time (as a column) where each row denotes the values at that time. The values of each features are the columns or the width, assuming HxW representation.
2. Number of channels: 1. This is because spectrogram consists of one scale of colors and not 3 like RGB image.
3. Convolution with the vertical filter with "same" padding
4. Apply ReLU
5. Max-pool the result

Forward (horizontal filter):

1. Input: 7x5 (HxW) where H is frequency and W is time.
2. Number of channels: 1. This is because spectrogram consists of one scale of colors and not 3 like RGB image.
3. Mean-pool the spectrogram input such that the output dimension has a height of 1 and a width of time
4. Convolution with horizonal filters with "same" padding
5. Apply ReLU

Note that in the vertical filter feed-forward, H is time while W is frequency but in the horizontal filter feed-forward, H is frequency while W is time. The rationale is described in Pons et al. (2018) where the vertical and horizontal filters are trying to learn timbre and temporal features. Which is why in both filters it is reducing the dimensionality of the frequency while keeping the time axis intact. This always results in something akin to time series data.

The result of vertical and horizontal convolutions are then concatenated to become a tensor with the time axis intact and the frequency bin axis reduced in dimensionality. This concatenation is the reason of separating the feed-forward for the front-end based on the filter shape. It will be clear during backpropagation.

---

// ADD VISUAL ILLUSTRATIONS FOR THE FEED-FORWARD OF VERTICAL AND HORIZONTAL FILTERS

## Feed-forward of vertical filter convolution

Given an input size of 7x5 with "same" padding, the output shape of the convolution is the same as the input shape. "Same" padding is also known as half padding. The amount of padding needed to achieve this is described in Domoulin (2016) where:

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

Gambar X.X \<PADDED MATRIX\>

Note:

1. p_H is the amount of padding to be applied on the height boundaries (top and bottom) of the input matrix
2. k_H is the height of the kernel
3. p_W is the amount of padding to be applied on the width boundaries (left and right) of the input matrix
4. k_W is the width of the kernel

The max-pooling operation is not padded but has a kernel and stride size of 1x5 where 5 is the width of the frequency bin axis. Max-pooling takes a sliding window which strides to capture a portion of the matrix based on the kernel size. The operation finds the maximum value in that portion of the matrix. The operation is mathematically defined as follows:

$$\text{out}_\text{MaxPool}(h,w)=\max_{m=0,\dots,k_H-1}\max_{n=0,\dots,k_W-1}\text{in}(\text{stride}[0]\times h+m,\text{stride}[1] \times w+n)$$

Note:

1. $\text{out}(\cdot)$ means the output function of max-pooling
2. $\text{in}(\cdot)$ means the input matrix of max-pooling
3. stride is a tuple with elements on the 0th and 1st index which signifies how much the window slides over all locations in the input
4. h is the height of the input matrix and as the parameter of the output
5. w is the width of the input matrix and as the parameter of the output
6. m is the row index of the input matrix
7. n is the column index of the input matrix
8. k_H is the height of the kernel
9. k_W is the width of the kernel

The operation has output shape mathematically defined as follows:

$$H_{out}=\left\lfloor \frac{H_{in}+2*\text{padding}[0]-\text{dilation}[0]\times(\text{kernel\_size}[0]-1)-1}{\text{stride}[0]}+1 \right\rfloor$$

Note:

1. H_out is the height of the output
2. H_in is the height of the input
3. padding[0] means the first index of the padding tuple or how much is the input padded on the top and bottom
4. dilation[0] means the first index of the dilation tuple or the rate of which the size of the kernel increases with which elements of the input are also skipped. Dilation value set to 1 is the same as not applying any dilation.
5. kernel_size[0] means the first index of the kernel_size tuple or the size of the kernel used as the window to calculate the max value at a given window
6. stride[0] means the first index of the stride tuple or how much the window slides over all locations in the input

$$W_{out}=\left\lfloor \frac{W_{in}+2*\text{padding}[1]-\text{dilation}[1]\times(\text{kernel\_size}[1]-1)-1}{\text{stride}[1]}+1 \right\rfloor$$

Note:

1. W_out is the width of the output
2. W_in is the width of the input
3. padding[1] means the second index of the padding tuple
4. dilation[1] means the second index of the dilation tuple
5. kernel_size[1] means the second index of the kernel_size tuple
6. stride[1] means the second index of the stride tuple

Therefore, the shape of the output after the max-pooling operation is shown to be of shape 6x1 as calculated: (2 separate eqs)

$$H_{out}=\left\lfloor \frac{7+2*0-1\times(1-1)-1}{1}+1 \right\rfloor=6$$

$$W_{out}=\left\lfloor \frac{5+2*0-1\times(5-1)-1}{5}+1 \right\rfloor=1$$

Putting it all together, the final shape (including the channel axis) of the vertical filter feed-forward is 1x6x1 with dimensions: channel, time (height), and frequency (width).

## Feed-forward of horizontal filter convolution

Given an input shape of 7x5, the mean-pooling operation is not padded but has a kernel and stride size of 1x5 where 5 is the width of the frequency bin axis. In a similar fashion to max-pooling, mean-pooling takes the average of all values in the portion of the matrix. The operation has output shape is the same as defined in max-pooling. Therefore, the input of the convolution has a shape of 1x6. The operation is mathematically defined as follows:

$$\text{out}_\text{MeanPool}(h,w)=\frac{1}{k_H+k_W}\sum_{m=0}^{k_H-1}\sum_{n=0}^{k_W-1}\text{in}(\text{stride}[0]\times h+m,\text{stride}[1] \times w+n)$$

The notes for this equation is the same as for max-pooling.

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

## Concatenation of vertical and horizontal filter convolutions

The concatenation process takes the output of both the vertical and horizontal filter convolutions. Since temporal convolution uses 1D convolution, a new dimension can be added after the time dimension to replace the frequency dimension that was dropped before the convolution. Therefore, the input shapes of both the convolutions are 1x6x1, therefore the resulting concatenation will have shape 1x6x2 which keeps the length of the time axis intact. The order of concatenation (assuming the matrix is read left to right) is the vertical and then the horizontal filter convolution. The dimensions are: channel, time, and frequency.

# Back-end

## CNN

The CNN back-end consists of 2 layers each with 1 filter of the same shape: 3xW. W is the width of the concatenated features; from the results of as shown in CHAPTER X.Y.Z, it is 2. The first convolution layer uses "valid" padding (no padding) while the second convolution layer uses "same" padding. Height is the time axis while the width is the concatenated feature axis. The activation function is ReLU. The output of this backend is the concatenated filters after ReLU.

// ADD VISUALISATION

## CNN with GRU

The CNN with GRU back-end consists of one layer of uni-directional GRU with 2 hidden layers. The input tensor is adapted to be of shape LxH_in where L is the sequence length or the height of the time axis and H_in is the width of the concatenated features; from the results of as shown in CHAPTER X.Y.Z, it is 2. The output of this backend is the final hidden state of GRU with shape LxH_out. H_out is set to 2.

// ADD VISUALISATION + FLOWCHART

## CNN with Self-Attention (implemented using Transformer)

The CNN with Self-Attention back-end consists of one layer of Self-Attention with 2 heads.

// ADD VISUALISATION + FLOWCHART

# Classifier

// ADD VISUALISATION

# Feed-forward

# Backpropagation

# Calculation of simplified network
