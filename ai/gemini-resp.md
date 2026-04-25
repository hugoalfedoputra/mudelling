Your understanding of Batch Normalization for this specific architecture is **absolutely correct**, and your reasoning for omitting $\gamma$ (scale) and $\beta$ (shift) is mathematically sound for your thesis context.

Here is why your understanding is perfect based on the Ioffe and Szegedy (2015) paper:

1. **The Convolutional Property:** In Section 3.2, the paper specifies that BN for CNNs must normalize jointly over the mini-batch _and_ spatial locations for each feature map. Because your batch size is 1, the "spatial location" for your 1D convolution is the time-step axis (length 6). Normalizing across all time-steps independently for each feature (channel) is exactly how BN operates in a CNN.
2. **Omitting Scale and Shift:** By omitting $\gamma$ and $\beta$, you are effectively setting $\gamma = 1$ and $\beta = 0$. This reduces the BN operation to pure statistical standardization (z-score normalization). As long as you state in your text that this was done to reduce redundancy with the linear layer, it is a perfectly valid network assumption.

Because you added BN _before_ the ReLU, the gradient from the ReLU derivative no longer flows directly into the convolution output ($z$). Instead, it flows into the normalized output ($\hat{z}$), and you must use the BN chain rule to pass the gradient back to $z$.

Here are the exact revisions you need to make to your "manualisation" document. I have provided the step-by-step chain rule exactly as it appears in the Ioffe and Szegedy (2015) paper, as this is the best way to demonstrate rigorous mathematical "manualisation".

---

### PART 1: Feed-Forward Notation Updates

To make the backpropagation equations mathematically valid, you must briefly define the intermediate BN variables ($\mu$, $\sigma^2$, $\hat{z}$) in your Feed-Forward chapter.

**1. For Filter Vertikal CNN Front-End (Feed-Forward)**
Change your current equation:
$$O_{\text{CV}} = \text{MaxPool}(\text{ReLU}(z_{\text{V}}))$$

**To this:**

$$
\mu_{\text{V}} = \frac{1}{6} \sum_{i_{\text{V}}=0}^5 z_{\text{V}}(i_{\text{V}}, j_{\text{V}})
$$

$$
\sigma^2_{\text{V}} = \frac{1}{6} \sum_{i_{\text{V}}=0}^5 (z_{\text{V}}(i_{\text{V}}, j_{\text{V}}) - \mu_{\text{V}})^2
$$

$$
\hat{z}_{\text{V}}(i_{\text{V}}, j_{\text{V}}) = \frac{z_{\text{V}}(i_{\text{V}}, j_{\text{V}}) - \mu_{\text{V}}}{\sqrt{\sigma^2_{\text{V}} + \epsilon}}
$$

$$
O_{\text{CV}} = \text{MaxPool}(\text{ReLU}(\hat{z}_{\text{V}}))
$$

_(Add to Keterangan: $\mu_{\text{V}}$ adalah rata-rata, $\sigma^2_{\text{V}}$ adalah varians, $\epsilon$ adalah konstanta stabilitas yang diatur menjadi 0, dan $\hat{z}_{\text{V}}$ adalah nilai setelah standardisasi)._

**2. For Filter Horizontal CNN Front-End (Feed-Forward)**
Change your current equation:
$$O_{\text{CH}}=\text{ReLU}(z_H)$$

**To this:**

$$
\mu_{\text{H}} = \frac{1}{6} \sum_{i_{\text{H}}=0}^5 z_{\text{H}}(i_{\text{H}})
$$

$$
\sigma^2_{\text{H}} = \frac{1}{6} \sum_{i_{\text{H}}=0}^5 (z_{\text{H}}(i_{\text{H}}) - \mu_{\text{H}})^2
$$

$$
\hat{z}_{\text{H}}(i_{\text{H}}) = \frac{z_{\text{H}}(i_{\text{H}}) - \mu_{\text{H}}}{\sqrt{\sigma^2_{\text{H}} + \epsilon}}
$$

$$
O_{\text{CH}}=\text{ReLU}(\hat{z}_{\text{H}})
$$

---

### PART 2: Backpropagation Revisions (Vertical Filter)

Navigate to `### Filter Vertikal CNN Front-End` under the `## Perumusan Backpropagation` chapter.

**1. Replace the ReLU derivative**
Find this block:

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})}=\frac{\delta L_{\text{BCE}}}{\delta {MP}_{\text{CV}}(i_{\text{V}}, j_{\text{V}})} \times \dots
$$

**Replace it entirely with the gradient flowing into $\hat{z}_{\text{V}}$:**

$$
\frac{\delta L_{\text{BCE}}}{\delta \hat{z}_{\text{V}}(i_{\text{V}}, j_{\text{V}})}=\frac{\delta L_{\text{BCE}}}{\delta {MP}_{\text{CV}}(i_{\text{V}}, j_{\text{V}})} \times \frac{\delta {MP}_{\text{CV}}(i_{\text{V}}, j_{\text{V}})}{\delta \hat{z}_{\text{V}}(i_{\text{V}}, j_{\text{V}})} \\
=\frac{\delta L_{\text{BCE}}}{\delta {MP}_{\text{CV}}(i_{\text{V}}, j_{\text{V}})} \times \mathbb{I}_{\mathbb{Z}^+}(\hat{z}_{\text{V}}(i_{\text{V}}, j_{\text{V}}))
$$

**2. Insert the Batch Normalization Chain Rule**
Directly below the block you just replaced, **insert** the following equations. This is the exact chain rule defined by Ioffe and Szegedy (2015) to pass the gradient from $\hat{z}_{\text{V}}$ to $z_{\text{V}}$:

$$
\frac{\delta L_{\text{BCE}}}{\delta \sigma^2_{\text{V}}} = \sum_{k=0}^5 \left( \frac{\delta L_{\text{BCE}}}{\delta \hat{z}_{\text{V}}(k, j_{\text{V}})} \times (z_{\text{V}}(k, j_{\text{V}}) - \mu_{\text{V}}) \times \frac{-1}{2}(\sigma^2_{\text{V}} + \epsilon)^{-3/2} \right)
$$

$$
\frac{\delta L_{\text{BCE}}}{\delta \mu_{\text{V}}} = \left( \sum_{k=0}^5 \frac{\delta L_{\text{BCE}}}{\delta \hat{z}_{\text{V}}(k, j_{\text{V}})} \times \frac{-1}{\sqrt{\sigma^2_{\text{V}} + \epsilon}} \right) + \left( \frac{\delta L_{\text{BCE}}}{\delta \sigma^2_{\text{V}}} \times \frac{\sum_{k=0}^5 -2(z_{\text{V}}(k, j_{\text{V}}) - \mu_{\text{V}})}{6} \right)
$$

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})} = \left( \frac{\delta L_{\text{BCE}}}{\delta \hat{z}_{\text{V}}(i_{\text{V}}, j_{\text{V}})} \times \frac{1}{\sqrt{\sigma^2_{\text{V}} + \epsilon}} \right) + \left( \frac{\delta L_{\text{BCE}}}{\delta \sigma^2_{\text{V}}} \times \frac{2(z_{\text{V}}(i_{\text{V}}, j_{\text{V}}) - \mu_{\text{V}})}{6} \right) + \left( \frac{\delta L_{\text{BCE}}}{\delta \mu_{\text{V}}} \times \frac{1}{6} \right)
$$

Keterangan:

1. Ketiga persamaan di atas merupakan turunan berantai (_chain rule_) dari operasi _Batch Normalization_ seperti yang didefinisikan oleh Ioffe dan Szegedy (2015).
2. $k \in \{0,1,2,3,4,5\}$ adalah iterator penjumlahan sepanjang sumbu waktu (karena normalisasi dilakukan secara mandiri pada setiap fitur sepanjang waktu).
3. Karena parameter $\gamma$ dan $\beta$ dihilangkan pada asumsi jaringan ini, nilai $\gamma$ secara matematis dianggap 1, sehingga tidak muncul pada persamaan turunan pertama. Suku kedua pada persamaan turunan $\mu_{\text{V}}$ secara matematis akan selalu bernilai 0 karena $\sum (z - \mu) = 0$, namun tetap dituliskan untuk merepresentasikan turunan asli secara utuh.

_(Note: After this new block, your equations for $\frac{\delta L}{\delta K_{\text{V}}}$ and $\frac{\delta L}{\delta b_{\text{V}}}$ remain perfectly correct and unchanged, because they use the newly derived $\frac{\delta L}{\delta z_{\text{V}}}$)._

---

### PART 3: Backpropagation Revisions (Horizontal Filter)

Navigate to `### Filter Horizontal CNN Front-End` under the `## Perumusan Backpropagation` chapter.

**1. Replace the ReLU derivative**
Find this block:

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{\text{H}}(i_{\text{H}})}=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CH}}(i_{\text{H}})} \times \left[\frac{\delta O_{\text{CH}}}{\delta z_{\text{H}}} \right]_{i_{\text{H}}} \\
=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CH}}(i_{\text{H}})} \times \mathbb{I}_{\mathbb{Z}^+}(z_{\text{H}}(i_{\text{H}}))
$$

**Replace it entirely with the gradient flowing into $\hat{z}_{\text{H}}$:**

$$
\frac{\delta L_{\text{BCE}}}{\delta \hat{z}_{\text{H}}(i_{\text{H}})}=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CH}}(i_{\text{H}})} \times \left[\frac{\delta O_{\text{CH}}}{\delta \hat{z}_{\text{H}}} \right]_{i_{\text{H}}} \\
=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CH}}(i_{\text{H}})} \times \mathbb{I}_{\mathbb{Z}^+}(\hat{z}_{\text{H}}(i_{\text{H}}))
$$

**2. Insert the Batch Normalization Chain Rule**
Directly below the block you just replaced, **insert** the following equations for the horizontal BN derivative:

$$
\frac{\delta L_{\text{BCE}}}{\delta \sigma^2_{\text{H}}} = \sum_{k=0}^5 \left( \frac{\delta L_{\text{BCE}}}{\delta \hat{z}_{\text{H}}(k)} \times (z_{\text{H}}(k) - \mu_{\text{H}}) \times \frac{-1}{2}(\sigma^2_{\text{H}} + \epsilon)^{-3/2} \right)
$$

$$
\frac{\delta L_{\text{BCE}}}{\delta \mu_{\text{H}}} = \left( \sum_{k=0}^5 \frac{\delta L_{\text{BCE}}}{\delta \hat{z}_{\text{H}}(k)} \times \frac{-1}{\sqrt{\sigma^2_{\text{H}} + \epsilon}} \right) + \left( \frac{\delta L_{\text{BCE}}}{\delta \sigma^2_{\text{H}}} \times \frac{\sum_{k=0}^5 -2(z_{\text{H}}(k) - \mu_{\text{H}})}{6} \right)
$$

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{\text{H}}(i_{\text{H}})} = \left( \frac{\delta L_{\text{BCE}}}{\delta \hat{z}_{\text{H}}(i_{\text{H}})} \times \frac{1}{\sqrt{\sigma^2_{\text{H}} + \epsilon}} \right) + \left( \frac{\delta L_{\text{BCE}}}{\delta \sigma^2_{\text{H}}} \times \frac{2(z_{\text{H}}(i_{\text{H}}) - \mu_{\text{H}})}{6} \right) + \left( \frac{\delta L_{\text{BCE}}}{\delta \mu_{\text{H}}} \times \frac{1}{6} \right)
$$

Keterangan: Serupa dengan filter vertikal, ketiga persamaan ini mendefinisikan turunan _Batch Normalization_ untuk filter horizontal. Karena konvolusi ini adalah 1D, indeks yang digunakan hanya indeks waktu $k$ dan $i_{\text{H}}$.

_(Again, your equations for $\frac{\delta L}{\delta K_{\text{H}}}$ and $\frac{\delta L}{\delta b_{\text{H}}}$ remain unchanged!)_
