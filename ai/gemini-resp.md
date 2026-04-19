# Identifikasi Permasalahan

Eksperimen ini bertujuan untuk mengotomatisasi pelabelan emosi dan tema pada musik. File musik atau digital audio dapat digunakan sebagai input untuk sistem tersebut. Namun, diskusi pada bab 2.2.1 telah menunjukkan bahwa mengubah digital audio menjadi spectrogram lebih bermanfaat untuk deep neural networks. Dalam signal processing, digital audio terdiri dari nilai-nilai diskrit pada setiap time-step berdasarkan sample rate. Nilai-nilai diskrit tersebut memperhitungkan amplitude, frequency, dan offset dari sinusoid. Waktu dalam digital audio didasarkan pada sample rate atau pada fraksi diskrit dari satu detik di mana sinyal direkam. Untuk file MP3, sample rate yang biasa digunakan adalah 44100 Hz, yang berarti terdapat 44100 nilai diskrit dalam satu detik. Nilai-nilai ini dapat direpresentasikan sebagai sebuah array di mana indeks dari array tersebut adalah time-step.

Spectrogram umumnya digunakan untuk merepresentasikan digital audio dalam speech recognition dan MER. Dalam hal ini, jenis spectrogram yang digunakan adalah log-mel spectrogram. Mengonversi menjadi log-mel spectrogram membutuhkan perhitungan STFT. STFT menghitung DFT untuk frame pendek dengan panjang berupa fraksi arbitrer dari sample rate. STFT mengasumsikan bahwa sinyal tersebut tidak atau kemungkinan tidak akan berulang. DFT mengasumsikan sebaliknya, oleh karena itu tidak layak untuk diterapkan pada digital audio. STFT mengambil parameter: sample rate, frame length, hop length, dan jumlah mel band. Output-nya adalah matriks 2D di mana setiap baris merepresentasikan mel band dan setiap kolom merepresentasikan time-frame. Data pada setiap mel band pada suatu time-frame adalah magnitude dalam dB. Magnitude tersebut kemudian diskalakan secara logaritmik.

Data yang disediakan dalam dataset MTG-Jamendo terdiri dari metadata dan digital audio. File digital audio dan metadata diunduh secara terpisah. Tabel X.X menunjukkan lima baris pertama dari metadata lagu. Dataset ini terdiri dari 55.609 file audio yang telah melalui proses preprocessing oleh penulis dataset seperti yang didiskusikan pada bab 3.2.2. Dataset yang akan digunakan dalam eksperimen ini adalah subset "mood/theme" yang hanya memiliki 18.486 file. Tidak ada informasi identifikasi untuk setiap file audio sehingga training tidak menunjukkan efek artist dan album (bogdanov 2019 mtg jamendo). Split training, validation, dan testing untuk subset ini telah diatur oleh penulis dataset. Perkiraan split adalah 60% untuk training serta masing-masing 20% untuk validation dan testing. Split dilakukan secara acak tetapi dipastikan bahwa tidak ada track yang muncul di lebih dari satu set dan tidak ada track di set mana pun yang berasal dari artist yang sama yang ada di set lainnya, semua label ada di ketiga split, dan setiap label di setiap split diwakili oleh setidaknya 40 file dan 10 artist pada split training serta masing-masing 20 file dan 5 artist pada split validation dan testing. Selama training, validation, dan inference, data dibagi menjadi chunk berdurasi 15 detik untuk keseluruhan durasi lagu.

Tabel X.X Lima baris pertama metadata lagu (image in Google Docs)

Ketiga back-end dalam eksperimen ini adalah aspek variabel yang memengaruhi skor PR-AUC dan ROC-AUC. Perbandingan dari back-end yang berbeda dirancang agar adil dalam hal parameter count dari model. Baseline dari ketiganya adalah CNN back-end seperti yang diusulkan oleh Pons et al (2018). Ini terdiri dari 3 layer convolution dengan satu max-pooling yang disisipkan di antara layer ke-2 dan ke-3 di mana setiap layer memiliki 64 filter. Dua back-end lainnya memiliki jumlah layer yang sama namun tanpa layer max-pooling di antaranya. Parameter untuk back-end lainnya diatur agar serupa dengan maksimum standard deviance sebesar 5% terhadap CNN back-end serupa dengan pengaturan oleh Shim dan Sung (2022).

# Perancangan Algoritme

Alur algoritme seperti yang ditunjukkan pada Gambar X.1 terdiri dari tiga langkah utama: preprocessing, perhitungan spectrogram, dan modeling. Preprocessing telah didiskusikan pada bab 3.2.4. Setelah menghitung STFT, output spectrogram kemudian dinormalisasi menggunakan z-score normalisation. Tanpa batching, semua split 15 detik dari audio untuk setiap file training akan digunakan karena setiap model dapat menerima time length yang bervariasi. Namun, karena batching digunakan dalam implementasinya, split terakhir dari audio training dibuang karena batching membutuhkan time length yang sama untuk semua input di dalam batch. Batch size-nya adalah 32. Hal ini merefleksikan literatur terdahulu seperti Choi et al. (2016conv) dan Pons et al. (2018) yang menggunakan batch size bawaan sebesar 32 yang disediakan oleh TensorFlow. Network yang disederhanakan mengasumsikan tidak adanya penggunaan mini-batching dan karenanya juga tidak ada batch normalisation dan layer normalisation. Dataset contoh untuk tahap modeling akan terdiri dari 5 baris data arbitrer yang disampel dari subset yang terdiri dari 3 label arbitrer: happy, sad, dan tense. Dataset contoh ditunjukkan pada Tabel X.X. Amplitude log-mel spectrogram pada setiap frequency bin diobservasi dalam eksplorasi data awal berada di sekitar -80 dan 10 (dalam dB). Dataset contoh merefleksikan rentang ini.

Gambar X.1 \<FLOWCHART 1\> (drawio:perancangan1)

Tabel X.X \<EXAMPLE DATASET\>

Gambar X.2 \<FLOWCHART PELATIHAN DETAIL\> (drawio:pelatihan1)

Untuk membantu selama training, dataset MTG-Jamendo menyediakan split resmi untuk training, validation, dan testing. Split-nya berturut-turut adalah 60%, 20%, dan 20%. Gambar X.1 dan Gambar X.2 mengilustrasikan kapan setiap split dataset digunakan. Secara khusus, validation set digunakan setelah satu epoch dari training selesai. Meskipun ini dimaksudkan untuk eksperimentasi model dalam literatur yang menggunakan dataset ini, validation set akan digunakan untuk menentukan checkpoint model dan membatasi epoch. Training berakhir ketika jumlah maksimum epoch telah tercapai. Checkpoint dengan performa terbaik di dalam model-model tersebut dimuat dan diuji. Hasilnya kemudian dibandingkan dan dianalisis sesuai dengan pertanyaan penelitian. Publikasi serupa oleh Pons et al. (2018) dan Choi et al. (2016conv) yang menjadi dasar untuk eksperimen ini berturut-turut menggunakan learning rate sebesar 0.001 dan 0.005. Penelitian serupa menggunakan CNN dari Choi et al. (2016automatic) menunjukkan diminishing returns setelah 40 epoch menggunakan optimizer ADAM. Keduanya mengobservasi performa dari model per epoch dan mengakhiri training berdasarkan keputusan arbitrer. Untuk model attention, Sukhavasi dan Adapa (2019) membatasi hingga 60 epoch untuk ADAM dengan learning rate sebesar 0.001 sebelum mengimplementasikan penyesuaian learning rate menggunakan metode lain. Won et al. (2019toward) setuju dengan jumlah maksimum epoch dan menetapkan learning rate menjadi 0.0001. Contoh perhitungan dan implementasi dari eksperimen ini akan menggunakan learning rate bawaan ADAM sebesar 0.001. Implementasi ini akan membatasi epoch hingga 60.

# Front-End

Gambar X.X (drawio:cnn-fe)

<!-- // FLESH THIS SECTION OUT TO BE A COMPARISON TABLE AND NOT JUST BULLET POINTS AND NUMBERED LIST

- Number of filters: 2
- Input is split to 2 types: for the vertical and horizontal filters
- For the vertical filter type: max-pool AFTER convolution
- For the horizontal filter type: mean-pool BEFORE convolution
- Filter 1 (namely $F_V$) size: 5x3 (vertical)
- Filter 2 (namely $F_H$) size: 1x3 (horizontal) -->

Asumsi forward (filter vertikal) adalah sebagai berikut:

1. Input: 6x5 (HxW) di mana H adalah waktu dan W adalah frekuensi. Ukuran filter adalah 5x3 (vertikal).
2. Jumlah channel: 1. Hal ini dikarenakan spectrogram terdiri dari satu skala warna dan bukan 3 seperti gambar RGB.
3. Convolution dengan filter vertikal dengan padding "same".
4. Menggunakan ReLU setelah convolution.
5. Lakukan max-pool pada hasilnya.

Asumsi forward (filter horizontal) adalah sebagai berikut:

1. Input: 6x5 (HxW) di mana H adalah frekuensi dan W adalah waktu. Ukuran filter adalah 1x3 (horizontal).
2. Jumlah channel: 1. Hal ini dikarenakan spectrogram terdiri dari satu skala warna dan bukan 3 seperti gambar RGB.
3. Lakukan mean-pool pada input spectrogram sehingga dimensi output memiliki height sebesar 1 dan width sebesar waktu.
4. Convolution dengan filter horizontal dengan padding "same".
5. Menggunakan ReLU setelah convolution.

Gambar X.X memvisualisasikan asumsi network front-end. Perhatikan bahwa dalam feed-forward filter vertikal, H adalah waktu sedangkan W adalah frekuensi, tetapi dalam feed-forward filter horizontal, H adalah frekuensi sedangkan W adalah waktu. Hal ini meniru modeling data time series di mana representasi tabular memiliki waktu (sebagai kolom) di mana setiap baris menunjukkan nilai-nilai pada waktu tersebut. Nilai dari setiap feature adalah kolom-kolomnya atau width, dengan asumsi representasi HxW. Rasionalisasi ini juga dijelaskan dalam Pons et al. (2018) di mana filter bertujuan untuk mempelajari feature di sepanjang sumbu waktu. Kedua filter tersebut mereduksi dimensionalitas dari frekuensi sambil menjaga sumbu waktu tetap utuh. Ini selalu menghasilkan sesuatu yang mirip dengan data time series. Hasil dari convolution vertikal dan horizontal kemudian di-concatenate untuk menjadi sebuah tensor dengan sumbu waktu yang utuh dan sumbu frequency bin yang direduksi dimensionalitasnya. Concatenation ini adalah alasan untuk memisahkan feed-forward untuk front-end berdasarkan bentuk filter.

<!-- --- -->

<!-- // ADD VISUAL ILLUSTRATIONS FOR THE FEED-FORWARD OF VERTICAL AND HORIzONTAL FILTERS -->

## Feed-Forward dari Convolution Filter Vertikal

Diberikan ukuran input 6x5 dengan padding "same", bentuk output dari convolution sama dengan bentuk input. Padding "same" juga dikenal sebagai half padding. Jumlah padding yang dibutuhkan untuk mencapai ini dijelaskan dalam Domoulin (2018) di mana:

$$p=\left\lfloor{\frac{k}{2}}\right\rfloor$$

Catatan:

1. p adalah jumlah padding yang akan diterapkan pada setiap boundary
2. k adalah ukuran kernel dengan asumsi kernel persegi

Dengan kernel persegi panjang 2D, persamaan dapat diekspansi menjadi:

$$p_H=\left\lfloor{\frac{k_H}{2}}\right\rfloor$$
$$p_W=\left\lfloor{\frac{k_W}{2}}\right\rfloor$$

Dengan ukuran kernel 5x3 dengan padding "same", jumlah padding pada matriks input adalah sebagai berikut:

$$p_H=\left\lfloor{\frac{5}{2}}\right\rfloor=2$$
$$p_W=\left\lfloor{\frac{3}{2}}\right\rfloor=1$$

Secara visual, matriks input yang di-pad ditunjukkan pada Gambar X.X

Gambar X.X \<PADDED MATRIX\> (drawio:paddedmatrix)

Catatan:

1. p_H adalah jumlah padding yang akan diterapkan pada boundary height (atas dan bawah) dari matriks input
2. k_H adalah height dari kernel
3. p_W adalah jumlah padding yang akan diterapkan pada boundary width (kiri dan kanan) dari matriks input
4. k_W adalah width dari kernel

Operasi max-pooling tidak menggunakan padding tetapi memiliki ukuran kernel dan stride 1x5 di mana 5 adalah width dari sumbu frequency bin. Max-pooling menggunakan sliding window yang melakukan stride untuk menangkap sebagian matriks berdasarkan ukuran kernel. Operasi ini menemukan nilai maksimum pada bagian matriks tersebut. Operasi ini secara matematis didefinisikan sebagai berikut:

$$
\text{MaxPool}(I)\\
=\max_{m=0,\dots,k_H-1}\max_{n=0,\dots,k_W-1}\text{in}(\text{stride}[0]\times h+m,\text{stride}[1] \times w+n)
$$

Catatan:

1. I adalah matriks (atau vektor) input yang akan di-max-pool
1. $\text{in}(\cdot)$ berarti matriks input dari max-pooling
1. stride adalah sebuah tuple dengan elemen pada indeks ke-0 dan ke-1 yang menandakan seberapa jauh window tersebut bergeser (slide) melintasi semua lokasi di dalam input
1. h adalah height dari matriks input dan sebagai parameter dari fungsi output
1. w adalah width dari matriks input dan sebagai parameter dari fungsi output
1. m adalah indeks baris dari matriks input
1. n adalah indeks kolom dari matriks input
1. k_H adalah height dari kernel
1. k_W adalah width dari kernel

Operasi tersebut memiliki bentuk output yang secara matematis didefinisikan sebagai berikut:

$$H_{out}=\left\lfloor \frac{H_{in}+2*\text{padding}[0]-\text{dilation}[0]\times(\text{k}[0]-1)-1}{\text{stride}[0]}+1 \right\rfloor$$

Catatan:

1. H_out adalah height dari output
2. H_in adalah height dari input
3. padding[0] berarti indeks pertama dari tuple padding atau seberapa banyak input di-pad pada bagian atas dan bawah
4. dilation[0] berarti indeks pertama dari tuple dilation atau laju pertambahan ukuran kernel seiring dengan dilewatinya elemen-elemen input. Nilai dilation yang diatur ke 1 sama dengan tidak menerapkan dilation sama sekali.
5. k[0] berarti indeks pertama dari tuple k atau ukuran kernel yang digunakan sebagai window untuk menghitung nilai max pada window tertentu
6. stride[0] berarti indeks pertama dari tuple stride atau seberapa jauh window tersebut bergeser (slide) melintasi semua lokasi di dalam input

$$W_{out}=\left\lfloor \frac{W_{in}+2*\text{padding}[1]-\text{dilation}[1]\times(\text{k}[1]-1)-1}{\text{stride}[1]}+1 \right\rfloor$$

Catatan:

1. W_out adalah width dari output
2. W_in adalah width dari input
3. padding[1] berarti indeks kedua dari tuple padding
4. dilation[1] berarti indeks kedua dari tuple dilation
5. k[1] berarti indeks kedua dari tuple k
6. stride[1] berarti indeks kedua dari tuple stride

Oleh karena itu, bentuk dari output setelah operasi max-pooling ditunjukkan berupa bentuk 6x1 seperti yang dihitung: (2 separate eqs)

$$H_{out}=\left\lfloor \frac{6+2*0-1\times(1-1)-1}{1}+1 \right\rfloor=6$$

$$W_{out}=\left\lfloor \frac{5+2*0-1\times(5-1)-1}{5}+1 \right\rfloor=1$$

Menggabungkan semuanya, bentuk akhir (termasuk sumbu channel) dari feed-forward filter vertikal adalah 1x6x1 dengan dimensi: channel, waktu (height), dan frekuensi (width).

## Feed-Forward dari Convolution Filter Horizontal

Diberikan bentuk input 6x5, operasi mean-pooling tidak menggunakan padding tetapi memiliki ukuran kernel dan stride 1x5 di mana 5 adalah width dari sumbu frequency bin. Dengan cara yang serupa dengan max-pooling, mean-pooling mengambil rata-rata dari semua nilai pada bagian matriks tersebut. Operasi ini memiliki bentuk output yang sama seperti yang didefinisikan pada max-pooling. Oleh karena itu, input dari convolution memiliki bentuk 1x6. Operasi ini secara matematis didefinisikan sebagai berikut:

$$
\text{MeanPool}(I)\\
=\frac{1}{k_H+k_W}\sum_{m=0}^{k_H-1}\sum_{n=0}^{k_W-1}\text{in}(\text{stride}[0]\times h+m,\text{stride}[1] \times w+n)
$$

Catatan:

1. k_H adalah height dari kernel
2. k_W adalah width dari kernel
3. Variabel lainnya memiliki catatan yang serupa dengan operasi max-pooling yang didefinisikan dalam Persamaan X.X (\<POINT TO THE MAXPOOL OP ABOVE\>).

Output dari mean-pooling adalah input dari layer convolution dengan filter horizontal. Convolution ini sekarang menjadi convolution 1D di mana frequency bin telah dirata-ratakan per time-frame. Namun, untuk mengakomodasi bentuk input dari class PyTorch Conv1d yang berupa: channel dan length, dimensi frekuensi dihilangkan (dropped) sementara dimensi length menjadi sumbu waktu. Hal ini menyesuaikan sumbu waktu sebagai dimensi length dari sequence yang terdiri dari 6 nilai dari frequency bin yang telah dirata-ratakan. Selain itu, convolution ini memiliki padding "same", oleh karena itu ukuran output-nya sama dengan ukuran input-nya, yaitu 1x6 (sebuah array dengan 6 nilai). Operasi ini memiliki bentuk output yang secara matematis didefinisikan sebagai berikut:

$$L_{out}=\left\lfloor \frac{L_{in}+2*\text{padding}-\text{dilation}\times(\text{kernel\_size}-1)-1}{\text{stride}}+1 \right\rfloor$$

Catatan:

1. L_out adalah length dari output sequence
2. L_in adalah length dari input sequence
3. padding berarti seberapa banyak input di-pad pada kedua ujung sequence
4. dilation berarti indeks pertama dari tuple dilation
5. kernel_size berarti indeks pertama dari tuple kernel_size
6. stride berarti indeks pertama dari tuple stride

Karena ukuran kernel didefinisikan dalam dua dimensi (1x3), nilai yang digunakan untuk kernel_size dan stride adalah yang lebih besar di antara keduanya. Hal ini dikarenakan convolution 1D pada dasarnya men-convolve data melalui length-nya daripada height-nya, mengasumsikan data tersebut menyerupai array. Oleh karena itu, bentuk output setelah operasi convolution ditunjukkan memiliki length 6 seperti yang dihitung:

$$L_{out}=\left\lfloor \frac{6+2*1-1\times(3-1)-1}{1}+1 \right\rfloor=6$$

Menggabungkan semuanya, bentuk akhir (termasuk sumbu channel) dari feed-forward filter horizontal adalah 1x6 dengan dimensi: channel dan waktu (height)

## Concatenation dari Convolution Filter Vertikal dan Horizontal

Proses concatenation mengambil output dari convolution filter vertikal maupun horizontal. Karena temporal convolution menggunakan convolution 1D, dimensi baru dapat ditambahkan setelah dimensi waktu untuk menggantikan dimensi frekuensi yang dihilangkan sebelum convolution. Oleh karena itu, bentuk input dari kedua convolution tersebut adalah 1x6x1, sehingga hasil concatenation-nya akan memiliki bentuk 1x6x2 yang menjaga length dari sumbu waktu tetap utuh. Urutan dari concatenation (dengan asumsi matriks dibaca dari kiri ke kanan) adalah convolution filter vertikal lalu horizontal. Dimensinya adalah: channel, waktu, dan frekuensi.

# Back-End

## CNN

CNN back-end terdiri dari 1 layer convolution masing-masing dengan 1 filter berbentuk 3xW. W adalah width dari concatenated feature; dari hasil seperti yang ditunjukkan pada CHAPTER X.Y.Z, nilainya adalah 2. Back-end ini menerima bentuk input 6x2 dari front-end. Height dari kernel adalah sumbu waktu sementara width adalah sumbu concatenated feature. Layer convolution ini menggunakan padding "same". Fungsi aktivasi untuk kedua layer adalah ReLU. Output dari back-end ini adalah filter yang di-concatenate setelah ReLU.

Gambar X.X (drawio:cnn-be)

## CNN dengan GRU

Back-end CNN dengan GRU terdiri dari 1 layer GRU uni-directional dengan satu hidden layer. Back-end ini menerima bentuk input 6x2 dari front-end. Desain ini sejalan dengan Cho et al. (2014). Tensor input diadaptasi menjadi bentuk LxH_in di mana L adalah sequence length atau height dari sumbu waktu dan H_in adalah width dari concatenated feature; dari hasil seperti yang ditunjukkan pada CHAPTER X.Y.Z, nilainya adalah 2. Output dari back-end ini adalah final hidden state dari GRU dengan bentuk LxH_out. H_out ditetapkan menjadi 2. Output dari back-end ini mengembalikan feature dari hidden state terakhir untuk setiap time-step. Bentuknya sama dengan input.

Gambar X.X (drawio:cnn-gru-be)

## CNN dengan Self-Attention

Back-end CNN dengan Self-Attention terdiri dari 1 layer Self-Attention dengan 2 head. Self-Attention diimplementasikan sebagai bagian dari bagian encoder dari Transformer. Transformer mengambil input berupa embedding vector. Mengingat output dari front-end berbentuk 6x2, setiap time-step darinya dapat diperlakukan sebagai 6 embedding vector yang masing-masing berbentuk 1x2. Dalam perhitungannya, ini dilakukan secara bersamaan (simultaneously). Dimensi dari query, keys, dan value ditentukan oleh Persamaan 2.12, di mana d_model sama dengan 2 karena width dari concatenated feature adalah 2 seperti yang ditunjukkan pada CHAPTER X.Y.Z dan h (atau head) sama dengan 2. Oleh karena itu, d_k dan d_v sama dengan 1. Mengingat ada dua head, ini berarti setiap head mendapat input dengan bentuk 6x1. Setelah menghitung Self-Attention, hasil dari setiap head di-concatenate menjadi 6x2 lagi untuk dilanjutkan ke layer Feed-Forward Network (FFN) dari Transformer. Fungsi aktivasi pada bagian feed-forward network dari arsitektur tersebut adalah ReLU. Output dari back-end ini.

Gambar X.X (drawio:cnn-attn-be)

# Classifier

Classifier terdiri dari sebuah layer FC yang mengambil 12 node sebagai input-nya. Output back-end akan di-concatenate menjadi bentuk 12x1. Output tersebut menjadi input dari output layer dengan 3 node yang berkorespondensi dengan tiga label arbitrer yang ditetapkan untuk perhitungan ini sesuai dengan Tabel X.X (\<DUMMY DATASET\>). Fungsi aktivasi untuk FC dan output layer adalah sigmoid.

Gambar X.X (drawio:classifier)

<!-- This is basically Manualisasi -->

# Perumusan Feed-Forward dan Backpropagation

---

// ADD THIS IN TINJAUAN TEORI NOT HERE

Dalam deep learning, convolution biasanya diimplementasikan sebagai cross-correlation. Convolution membalikkan (flip) kernel sementara cross-correlation tidak (Goodfellow 2016). Cross-correlation (convolution dalam deep learning) secara matematis didefinisikan sebagai:

$$S(i,j)=\sum_{m=0}^{M-1}\sum_{n=0}^{N-1}{I(i+m,j+n)\times K(m,n)}$$

Catatan:

1. i dan j berturut-turut adalah indeks baris dan kolom dari input, keduanya dimulai dari 0.
2. m dan n berturut-turut adalah indeks baris dan kolom dari kernel
3. M dan N berturut-turut adalah height dan width dari kernel
4. S(i,j) adalah hasil convolution pada indeks i dan j dari input
5. I(i+m,j+n) berturut-turut adalah indeks baris dan kolom dari input
6. K(m,n) berturut-turut adalah indeks baris dan kolom dari kernel

---

Network yang disederhanakan yang akan digunakan untuk perhitungan numerik penuh adalah model CNN. Definisi feed-forward dan backpropagation untuk model CNN dengan GRU dan model CNN dengan Self-Attention tetap disertakan. Model CNN terdiri dari CNN front-end, CNN back-end, dan classifier. Loss function didefinisikan pada Persamaan 2.24 berdasarkan nilai prediksi. Untuk memudahkan pemahaman, Gambar X.X memvisualisasikan setiap variabel yang merupakan bagian dari model CNN. Pada definisi matematis di CHAPTER X.Y.Z (\<THE TWO SUBCHAPTERS FOR FF AND BPROP BELOW\>), tanda perkalian digunakan untuk menunjukkan perkalian skalar (scalar multiplication). Jika tidak ada, maka itu menunjukkan perkalian matriks (matrix multiplication).

## Perumusan Feed-Forward

Persamaan feed-forward didefinisikan mulai dari fungsi weighted BCE loss ke arah input. Beberapa catatan persamaan tidak lengkap karena telah dicatat pada persamaan sebelumnya.

GAMBAR X.X \<VIS CNN TAPI ADA VARIABLE VARIABLE\> (drawio:cnn-ff-bprop)

### Classifier

$$y_n=\sigma(z_O)$$

Catatan:

1. y_n adalah predicted output
2. $\sigma$ adalah fungsi aktivasi sigmoid
3. z_O adalah output dari output layer sebelum menerapkan fungsi aktivasi seperti yang divisualisasikan pada Gambar X.X

$$z_O=O_{\text{FC}}^T W_O + b_O$$

Catatan:

1. O_FC adalah output dari layer "FC"
2. W_FC adalah bobot (weights) untuk layer "FC"
3. b_FC adalah bias untuk layer "FC"

$$O_{\text{FC}} = \text{Flatten}(O_{\text{c}})$$

Catatan:

1. $O_{\text{c}}$ adalah output dari layer convolution kedua di back-end
2. Flatten(.) adalah operasi berdasar baris (row-wise) yang me-reshape output matriks 2D dari $O_{\text{c}}$ berbentuk 6x2 menjadi vektor 1D berbentuk 12x1.

### CNN Back-End

$$O_{\text{c}}=\operatorname*{ReLU}(z_{\text{c}})$$

Catatan: $z_{\text{c}}$ adalah output dari layer convolution kedua (dinotasikan c) di back-end sebelum menerapkan fungsi aktivasi

Berikut ini menggunakan kembali (re-purposes) definisi matematis dari operasi convolution seperti yang didefinisikan pada Persamaan 2.X (DI BAB 2) untuk digunakan dalam kasus ini:

$$
z_{\text{c}}=S_{\text{c}}(i_{\text{c}},j_{\text{c}}) + b_{c} \\
=\sum_{m_{\text{c}}=0}^{M_{\text{c}}-1}\sum_{n_{\text{c}}=0}^{N_{\text{c}}-1} \big(O_{\text{Cc}}(i_{\text{c}}+m_{\text{c}},j_{\text{c}}+n_{\text{c}})\times K_{\text{c}}(m_{\text{c}},n_{\text{c}}) \big) + b_{c}
$$

Catatan: serupa dengan Persamaan 2.X; I yang didefinisikan pada Persamaan 2.X diganti dengan O_Cc yang merupakan output dari layer convolution pertama di back-end.

$$
O_{\text{Cc}}=\text{Concat}(O_{\text{CV}},O_{\text{CH}})
$$

Catatan:

1. Concat(.) adalah fungsi concatenation berdasar kolom (column-wise) yang diberikan dua input O_CV dan O_CH
2. O_CV adalah output dari convolution vertikal dari front-end; sebaliknya, O_CH adalah output dari convolution horizontal.

### CNN dengan GRU Back-End

Operasi ini menggunakan kembali (repurpose) definisi matematis dari GRU yang diusulkan oleh Cho et al. (2014):

$$
O_{\text{FC}} = \operatorname*{Flatten}(O^T_{\text{GRU}}) \\

O_{\text{GRU}}=\operatorname*{Concat}(h_1,h_2,\dots,h_L)
$$

Catatan:

1. $L=6$ karena ada 6 time-step dalam data tersebut.
2. Concat(.) melakukan concatenate secara column-wise.
3. Output dari layer GRU adalah hidden state pada semua $t$. Masing-masing di-transpose (lihat Gambar X.X (drawio:cnn-gru-be)) lalu di-flatten. Dalam kasus ini, output-nya adalah 6x2 kemudian di-flatten menjadi 12x1.

$$h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t$$

Catatan:

1. $t \in \{1, 2, 3, 4, 5, 6\}$ karena ukuran matriks input adalah 6x2 di mana 6 adalah sumbu waktu. Terdapat 6 time-step.
2. $h_t$ adalah final hidden state dari GRU pada time-step $t$.
3. $\odot$ menunjukkan perkalian element-wise (Hadamard product).
4. $h_0$ diinisialisasi sebagai all-zero vector.

$$\tilde{h}_t = \tanh(W_h  O_{\text{Cc}(t)} + U_h (r_t \odot h_{t-1}) + b_h)$$

Catatan:

1. $\tilde{h}_t$ adalah candidate hidden state pada time-step $t$.
2. $O_{\text{Cc}(t)}$ adalah baris ke-$t$ dari output concatenation front-end $O_{\text{Cc}}$ yang merepresentasikan feature pada time-step $t$.
3. $W_h$ dan $U_h$ adalah weight matrices untuk candidate hidden state, dan $b_h$ adalah bias. Dalam Cho et al. (2014), hal ini dinotasikan hanya sebagai $W$.

$$z_t = \sigma(W_z O_{\text{Cc}(t)}+ U_z h_{t-1} + b_z)$$
$$r_t = \sigma(W_r O_{\text{Cc}(t)}+ U_r h_{t-1} + b_r)$$

Catatan:

1. $z_t$ adalah update gate vector pada time-step $t$.
2. $r_t$ adalah reset gate vector pada time-step $t$.
3. $W_z, U_z, W_r, U_r$ adalah masing-masing weight matrices untuk update dan reset gate, sementara $b_z, b_r$ adalah masing-masing bias-nya.

### CNN dengan Self-Attention Back-End

Operasi ini menggunakan kembali (repurpose) definisi matematis dari Transformer encoder yang diusulkan oleh Vaswani et al. (2017):

$$O_{\text{FC}} = \text{Flatten}(O_{\text{Attn}})$$

Catatan: $O_{\text{Attn}}$ adalah matriks output 6x2 akhir dari back-end Self-Attention.

Layer Position-Wise FFN seperti yang diusulkan dalam Vaswani et al. (2017) terdiri dari dua layer. Layer pertama dan layer kedua memiliki weights dan bias yang berbeda. Layer pertama diterapkan ReLU sebagai fungsi aktivasi sementara yang kedua tidak mendapatkan aktivasi. Seperti yang dicatat dalam paper tersebut, dimensionalitas dari input dan output dari FFN didasarkan pada d_model, dalam kasus ini bernilai 2. Inner layer dari FFN mengambil parameter d_ff atau dimensi dari feed-forward, dalam kasus ini juga diatur menjadi 2. Gambar X.X (drawio:cnn-attn-be) memvisualisasikan perhitungan yang dilakukan di dalam FFN.

$$
O_{\text{Attn}} = O_{\text{MHA}} + \operatorname*{ReLU} (O_{\text{MHA}} W_1 + b_1) W_2 + b_2
$$

Catatan:

1. $O_{\text{Attn}}$ adalah Position-Wise FFN dengan residual connection ke O_MHA atau output dari blok Multi-Head Attention (MHA).
2. $W_1, b_1$ adalah weights dan bias untuk linear transformation pertama.
3. $W_2, b_2$ adalah weights dan bias untuk linear transformation kedua.

$$O_{\text{MHA}} = O_{\text{Cc}} + \text{Concat}(\text{head}_1, \text{head}_2) W^O$$

Catatan:

1. $O_{\text{MHA}}$ adalah output dari blok MHA dengan residual connection yang menambahkan input asli $O_{\text{Cc}}$.
2. $\text{head}_1$ dan $\text{head}_2$ adalah output dari dua individual attention head.
3. $W^O$ adalah output projection weight matrix.
4. $\text{Concat}(\cdot)$ melakukan concatenate dua head tersebut sepanjang dimensi feature.

$$\text{head}_i = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i$$

Catatan:

1. Ini adalah perhitungan scaled dot product attention untuk head ke-$i$, di mana $i \in \{1, 2\}$.
2. $d_k$ adalah dimensi dari keys (didefinisikan sebagai 1 dalam asumsi network).
3. Fungsi $\text{softmax}$ diterapkan secara row-wise.

$$Q_i = O_{\text{Cc}} W_i^Q$$

$$K_i = O_{\text{Cc}} W_i^K$$

$$V_i = O_{\text{Cc}} W_i^V$$

Catatan:

1. $Q_i, K_i, V_i$ adalah matriks query, key, dan value untuk head ke-$i$.
2. $W_i^Q, W_i^K, W_i^V$ adalah learned weight matrices untuk head ke-$i$.

### Filter Vertikal CNN Front-End

Catatan:

1. Concat(.) adalah operasi concatenation yang melakukan concatenate secara column-wise (menjaga length baris tetap sama)
2. O_CV adalah output dari layer convolution vertikal (dinotasikan CV) di front-end
3. O_CH adalah output dari layer convolution horizontal (dinotasikan CH) di front-end

$$O_{\text{CV}} = \text{MaxPool}(\operatorname*{ReLU}(z_{\text{V}}))$$

$$
z_{\text{V}}=S_{\text{V}}(i_{\text{V}},j_{\text{V}}) + b_{\text{V}} \\
=\sum_{m_{\text{V}}=0}^{M_{\text{V}}-1}\sum_{n_{\text{V}}=0}^{N_{\text{V}}-1} \big(I(i_{\text{V}}+m_{\text{V}},j_{\text{V}}+n_{\text{V}})\times K_{\text{V}}(m_{\text{V}},n_{\text{V}}) \big) + b_{\text{V}}
$$

Catatan:

1. Baik Persamaan X.X maupun Persamaan X.X (\<REFERRING TO THE 2 EQS ABOVE\>) serupa dengan Persamaan X.X (\<THE OG CONV EQ ABOVE THESE 2\>) hanya saja diperuntukkan bagi layer convolution vertikal di front-end.
2. MaxPool(.) didefinisikan pada Persamaan X.X (\<THE MAXPOOL DEF ABOVE\>)
3. $I$ adalah input log-mel spectrogram

### Filter Horizontal CNN Front-End

$$O_{\text{CH}}=\operatorname*{ReLU}(z_H)$$

$$
z_{\text{H}}=S_{\text{H}}(i_{\text{H}},j_{\text{H}}) + b_{\text{H}} \\
=\sum_{m_{\text{H}}=0}^{M_{\text{H}}-1}\sum_{n_{\text{H}}=0}^{N_{\text{H}}-1} \big(\text{MeanPool}(I)(i_{\text{H}}+m_{\text{H}},j_{\text{H}}+n_{\text{H}})\times K_{\text{H}}(m_{\text{H}},n_{\text{H}}) \big) + b_{\text{H}}
$$

Catatan:

1. MeanPool(.) didefinisikan pada Persamaan X.X (\<THE MEANPOOL DEF ABOVE\>)
2. Baik Persamaan X.X maupun Persamaan X.X (\<REFERRING TO THE 2 EQS ABOVE\>) serupa dengan Persamaan X.X (\<THE OG CONV EQ ABOVE THESE 2\>) hanya saja diperuntukkan bagi layer convolution horizontal di front-end.

## Perumusan Backpropagation

Dengan definisi persamaan feed-forward pada CHAPTER X.Y.Z (\<THE FF SUBCHAPTER DIRECTLY ABOVE THIS\>), persamaan backpropagation dapat didefinisikan sebagai turunan parsial (partial derivatives) terhadap masing-masing variabel dari setiap persamaan feed-forward. Beberapa nilai pada definisi tersebut telah disubstitusi berdasarkan asumsi network yang didiskusikan pada CHAPTER X.Y.Z (\<THE CHAPTER BEFORE THIS\>). Subscript dari variabel telah divisualisasikan pada Gambar X.X (\<THE DETAILED NETWORK IMAGE ON THE PREV CHAPTER\>)

### Classifier

$$
\frac{\delta L_{\text{BCE}}}{\delta y_n}=-\frac{1}{N} \times \left(P_w \times \frac{t_n}{y_n} - N_w \times \frac{1-t_n}{1-y_n} \right) \\
=-\frac{1}{3} \times \left(P_w \times \frac{t_n}{y_n} - N_w \times \frac{1-t_n}{1-y_n} \right)
$$

$$P_w=\frac{2}{1+p_n}$$
$$N_w=\frac{2\times p_n}{1+p_n}$$

Catatan:

1. N disubstitusi dengan 3 karena terdapat 3 class dalam asumsi network.
2. P_w dan N_w didefinisikan sebagai konstanta untuk membuat definisinya lebih ringkas
3. Variabel-variabel pada Persamaan X.X, Persamaan X.X, dan Persamaan X.X (\<THE 3 EQS ABOVE\>) adalah sama seperti yang didefinisikan pada Persamaan 2.24

$$
\frac{\delta L_{\text{BCE}}}{\delta z_O}=\frac{\delta L_{\text{BCE}}}{\delta y_n} \times \frac{\delta y_n}{\delta z_O} \\
=\frac{\delta L_{\text{BCE}}}{\delta y_n}\times \sigma(z_O) \times (1-\sigma(z_O)) \\
=-\frac{1}{3} \times \left(P_w \times \frac{t_n}{y_n} - N_w \times \frac{1-t_n}{1-y_n} \right) \times \big(y_n \times (1-y_n) \big) \\
=-\frac{1}{3} \times \left(P_w \times t_n \times (1-y_n) - N_w \times (1-t_n) \times y_n \right)
$$

Catatan: Variabel-variabel pada Persamaan X.X (\<THE 1 EQS ABOVE\>) adalah sama seperti yang didefinisikan pada Persamaan 2.24

$$
\frac{\delta L_{\text{BCE}}}{\delta W_{O(j,n)}}=\left[\frac{\delta L_{\text{BCE}}}{\delta y_n}\times \frac{\delta y_n}{\delta z_O}\right]_n \times\left[\frac{\delta z_O}{\delta W_{O(j,n)}}\right]_n \\
=\left[\frac{\delta L_{\text{BCE}}}{\delta z_O}\right]_n \times\left[\frac{\delta z_O}{\delta W_{O(j,n)}}\right]_n \\
=\left[\frac{\delta L_{\text{BCE}}}{\delta z_O}\right]_n \times\left[O_\text{FC}\right]_j
$$

Catatan:

1. n adalah class ke-n di dalam dataset sesuai dengan Persamaan 2.24
2. $W_{O(j,n)}$ adalah weight yang menghubungkan node ke-j dari sisi output FC ke output node ke-n (layer "O" pada Gambar X.X).
3. [.]\_n adalah gradien ke-n. Karena terdapat 3 class, akan ada N perhitungan weighted BCE loss seperti yang didefinisikan pada Persamaan X.X (\<THE EQ ABOVE THIS ONE\>).
4. $z_{O(n)}$ adalah output ke-n setelah perhitungan dengan weight pada output layer yang belum diaktivasi
5. $\left[O_\text{FC}\right]_j$ adalah node ke-j dari sisi output FC.

$$
\frac{\delta L_{\text{BCE}}}{\delta b_{O(n)}}=\left[\frac{\delta L_{\text{BCE}}}{\delta y_n}\times \frac{\delta y_n}{\delta z_O}\right]_n \times\left[\frac{\delta z_O}{\delta b_O}\right]_n \\
=\left[\frac{\delta L_{\text{BCE}}}{\delta z_O}\right]_n \times 1 \\
$$

Catatan: $b_{O(n)}$ adalah bias untuk output ("O" layer) node ke-n

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{FC}(i)}}=\sum_{n=1}^3 \left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{O}(n)}} \times \frac{\delta z_{\text{O}(n)}}{\delta O_{\text{FC}(i)}} \right) \\
=\sum_{n=1}^3 \left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{O}(n)}} \times W_{\text{O}(i,n)} \right)
$$

Catatan:

1. $i=1,2,3,\dots,12$.
2. O_FC(i) adalah output ke-i dari layer "FC" yang mana merupakan flatten layer dari back-end
3. Gradien yang di-propagate ke node ke-$i$ dari layer "FC" adalah hasil penjumlahan dari masing-masing 3 node di output layer atau layer "O" yang ditunjukkan pada Gambar X.X.

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{c}}} = \operatorname*{Reshape}\left( \frac{\delta L_{\text{BCE}}}{\delta O_{\text{FC}}}, (6, 2) \right)
$$

Catatan:

1. O_c adalah output dari layer convolutional (layer "c") dari back-end
2. Reshape(.) adalah untuk me-reshape sebuah vektor guna membentuk array dengan bentuk yang telah ditentukan sesuai dengan parameter kedua dari fungsi tersebut. Dalam kasus ini adalah 6x2, yang ditulis (6, 2) agar tidak dikacaukan dengan perkalian skalar.

Fungsi concatenation seperti yang didefinisikan pada Persamaan X.X (\<SEE THE CONCAT FORMULA IN PREV SUBCHAPTER\>) berarti gradien yang di-propagate harus di-reshape agar bentuknya sama dengan output feed-forward dari layer "c".

### CNN Back-End

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c}}(i_{\text{c}}, j_{\text{c}})}=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{c}}(i_{\text{c}}, j_{\text{c}})} \times \left[\frac{\delta O_{\text{c}}}{\delta z_{\text{c}}} \right]_{i_{\text{c}}, j_{\text{c}}} \\
=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{c}}(i_{\text{c}}, j_{\text{c}})} \times \mathbb{I}_{\mathbb{Z}^+}(z_{\text{c}}(i_{\text{c}}, j_{\text{c}}))
$$

Catatan:

1. $\frac{\delta O_{\text{c}}}{\delta z_{\text{c}}}$ adalah turunan dari fungsi aktivasi ReLU. $\mathbb{I}_{\mathbb{Z}^+}(\cdot)$ adalah fungsi indikator. Ini didefinisikan lebih lanjut pada Persamaan X.X (\<THE EQ BELOW\>)
2. $i_{\text{c}}$ dan $j_{\text{c}}$ adalah indeks baris dan kolom dari feature map yang dihasilkan untuk layer convolution kedua (c).
3. $\frac{\delta L_{\text{BCE}}}{\delta O_{\text{c}}}$ adalah gradient map yang di-propagate dari flattening layer (dihitung di akhir CHAPTER X.Y.Z (\<PREV SUBCHAPTER\>)), yang di-reshape kembali ke matriks 6x2.

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

Catatan: ${\mathbb{Z}^+}=\set{1,2,3,\dots}$

$$
\frac{\delta L_{\text{BCE}}}{\delta K_{\text{c}}(m_{\text{c}}, n_{\text{c}})}=\sum_{i_{\text{c}}=0}^5\sum_{j_{\text{c}}=0}^1\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c}}(i_{\text{c}}, j_{\text{c}})} \times \frac{\delta z_{\text{c}}(i_{\text{c}}, j_{\text{c}})}{\delta K_{\text{c}}(m_{\text{c}}, n_{\text{c}})}\right) \\
=\sum_{i_{\text{c}}=0}^5\sum_{j_{\text{c}}=0}^1\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c}}(i_{\text{c}}, j_{\text{c}})} \times O_{\text{Cc}}(i_{\text{c}}+m_{\text{c}}, j_{\text{c}}+n_{\text{c}})\right)
$$

Catatan:

1. $m_{\text{c}} = \set{0,1,2}$ dan $n_{\text{c}} = \set{0,1}$ berdasarkan batas atas penjumlahan (summation upper bounds) pada Persamaan X.X dan asumsi network bahwa $M_{\text{c}} = 3$ dan $N_{\text{c}} = 2$ yang didiskusikan pada CHAPTER X.Y.Z. (\<THE CNN BACKEND CHAPTER\>)
2. $m_{\text{c}}$ dan $n_{\text{c}}$ adalah indeks baris dan kolom dari kernel.
3. $\frac{\delta L_{\text{BCE}}}{\delta K_{\text{c}}(m_{\text{c}}, n_{\text{c}})}$ berarti bahwa turunannya memperbarui weight dari kernel pada indeks baris dan kolom spesifik.
4. $i_{\text{c}}$ dan $j_{\text{c}}$ adalah indeks baris dan kolom dari input. Batas atasnya berturut-turut adalah 5 dan 1 karena layer "c" memiliki bentuk 6x2.
5. O_Cc adalah output dari concatenation dari front-end.

$$
\frac{\delta L_{\text{BCE}}}{\delta b_{\text{c}}(m_{\text{c}}, n_{\text{c}})}
=\sum_{i_{\text{c}}=0}^5\sum_{j_{\text{c}}=0}^1 \left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c}}(i_{\text{c}}, j_{\text{c}})} \times \frac{\delta z_{\text{c}}(i_{\text{c}}, j_{\text{c}})}{\delta b_{\text{c}}(m_{\text{c}}, n_{\text{c}})} \right) \\
=\sum_{i_{\text{c}}=0}^5\sum_{j_{\text{c}}=0}^1 \left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c}}(i_{\text{c}}, j_{\text{c}})} \times 1\right)
$$

Catatan: $b_{\text{c}}$ adalah variabel bias untuk layer "c".

Ketika menghitung turunan untuk sebuah layer yang tidak berada di bagian paling bawah (bottom) dari network mengikuti Goodfellow (2016), perhitungan indeks dari gradien yang akan di-propagate (dari layer "c") memiliki tanda (sign) yang dibalik. Cross-correlation seperti pada Persamaan 2.X menunjukkan indeks input meningkat (karena penambahan), sementara backpropagation dari proses tersebut menurunkan indeks. Terlebih lagi, ketika dihadapkan dengan indeks yang tidak valid (indeks negatif atau di luar batas bentuk 6x2), nilainya adalah 0. Dalam kasus ini, diasumsikan bahwa O_Cc di-pad dengan nol hingga tak terbatas di semua sisinya. Untuk menunjukkan ini, fungsi aktivasi ReLU pada Persamaan X.X (\<FEEDFORWARD EQ O_Cc=ReLU(z_Cc)\>) dapat diabaikan karena fungsi tersebut tidak mengubah bentuk maupun memengaruhi propagasi gradien. Oleh karena itu, ini dapat didefinisikan bahwa:

$$
O_{\text{Cc}}(i_{\text{c}}+m_{\text{c}},j_{\text{c}}+n_{\text{c}}) = z_{\text{Cc}}(i_{\text{Cc}},j_{\text{Cc}})
$$

Catatan:

1. $O_{\text{Cc}}$ dari Persamaan X.X (\<THE BACKPROP BEFORE THIS\>)
2. $z_{\text{Cc}}$ dari Persamaan X.X (\<FROM FEEDFORWARD\>)

Dengan mencocokkan indeks dari masing-masing parameter $O_{\text{Cc}}$ dan $z_{\text{Cc}}$, dapat didefinisikan bahwa:

$$
i_{\text{c}}+m_{\text{c}}=i_{\text{Cc}} \ ; \quad j_{\text{c}}+n_{\text{c}}=j_{\text{Cc}} \\
i_{\text{c}}=i_{\text{Cc}}-m_{\text{c}} \ ; \quad j_2=j_{\text{Cc}}-n_{\text{c}}
$$

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}}(i_{\text{Cc}}, j_{\text{Cc}})}=\sum_{m_{\text{c}}=0}^2\sum_{n_{\text{c}}=0}^1\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c}}(i_{\text{c}},j_{\text{c}})} \times \frac{\delta z_{\text{c}}(i_{\text{c}},j_{\text{c}})}{\delta O_{\text{Cc}}(i_{\text{Cc}}, j_{\text{Cc}})}\right) \\
=\sum_{m_{\text{c}}=0}^2\sum_{n_{\text{c}}=0}^1\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c}}(i_{\text{Cc}}-m_{\text{c}}, j_{\text{Cc}}-n_{\text{c}})} \times K_{\text{c}}(m_{\text{c}}, n_{\text{c}})\right)
$$

Catatan:

1. $i_{\text{Cc}} \in \set{0,1,2,3,4,5};\  j_{\text{Cc}} \in \set{0,1}$
2. $i_{\text{Cc}}$ dan $j_{\text{Cc}}$ adalah indeks baris dan kolom untuk matriks $O_{\text{Cc}}$ yang di-concatenate.
3. Batas atas untuk penjumlahannya didiskusikan pada Persamaan X.X (\<THE BACKPROP BEFORE THIS\>)

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CV}}} = \frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}}(i_{\text{Cc}},j_{\text{Cc}})} = \frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}}(i_{\text{Cc}},0)}
$$

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CH}}} = \frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}}(i_{\text{Cc}},j_{\text{Cc}})} = \frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}}(i_{\text{Cc}},1)}
$$

Catatan:

1. $i_{\text{Cc}}=\set{0,1,2,3,4,5};\ j_{\text{Cc}}=\set{0,1}$
2. Operasi concatenation feed-forward Concat(.) yang didefinisikan pada Persamaan X.X (\<FF CONCAT ABOVE\>) menggabungkan $O_{\text{CV}}$ dan $O_{\text{CH}}$ secara column-wise (sepanjang dimensi frekuensi/sumbu width), backpropagation memecah gradient matrix kembali menjadi bentuk aslinya 6x1.
3. i_Cc adalah baris ke-i dari output yang di-concatenate dari dua convolution front-end
4. j_Cc adalah kolom ke-j dari output yang di-concatenate dari dua convolution front-end. Ini bernilai 0 atau 1 di mana indeks ke-0 adalah error matrix yang merepresentasikan gradien untuk output filter vertikal dan indeks ke-1 adalah untuk output filter horizontal.
5. $i_{\text{MPCV}}$ adalah baris ke-i dari output max-pooling (dengan $j = 0$) dari layer convolution filter vertikal front-end atau layer "CV"

### CNN dengan GRU Back-End

Gambar X.X (drawio:cnn-gru-be-bprop)

Dikarenakan perbedaan cara hasil dari GRU di-concatenate dan di-flatten dari CNN back-end, proses reshape didefinisikan sebagai:

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{GRU}}} = \operatorname*{Reshape}\left(\frac{\delta L_{\text{BCE}}}{\delta O_{\text{FC}}}, (6, 2) \right)
$$

Parameter-parameter yang diturunkan divisualisasikan pada Gambar X.X. Hidden state terakhir atau h_t dipengaruhi oleh hidden state sebelumnya. Oleh karena itu, gradien yang di-backpropagate pada suatu time-step adalah jumlah dari semua gradien setelahnya atau 0 jika itu merupakan time-step terakhir. Hal ini menciptakan penjumlahan rekursif (recursive sum) saat menghitung weights dan bias dari recurrent network (Zhang 2023). Recursion dari gradien tersebut didefinisikan sebagai:

$$
\frac{\delta L_{\text{BCE}}}{\delta h_t} = \left[\frac{\delta L_{\text{BCE}}}{\delta O_{\text{GRU}}} \right]_t + \frac{\delta O_{\text{GRU}}}{\delta h_{t+1}}\frac{\delta h_{t+1}}{\delta h_t}
$$

Catatan:

1. $t\in\set{1,2,3,4,5,6}$ karena length dari sequence adalah 6.
2. Jika $t=6$ (time-step terakhir), gradien dari masa depan $\frac{\delta L_{\text{BCE}}}{\delta h_{t+1}} = 0$. Backpropagation melakukan iterasi mundur (backwards) dari $t=6$ ke $t=1$.

Untuk memudahkan mendefinisikan gradien, persamaan feed-forward didefinisikan ulang menjadi:

$$
\tilde{h}_t=\operatorname*{tanh}(z_{\tilde{h}(t)}) \\
z_t=\sigma(z_{z(t)}) \\
r_t=\sigma(z_{r(t)})
$$

Catatan: semua persamaan didefinisikan pada Persamaan X.X sampai dengan X.X (\<THE FEEDFORWARD GRU EQS ABOVE\>)

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

Catatan: operasi $U_h (r_t \odot h_{t-1})$ yang didefinisikan pada Persamaan X.X (\<FEEDFORWARD EQ ABOVE\>) tidak dapat didistribusikan. Oleh karena itu, turunan parsialnya harus dicari terhadap nilai tersebut terlebih dahulu.

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{r(t)}} = \frac{\delta L_{\text{BCE}}}{\delta r_t} \odot \frac{\delta r_t}{\delta z_{r(t)}}
= \frac{\delta L_{\text{BCE}}}{\delta r_t} \odot (r_t \odot (1 - r_t))
$$

Seperti yang dicatat sebelumnya, penjumlahan dari setiap time-step bersifat rekursif. Base case-nya diketahui bernilai 0 ketika kasusnya adalah $t+1 \notin t$. Recursive case-nya didefinisikan sebagai penjumlahan dari turunan parsial dari seluruh persamaan yang didefinisikan pada Persamaan X.X sampai dengan X.X (\<THE FEEDFORWARD GRU EQS ABOVE\>) di mana $h_{t-1}$ muncul. Ini didefinisikan menjadi:

$$
\frac{\delta L_{\text{BCE}}}{\delta h_{t-1}} = \left(\frac{\delta L_{\text{BCE}}}{\delta h_t} \odot \frac{\delta h_t}{\delta h_{t+1}}\right) + \left(\frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}(t)}} \times \frac{\delta z_{\tilde{h}(t)}}{\delta h_{t+1}}\right) + \left(\frac{\delta L_{\text{BCE}}}{\delta z_{z(t)}} \times \frac{\delta z_{z(t)}}{\delta h_{t+1}}\right) + \left(\frac{\delta L_{\text{BCE}}}{\delta z_{r(t)}} \times \frac{\delta z_{r(t)}}{\delta h_{t+1}}\right) \\
= \left( \frac{\delta L_{\text{BCE}}}{\delta h_t} \odot z_t \right) + \left( \left( U_h^T \frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}(t)}}  \right) \odot r_t \right) + \left( U_z^T \frac{\delta L_{\text{BCE}}}{\delta z_{z(t)}}  \right) + \left( U_r^T \frac{\delta L_{\text{BCE}}}{\delta z_{r(t)}}  \right)
$$

Pada definisi di Persamaan X.X hingga X.X (\<FEEDFORWARD EQS ABOVE\>), parameter yang dipakai bersama (shared) di seluruh recurrent layer adalah: $W_h$, $U_h$, $b_h$, $W_z$, $U_z$, $b_z$, $W_r$, $U_r$, dan $b_r$. Gradien untuk parameter candidate hidden state didefinisikan menjadi:

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

Serupa dengan itu, gradien untuk update dan reset gate berturut-turut didefinisikan menjadi:

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

Serupa dengan itu, parameter O_Cc(t) muncul beberapa kali seperti yang didefinisikan pada Persamaan X.X hingga X.X (\<FEEDFORWARD EQS ABOVE\>). Turunan dari parameter ini digunakan untuk melakukan propagate gradien ke arah mundur (backwards) menuju front-end layer. Ini didefinisikan menjadi:

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}(t)}} = \left( \frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}(t)}} \times \frac{\delta z_{\tilde{h}(t)}}{\delta O_{\text{Cc}(t)}} \right) + \left( \frac{\delta L_{\text{BCE}}}{\delta z_{z(t)}} \times \frac{\delta z_{z(t)}}{\delta O_{\text{Cc}(t)}} \right) + \left( \frac{\delta L_{\text{BCE}}}{\delta z_{r(t)}} \times \frac{\delta z_{r(t)}}{\delta O_{\text{Cc}(t)}} \right) \\
= \left( W_h^T \frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}(t)}}  \right) + \left( W_z^T \frac{\delta L_{\text{BCE}}}{\delta z_{z(t)}}  \right) + \left( W_r^T \frac{\delta L_{\text{BCE}}}{\delta z_{r(t)}}  \right)
$$

Catatan: operasi ini dilakukan untuk seluruh $t \in \set{1,2,3,4,5,6}$. Perhatikan bahwa ini tidak dijumlahkan tidak seperti weights dan bias pada GRU.

Hasil dari Persamaan X.X (\<THE EQ DIRECTLY ABOVE THIS\>) di-concatenate secara column-wise. Dalam kasus ini, ini akan menghasilkan matriks dengan bentuk 6x2, bentuk yang sama dengan feed-forward-nya. Matriks ini kemudian di-backpropagate ke front-end dengan cara yang sama seperti yang didefinisikan pada Persamaan X.X (\<THE LAST EQ OF CNN BACKEND BACKPROP SUBCHAPTER WITH O_CV AND O_CH\>).

### CNN dengan Self-Attention Back-End

Gambar X.X (drawio:cnn-gru-be-bprop)

Untuk kasus attention back-end, proses reshape didefinisikan sebagai:

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{FFN}(2)}} = \operatorname*{Reshape}\left( \frac{\delta L_{\text{BCE}}}{\delta O_{\text{FC}}}, (6, 2) \right)
$$

Alur backpropagation divisualisasikan pada Gambar X.X (drawio:cnn-attn-be-bprop). Operasi-operasi untuk setiap langkah backpropagation didefinisikan sebagai berikut:

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

Catatan: $j=\set{0,1}$ karena $d_{\text{model}} = 2$

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

Catatan: $j=\set{0,1}$ karena $d_{ff} = 2$

<!-- omha -->

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{MHA}}} = \frac{\delta L_{\text{BCE}}}{\delta O_{\text{FFN}(2)}} + \frac{\delta L_{\text{BCE}}}{\delta z_{\text{FFN}(1)}} \times \frac{\delta z_{\text{FFN}(1)}}{\delta O_{\text{MHA}}} \\
=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{FFN}(2)}} + \left( \frac{\delta L_{\text{BCE}}}{\delta z_{\text{FFN}(1)}} W_1^T \right)
$$

Catatan: suku (term) pertama adalah residual

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

Catatan: $i=\set{1,2}$ karena ada 2 head seperti yang didiskusikan pada asumsi network.

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

Catatan: $\delta_{mn}$ adalah fungsi Kronecker delta yang digunakan pada turunan fungsi aktivasi softmax.

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{\text{Attn}(i)}} = \frac{\delta L_{\text{BCE}}}{\delta s_{\text{Attn}(i)}} \times \left[ \frac{\delta s_{\text{Attn}}}{\delta z_{\text{Attn}}} \right]_i \\
\frac{\delta L_{\text{BCE}}}{\delta z_{\text{Attn}(i, m, n)}}=\sum_{k=0}^{L-1} \left( \frac{\delta L_{\text{BCE}}}{\delta s_{\text{Attn}(i, m, k)}} s_{\text{Attn}(i, m, k)} (\delta_{np} - s_{\text{Attn}(i, m, n)}) \right)
$$

Catatan:

1. $m=\set{0,1,\dots,L-1}$ adalah indeks baris (time-step query sequence).
2. $n=\set{0,1,\dots,L-1}$ adalah indeks kolom (time-step key sequence) yang gradiennya sedang dihitung.
3. $k=\set{0,1,\dots,L-1}$ adalah iterator penjumlahan melintasi kolom-kolom output softmax.
4. L adalah sequence length input (sumbu waktu). Dalam kasus ini, nilainya 6.
5. Persamaan tersebut dievaluasi untuk seluruh $m$ dan seluruh $n$. Bentuk hasil dari turunannya adalah LxL, dalam kasus ini adalah 6x6.

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

Dengan cara yang serupa:

$$
\frac{\delta L_{\text{BCE}}}{\delta W_i^K} = O_{\text{Cc}}^T \frac{\delta L_{\text{BCE}}}{\delta K_i}
$$

$$
\frac{\delta L_{\text{BCE}}}{\delta W_i^V} = O_{\text{Cc}}^T \frac{\delta L_{\text{BCE}}}{\delta V_i}
$$

Catatan: ini adalah gradien-gradien untuk matriks Query, Key, dan Value linear projection weight untuk head $i$. Semuanya memiliki bentuk 2x1. $O_{\text{Cc}}^T$ memiliki bentuk 2x6.

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}}} = \frac{\delta L_{\text{BCE}}}{\delta O_{\text{MHA}}} \\
+ \sum_{i=1}^2 \left( \left[ \frac{\delta L_{\text{BCE}}}{\delta Q_i} \times \frac{\delta Q_i}{\delta O_{\text{Cc}}} \right] + \left[ \frac{\delta L_{\text{BCE}}}{\delta K_i} \times \frac{\delta K_i}{\delta O_{\text{Cc}}} \right] + \left[ \frac{\delta L_{\text{BCE}}}{\delta V_i} \times \frac{\delta V_i}{\delta O_{\text{Cc}}} \right] \right) \\
= \frac{\delta L_{\text{BCE}}}{\delta O_{\text{MHA}}} + \sum_{i=1}^2 \left( \frac{\delta L_{\text{BCE}}}{\delta Q_i} (W_i^Q)^T + \frac{\delta L_{\text{BCE}}}{\delta K_i} (W_i^K)^T + \frac{\delta L_{\text{BCE}}}{\delta V_i} (W_i^V)^T \right)
$$

Catatan:

1. $\frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}}}$ (bentuk 6x2) adalah gradien akumulasi total yang diteruskan kembali (passed back) ke Front-End concatenation layer.
2. Karena $O_{\text{Cc}}$ bercabang (branches out) menuju residual connection, dan menuju matriks $Q_i, K_i, V_i$ untuk kedua head, gradien yang akan di-backpropagate adalah jumlah dari seluruh gradien yang di-propagate mundur (backward) dari semua 7 jalur, lihat Gambar X.X (drawio:cnn-attn-be-bprop).

### Filter Vertikal CNN Front-End

$$
\frac{\delta L_{\text{BCE}}}{\delta {MP}_{\text{CV}}(i_{\text{MPCV}}, j_{\text{MPCV}})}=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CV}}(i_{\text{MPCV}}, 0)} \times \frac{\delta O_{\text{CV}}(i_{\text{MPCV}}, 0)}{\delta {MP}_{\text{CV}}(i_{\text{MPCV}}, j_{\text{MPCV}})} \\
= \begin{cases}
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CV}}(i_{\text{MPCV}}, 0)} \times 1 & \text{jika } j_{\text{MPCV}} = \operatorname*{argmax}(\operatorname*{ReLU}(z_{\text{V}})) \\
0 & \text{sebaliknya}
\end{cases}
$$

Catatan:

1. $i_{\text{MPCV}}=i_{\text{Cc}}$
2. $j_{\text{MPCV}}=\set{0,1,2,3,4}$ karena width dari output convolution filter vertikal adalah 5.
3. j_MPCV adalah kolom ke-j dari output convolution filter vertikal.
4. argmax(.) adalah fungsi yang mengembalikan indeks di mana nilainya berada pada titik maksimum dalam input yang diberikan. Dalam kasus ini, inputnya adalah vektor atau matriks.

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})}=\frac{\delta L_{\text{BCE}}}{\delta {MP}_{\text{CV}}(i_{\text{V}}, j_{\text{V}})} \times \frac{\delta {MP}_{\text{CV}}(i_{\text{V}}, j_{\text{V}})}{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})} \\
=\frac{\delta L_{\text{BCE}}}{\delta {MP}_{\text{CV}}(i_{\text{V}}, j_{\text{V}})} \times \mathbb{I}_{\mathbb{Z}^+}(z_{\text{V}}(i_{\text{V}}, j_{\text{V}}))
$$

Catatan:

1. $i_{\text{V}}=i_{\text{MPCV}}$
2. $j_{\text{V}}=j_{\text{MPCV}}$

$$
\frac{\delta L_{\text{BCE}}}{\delta K_{\text{V}}(m_{\text{V}}, n_{\text{V}})}=\sum_{i_{\text{V}}=0}^4\sum_{j_{\text{V}}=0}^2\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})} \times \frac{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})}{\delta K_{\text{V}}(m_{\text{V}}, n_{\text{V}})}\right) \\
=\sum_{i_{\text{V}}=0}^4\sum_{j_{\text{V}}=0}^2\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})} \times I(i_{\text{V}}+m_{\text{V}}, j_{\text{V}}+n_{\text{V}})\right)
$$

Catatan:

1. Ini serupa dengan Persamaan X.X (\<REFERRING TO CNN BACKEND KERNEL BACKPROP ABOVE\>). Bentuk dari error matrix (karena padding "same" selama feed-forward) berarti batas atas dari penjumlahannya berturut-turut adalah 4 dan 2.
2. $I$ adalah input log-mel spectrogram awal.

$$
\frac{\delta L_{\text{BCE}}}{\delta b_{\text{V}}}=\sum_{i_{\text{V}}=0}^4\sum_{j_{\text{V}}=0}^2\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})} \times \frac{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})}{\delta b_{\text{V}}}\right) \\
=\sum_{i_{\text{V}}=0}^4\sum_{j_{\text{V}}=0}^2\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})} \times 1\right)
$$

Catatan: ini serupa dengan Persamaan X.X (\<REFERRING TO THE CNN BACKEND BIAS BACKPROP ABOVE\>)

### Filter Horizontal CNN Front-End

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{\text{H}}(i_{\text{H}})}=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CH}}(i_{\text{H}})} \times \left[\frac{\delta O_{\text{CH}}}{\delta z_{\text{H}}} \right]_{i_{\text{H}}} \\
=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CH}}(i_{\text{H}})} \times \mathbb{I}_{\mathbb{Z}^+}(z_{\text{H}}(i_{\text{H}}))
$$

Catatan:

1. $i_{\text{H}}=i_{\text{Cc}}$
2. Hanya terdapat 1 parameter indeks karena convolution-nya adalah 1D.

$$
\frac{\delta L_{\text{BCE}}}{\delta K_{\text{H}}(m_{\text{H}})}=\sum_{i_{\text{H}}=0}^5\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{H}}(i_{\text{H}})} \times \frac{\delta z_{\text{H}}(i_{\text{H}})}{\delta K_{\text{H}}(m_{\text{H}})}\right) \\
=\sum_{i_{\text{H}}=0}^5\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{H}}(i_{\text{H}})} \times M_{\text{in}}(i_{\text{H}}+m_{\text{H}})\right)
$$

Catatan:

1. $i_{\text{H}}\in \set{0,1,2,3,4,5}$ karena height dari output vector mean-pooling adalah 6.
2. $m_{\text{H}}=\set{0,1,2}$ karena width kernel horizontal adalah 3.
3. $M_{\text{in}}$ adalah output vector menengah (intermediate) dari mean-pooling yang bertindak sebagai input bagi convolution filter horizontal.

$$
\frac{\delta L_{\text{BCE}}}{\delta b_{\text{H}}}=\sum_{i_{\text{H}}=0}^5\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{H}}(i_{\text{H}})} \times \frac{\delta z_{\text{H}}(i_{\text{H}})}{\delta b_{\text{H}}}\right) \\
=\sum_{i_{\text{H}}=0}^5\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{H}}(i_{\text{H}})} \times 1\right)
$$

Catatan: ini serupa dengan Persamaan X.X (\<REFERRING TO THE CNN BACKEND BIAS BACKPROP ABOVE\>)

# Penghitungan Rancangan Model
