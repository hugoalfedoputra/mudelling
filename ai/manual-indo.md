# Identifikasi Permasalahan

Penelitian ini bertujuan untuk mengotomatiskan pelabelan perasaaan dan tema pada musik. Sistem ini menggunakan berkas audio digital sebagai masukan. Namun, pembahasan pada Bab 2.2.1 telah menunjukkan bahwa perubahan audio digital menjadi spektogram lebih bermanfaat bagi deep neural jaringan. Dalam pemrosesan sinyal, audio digital terdiri dari nilai-nilai diskrit pada setiap time-step berdasarkan sample rate. Nilai-nilai diskrit tersebut memperhitungkan amplitudo, frekuensi, dan offset dari sinusoid. Digital audio menetapkan waktu berdasarkan sample rate atau pada fraksi diskrit detik ke berapa sistem merekam sinyal. Untuk berkas MP3, sample rate yang umum adalah 44100 Hz, yang berarti terdapat 44100 nilai diskrit dalam satu detik. Kita dapat merepresentasikan nilai-nilai ini sebagai sebuah array di mana indeks dari array tersebut merupakan time-step.

Pengenalan ucapan dan MER umumnya menggunakan spektogram untuk merepresentasikan audio digital. Dalam hal ini, jenis spektogram yang penelitian ini gunakan adalah spektogram log-mel. Perubahan ke spektogram log-mel memerlukan perhitungan STFT. STFT menghitung DFT untuk frame pendek dengan panjang fraksi arbitrer tertentu dari sample rate. STFT berasumsi bahwa sinyal tidak atau kecil kemungkinannya untuk berulang; DFT mengasumsikan sebaliknya. Oleh karena itu, DFT tidak dapat diterapkan pada audio digital. STFT menerima parameter: sample rate, frame panjang, hop panjang, dan banyak pita mel. Keluarannya merupakan matriks 2D di mana setiap baris merepresentasikan pita mel dan setiap kolom merepresentasikan time-frame. Data pada setiap pita mel di suatu time-frame merupakan magnitude dalam dB. Sistem kemudian menskalakan magnitude tersebut secara logaritmik.

Dataset MTG-Jamendo menyediakan metadata dan berkas audio digital. Berkas audio digital dan metadata diunduh secara terpisah. Tabel X.X menunjukkan lima baris pertama dari metadata lagu. Dataset ini terdiri dari 55.609 berkas audio yang telah di-preprocess seperti yang telah dibahas pada Bab 3.2.2. Dataset yang digunakan oleh penelitian ini adalah subset "mood/theme" yang hanya memiliki 18.486 berkas. Setiap berkas audio tidak memiliki informasi pengenal sehingga model tidak menunjukkan efek artis dan album (bogdanov 2019 mtg jamendo). Pembuat dataset telah mengatur pembagian set untuk pelatihan, validasi, dan pengujian untuk subset ini. Perkiraan pembagiannya adalah 60% untuk pelatihan dan masing-masing 20% untuk validasi dan pengujian. Pembagian tersebut bersifat acak tetapi telah dipastikan bahwa tidak ada berkas yang muncul di lebih dari satu set dan tidak ada berkas di set manapun yang berasal dari pencipta atau artis yang sama dengan set lainnya, semua label hadir di ketiga pembagian, dan dataset merepresentasikan setiap label di setiap pembagian dengan setidaknya 40 berkas dan 10 pencipta pada pembagian pelatihan dan masing-masing 20 berkas dan 5 pencipta pada pembagian validasi dan pengujian. Selama pelatihan, validasi, dan inferensi, setiap lagu dipotong menjadi potongan-potongan dengan durasi 15 detik.

Tabel X.X Lima baris pertama dari metadata lagu (image in Google Docs)

Tiga back-end dari penelitian ini merupakan variabel bebas yang memengaruhi skor PR-AUC dan ROC-AUC. Pembandingan berbagai back-end yang berbeda dirancang agar adil dalam hal banyak parameter model. Baseline pembandingan adalah CNN back-end seperti yang diusulkan oleh Pons et al. (2018). Model ini terdiri dari 3 lapis konvolusi dengan satu kali max-pooling yang disisipkan di antara lapisan ke-2 dan ke-3. Tiap lapisan konvolusi memiliki 64 filter. Dua back-end lainnya menggunakan banyak lapisan yang sama tetapi tanpa lapisan max-pooling di antaranya. Peneliti mengatur parameter untuk back-end lainnya agar serupa dengan standar deviasi maksimum 5% terhadap CNN back-end, mirip dengan pengaturan oleh Shim dan Sung (2022).

# Perancangan Algoritme

Alur algoritme seperti yang Gambar X.1 tunjukkan terdiri dari tiga langkah utama: preprocessing, penghitungan spektogram, dan pemodelan. Bab 3.2.4 telah membahas preprocessing. Setelah menghitung STFT, sistem menormalisasi keluaran spektogram menggunakan normalisasi Z-score. Tanpa batching, model akan menggunakan semua potongan 15 detik audio untuk setiap berkas pelatihan karena setiap model dapat menerima panjang waktu yang bervariasi. Namun, karena implementasi menggunakan batching, potongan terakhir dari audio pelatihan tidak digunakan karena batching membutuhkan panjang waktu yang sama untuk semua masukan dalam sebuah batch. Ukuran batch adalah 32. Hal ini mencerminkan penelitian sebelumnya oleh Choi et al. (2016conv) dan Pons et al. (2018) yang menggunakan ukuran batch bawaan sebesar 32 seperti pada pustaka TensorFlow. Contoh jaringan tidak menggunakan batching dan LN. Dataset contoh untuk tahap pemodelan akan terdiri dari 5 baris data arbitrer yang disampel dari subset yang terdiri dari 3 label arbitrer: "gembira", "sedih", dan "tegang". Tabel X.X menunjukkan dataset contoh tersebut. Eksplorasi data awal mengamati bahwa amplitudo spektogram log-mel pada setiap pita frekuensi berada di sekitar -80 dan 10 dB. Dataset contoh dibuat dengan mempertimbangkan rentang tersebut.

Gambar X.1 \<FLOWCHART 1\> (drawio:perancangan1)

Tabel X.X \<EXAMPLE DATASET\>

Gambar X.2 \<FLOWCHART PELATIHAN DETAIL\> (drawio:pelatihan1)

Dataset MTG-Jamendo menyediakan pembagian resmi untuk pelatihan, validasi, dan pengujian. Pembagiannya masing-masing adalah 60%, 20%, dan 20%. Gambar X.1 dan Gambar X.2 mengilustrasikan kapan penggunaan setiap set. Perlu dicatat bahwa set validasi digunakan setelah selesai satu epoch pelatihan. Set validasi ini digunakan untuk menyimpan model pada checkpoint-checkpoint tertentu dan membatasi banyak epoch pelatihan. Pelatihan berakhir ketika model telah mencapai batas epoch. Checkpoint pada model dengan kinerja terbaik saat penelitian dimuat lalu dibandingkan dan dianalisis sesuai dengan rumusan permasalahan. Dalam hal pengoptimalan model, penelitian serupa oleh Pons et al. (2018) dan Choi et al. (2016conv) yang menjadi dasar penelitian ini menggunakan learning rate masing-masing sebesar 0.001 dan 0.005. Penelitian serupa menggunakan CNN dari Choi et al. (2016automatic) menunjukkan penurunan kinerja setelah 40 epoch saat menggunakan ADAM. Kedua penelitian tersebut mengamati kinerja model tiap epoch dan mengakhiri pelatihan sewenang-wenang. Pada model attention, Sukhavasi dan Adapa (2019) membatasi pelatihan hingga 60 epoch untuk ADAM dengan learning rate sebesar 0.001 sebelum menerapkan penyesuaian learning rate menggunakan metode lain. Won et al. (2019toward) sejalan dengan banyak maksimum epoch sementara learning rate ditetapkan sebesar 0.0001. Contoh penghitungan dan penerapan sesungguhnya menggunakan learning rate bawaan untuk ADAM sebesar 0.001. Pelatihan pula dibatasi hingga 60 epoch.

## Front-End

Gambar X.X (drawio:cnn-fe)

<!-- // FLESH THIS SECTION OUT TO BE A COMPARISON TABLE AND NOT JUST BULLET POINTS AND NUMBERED LIST

- Number of filters: 2
- Masukan is split to 2 types: for the vertical and horizontal filters
- For the filter vertikal type: max-pool AFTER konvolusi
- For the filter horizontal type: mean-pool BEFORE konvolusi
- Filter 1 (namely $F_V$) size: 5x3 (vertical)
- Filter 2 (namely $F_H$) size: 1x3 (horizontal) -->

Konfigurasi feed-forward (filter vertikal) untuk jaringan contoh adalah sebagai berikut:

1. Masukan berukuran 6x5 (HxW) di mana H adalah waktu dan W adalah frekuensi. Ukuran filter adalah 5x3 (vertical).
2. Banyak channel adalah 1 karena spektogram terdiri dari satu skala warna.
3. Konvolusi dengan filter vertikal menggunakan padding berjenis "same".
4. ReLU digunakan sebagai fungsi aktivasi setelah konvolusi.
5. Max-pool dilakukan pada hasil setelah aktivasi.

Konfigurasi feed-forward (filter horizontal) untuk jaringan contoh adalah sebagai berikut:

1. Masukan: 6x5 (HxW) di mana H adalah frekuensi dan W adalah waktu. Ukuran filter adalah 1x3 (horizontal).
2. Banyak channel: 1
3. Mean-pool dilakukan pada masukan spektogram sehingga dimensi keluaran memiliki tinggi 1 dan lebar senilai waktu.
4. Konvolusi dengan filter horizontal menggunakan padding berjenis "same".
5. ReLU digunakan sebagai fungsi aktivasi setelah konvolusi.

Gambar X.X menggambarkan contoh jaringan front-end. Pada feed-forward filter vertikal, tinggi masukan adalah waktu sedangkan lebarnya adalah frekuensi. Hal ini meniru pemodelan data time-series yang menempatkan waktu pada kolom dengan tiap baris menunjukkan nilai-nilai pada waktu tersebut. Pons et al. (2018) menjelaskan bahwa filter berfungsi untuk mempelajari fitur di sepanjang sumbu waktu. Kedua filter mengurangi dimensi frekuensi dan mempertahankan panjang (tinggi) dari sumbu waktu. Hasil dari konvolusi vertikal dan horizontal digabungkan (penggabungan) menjadi sebuah tensor dengan sumbu waktu yang utuh dan sumbu pita frekuensi yang berkurang dimensinya.

<!-- --- -->

<!-- // ADD VISUAL ILLUSTRATIONS FOR THE FEED-FORWARD OF VERTICAL AND HORIzONTAL FILTERS -->

### Feed-Forward of Filter vertikal Konvolusi

Dengan ukuran masukan 6x5 menggunakan padding berjenis "same", bentuk keluaran dari konvolusi adalah sama dengan bentuk masukan. Pustaka juga menyebut padding berjenis "same" sebagai "half padding". Domoulin (2018) menjelaskan banyak padding yang dibutuhkan secara matematis:

$$p=\left\lfloor{\frac{k}{2}}\right\rfloor$$

Keterangan:

1. p adalah banyak padding yang model terapkan pada setiap batasan
2. k adalah ukuran kernel dengan asumsi kernel berbentuk persegi

Apabila kernel 2D berbentuk persegi panjang, persamaan tersebut diperluas menjadi:

$$p_H=\left\lfloor{\frac{k_H}{2}}\right\rfloor$$
$$p_W=\left\lfloor{\frac{k_W}{2}}\right\rfloor$$

Banyak padding pada matriks masukan dengan ukuran kernel 5x3 menggunakan padding berjenis "same" adalah sebagai berikut:

$$p_H=\left\lfloor{\frac{5}{2}}\right\rfloor=2$$
$$p_W=\left\lfloor{\frac{3}{2}}\right\rfloor=1$$

Gambar X.X menggambarkan matriks masukan yang telah mendapatkan padding.

Gambar X.X \<PADDED MATRIX\> (drawio:paddedmatrix)

Keterangan:

1. p_H adalah banyak padding yang diterapkan pada batasan tinggi (atas dan bawah) matriks masukan
2. k_H adalah tinggi dari kernel
3. p_W adalah banyak padding yang diterapkan pada batasan lebar (kiri dan kanan) matriks masukan
4. k_W adalah lebar dari kernel

Operasi max-pooling menggunakan kernel yang melangkah (stride) untuk mengembalikan nilai maksimum pada suatu masukan. Operasi ini didefinisikan secara matematis:

$$
\text{MaxPool}(I)\\
=\max_{m=0,\dots,k_H-1}\max_{n=0,\dots,k_W-1}\text{in}(\text{stride}[0]\times h+m,\text{stride}[1] \times w+n)
$$

Keterangan:

1. I adalah matriks masukan (atau vektor) yang akan model max-pool
1. $\text{in}(\cdot)$ berarti matriks masukan dari max-pooling
1. stride adalah tuple dengan elemen pada indeks ke-0 dan ke-1 yang menandakan seberapa banyak window bergeser melewati semua lokasi pada masukan
1. h adalah tinggi matriks masukan dan sebagai parameter fungsi keluaran
1. w adalah lebar matriks masukan dan sebagai parameter fungsi keluaran
1. m adalah indeks baris dari matriks masukan
1. n adalah indeks kolom dari matriks masukan
1. k_H adalah tinggi kernel
1. k_W adalah lebar kernel

Bentuk keluaran berupa tinggi dan lebar dari operasi tersebut didefinisikan secara matematis:

$$H_{out}=\left\lfloor \frac{H_{in}+2*\text{padding}[0]-\text{dilation}[0]\times(\text{k}[0]-1)-1}{\text{stride}[0]}+1 \right\rfloor$$

Keterangan:

1. H_out adalah tinggi keluaran
2. H_in adalah tinggi masukan
3. padding[0] berarti indeks pertama dari tuple padding atau seberapa banyak model memberi padding pada masukan di bagian atas dan bawah
4. dilation[0] berarti indeks pertama dari tuple dilation atau tingkat di mana ukuran kernel meningkat sekaligus melewati elemen-elemen masukan. Nilai dilation yang pengguna atur ke 1 sama dengan tidak menerapkan dilation sama sekali.
5. k[0] berarti indeks pertama dari tuple k atau ukuran kernel yang model gunakan sebagai window untuk menghitung nilai maksimum pada window tertentu
6. stride[0] berarti indeks pertama dari tuple stride atau seberapa banyak window bergeser melewati semua lokasi pada masukan

$$W_{out}=\left\lfloor \frac{W_{in}+2*\text{padding}[1]-\text{dilation}[1]\times(\text{k}[1]-1)-1}{\text{stride}[1]}+1 \right\rfloor$$

Keterangan:

1. W_out adalah lebar keluaran
2. W_in adalah lebar masukan
3. padding[1] berarti indeks kedua dari tuple padding
4. dilation[1] berarti indeks kedua dari tuple dilation
5. k[1] berarti indeks kedua dari tuple k
6. stride[1] berarti indeks kedua dari tuple stride

Oleh karena itu, operasi ini menunjukkan bentuk keluaran setelah max-pooling menjadi berukuran 6x1 seperti perhitungan berikut: (2 persamaan terpisah)

$$H_{out}=\left\lfloor \frac{6+2*0-1\times(1-1)-1}{1}+1 \right\rfloor=6$$

$$W_{out}=\left\lfloor \frac{5+2*0-1\times(5-1)-1}{5}+1 \right\rfloor=1$$

Dengan demikian, bentuk akhir (termasuk sumbu channel) dari feed-forward filter vertikal adalah 1x6x1 dengan dimensi: channel, waktu (tinggi), dan frekuensi (lebar).

### Feed-Forward of Filter horizontal Konvolusi

Operasi mean-pooling mirip dengan max-pooling dari segi penghitungan bentuk keluaran. Perbedaannya adalah pada pengambilan rata-rata dari semua nilai di dalam kernel yang bergeser. Oleh karena itu, masukan dari konvolusi berbentuk 1x6 pula. Operasi mean-pooling didefinisikan secara matematis:

$$
\text{MeanPool}(I)\\
=\frac{1}{k_H+k_W}\sum_{m=0}^{k_H-1}\sum_{n=0}^{k_W-1}\text{in}(\text{stride}[0]\times h+m,\text{stride}[1] \times w+n)
$$

Keterangan:

1. k_H adalah tinggi kernel
2. k_W adalah lebar kernel
3. Variabel lainnya memiliki keterangan yang serupa dengan operasi max-pooling yang didefinisikan pada Persamaan X.X (\<POINT TO THE MAXPOOL OP ABOVE\>).

Keluaran dari mean-pooling merupakan masukan dari lapisan konvolusi dengan filter horizontal. Konvolusi tersebut sekarang menjadi konvolusi 1D dengan pita-pita frekuensi yang telah dirata-ratakan untuk tiap time-frame. Karena itu, dimensi frekuensi dapat dihiraukan. Konvolusi ini memiliki padding berjenis "same", sehingga ukuran keluarannya sama dengan ukuran masukannya, yaitu 1x6. Bentuk keluaran dari operasi mean-pooling didefinisikan secara matematis:

$$L_{out}=\left\lfloor \frac{L_{in}+2*\text{padding}-\text{dilation}\times(\text{kernel\_size}-1)-1}{\text{stride}}+1 \right\rfloor$$

Keterangan:

1. L_out adalah panjang dari keluaran sequence
2. L_in adalah panjang dari masukan sequence
3. padding berarti seberapa banyak model memberi padding pada masukan di setiap ujung sequence
4. dilation berarti indeks pertama dari tuple dilation
5. kernel_size berarti indeks pertama dari tuple kernel_size
6. stride berarti indeks pertama dari tuple stride

Karena ukuran kernel didefinisikan dalam dua dimensi (1x3), nilai yang digunakan untuk kernel_size dan stride adalah yang lebih besar di antara keduanya. Hal ini dikarenakan konvolusi 1D melakukan konvolusi pada data melalui panjangnya daripada tingginya. Oleh karena itu, bentuk keluaran setelah operasi konvolusi ditunjukkan memiliki panjang 6 seperti dihitung:

$$L_{out}=\left\lfloor \frac{6+2*1-1\times(3-1)-1}{1}+1 \right\rfloor=6$$

Maka, bentuk akhir (termasuk sumbu channel) dari feed-forward filter horizontal adalah 1x6 dengan dimensi: channel dan waktu.

### Penggabungan dari Konvolusi Filter Vertikal dan Horizontal

Proses penggabungan menggunakan keluaran dari konvolusi filter vertikal maupun horizontal. Karena konvolusi temporal dalam kasus ini menggunakan konvolusi 1D, suatu dimensi baru dapat ditambahkan setelah dimensi waktu untuk memperjelas keberadaan dimensi fitur yang dihiraukan sebelum konvolusi. Maka, bentuk masukan dari kedua konvolusi tersebut adalah 1x6x1 (channel, waktu, fitur). Hasil penggabungan dari kedua keluaran konvolusi memiliki bentuk 1x6x2 yang menjaga panjang dari sumbu waktu tetap utuh. Urutan dari penggabungan (dengan asumsi matriks dibaca dari kiri ke kanan) adalah konvolusi filter vertikal lalu horizontal.

## Back-End

### CNN

CNN back-end terdiri dari 1 lapisan konvolusi masing-masing dengan 1 filter berbentuk 3xW. W adalah lebar dari fitur yang telah digabungkan; dari hasil seperti yang ditunjukkan pada CHAPTER X.Y.Z, nilainya adalah 2. Back-end ini menerima bentuk masukan 6x2 dari front-end. Tinggi dari kernel adalah sumbu waktu sementara lebar adalah sumbu fitur yang telah digabungkan. Lapisan konvolusi ini menggunakan padding berjenis "same". Fungsi aktivasi untuk kedua lapisan adalah ReLU. Keluaran dari back-end ini adalah filter yang digabungkan setelah ReLU. Operasi-operasi yang dilakukan pada back-end ini dengan gambaran ukuran matriks tiap tahapannya terlihat pada Gambar X.X.

Gambar X.X (drawio:cnn-be)

### CNN dengan GRU

Back-end CNN dengan GRU terdiri dari 1 lapisan GRU satu arah (uni-directional) dengan satu lapisan hidden. Back-end ini menerima bentuk masukan 6x2 dari front-end. Desain ini sejalan dengan Cho et al. (2014). Tensor masukan diadaptasi menjadi bentuk LxH_in di mana L adalah sequence panjang atau tinggi dari sumbu waktu dan H_in adalah lebar dari fitur yang telah digabungkan; dari hasil seperti yang ditunjukkan pada CHAPTER X.Y.Z, nilainya adalah 2. Keluaran dari back-end ini adalah final hidden state dari GRU dengan bentuk LxH_out. H_out ditetapkan menjadi 2. Keluaran dari back-end ini mengembalikan fitur dari hidden state terakhir untuk setiap time-step. Bentuknya sama dengan masukan. Operasi-operasi yang dilakukan pada back-end ini dengan gambaran ukuran matriks tiap tahapannya terlihat pada Gambar X.X.

Gambar X.X (drawio:cnn-gru-be)

### CNN dengan Self-Attention

Back-end CNN dengan Self-Attention terdiri dari 1 lapisan Self-Attention dengan 2 head. Self-Attention diimplementasikan sebagai bagian dari bagian encoder dari Transformer. Transformer mengambil masukan berupa vektor embedding. Mengingat keluaran dari front-end berbentuk 6x2, setiap time-step darinya dapat diperlakukan sebagai 6 vektor embedding yang masing-masing berbentuk 1x2. Dalam perhitungannya, ini dilakukan secara bersamaan. Dimensi dari query, key, dan value ditentukan oleh Persamaan 2.12, di mana d_model sama dengan 2 karena lebar dari fitur yang telah digabungkan adalah 2 seperti yang ditunjukkan pada CHAPTER X.Y.Z dan h (head) sama dengan 2. Oleh karena itu, d_k dan d_v sama dengan 1. Karena terdapat dua head, setiap head menerima masukan dengan bentuk 6x1. Setelah menghitung Self-Attention, hasil dari setiap head digabungkan menjadi 6x2 untuk dilanjutkan ke lapisan Feed-Forward Jaringan (FFN) dari Transformer. Fungsi aktivasi pada bagian FFN dari arsitektur tersebut adalah ReLU. Keluaran dari back-end ini berupa hasil operasi dengan bobot dan bias pada lapisan FFN ke-2 tanpa diaktivasi. Operasi-operasi yang dilakukan pada back-end ini dengan gambaran ukuran matriks tiap tahapannya terlihat pada Gambar X.X.

Gambar X.X (drawio:cnn-attn-be)

## Pengklasifikasi

Pengklasifikasi terdiri dari sebuah lapisan FC yang mengambil 12 node sebagai masukan-nya. Keluaran back-end akan digabungkan menjadi bentuk 12x1. Keluaran tersebut menjadi masukan dari keluaran lapisan dengan 3 node yang berkorespondensi dengan tiga label arbitrer yang ditetapkan untuk perhitungan ini sesuai dengan Tabel X.X (\<DUMMY DATASET\>). Fungsi aktivasi untuk FC dan keluaran lapisan adalah sigmoid.

Gambar X.X (drawio:classifier)

<!-- This is basically Manualisasi -->

# Perumusan Feed-Forward dan Backpropagation

---

// ADD THIS IN TINJAUAN TEORI NOT HERE

Dalam konteks deep learning, fungsi konvolusi pada pustaka pemrograman diterapkan sebagai cross-correlation. Konvolusi membalikkan (flip) kernel sementara cross-correlation tidak (Goodfellow 2016). Cross-correlation (dalam konteks deep learning) didefinisikan secara matematis:

$$S(i,j)=\sum_{m=0}^{M-1}\sum_{n=0}^{N-1}{I(i+m,j+n)\times K(m,n)}$$

Keterangan:

1. i dan j berturut-turut adalah indeks baris dan kolom dari masukan, keduanya dimulai dari 0.
2. m dan n berturut-turut adalah indeks baris dan kolom dari kernel
3. M dan N berturut-turut adalah tinggi dan lebar dari kernel
4. S(i,j) adalah hasil konvolusi pada indeks i dan j dari masukan
5. I(i+m,j+n) berturut-turut adalah indeks baris dan kolom dari masukan
6. K(m,n) berturut-turut adalah indeks baris dan kolom dari kernel

---

Jaringan contoh yang digunakan untuk perhitungan numerik penuh adalah model CNN. Definisi feed-forward dan backpropagation untuk model CNN dengan GRU dan model CNN dengan Self-Attention juga dirincikan. Model CNN terdiri dari CNN front-end, CNN back-end, dan pengklasifikasi. Fungsi rugi didefinisikan pada Persamaan 2.24 dengan masukan berupa nilai prediksi setelah feed-forward. Definisi persamaan-persamaan pada CHAPTER X.Y.Z (\<THE TWO SUBCHAPTERS FOR FF AND BPROP BELOW\>), tanda perkalian digunakan untuk menunjukkan perkalian skalar. Jika tidak ada, maka itu menunjukkan perkalian matriks.

## Perumusan Feed-Forward

GAMBAR X.X \<VIS CNN TAPI ADA VARIABLE VARIABLE\> (drawio:cnn-ff-bprop)

### Pengklasifikasi

Persamaan feed-forward digambarkan pada Gambar X.X. Alur feed-forward sengaja didefinisikan mulai dari fungsi rugi BCE berbobot ke arah masukan.

$$y_n=\sigma(z_O)$$

Keterangan:

1. y_n adalah nilai prediksi keluaran
2. $\sigma$ adalah fungsi aktivasi sigmoid
3. z_O adalah keluaran dari keluaran lapisan sebelum menerapkan fungsi aktivasi seperti yang digambarkan pada Gambar X.X

$$z_O=O_{\text{FC}}^T W_O + b_O$$

Keterangan:

1. O_FC adalah keluaran dari lapisan "FC"
2. W_FC adalah bobot (weights) untuk lapisan "FC"
3. b_FC adalah bias untuk lapisan "FC"

$$O_{\text{FC}} = \text{Flatten}(O_{\text{c}})$$

Keterangan:

1. $O_{\text{c}}$ adalah keluaran dari lapisan konvolusi kedua di back-end
2. Flatten(.) adalah operasi berdasar baris (row-wise) yang me-reshape keluaran matriks 2D dari $O_{\text{c}}$ berbentuk 6x2 menjadi vektor 1D berbentuk 12x1.

### CNN Back-End

$$O_{\text{c}}=\text{ReLU}(z_{\text{c}})$$

Keterangan: $z_{\text{c}}$ adalah keluaran dari lapisan konvolusi kedua (dinotasikan c) di back-end sebelum menerapkan fungsi aktivasi

Berikut ini menggunakan kembali (re-purposes) definisi matematis dari operasi konvolusi seperti yang didefinisikan pada Persamaan 2.X (DI BAB 2) untuk digunakan dalam kasus ini:

$$
z_{\text{c}}=S_{\text{c}}(i_{\text{c}},j_{\text{c}}) + b_{c} \\
=\sum_{m_{\text{c}}=0}^{M_{\text{c}}-1}\sum_{n_{\text{c}}=0}^{N_{\text{c}}-1} \big(O_{\text{Cc}}(i_{\text{c}}+m_{\text{c}},j_{\text{c}}+n_{\text{c}})\times K_{\text{c}}(m_{\text{c}},n_{\text{c}}) \big) + b_{c}
$$

Keterangan: serupa dengan Persamaan 2.X; nilai $I$ yang didefinisikan pada Persamaan 2.X diganti dengan O_Cc yang merupakan keluaran dari lapisan konvolusi pertama di back-end.

$$
O_{\text{Cc}}=\text{Concat}(O_{\text{CV}},O_{\text{CH}})
$$

Keterangan:

1. Concat(.) adalah fungsi penggabungan berdasarkan kolom (column-wise) yang diberikan dua masukan O_CV dan O_CH
2. O_CV adalah keluaran dari konvolusi vertikal dari front-end; O_CH adalah keluaran dari konvolusi horizontal.

### CNN dengan GRU Back-End

Operasi ini menggunakan kembali definisi matematis dari GRU yang diusulkan oleh Cho et al. (2014):

$$
O_{\text{FC}} = \text{Flatten}(O^T_{\text{GRU}}) \\

O_{\text{GRU}}=\text{Concat}(h_1,h_2,\dots,h_L)
$$

Keterangan:

1. $L=6$ karena ada 6 time-step dalam data tersebut.
2. Concat(.) melakukan penggabungan berdasarkan kolom.
3. Keluaran dari lapisan GRU adalah hidden state pada semua $t$. Masing-masing di-transpose (lihat Gambar X.X (drawio:cnn-gru-be)) lalu di-flatten. Dalam kasus ini, keluaran-nya adalah 6x2 kemudian di-flatten menjadi 12x1.

$$h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t$$

Keterangan:

1. $t \in \{1, 2, 3, 4, 5, 6\}$ karena ukuran matriks masukan adalah 6x2 di mana 6 adalah sumbu waktu. Terdapat 6 time-step.
2. $h_t$ adalah final hidden state dari GRU pada time-step $t$.
3. $\odot$ menunjukkan perkalian element-wise (Hadamard product).
4. $h_0$ diinisialisasi sebagai vektor dengan semua elemennya bernilai 0.

$$\tilde{h}_t = \tanh(W_h  O_{\text{Cc}(t)} + U_h (r_t \odot h_{t-1}) + b_h)$$

Keterangan:

1. $\tilde{h}_t$ adalah kandidat hidden state pada time-step $t$.
2. $O_{\text{Cc}(t)}$ adalah baris ke-$t$ dari keluaran penggabungan front-end $O_{\text{Cc}}$ yang merepresentasikan fitur pada time-step $t$.
3. $W_h$ dan $U_h$ adalah weight matrices untuk kandidat hidden state, dan $b_h$ adalah bias. Dalam Cho et al. (2014), hal ini dinotasikan hanya sebagai $W$.

$$z_t = \sigma(W_z O_{\text{Cc}(t)}+ U_z h_{t-1} + b_z)$$
$$r_t = \sigma(W_r O_{\text{Cc}(t)}+ U_r h_{t-1} + b_r)$$

Keterangan:

1. $z_t$ adalah vektor update gate pada time-step $t$.
2. $r_t$ adalah vektor reset gate pada time-step $t$.
3. $W_z, U_z, W_r, U_r$ adalah masing-masing weight matrices untuk update dan reset gate, sementara $b_z, b_r$ adalah masing-masing bias-nya.

### CNN dengan Self-Attention Back-End

Operasi ini menggunakan kembali (repurpose) definisi matematis dari Transformer encoder yang diusulkan oleh Vaswani et al. (2017):

$$O_{\text{FC}} = \text{Flatten}(O_{\text{Attn}})$$

Keterangan: $O_{\text{Attn}}$ adalah matriks keluaran 6x2 akhir dari back-end Self-Attention.

Lapisan Position-Wise FFN seperti yang diusulkan dalam Vaswani et al. (2017) terdiri dari dua lapisan. Lapisan pertama dan lapisan kedua memiliki weights dan bias yang berbeda. Lapisan pertama diterapkan ReLU sebagai fungsi aktivasi sementara yang kedua tidak mendapatkan aktivasi. Seperti yang dicatat dalam paper tersebut, dimensionalitas dari masukan dan keluaran dari FFN didasarkan pada d_model, dalam kasus ini bernilai 2. Inner lapisan dari FFN mengambil parameter d_ff atau dimensi dari feed-forward, dalam kasus ini juga diatur menjadi 2. Gambar X.X (drawio:cnn-attn-be) menggambarkan perhitungan yang dilakukan di dalam FFN.

$$
O_{\text{Attn}} = O_{\text{MHA}} + \text{ReLU} (O_{\text{MHA}} W_1 + b_1) W_2 + b_2
$$

Keterangan:

1. $O_{\text{Attn}}$ adalah Position-Wise FFN dengan residual connection ke O_MHA atau keluaran dari blok Multi-Head Attention (MHA).
2. $W_1, b_1$ adalah weights dan bias untuk linear transformation pertama.
3. $W_2, b_2$ adalah weights dan bias untuk linear transformation kedua.

$$O_{\text{MHA}} = O_{\text{Cc}} + \text{Concat}(\text{head}_1, \text{head}_2) W^O$$

Keterangan:

1. $O_{\text{MHA}}$ adalah keluaran dari blok MHA dengan residual connection yang menambahkan masukan asli $O_{\text{Cc}}$.
2. $\text{head}_1$ dan $\text{head}_2$ adalah keluaran dari dua individual attention head.
3. $W^O$ adalah keluaran projection weight matrix.
4. $\text{Concat}(\cdot)$ melakukan penggabungan dua head tersebut sepanjang dimensi fitur.

$$\text{head}_i = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i$$

Keterangan:

1. Ini adalah perhitungan scaled dot product attention untuk head ke-$i$, di mana $i \in \{1, 2\}$.
2. $d_k$ adalah dimensi dari keys (didefinisikan sebagai 1 dalam jaringan contoh).
3. Fungsi $\text{softmax}$ diterapkan secara row-wise.

$$Q_i = O_{\text{Cc}} W_i^Q$$

$$K_i = O_{\text{Cc}} W_i^K$$

$$V_i = O_{\text{Cc}} W_i^V$$

Keterangan:

1. $Q_i, K_i, V_i$ adalah matriks query, key, dan value untuk head ke-$i$.
2. $W_i^Q, W_i^K, W_i^V$ adalah learned weight matrices untuk head ke-$i$.

### Filter Vertikal CNN Front-End

Keterangan:

1. Concat(.) adalah operasi penggabungan yang melakukan penggabungan berdasarkan kolom (menjaga panjang baris tetap sama)
2. O_CV adalah keluaran dari lapisan konvolusi vertikal (dinotasikan CV) di front-end
3. O_CH adalah keluaran dari lapisan konvolusi horizontal (dinotasikan CH) di front-end

$$O_{\text{CV}} = \text{MaxPool}(\text{ReLU}(z_{\text{V}}))$$

$$
z_{\text{V}}=S_{\text{V}}(i_{\text{V}},j_{\text{V}}) + b_{\text{V}} \\
=\sum_{m_{\text{V}}=0}^{M_{\text{V}}-1}\sum_{n_{\text{V}}=0}^{N_{\text{V}}-1} \big(I(i_{\text{V}}+m_{\text{V}},j_{\text{V}}+n_{\text{V}})\times K_{\text{V}}(m_{\text{V}},n_{\text{V}}) \big) + b_{\text{V}}
$$

Keterangan:

1. Baik Persamaan X.X maupun Persamaan X.X (\<REFERRING TO THE 2 EQS ABOVE\>) serupa dengan Persamaan X.X (\<THE OG CONV EQ ABOVE THESE 2\>) hanya saja diperuntukkan bagi lapisan konvolusi vertikal di front-end.
2. MaxPool(.) didefinisikan pada Persamaan X.X (\<THE MAXPOOL DEF ABOVE\>)
3. $I$ adalah masukan log-mel spectrogram

### Filter Horizontal CNN Front-End

$$O_{\text{CH}}=\text{ReLU}(z_H)$$

$$
z_{\text{H}}=S_{\text{H}}(i_{\text{H}},j_{\text{H}}) + b_{\text{H}} \\
=\sum_{m_{\text{H}}=0}^{M_{\text{H}}-1}\sum_{n_{\text{H}}=0}^{N_{\text{H}}-1} \big(\text{MeanPool}(I)(i_{\text{H}}+m_{\text{H}},j_{\text{H}}+n_{\text{H}})\times K_{\text{H}}(m_{\text{H}},n_{\text{H}}) \big) + b_{\text{H}}
$$

Keterangan:

1. MeanPool(.) didefinisikan pada Persamaan X.X (\<THE MEANPOOL DEF ABOVE\>)
2. Baik Persamaan X.X maupun Persamaan X.X (\<REFERRING TO THE 2 EQS ABOVE\>) serupa dengan Persamaan X.X (\<THE OG CONV EQ ABOVE THESE 2\>) hanya saja diperuntukkan bagi lapisan konvolusi horizontal di front-end.

## Perumusan Backpropagation

Dengan definisi persamaan feed-forward pada CHAPTER X.Y.Z (\<THE FF SUBCHAPTER DIRECTLY ABOVE THIS\>), persamaan backpropagation dapat didefinisikan sebagai turunan parsial (partial derivatives) terhadap masing-masing variabel dari setiap persamaan feed-forward. Beberapa nilai pada definisi tersebut telah disubstitusi berdasarkan jaringan contoh yang telah dibahas pada CHAPTER X.Y.Z (\<THE CHAPTER BEFORE THIS\>). Variabel-variabel telah digambarkan pada Gambar X.X (\<THE DETAILED JARINGAN IMAGE ON THE PREV CHAPTER\>)

### Pengklasifikasi

$$
\frac{\delta L_{\text{BCE}}}{\delta y_n}=-\frac{1}{N} \times \left(P_w \times \frac{t_n}{y_n} - N_w \times \frac{1-t_n}{1-y_n} \right) \\
=-\frac{1}{3} \times \left(P_w \times \frac{t_n}{y_n} - N_w \times \frac{1-t_n}{1-y_n} \right)
$$

$$P_w=\frac{2}{1+p_n}$$
$$N_w=\frac{2\times p_n}{1+p_n}$$

Keterangan:

1. N disubstitusi dengan 3 karena terdapat 3 class dalam jaringan contoh.
2. P_w dan N_w didefinisikan sebagai konstanta untuk membuat definisinya lebih ringkas
3. Variabel-variabel pada Persamaan X.X, Persamaan X.X, dan Persamaan X.X (\<THE 3 EQS ABOVE\>) telah diterangkan pada Persamaan 2.24

$$
\frac{\delta L_{\text{BCE}}}{\delta z_O}=\frac{\delta L_{\text{BCE}}}{\delta y_n} \times \frac{\delta y_n}{\delta z_O} \\
=\frac{\delta L_{\text{BCE}}}{\delta y_n}\times \sigma(z_O) \times (1-\sigma(z_O)) \\
=-\frac{1}{3} \times \left(P_w \times \frac{t_n}{y_n} - N_w \times \frac{1-t_n}{1-y_n} \right) \times \big(y_n \times (1-y_n) \big) \\
=-\frac{1}{3} \times \left(P_w \times t_n \times (1-y_n) - N_w \times (1-t_n) \times y_n \right)
$$

Keterangan: Variabel-variabel pada Persamaan X.X (\<THE 1 EQS ABOVE\>) telah diterangkan pada Persamaan 2.24

$$
\frac{\delta L_{\text{BCE}}}{\delta W_{O(j,n)}}=\left[\frac{\delta L_{\text{BCE}}}{\delta y_n}\times \frac{\delta y_n}{\delta z_O}\right]_n \times\left[\frac{\delta z_O}{\delta W_{O(j,n)}}\right]_n \\
=\left[\frac{\delta L_{\text{BCE}}}{\delta z_O}\right]_n \times\left[\frac{\delta z_O}{\delta W_{O(j,n)}}\right]_n \\
=\left[\frac{\delta L_{\text{BCE}}}{\delta z_O}\right]_n \times\left[O_\text{FC}\right]_j
$$

Keterangan:

1. n adalah class ke-n di dalam dataset sesuai dengan Persamaan 2.24
2. $W_{O(j,n)}$ adalah weight yang menghubungkan node ke-j dari sisi keluaran FC ke keluaran node ke-n (lapisan "O" pada Gambar X.X).
3. [.]\_n adalah gradien ke-n. Karena terdapat 3 class, akan ada N perhitungan weighted BCE loss seperti yang didefinisikan pada Persamaan X.X (\<THE EQ ABOVE THIS ONE\>).
4. $z_{O(n)}$ adalah keluaran ke-n setelah perhitungan dengan weight pada keluaran lapisan yang belum diaktivasi
5. $\left[O_\text{FC}\right]_j$ adalah node ke-j dari sisi keluaran FC.

$$
\frac{\delta L_{\text{BCE}}}{\delta b_{O(n)}}=\left[\frac{\delta L_{\text{BCE}}}{\delta y_n}\times \frac{\delta y_n}{\delta z_O}\right]_n \times\left[\frac{\delta z_O}{\delta b_O}\right]_n \\
=\left[\frac{\delta L_{\text{BCE}}}{\delta z_O}\right]_n \times 1 \\
$$

Keterangan: $b_{O(n)}$ adalah bias untuk node keluaran yang ke-n.

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{FC}(i)}}=\sum_{n=1}^3 \left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{O}(n)}} \times \frac{\delta z_{\text{O}(n)}}{\delta O_{\text{FC}(i)}} \right) \\
=\sum_{n=1}^3 \left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{O}(n)}} \times W_{\text{O}(i,n)} \right)
$$

Keterangan:

1. $i=1,2,3,\dots,12$.
2. O_FC(i) adalah keluaran ke-i dari lapisan "FC" yang mana merupakan flatten lapisan dari back-end
3. Gradien yang di-propagate ke node ke-$i$ dari lapisan "FC" adalah hasil penjumlahan dari masing-masing 3 node di keluaran lapisan atau lapisan "O" yang ditunjukkan pada Gambar X.X.

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{c}}} = \text{Reshape}\left( \frac{\delta L_{\text{BCE}}}{\delta O_{\text{FC}}}, (6, 2) \right)
$$

Keterangan:

1. O_c adalah keluaran dari lapisan convolutional (lapisan "c") dari back-end
2. Reshape(.) adalah untuk me-reshape sebuah vektor guna membentuk array dengan bentuk yang telah ditentukan sesuai dengan parameter kedua dari fungsi tersebut. Dalam kasus ini adalah 6x2, yang ditulis (6, 2) agar tidak dikacaukan dengan perkalian skalar.

Fungsi penggabungan seperti yang didefinisikan pada Persamaan X.X (\<SEE THE CONCAT FORMULA IN PREV SUBCHAPTER\>) berarti gradien yang di-propagate harus di-reshape agar bentuknya sama dengan keluaran feed-forward dari lapisan "c".

### CNN Back-End

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c}}(i_{\text{c}}, j_{\text{c}})}=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{c}}(i_{\text{c}}, j_{\text{c}})} \times \left[\frac{\delta O_{\text{c}}}{\delta z_{\text{c}}} \right]_{i_{\text{c}}, j_{\text{c}}} \\
=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{c}}(i_{\text{c}}, j_{\text{c}})} \times \mathbb{I}_{\mathbb{Z}^+}(z_{\text{c}}(i_{\text{c}}, j_{\text{c}}))
$$

Keterangan:

1. $\frac{\delta O_{\text{c}}}{\delta z_{\text{c}}}$ adalah turunan dari fungsi aktivasi ReLU. $\mathbb{I}_{\mathbb{Z}^+}(\cdot)$ adalah fungsi indikator. Ini didefinisikan lebih lanjut pada Persamaan X.X (\<THE EQ BELOW\>)
2. $i_{\text{c}}$ dan $j_{\text{c}}$ adalah indeks baris dan kolom dari fitur map yang dihasilkan untuk lapisan konvolusi kedua (c).
3. $\frac{\delta L_{\text{BCE}}}{\delta O_{\text{c}}}$ adalah gradient map yang di-propagate dari flattening lapisan (dihitung di akhir CHAPTER X.Y.Z (\<PREV SUBCHAPTER\>)), yang di-reshape kembali ke matriks 6x2.

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

Keterangan: ${\mathbb{Z}^+}=\set{1,2,3,\dots}$

$$
\frac{\delta L_{\text{BCE}}}{\delta K_{\text{c}}(m_{\text{c}}, n_{\text{c}})}=\sum_{i_{\text{c}}=0}^5\sum_{j_{\text{c}}=0}^1\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c}}(i_{\text{c}}, j_{\text{c}})} \times \frac{\delta z_{\text{c}}(i_{\text{c}}, j_{\text{c}})}{\delta K_{\text{c}}(m_{\text{c}}, n_{\text{c}})}\right) \\
=\sum_{i_{\text{c}}=0}^5\sum_{j_{\text{c}}=0}^1\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c}}(i_{\text{c}}, j_{\text{c}})} \times O_{\text{Cc}}(i_{\text{c}}+m_{\text{c}}, j_{\text{c}}+n_{\text{c}})\right)
$$

Keterangan:

1. $m_{\text{c}} = \set{0,1,2}$ dan $n_{\text{c}} = \set{0,1}$ berdasarkan batas atas penjumlahan (summation upper bounds) pada Persamaan X.X dan jaringan contoh bahwa $M_{\text{c}} = 3$ dan $N_{\text{c}} = 2$ yang telah dibahas pada CHAPTER X.Y.Z. (\<THE CNN BACKEND CHAPTER\>)
2. $m_{\text{c}}$ dan $n_{\text{c}}$ adalah indeks baris dan kolom dari kernel.
3. $\frac{\delta L_{\text{BCE}}}{\delta K_{\text{c}}(m_{\text{c}}, n_{\text{c}})}$ berarti bahwa turunannya memperbarui weight dari kernel pada indeks baris dan kolom spesifik.
4. $i_{\text{c}}$ dan $j_{\text{c}}$ adalah indeks baris dan kolom dari masukan. Batas atasnya berturut-turut adalah 5 dan 1 karena lapisan "c" memiliki bentuk 6x2.
5. O_Cc adalah keluaran dari penggabungan dari front-end.

$$
\frac{\delta L_{\text{BCE}}}{\delta b_{\text{c}}(m_{\text{c}}, n_{\text{c}})}
=\sum_{i_{\text{c}}=0}^5\sum_{j_{\text{c}}=0}^1 \left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c}}(i_{\text{c}}, j_{\text{c}})} \times \frac{\delta z_{\text{c}}(i_{\text{c}}, j_{\text{c}})}{\delta b_{\text{c}}(m_{\text{c}}, n_{\text{c}})} \right) \\
=\sum_{i_{\text{c}}=0}^5\sum_{j_{\text{c}}=0}^1 \left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{c}}(i_{\text{c}}, j_{\text{c}})} \times 1\right)
$$

Keterangan: $b_{\text{c}}$ adalah variabel bias untuk lapisan "c".

Ketika menghitung turunan untuk sebuah lapisan yang tidak berada di bagian paling bawah (bottom) dari jaringan mengikuti Goodfellow (2016), perhitungan indeks dari gradien yang akan di-propagate (dari lapisan "c") memiliki tanda (sign) yang dibalik. Cross-correlation seperti pada Persamaan 2.X menunjukkan indeks masukan meningkat (karena penambahan), sementara backpropagation dari proses tersebut menurunkan indeks. Terlebih lagi, ketika dihadapkan dengan indeks yang tidak valid (indeks negatif atau di luar batas bentuk 6x2), nilainya adalah 0. Dalam kasus ini, diasumsikan bahwa O_Cc di-pad dengan nol hingga tak terbatas di semua sisinya. Untuk menunjukkan ini, fungsi aktivasi ReLU pada Persamaan X.X (\<FEEDFORWARD EQ O_Cc=ReLU(z_Cc)\>) dapat diabaikan karena fungsi tersebut tidak mengubah bentuk maupun memengaruhi propagasi gradien. Oleh karena itu, ini dapat didefinisikan bahwa:

$$
O_{\text{Cc}}(i_{\text{c}}+m_{\text{c}},j_{\text{c}}+n_{\text{c}}) = z_{\text{Cc}}(i_{\text{Cc}},j_{\text{Cc}})
$$

Keterangan:

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

Keterangan:

1. $i_{\text{Cc}} \in \set{0,1,2,3,4,5};\  j_{\text{Cc}} \in \set{0,1}$
2. $i_{\text{Cc}}$ dan $j_{\text{Cc}}$ adalah indeks baris dan kolom untuk matriks $O_{\text{Cc}}$ yang digabungkan.
3. Batas atas untuk penjumlahannya telah dibahas pada Persamaan X.X (\<THE BACKPROP BEFORE THIS\>)

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CV}}} = \frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}}(i_{\text{Cc}},j_{\text{Cc}})} = \frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}}(i_{\text{Cc}},0)}
$$

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CH}}} = \frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}}(i_{\text{Cc}},j_{\text{Cc}})} = \frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}}(i_{\text{Cc}},1)}
$$

Keterangan:

1. $i_{\text{Cc}}=\set{0,1,2,3,4,5};\ j_{\text{Cc}}=\set{0,1}$
2. Operasi penggabungan feed-forward Concat(.) yang didefinisikan pada Persamaan X.X (\<FF CONCAT ABOVE\>) menggabungkan $O_{\text{CV}}$ dan $O_{\text{CH}}$ berdasarkan kolom (sepanjang dimensi frekuensi/sumbu lebar), backpropagation memecah gradient matrix kembali menjadi bentuk aslinya 6x1.
3. i_Cc adalah baris ke-i dari keluaran yang digabungkan dari dua konvolusi front-end
4. j_Cc adalah kolom ke-j dari keluaran yang digabungkan dari dua konvolusi front-end. Ini bernilai 0 atau 1 di mana indeks ke-0 adalah error matrix yang merepresentasikan gradien untuk keluaran filter vertikal dan indeks ke-1 adalah untuk keluaran filter horizontal.
5. $i_{\text{MPCV}}$ adalah baris ke-i dari keluaran max-pooling (dengan $j = 0$) dari lapisan konvolusi filter vertikal front-end atau lapisan "CV"

### CNN dengan GRU Back-End

Gambar X.X (drawio:cnn-gru-be-bprop)

Dikarenakan perbedaan cara hasil dari GRU digabungkan dan di-flatten dari CNN back-end, proses reshape didefinisikan sebagai:

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{GRU}}} = \text{Reshape}\left(\frac{\delta L_{\text{BCE}}}{\delta O_{\text{FC}}}, (6, 2) \right)
$$

Parameter-parameter yang diturunkan digambarkan pada Gambar X.X. Hidden state terakhir atau h_t dipengaruhi oleh hidden state sebelumnya. Oleh karena itu, gradien yang di-backpropagate pada suatu time-step adalah jumlah dari semua gradien setelahnya atau 0 jika itu merupakan time-step terakhir. Hal ini menciptakan penjumlahan rekursif (recursive sum) saat menghitung weights dan bias dari recurrent jaringan (Zhang 2023). Recursion dari gradien tersebut didefinisikan sebagai:

$$
\frac{\delta L_{\text{BCE}}}{\delta h_t} = \left[\frac{\delta L_{\text{BCE}}}{\delta O_{\text{GRU}}} \right]_t + \frac{\delta O_{\text{GRU}}}{\delta h_{t+1}}\frac{\delta h_{t+1}}{\delta h_t}
$$

Keterangan:

1. $t\in\set{1,2,3,4,5,6}$ karena panjang dari sequence adalah 6.
2. Jika $t=6$ (time-step terakhir), gradien dari masa depan $\frac{\delta L_{\text{BCE}}}{\delta h_{t+1}} = 0$. Backpropagation melakukan iterasi mundur (backwards) dari $t=6$ ke $t=1$.

Untuk memudahkan mendefinisikan gradien, persamaan feed-forward didefinisikan ulang menjadi:

$$
\tilde{h}_t=\text{tanh}(z_{\tilde{h}(t)}) \\
z_t=\sigma(z_{z(t)}) \\
r_t=\sigma(z_{r(t)})
$$

Keterangan: semua persamaan didefinisikan pada Persamaan X.X sampai dengan X.X (\<THE FEEDFORWARD GRU EQS ABOVE\>)

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

Keterangan: operasi $U_h (r_t \odot h_{t-1})$ yang didefinisikan pada Persamaan X.X (\<FEEDFORWARD EQ ABOVE\>) tidak dapat didistribusikan. Oleh karena itu, turunan parsialnya harus dicari terhadap nilai tersebut terlebih dahulu.

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{r(t)}} = \frac{\delta L_{\text{BCE}}}{\delta r_t} \odot \frac{\delta r_t}{\delta z_{r(t)}}
= \frac{\delta L_{\text{BCE}}}{\delta r_t} \odot (r_t \odot (1 - r_t))
$$

Seperti yang dicatat sebelumnya, penjumlahan dari setiap time-step bersifat rekursif. Base case-nya diketahui bernilai 0 ketika kasusnya adalah $t+1 \notin t$. Recursive case-nya didefinisikan sebagai penjumlahan dari turunan parsial dari seluruh persamaan yang didefinisikan pada Persamaan X.X sampai dengan X.X (\<THE FEEDFORWARD GRU EQS ABOVE\>) di mana $h_{t-1}$ muncul. Ini didefinisikan menjadi:

$$
\frac{\delta L_{\text{BCE}}}{\delta h_{t-1}} = \left(\frac{\delta L_{\text{BCE}}}{\delta h_t} \odot \frac{\delta h_t}{\delta h_{t+1}}\right) + \left(\frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}(t)}} \times \frac{\delta z_{\tilde{h}(t)}}{\delta h_{t+1}}\right) + \left(\frac{\delta L_{\text{BCE}}}{\delta z_{z(t)}} \times \frac{\delta z_{z(t)}}{\delta h_{t+1}}\right) + \left(\frac{\delta L_{\text{BCE}}}{\delta z_{r(t)}} \times \frac{\delta z_{r(t)}}{\delta h_{t+1}}\right) \\
= \left( \frac{\delta L_{\text{BCE}}}{\delta h_t} \odot z_t \right) + \left( \left( U_h^T \frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}(t)}}  \right) \odot r_t \right) + \left( U_z^T \frac{\delta L_{\text{BCE}}}{\delta z_{z(t)}}  \right) + \left( U_r^T \frac{\delta L_{\text{BCE}}}{\delta z_{r(t)}}  \right)
$$

Pada definisi di Persamaan X.X hingga X.X (\<FEEDFORWARD EQS ABOVE\>), parameter yang dipakai bersama (shared) di seluruh recurrent lapisan adalah: $W_h$, $U_h$, $b_h$, $W_z$, $U_z$, $b_z$, $W_r$, $U_r$, dan $b_r$. Gradien untuk parameter kandidat hidden state didefinisikan menjadi:

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

Serupa dengan itu, parameter O_Cc(t) muncul beberapa kali seperti yang didefinisikan pada Persamaan X.X hingga X.X (\<FEEDFORWARD EQS ABOVE\>). Turunan dari parameter ini digunakan untuk melakukan propagate gradien ke arah mundur (backwards) menuju front-end lapisan. Ini didefinisikan menjadi:

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}(t)}} = \left( \frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}(t)}} \times \frac{\delta z_{\tilde{h}(t)}}{\delta O_{\text{Cc}(t)}} \right) + \left( \frac{\delta L_{\text{BCE}}}{\delta z_{z(t)}} \times \frac{\delta z_{z(t)}}{\delta O_{\text{Cc}(t)}} \right) + \left( \frac{\delta L_{\text{BCE}}}{\delta z_{r(t)}} \times \frac{\delta z_{r(t)}}{\delta O_{\text{Cc}(t)}} \right) \\
= \left( W_h^T \frac{\delta L_{\text{BCE}}}{\delta z_{\tilde{h}(t)}}  \right) + \left( W_z^T \frac{\delta L_{\text{BCE}}}{\delta z_{z(t)}}  \right) + \left( W_r^T \frac{\delta L_{\text{BCE}}}{\delta z_{r(t)}}  \right)
$$

Keterangan: operasi ini dilakukan untuk seluruh $t \in \set{1,2,3,4,5,6}$. Perhatikan bahwa ini tidak dijumlahkan tidak seperti weights dan bias pada GRU.

Hasil dari Persamaan X.X (\<THE EQ DIRECTLY ABOVE THIS\>) digabungkan berdasarkan kolom. Dalam kasus ini, ini akan menghasilkan matriks dengan bentuk 6x2, bentuk yang sama dengan feed-forward-nya. Matriks ini kemudian di-backpropagate ke front-end dengan cara yang sama seperti yang didefinisikan pada Persamaan X.X (\<THE LAST EQ OF CNN BACKEND BACKPROP SUBCHAPTER WITH O_CV AND O_CH\>).

### CNN dengan Self-Attention Back-End

Gambar X.X (drawio:cnn-gru-be-bprop)

Untuk kasus attention back-end, proses reshape didefinisikan sebagai:

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{FFN}(2)}} = \text{Reshape}\left( \frac{\delta L_{\text{BCE}}}{\delta O_{\text{FC}}}, (6, 2) \right)
$$

Alur backpropagation digambarkan pada Gambar X.X (drawio:cnn-attn-be-bprop). Operasi-operasi untuk setiap langkah backpropagation didefinisikan sebagai berikut:

$$
z_{\text{FFN}(1)}=O_{\text{MHA}} W_1+b_1
$$

$$
O_{\text{FFN}(1)}=\text{ReLU}(z_{\text{FFN}(1)})
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

Keterangan: $j=\set{0,1}$ karena $d_{\text{model}} = 2$

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

Keterangan: $j=\set{0,1}$ karena $d_{ff} = 2$

<!-- omha -->

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{MHA}}} = \frac{\delta L_{\text{BCE}}}{\delta O_{\text{FFN}(2)}} + \frac{\delta L_{\text{BCE}}}{\delta z_{\text{FFN}(1)}} \times \frac{\delta z_{\text{FFN}(1)}}{\delta O_{\text{MHA}}} \\
=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{FFN}(2)}} + \left( \frac{\delta L_{\text{BCE}}}{\delta z_{\text{FFN}(1)}} W_1^T \right)
$$

Keterangan: suku (term) pertama adalah residual

$$
C_h = \text{Concat}(\text{head}_1, \text{head}_2)
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
s_{\text{Attn}}=\text{softmax}(z_{\text{Attn}})
$$

$$
C_{h(i)}=s_{\text{Attn}}V
$$

Keterangan: $i=\set{1,2}$ karena ada 2 head seperti yang telah dibahas pada jaringan contoh.

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

Keterangan: $\delta_{mn}$ adalah fungsi Kronecker delta yang digunakan pada turunan fungsi aktivasi softmax.

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{\text{Attn}(i)}} = \frac{\delta L_{\text{BCE}}}{\delta s_{\text{Attn}(i)}} \times \left[ \frac{\delta s_{\text{Attn}}}{\delta z_{\text{Attn}}} \right]_i \\
\frac{\delta L_{\text{BCE}}}{\delta z_{\text{Attn}(i, m, n)}}=\sum_{k=0}^{L-1} \left( \frac{\delta L_{\text{BCE}}}{\delta s_{\text{Attn}(i, m, k)}} s_{\text{Attn}(i, m, k)} (\delta_{np} - s_{\text{Attn}(i, m, n)}) \right)
$$

Keterangan:

1. $m=\set{0,1,\dots,L-1}$ adalah indeks baris (time-step query sequence).
2. $n=\set{0,1,\dots,L-1}$ adalah indeks kolom (time-step key sequence) yang gradiennya sedang dihitung.
3. $k=\set{0,1,\dots,L-1}$ adalah iterator penjumlahan melintasi kolom-kolom keluaran softmax.
4. L adalah sequence panjang masukan (sumbu waktu). Dalam kasus ini, nilainya 6.
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

Keterangan: ini adalah gradien-gradien untuk matriks Query, Key, dan Value linear projection weight untuk head $i$. Semuanya memiliki bentuk 2x1. $O_{\text{Cc}}^T$ memiliki bentuk 2x6.

$$
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}}} = \frac{\delta L_{\text{BCE}}}{\delta O_{\text{MHA}}} \\
+ \sum_{i=1}^2 \left( \left[ \frac{\delta L_{\text{BCE}}}{\delta Q_i} \times \frac{\delta Q_i}{\delta O_{\text{Cc}}} \right] + \left[ \frac{\delta L_{\text{BCE}}}{\delta K_i} \times \frac{\delta K_i}{\delta O_{\text{Cc}}} \right] + \left[ \frac{\delta L_{\text{BCE}}}{\delta V_i} \times \frac{\delta V_i}{\delta O_{\text{Cc}}} \right] \right) \\
= \frac{\delta L_{\text{BCE}}}{\delta O_{\text{MHA}}} + \sum_{i=1}^2 \left( \frac{\delta L_{\text{BCE}}}{\delta Q_i} (W_i^Q)^T + \frac{\delta L_{\text{BCE}}}{\delta K_i} (W_i^K)^T + \frac{\delta L_{\text{BCE}}}{\delta V_i} (W_i^V)^T \right)
$$

Keterangan:

1. $\frac{\delta L_{\text{BCE}}}{\delta O_{\text{Cc}}}$ (bentuk 6x2) adalah gradien akumulasi total yang diteruskan kembali (passed back) ke Front-End penggabungan lapisan.
2. Karena $O_{\text{Cc}}$ bercabang (branches out) menuju residual connection, dan menuju matriks $Q_i, K_i, V_i$ untuk kedua head, gradien yang akan di-backpropagate adalah jumlah dari seluruh gradien yang di-propagate mundur (backward) dari semua 7 jalur, lihat Gambar X.X (drawio:cnn-attn-be-bprop).

### Filter Vertikal CNN Front-End

$$
\frac{\delta L_{\text{BCE}}}{\delta {MP}_{\text{CV}}(i_{\text{MPCV}}, j_{\text{MPCV}})}=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CV}}(i_{\text{MPCV}}, 0)} \times \frac{\delta O_{\text{CV}}(i_{\text{MPCV}}, 0)}{\delta {MP}_{\text{CV}}(i_{\text{MPCV}}, j_{\text{MPCV}})} \\
= \begin{cases}
\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CV}}(i_{\text{MPCV}}, 0)} \times 1 & \text{jika } j_{\text{MPCV}} = \text{argmax}(\text{ReLU}(z_{\text{V}})) \\
0 & \text{sebaliknya}
\end{cases}
$$

Keterangan:

1. $i_{\text{MPCV}}=i_{\text{Cc}}$
2. $j_{\text{MPCV}}=\set{0,1,2,3,4}$ karena lebar dari keluaran konvolusi filter vertikal adalah 5.
3. j_MPCV adalah kolom ke-j dari keluaran konvolusi filter vertikal.
4. argmax(.) adalah fungsi yang mengembalikan indeks di mana nilainya berada pada titik maksimum dalam masukan yang diberikan. Dalam kasus ini, inputnya adalah vektor atau matriks.

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})}=\frac{\delta L_{\text{BCE}}}{\delta {MP}_{\text{CV}}(i_{\text{V}}, j_{\text{V}})} \times \frac{\delta {MP}_{\text{CV}}(i_{\text{V}}, j_{\text{V}})}{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})} \\
=\frac{\delta L_{\text{BCE}}}{\delta {MP}_{\text{CV}}(i_{\text{V}}, j_{\text{V}})} \times \mathbb{I}_{\mathbb{Z}^+}(z_{\text{V}}(i_{\text{V}}, j_{\text{V}}))
$$

Keterangan:

1. $i_{\text{V}}=i_{\text{MPCV}}$
2. $j_{\text{V}}=j_{\text{MPCV}}$

$$
\frac{\delta L_{\text{BCE}}}{\delta K_{\text{V}}(m_{\text{V}}, n_{\text{V}})}=\sum_{i_{\text{V}}=0}^4\sum_{j_{\text{V}}=0}^2\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})} \times \frac{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})}{\delta K_{\text{V}}(m_{\text{V}}, n_{\text{V}})}\right) \\
=\sum_{i_{\text{V}}=0}^4\sum_{j_{\text{V}}=0}^2\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})} \times I(i_{\text{V}}+m_{\text{V}}, j_{\text{V}}+n_{\text{V}})\right)
$$

Keterangan:

1. Ini serupa dengan Persamaan X.X (\<REFERRING TO CNN BACKEND KERNEL BACKPROP ABOVE\>). Bentuk dari error matrix (karena padding berjenis "same" selama feed-forward) berarti batas atas dari penjumlahannya berturut-turut adalah 4 dan 2.
2. $I$ adalah masukan log-mel spectrogram awal.

$$
\frac{\delta L_{\text{BCE}}}{\delta b_{\text{V}}}=\sum_{i_{\text{V}}=0}^4\sum_{j_{\text{V}}=0}^2\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})} \times \frac{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})}{\delta b_{\text{V}}}\right) \\
=\sum_{i_{\text{V}}=0}^4\sum_{j_{\text{V}}=0}^2\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{V}}(i_{\text{V}}, j_{\text{V}})} \times 1\right)
$$

Keterangan: ini serupa dengan Persamaan X.X (\<REFERRING TO THE CNN BACKEND BIAS BACKPROP ABOVE\>)

### Filter Horizontal CNN Front-End

$$
\frac{\delta L_{\text{BCE}}}{\delta z_{\text{H}}(i_{\text{H}})}=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CH}}(i_{\text{H}})} \times \left[\frac{\delta O_{\text{CH}}}{\delta z_{\text{H}}} \right]_{i_{\text{H}}} \\
=\frac{\delta L_{\text{BCE}}}{\delta O_{\text{CH}}(i_{\text{H}})} \times \mathbb{I}_{\mathbb{Z}^+}(z_{\text{H}}(i_{\text{H}}))
$$

Keterangan:

1. $i_{\text{H}}=i_{\text{Cc}}$
2. Hanya terdapat 1 parameter indeks karena konvolusi-nya adalah 1D.

$$
\frac{\delta L_{\text{BCE}}}{\delta K_{\text{H}}(m_{\text{H}})}=\sum_{i_{\text{H}}=0}^5\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{H}}(i_{\text{H}})} \times \frac{\delta z_{\text{H}}(i_{\text{H}})}{\delta K_{\text{H}}(m_{\text{H}})}\right) \\
=\sum_{i_{\text{H}}=0}^5\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{H}}(i_{\text{H}})} \times M_{\text{in}}(i_{\text{H}}+m_{\text{H}})\right)
$$

Keterangan:

1. $i_{\text{H}}\in \set{0,1,2,3,4,5}$ karena tinggi dari keluaran vektor mean-pooling adalah 6.
2. $m_{\text{H}}=\set{0,1,2}$ karena lebar kernel horizontal adalah 3.
3. $M_{\text{in}}$ adalah keluaran operasi mean-pooling sebagai masukan konvolusi filter horizontal.

$$
\frac{\delta L_{\text{BCE}}}{\delta b_{\text{H}}}=\sum_{i_{\text{H}}=0}^5\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{H}}(i_{\text{H}})} \times \frac{\delta z_{\text{H}}(i_{\text{H}})}{\delta b_{\text{H}}}\right) \\
=\sum_{i_{\text{H}}=0}^5\left(\frac{\delta L_{\text{BCE}}}{\delta z_{\text{H}}(i_{\text{H}})} \times 1\right)
$$

Keterangan: ini serupa dengan Persamaan X.X (\<REFERRING TO THE CNN BACKEND BIAS BACKPROP ABOVE\>)

# Penghitungan Rancangan Model
