def calc_melspec(subfolder, item_in_question, split, naive_interval=15):
    def znorm(col):
        return (col - numpy.average(col)) / (numpy.sqrt(numpy.var(col) + 10e-12))

    interval = 16000 * naive_interval

    y, sr = librosa.load("./temp/" + subfolder + "/" + item_in_question, sr=16000)

    melspecs: list[numpy.ndarray] = []

    for i in range(len(y) // interval):
        lower = i * interval
        upper = (i + 1) * interval

        if upper > len(y):
            upper = -1

        si = librosa.feature.melspectrogram(y=y[lower:upper], sr=sr, n_mels=96)
        s_dbi = librosa.amplitude_to_db(si, ref=10e-12)

        s_dbi_norm = numpy.apply_along_axis(znorm, 1, s_dbi)

        melspecs.append(s_dbi_norm)

    split_dir = f"./temp/{naive_interval}_melspec{subfolder}/{split}"
    os.makedirs(split_dir, exist_ok=True)

    for chunk_idx, chunk_arr in enumerate(melspecs):
        file_stem = f"{naive_interval}{item_in_question.split('.')[0]}_{chunk_idx}"

        npy_path = f"{split_dir}/{file_stem}.npy"
        with open(npy_path, "wb") as f:
            numpy.save(f, chunk_arr)
