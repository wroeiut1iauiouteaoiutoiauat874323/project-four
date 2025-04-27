def labelling_selected(my_df):  # Fungsi untuk melabeli data tertentu dalam DataFrame
    def pelabelan(skor):  # Fungsi internal untuk menentukan label berdasarkan skor
        if skor < 3:  # Jika skor kurang dari 3
            return "negatif"  # Beri label "negatif"
        elif skor == 4:  # Jika skor sama dengan 4
            return "positif"  # Beri label "positif"
        elif skor == 5:  # Jika skor sama dengan 5
            return "positif"  # Beri label "positif"

    my_df["Label"] = my_df["score"].apply(pelabelan)  # Terapkan fungsi pelabelan ke kolom "score" dan simpan hasilnya di kolom "Label"
    my_df = my_df.dropna()  # Hapus baris yang memiliki nilai NaN

    my_df.to_csv("sample_data/the_data_hasillabeling.csv", index=False)  # Simpan DataFrame ke file CSV tanpa menyertakan indeks

    return my_df  # Kembalikan DataFrame yang sudah dilabeli


def labelling_all(my_df):  # Fungsi untuk melabeli seluruh data dalam DataFrame
    def pelabelan(skor):  # Fungsi internal untuk menentukan label berdasarkan skor
        if skor < 3:  # Jika skor kurang dari 3
            return "negatif"  # Beri label "negatif"
        elif skor == 4:  # Jika skor sama dengan 4
            return "positif"  # Beri label "positif"
        elif skor == 5:  # Jika skor sama dengan 5
            return "positif"  # Beri label "positif"

    my_df["Label"] = my_df["score"].apply(pelabelan)  # Terapkan fungsi pelabelan ke kolom "score" dan simpan hasilnya di kolom "Label"
    my_df = my_df.dropna()  # Hapus baris yang memiliki nilai NaN

    my_df.to_csv("sample_data/the_data_all_hasillabeling.csv", index=False)  # Simpan DataFrame ke file CSV tanpa menyertakan indeks

    return my_df  # Kembalikan DataFrame yang sudah dilabeli
