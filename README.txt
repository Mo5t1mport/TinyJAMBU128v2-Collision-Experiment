Kode dan data revisi - Evaluasi Empiris Kolisi Generik pada Tag Autentikasi 64 Bit TinyJAMBU-128

Persyaratan: Python 3.11+, numpy, numba, scipy, pandas, psutil, matplotlib.
Jalankan semua perintah dari dalam folder ini.

1. Validasi implementasi (Subbab 3.2)
   python validate_kat.py        -> 1089/1089 KAT (implementasi spesifikasi), 1/1 KAT pesan 8 byte,
                                    10000/10000 kecocokan dengan implementasi penulis
   python validate_fast.py       -> implementasi cepat == implementasi penulis
2. Eksperimen 1 (Tabel 2): skrip asli penulis
   python TinyJAMBU128v2_collision.py --trials 5 --output results/original_run
3. Eksperimen 2 (Tabel 3, Gambar 2)
   python truncated_tag_experiment.py   -> results/truncated_tags.json
   python fig_truncated.py               -> results/fig2_truncated.png
4. Eksperimen 3 (Tabel 4)
   python rho_collision_search.py       -> results/rho_collision.json (~11 menit pada 2 inti)
   python verify_collision.py           -> verifikasi kolisi dengan implementasi spesifikasi
5. Ukuran memori dictionary (Subbab 3.4)
   python mem_dict.py                    -> results/dict_memory.json

Sumber vektor KAT: LWC_AEAD_KAT_128_96_TinyJAMBU-128.txt (format NIST LWC, 1.089 vektor),
diambil dari repositori rweather/lwc-finalists; sebaiknya dicocokkan dengan paket resmi NIST.
Seed kunci/nonce: 20260726 (fungsi make_key_nonce pada skrip penulis). Seed titik awal rantai: 20261107.
