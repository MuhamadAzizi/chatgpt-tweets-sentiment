# Analisis Sentimen Opini Masyarakat terhadap ChatGPT sebagai Aplikasi Natural Language Processing

Dataset yang digunakan berjumlah `15.000` data, kemudian dibagi menjadi data train dan data testing dengan rasio `80:20`.
Kemudian data tersebut di train dengan menggunakan model `Random Forest` dan `LightGBM` serta menerapkan hyperparameter tuning untuk 
mendapatkan hasil yang lebih baik. Berikut adalah hasil training yang sudah dilakukan.

|Model                            |Accuracy|Precision|Recall|Training Time                     |
|---------------------------------|--------|---------|------|----------------------------------|
|Random Forest                    |78.13%  |78.50%   |78.32%|<p align="right">48.246123</p>    |
|LightGBM                         |77.11%  |77.97%   |77.36%|<p align="right">1.214324</p>     |
|Random Forest with Hyperparameter|74.00%  |75.00%   |74.11%|<p align="right">9.784740</p>     |
|LightGBM with Hyperparameter     |77.73%  |78.45%   |77.94%|<p align="right">2.802975</p>     |

**Catatan**: Training time dapat bervariasi bergantung pada spesifikasi komputer yang dijalankan untuk training model.

## Laporan
<img src="https://github.com/MuhamadAzizi/chatgpt-tweets-sentiment/blob/main/reports/bar_plot_model_performance.png?raw=true">
