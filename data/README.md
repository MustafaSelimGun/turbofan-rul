# Veri

`raw/CMAPSSData/` klasöründe NASA'nın **Turbofan Engine Degradation Simulation (C-MAPSS)** veri seti var.

- Kaynak: NASA Ames Prognostics Center of Excellence (PCoE) veri deposu
  https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- Referans: A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft Engine
  Run-to-Failure Simulation", PHM08, 2008.
- Dosya bütünlüğü (satır sayıları): train_FD001 20631, test_FD001 13096, RUL_FD001 100;
  train_FD002 53759, train_FD003 24720, train_FD004 61249.

| Alt küme | Eğitim motoru | Test motoru | Çalışma koşulu | Arıza modu |
|---|---|---|---|---|
| FD001 | 100 | 100 | 1 | 1 (HPC) |
| FD002 | 260 | 259 | 6 | 1 (HPC) |
| FD003 | 100 | 100 | 1 | 2 (HPC, fan) |
| FD004 | 248 | 249 | 6 | 2 (HPC, fan) |

Her dosyada 26 sütun var: motor numarası, döngü, 3 çalışma ayarı ve 21 sensör.
Eğitim serileri arızaya kadar sürer; test serileri arızadan önce kesilir ve gerçek kalan ömür `RUL_FDxxx.txt` dosyasındadır.
