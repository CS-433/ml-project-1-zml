# ml-project-1-zml

- [x] baseline functions and script to run
- [ ] unbalanced data: test with sklearn, not improved much
- [ ] missing value : 
  - [x] mean, median, zero: zero is the best (or maybe +999)
  - [ ] predict to fill, missing value have relationship to predictions
  - there should be some columns with missing value have huge 'influence' on the result, find them and drop or using different classifier
- [x] outliers 
- [ ] Polynomial test with sklearn, not improved much
- [x] Standardization
- [ ] Voting system


https://github.com/shannonrush/HiggsBoson/blob/master/Summary.md

| No | Name | Note |
| ---- | --------------------------- | ------------------------------------------------------------ |
| 0    | DER_mass_MMC                | 97% of s lie in range 75 to 175, 93% of  undefined are b     |
| 1    | DER_mass_transverse_met_lep | s < b                                                        |
| 2    | DER_mass_vis                | 95% of all s observations found in range 50 to 125           |
| 3    | DER_pt_h                    | s > b                                                        |
| 4    | DER_deltaeta_jet_jet        | 70% of undefined observations are b, else s > b              |
| 5    | DER_mass_jet_jet            | 70% of undefined observations are b, else s > b              |
| 6    | DER_prodeta_jet_jet         | 70% of undefined observations are b, else s < b              |
| 7    | DER_deltar_tau_lep          | similar distribution of s and b                              |
| 8    | DER_pt_tot                  | similar distribution of s and b                              |
| 9    | DER_sum_pt                  | similar distribution of s and b , s slightly > b             |
| 10   | DER_pt_ratio_lep_tau        | similar distribution                                         |
| 11   | DER_met_phi_centrality      | similar distribution                                         |
| 12   | DER_lep_eta_centrality      | 70% of undefined observations are b                          |
| 13   | PRI_tau_pt                  | s > b, result of the square root                             |
| 14   | PRI_tau_eta                 | 80% of s in the -1.5 to 1.5, 74% of data outside -1.5 to 1.5 are b |
| 15   | PRI_tau_phi                 | angle                                                        |
| 16   | PRI_lep_pt                  | result of the square root                                    |
| 17   | PRI_lep_eta                 | similar to 14                                                |
| 18   | PRI_lep_phi                 | angle                                                        |
| 19   | PRI_met                     | based on plot might be better to have square                 |
| 20   | PRI_met_phi                 | angle                                                        |
| 21   | PRI_met_sumet               | similar distribution of s and b , s slightly > b             |
| 22   | PRI_jet_num                 | Catological                                                  |
| 23   | PRI_jet_leading_pt          | result of the square root, 75% of undefined are background.  |
| 24   | PRI_jet_leading_eta         | 75% of undefined are background.67% of defined in the -1.5 to 1.5 range  are b |
| 25   | PRI_jet_leading_phi         | angle, 75% of undefined are background                       |
| 26   | PRI_jet_subleading_pt       | result of the square root, 70% of undefined are background   |
| 27   | PRI_jet_subleading_eta      | [todo]                                                       |
| 28   | PRI_jet_subleading_phi      | angle, 75% of undefined are b                                |
| 29   | PRI_jet_all_pt              | gaps between 0 and other value                               |
