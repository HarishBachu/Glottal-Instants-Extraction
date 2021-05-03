# Glottal-Instants-Extraction
SAP Project (in progress)

Dataset Source: CMU Arctic Database 

Link: <href>http://festvox.org/cmu_arctic/</href>

Datasets Downloaded: bdl, jmk, slt

Task: Glottal Instants Extraction using SEGAN and SIGMA Algorithm

[Extracting Dataset](split_egg_waveform.py)\
[Dataset](Dataset)\
[Autoencoder Class declaration](segan_utils.py)\
[Creating and training loop for Autoencoder](train_gan.py)\
[Training and saving Autoencoder model checkpoints](train_autoencoder.ipynb)\
[Trained Autoencoder Models](models)\
[Viewing model performance for given checkpoint](view_performance.py)\
[GCI and GOI extraction and Naylor's metrics](sigma.py)\
[Metrics Evaluation for trained models](SIGMA_Algorithm.ipynb)
