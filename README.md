**Note** : We are still in the process of implementing. Use it at your own risk.


This repository contains an implementation of the [Representation Learning for Information Extraction From Form Like Documents](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/59f3bb33216eae711b36f3d8b3ee3cc67058803f.pdf) paper.

## Project setup
```
python -m virtualenv -p python3.8 venv
source venv/bin/activate
pip install -e .
gdown --id 10r9y17wg8Elo-3Zi61xA_8QDaKix8giN -O data.tar.xz
tar -xf data.tar.xz
gdown --id 16FzDxLOFxNmYi3JNXaYCmnZvR4x5T54I -O ocr_modified_files.tar.xz
tar -xf ocr_modified_files.tar.xz && mv ocr_modified_files data/

python data_processing.py
```

At this point your `data` dir should have `box`, `img`, `key`, `new_processed_files`, and `ocr_modified_files`


If you are interested about the paper or implementation details you can this [report](https://wandb.ai/tulasi1729/information_extraction/reports/Information-Extraction-From-Documents---Vmlldzo0MDc3MDQ) published in Weights and Biases.
