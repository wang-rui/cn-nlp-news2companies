# NER Model Performance

## Origin Parameters

| Parameter   | Value | Description                                    |
| ----------- | ----- | ---------------------------------------------- |
| seg_dim     | 20    | Embedding size for segmentation, 0 if not used |
| char_dim    | 100   | Embedding size for characters                  |
| lstm_dim    | 100   | Num of hidden units in LSTM                    |
| tag_schema  | iob   | Tagging schema iobes or iob                    |
| clip        | 5     | Gradient clip                                  |
| dropout     | 0.5   | Dropout rate                                   |
| batch_size  | 128   | batch size                                     |
| lr          | 0.001 | Initial learning rate                          |
| optimizer   | adam  | Optimizer for training                         |
| max_epoch   | 100   | Maximum training epochs                        |
| steps_check | 100   | Steps per checkpoint                           |

## Train Records

| Model Path        | Adjusted Parameters | Train Set                                                    | Test Set           | Dim of word2vec | Remark                         |
| ----------------- | ------------------- | ------------------------------------------------------------ | ------------------ | :-------------: | ------------------------------ |
| output1129_121449 | optimizer: sgd      | 100w data from origin project                                | labeled 62558 data |       100       |                                |
| output1129_140548 | optimizer: adagrad  | 100w data from origin project                                | labeled 62558 data |       100       |                                |
| output1129_163241 | lstm_dim: 128       | 100w data from origin project                                | labeled 62558 data |       100       | Better performance             |
| output1129_163419 | lr: 0.01            | 100w data from origin project                                | labeled 62558 data |       100       |                                |
| output1129_174130 | None                | 100w data from origin project                                | labeled 62558 data |       100       |                                |
| output1130_151428 | lstm_dim: 128       | 100w data from origin project + labeled 62558 data           | 62558 - 92870 data |       100       |                                |
| output1130_151458 | lr: 0.0005          | 100w data from origin project + labeled 62558 data           | 62558 - 92870 data |       100       |                                |
| output1130_162044 | None                | 100w data from origin project + labeled 62558 data           | 62558 - 92870 data |       100       |                                |
| output1203_114137 | lstm_dim: 256       | 100w data from origin project + labeled 62558 data           | 62558 - 92870 data |       100       | Worse than lstm_dim = 128      |
| output1203_130540 | lstm_dim: 256       | 100w data from origin project + labeled 92870 data           | 92870 - end data   |       100       | Best performance till now      |
| output1203_130554 | None                | 100w data from origin project + labeled 92870 data           | 92870 - end data   |       100       |                                |
| output1203_142000 | lstm_dim: 256       | 100w data from origin project + labeled 92870 data           | 92870 - end data   |       100       | May not better than lstm = 128 |
| output1203_153243 | lstm_dim: 256       | 100w data from origin project + labeled 92870 data           | 92870 - end data   |       200       | Worse than dim of w2v = 100    |
| output1203_153310 | None                | 100w data from origin project + labeled 92870 data           | 92870 - end data   |       200       |                                |
| output1204_145939 | lstm_dim: 256       | 100w data from origin project + labeled 92870 data + new 7w data | 92870 - end data   |       100       |                                |

