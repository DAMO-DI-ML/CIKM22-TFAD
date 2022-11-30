# CIKM22-TFAD
Source code of CIKM'22 paper: TFAD: A Decomposition Time Series Anomaly Detection Architecture with Frequency Analysis
* Chaoli Zhang, Tian Zhou, Qingsong Wen, Liang Sun, "TFAD: A Decomposition Time Series Anomaly Detection Architecture with Time-Freq Analysis,” in Proc. 31st ACM International Conference on Information and Knowledge Management (CIKM 2022), Atlanta, GA, Oct. 2022.

Time series anomaly detection has been widely studied recent years. Deep methods achieves success in many multi-variate time series scenarios. However, we know few about why these sophisticated and complex deep methods work well and thus it is usually hard to apply in reality. What’s more, deep methods rely on vast amounts of data which limits its application. Time series decomposition, data augmentation and frequency analysis are widely used in time series analysis while the combination with neural network are not fully considered. In this paper, we activate classical time series analysis techniques with a simple TCN representation network under the window-based framework. The design of our decomposition time series Anomaly Detection architecture with Time-Freq Analysis (TFAD) is concise and the SOTA performance of TFAD is impressive.

## TFAD model architecture
![avatar](https://github.com/DAMO-DI-ML/CIKM22-TFAD/blob/main/img_folder/TFAD-Art.png)

## Main Results
![avatar](https://github.com/DAMO-DI-ML/CIKM22-TFAD/blob/main/img_folder/results.png)
![avatar](https://github.com/DAMO-DI-ML/CIKM22-TFAD/blob/main/img_folder/ablation_study.png)
## Get Started 
This model follows the code of NCAD (adding_ncad_to_nursery branch in https://github.com/Francois-Aubet/gluon-ts.git )

1、
```
conda create --name TFAD python=3.8
conda activate TFAD
pip install -e tfad  
```
2、
```
python3 examples/article/run_all_experiments.py \
--tfad_dir='tfad' \
--data_dir='tfad/tfad_datasets' \
--hparams_dir='tfad/examples/article/hparams' \
--out_dir='tfad/output' \
--download_data=True \
--number_of_trials=10 \
--run_swat=False \
--run_yahoo=False
```

## Citation
If you find this repo useful, please cite our paper.

```
@inproceedings{zhang2022TFAD,
  title={{TFAD}: A Decomposition Time Series Anomaly Detection Architecture with Time-Freq Analysis},
  author={Chaoli, Zhang and Tian, Zhou and Qingsong, Wen and Liang, Sun},
  booktitle={31st ACM International Conference on Information and Knowledge Management (CIKM 2022)},
  location = {Atlanta, GA},
  pages={},
  year={2022}
}
```

## Contact
If you have any question or want to use the code, please contact chaoli.zcl@alibaba-inc.com .
                                     
