* Chaoli Zhang, Tian Zhou, Qingsong Wen, Liang Sun, "TFAD: A Decomposition Time Series Anomaly Detection Architecture with Time-Freq Analysis,” in Proc. 31st ACM International Conference on Information and Knowledge Management (CIKM 2022), Atlanta, GA, Oct. 2022.

Time series anomaly detection has been widely studied recent years. Deep methods achieves success in many multi-variate time series scenarios. However, we know few about why these sophisticated and complex deep methods work well and thus it is usually hard to apply in reality. What’s more, deep methods rely on vast amounts of data which limits its application. Time series decomposition, data augmentation and frequency analysis are widely used in time series analysis while the combination with neural network are not fully considered. In this paper, we activate classical time series analysis techniques with a simple TCN representation network under the window-based framework. The design of our decomposition time series Anomaly Detection architecture with Time-Freq Analysis (TFAD) is concise and the SOTA performance of TFAD is impressive.

## TFAD model architecture
![avatar](https://github.com/DAMO-DI-ML/CIKM22-TFAD/blob/main/img_folder/TFAD.png)

## Main Results
![avatar](https://github.com/DAMO-DI-ML/CIKM22-TFAD/blob/main/img_folder/results.png)
![avatar](https://github.com/DAMO-DI-ML/CIKM22-TFAD/blob/main/img_folder/ablation_study.png)
## Get Started 
This model follows the code of NCAD (adding_ncad_to_nursery branch in https://github.com/Francois-Aubet/gluon-ts.git )

1、

```
git clone -b adding_ncad_to_nursery https://github.com/Francois-Aubet/gluon-ts.git
```
2、Replace /ncad/examples folder with our /TFAD/examples folder;     
3、Replace /ncad/src/ncad folder with our /TFAD/src/ncad folder;      
4、
```
pip install -e gluon-ts/src/gluonts/nursery/ncad
```
5、
```
python3 examples/article/run_all_experiments.py \
--ncad_dir='~/gluon-ts/src/gluonts/nursery/ncad' \
--data_dir='~/gluon-ts/src/gluonts/nursery/ncad_datasets' \
--hparams_dir='~/gluon-ts/src/gluonts/nursery/ncad/examples/article/hparams' \
--out_dir='~/gluon-ts/src/gluonts/nursery/ncad_output' \
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
                                     
