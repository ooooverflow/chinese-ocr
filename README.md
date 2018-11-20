# chinese-ocr
基于CTPN（tensorflow）+CRNN（pytorch）+CTC的不定长文本检测和识别

## 环境部署
    sh setup.sh  
      
    使用环境： python 3.6 + tensorflow 1.10 + pytorch 0.4.1
* 注：CPU环境执行前需注释掉for gpu部分，并解开for cpu部分的注释

## Demo
    python demo.py    
    
下载 [预训练模型](https://pan.baidu.com/s/1b2Fsf1oYgZOueYW2kvUpfg)  
### CRNN 
将pytorch-crnn.pth放入/train/models中  
### CTPN
将checkpoints.zip解压后的内容放入/ctpn/checkpoints中

## 模型训练
### warp-ctc安装pytorch版
详见 [warp-ctc.pytorch](https://github.com/SeanNaren/warp-ctc)

### CTPN训练
详见 [tensorflow-ctpn](https://github.com/eragonruan/text-detection-ctpn)
### CRNN训练
#### 1.数据准备
下载[训练集](https://pan.baidu.com/s/1E_1iFERWr9Ro-dmlSVY8pA)

* 共约364万张图片，按照99:1划分成训练集和验证集  
* 数据利用中文语料库（新闻 + 文言文），通过字体、大小、灰度、模糊、透视、拉伸等变化随机生成  
* 包含汉字、英文字母、数字和标点共5990个字符  
* 每个样本固定10个字符，字符随机截取自语料库中的句子  
* 图片分辨率统一为280x32  

修改/train/config.py中`train_data_root`，`validation_data_root`以及`image_path`  
#### 2.训练
    cd train  
    python train.py
#### 3.训练结果
![](https://github.com/ooooverflow/chinese-ocr/blob/master/demo/val.png)
![](https://github.com/ooooverflow/chinese-ocr/blob/master/demo/ocr.png)

## 效果展示
### CTPN
![](https://github.com/ooooverflow/chinese-ocr/blob/master/demo/demo.jpg)
### OCR
![](https://github.com/ooooverflow/chinese-ocr/blob/master/demo/demo.png)

## 参考
[warp-ctc-pytorch](https://github.com/SeanNaren/warp-ctc)  
[chinese_ocr-(tensorflow+keras)](https://github.com/YCG09/chinese_ocr)  
[CTPN-tensorflow](https://github.com/eragonruan/text-detection-ctpn)  
[crnn-pytorch](https://github.com/meijieru/crnn.pytorch)
