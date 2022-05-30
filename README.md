# cifar100

首先沿用之前手写数字识别的LeNet，训练50次达到了30%出头的准确率，考虑到结构太简单，拟合效果不好

于是试用了其他有名的模型，如vgg16，vgg19，但效果均不好，于是不断改变学习率衰减，学习率以及训练次数，效果提升不大

然后用resnet34和resnet50做了测试，发现通过改变学习率衰减和训练次数，可以达到50出头的准确率

最后使用了GoogleNet，lr=0.001，逢10折半，训练50次能有60%出头，然后不断改变学习率衰减方法，如等间隔衰减，三次不下降衰减等，以及去掉dropout2d层，达到63%左右

其实通过多次训练，我发现每次训练到后面准确率一直下降，直到学习率折半才可能不下降，因此想通过更多次改变学习率来提高准确率，改学习率衰减方法为：连续三次测试集准确率不下降就折半。这样一来训练到50次也会处在拟合比较好的状态（64%左右），取1-50次里拟合效果最好的就是保存的模型，即准确率65.33%，训练时间约为1.22h

以下是训练记录（仅放出最终模型）：

```
epoch: 0
        batch: 0, loss: 4.6462
        batch: 200, loss: 3.6738
train acc: 13.00%, loss: 3.6400
test  acc: 20.63%, loss: 3.1256
耗时：96.41377019882202s
epoch: 1
        batch: 0, loss: 3.0469
        batch: 200, loss: 2.5448
train acc: 29.01%, loss: 2.6681
test  acc: 29.45%, loss: 2.7300
耗时：92.19750213623047s
epoch: 2
        batch: 0, loss: 2.3202
        batch: 200, loss: 2.3444
train acc: 40.15%, loss: 2.1693
test  acc: 40.81%, loss: 2.1745
耗时：94.79819011688232s
epoch: 3
        batch: 0, loss: 1.8423
        batch: 200, loss: 1.9425
train acc: 47.11%, loss: 1.8692
test  acc: 41.86%, loss: 2.1509
耗时：95.43264150619507s
epoch: 4
        batch: 0, loss: 1.5791
        batch: 200, loss: 1.6362
train acc: 52.19%, loss: 1.6693
test  acc: 46.60%, loss: 1.9198
耗时：97.20437788963318s
epoch: 5
        batch: 0, loss: 1.6566
        batch: 200, loss: 1.5505
train acc: 55.88%, loss: 1.5207
test  acc: 49.39%, loss: 1.8331
耗时：97.45988202095032s
epoch: 6
        batch: 0, loss: 1.2443
        batch: 200, loss: 1.3602
train acc: 59.11%, loss: 1.3906
test  acc: 46.45%, loss: 2.0054
耗时：97.44350457191467s
epoch: 7
        batch: 0, loss: 1.2061
        batch: 200, loss: 1.1713
train acc: 61.85%, loss: 1.2796
test  acc: 53.47%, loss: 1.6636
耗时：96.85943841934204s
epoch: 8
        batch: 0, loss: 0.9694
        batch: 200, loss: 1.2675
train acc: 64.72%, loss: 1.1844
test  acc: 48.61%, loss: 1.9489
耗时：97.57309579849243s
epoch: 9
        batch: 0, loss: 1.2389
        batch: 200, loss: 1.1513
train acc: 66.98%, loss: 1.0974
test  acc: 54.73%, loss: 1.6612
耗时：96.81271886825562s
epoch: 10
        batch: 0, loss: 0.9167
        batch: 200, loss: 1.0617
train acc: 68.90%, loss: 1.0288
test  acc: 54.49%, loss: 1.6563
耗时：96.89318346977234s
epoch: 11
        batch: 0, loss: 0.7213
        batch: 200, loss: 0.9613
train acc: 71.08%, loss: 0.9505
test  acc: 56.66%, loss: 1.5526
耗时：97.51925873756409s
epoch: 12
        batch: 0, loss: 0.6592
        batch: 200, loss: 0.7883
train acc: 72.87%, loss: 0.8818
test  acc: 55.10%, loss: 1.6546
耗时：98.10981702804565s
epoch: 13
        batch: 0, loss: 0.8978
        batch: 200, loss: 0.7252
train acc: 74.55%, loss: 0.8290
test  acc: 58.14%, loss: 1.5324
耗时：110.21925354003906s
epoch: 14
        batch: 0, loss: 0.6929
        batch: 200, loss: 0.8259
train acc: 75.91%, loss: 0.7746
test  acc: 55.73%, loss: 1.6888
耗时：102.14366960525513s
epoch: 15
        batch: 0, loss: 0.6101
        batch: 200, loss: 0.6381
train acc: 77.74%, loss: 0.7171
test  acc: 51.19%, loss: 1.9838
耗时：103.62581586837769s
epoch: 16
        batch: 0, loss: 0.6976
        batch: 200, loss: 0.8209
train acc: 78.70%, loss: 0.6823
test  acc: 56.95%, loss: 1.6551
耗时：108.24550342559814s
lr now changes:0.0005
epoch: 17
        batch: 0, loss: 0.5303
        batch: 200, loss: 0.3596
train acc: 88.86%, loss: 0.3782
test  acc: 63.20%, loss: 1.3997
耗时：105.62753820419312s
epoch: 18
        batch: 0, loss: 0.1966
        batch: 200, loss: 0.2055
train acc: 93.18%, loss: 0.2464
test  acc: 62.37%, loss: 1.5042
耗时：103.07838201522827s
epoch: 19
        batch: 0, loss: 0.1508
        batch: 200, loss: 0.1760
train acc: 94.05%, loss: 0.2192
test  acc: 61.43%, loss: 1.5847
耗时：102.41235017776489s
epoch: 20
        batch: 0, loss: 0.2043
        batch: 200, loss: 0.2321
train acc: 93.00%, loss: 0.2403
test  acc: 60.78%, loss: 1.6075
耗时：102.9318299293518s
lr now changes:0.00025
epoch: 21
        batch: 0, loss: 0.1756
        batch: 200, loss: 0.0957
train acc: 97.84%, loss: 0.1071
test  acc: 65.11%, loss: 1.4468
耗时：93.78915691375732s
epoch: 22
        batch: 0, loss: 0.0428
        batch: 200, loss: 0.0425
train acc: 99.59%, loss: 0.0475
test  acc: 65.06%, loss: 1.4660
耗时：89.92707109451294s
epoch: 23
        batch: 0, loss: 0.0427
        batch: 200, loss: 0.0184
train acc: 99.82%, loss: 0.0348
test  acc: 65.15%, loss: 1.4824
耗时：92.15653419494629s
lr now changes:0.000125
epoch: 24
        batch: 0, loss: 0.0210
        batch: 200, loss: 0.0209
train acc: 99.94%, loss: 0.0243
test  acc: 65.33%, loss: 1.4489
耗时：98.06640720367432s
epoch: 25
        batch: 0, loss: 0.0200
        batch: 200, loss: 0.0180
train acc: 99.96%, loss: 0.0202
test  acc: 65.07%, loss: 1.4530
耗时：97.63865780830383s
epoch: 26
        batch: 0, loss: 0.0143
        batch: 200, loss: 0.0140
train acc: 99.96%, loss: 0.0204
test  acc: 64.76%, loss: 1.4699
耗时：97.56008672714233s
lr now changes:6.25e-05
epoch: 27
        batch: 0, loss: 0.0170
        batch: 200, loss: 0.0156
train acc: 99.97%, loss: 0.0178
test  acc: 65.22%, loss: 1.4557
耗时：98.52833867073059s
epoch: 28
        batch: 0, loss: 0.0169
        batch: 200, loss: 0.0150
train acc: 99.98%, loss: 0.0175
test  acc: 65.07%, loss: 1.4539
耗时：98.48674821853638s
epoch: 29
        batch: 0, loss: 0.0126
        batch: 200, loss: 0.0130
train acc: 99.97%, loss: 0.0174
test  acc: 64.71%, loss: 1.4658
耗时：101.81504011154175s
lr now changes:3.125e-05
epoch: 30
        batch: 0, loss: 0.0145
        batch: 200, loss: 0.0133
train acc: 99.98%, loss: 0.0160
test  acc: 64.91%, loss: 1.4638
耗时：98.61040329933167s
epoch: 31
        batch: 0, loss: 0.0111
        batch: 200, loss: 0.0118
train acc: 99.97%, loss: 0.0155
test  acc: 64.72%, loss: 1.4628
耗时：98.52904939651489s
epoch: 32
        batch: 0, loss: 0.0284
        batch: 200, loss: 0.0144
train acc: 99.97%, loss: 0.0159
test  acc: 64.45%, loss: 1.4676
耗时：98.48794507980347s
lr now changes:1.5625e-05
epoch: 33
        batch: 0, loss: 0.0123
        batch: 200, loss: 0.0119
train acc: 99.98%, loss: 0.0146
test  acc: 64.53%, loss: 1.4624
耗时：98.74755001068115s
epoch: 34
        batch: 0, loss: 0.0177
        batch: 200, loss: 0.0139
train acc: 99.97%, loss: 0.0149
test  acc: 64.62%, loss: 1.4668
耗时：98.71589016914368s
epoch: 35
        batch: 0, loss: 0.0097
        batch: 200, loss: 0.0109
train acc: 99.97%, loss: 0.0149
test  acc: 64.31%, loss: 1.4697
耗时：99.26207375526428s
lr now changes:7.8125e-06
epoch: 36
        batch: 0, loss: 0.0116
        batch: 200, loss: 0.0156
train acc: 99.98%, loss: 0.0144
test  acc: 64.16%, loss: 1.4735
耗时：93.76258397102356s
epoch: 37
        batch: 0, loss: 0.0129
        batch: 200, loss: 0.0159
train acc: 99.98%, loss: 0.0141
test  acc: 64.18%, loss: 1.4668
耗时：96.47436952590942s
epoch: 38
        batch: 0, loss: 0.0129
        batch: 200, loss: 0.0122
train acc: 99.97%, loss: 0.0143
test  acc: 64.37%, loss: 1.4731
耗时：97.20536828041077s
lr now changes:3.90625e-06
epoch: 39
        batch: 0, loss: 0.0115
        batch: 200, loss: 0.0136
train acc: 99.98%, loss: 0.0138
test  acc: 64.11%, loss: 1.4668
耗时：89.60581064224243s
epoch: 40
        batch: 0, loss: 0.0141
        batch: 200, loss: 0.0143
train acc: 99.98%, loss: 0.0138
test  acc: 64.32%, loss: 1.4705
耗时：89.07503771781921s
epoch: 41
        batch: 0, loss: 0.0123
        batch: 200, loss: 0.0140
train acc: 99.98%, loss: 0.0140
test  acc: 64.21%, loss: 1.4709
耗时：88.57020854949951s
lr now changes:1.953125e-06
epoch: 42
        batch: 0, loss: 0.0141
        batch: 200, loss: 0.0137
train acc: 99.98%, loss: 0.0137
test  acc: 64.20%, loss: 1.4686
耗时：88.80091738700867s
epoch: 43
        batch: 0, loss: 0.0131
        batch: 200, loss: 0.0152
train acc: 99.98%, loss: 0.0138
test  acc: 64.18%, loss: 1.4715
耗时：89.24178743362427s
epoch: 44
        batch: 0, loss: 0.0152
        batch: 200, loss: 0.0125
train acc: 99.97%, loss: 0.0138
test  acc: 64.02%, loss: 1.4748
耗时：89.97269010543823s
lr now changes:9.765625e-07
epoch: 45
        batch: 0, loss: 0.0155
        batch: 200, loss: 0.0143
train acc: 99.98%, loss: 0.0137
test  acc: 64.23%, loss: 1.4723
耗时：89.22952270507812s
epoch: 46
        batch: 0, loss: 0.0138
        batch: 200, loss: 0.0183
train acc: 99.97%, loss: 0.0138
test  acc: 64.13%, loss: 1.4721
耗时：88.76394462585449s
epoch: 47
        batch: 0, loss: 0.0120
        batch: 200, loss: 0.0109
train acc: 99.99%, loss: 0.0136
test  acc: 64.06%, loss: 1.4725
耗时：88.77308416366577s
lr now changes:4.8828125e-07
epoch: 48
        batch: 0, loss: 0.0125
        batch: 200, loss: 0.0148
train acc: 99.98%, loss: 0.0137
test  acc: 64.06%, loss: 1.4719
耗时：88.75014734268188s
epoch: 49
        batch: 0, loss: 0.0116
        batch: 200, loss: 0.0118
train acc: 99.98%, loss: 0.0138
test  acc: 64.26%, loss: 1.4706
耗时：88.99230933189392s
```

> 还有结果可视化的问题，我尝试了一下可视化，结果标签胡成一团，图片放大了也看不清（毕竟1000张图片合成一张图片），感觉有点恶心，蛮放上去吧



说实话最近有点忙，这是我大概半个月前的实践经历，之后一直在复习这个复习那个，最近也没空继续改良了，今天就直接交了吧，专心复习。（希望大伙给我点面子，别让我垫底呜呜呜）