# CNN总结
@(pytorch与深度学习)[机器学习, 深度学习, 卷积神经网络, TODO]

---------------------------
[TOC]


##共享权值
![Alt text](./1552049113919.png)

##信号上面的卷积运算

##卷积神经网络的流程
- **Input_channels:**输入通道
- **Kernel_channels:2ch**:卷积核
- **Kerne_Size**
- **Stride**
- **Padding**

-----
 
给个例子:
>我们有
>理解
>- $Input\_channels:$输入通道(比如黑白照片通道为1,彩色照片通道一般为3)
>- $Kernel\_channels:$:比如边缘检测核,模糊核($Blur,Edge$)`每一个kernel有输入通道的通道，假设输入通道3，Kernel_channels为2,则总共有2*3个核，其中每个检测核重复3层。假设输入为x:[b,3,28,28],那么一个核为[3,3,3]，多个核为[16,3,3,3]，这里的Kernel_channels为16,该层的偏置也是向量[16]`[详情](https://study.163.com/course/courseLearn.htm?courseId=1208894818&share=1&shareId=11055994#/learn/video?lessonId=1278346780&courseId=1208894818)
>>- $Kernel\_size:$核尺寸($3\times 3$,$5\times 5$)
>- $Padding:$在原始的图片周围的填充
>- $Stride:$做卷积运算时移动步长
>$$Input(N,C_{in},H_{in},W_{in})$$
>$$Output(N,C_{out},H_{out},W_{out})$$
>$$H_{out} =\lfloor \frac{H_{in} + 2 \times padding[0]-dilation[0]\times (kernel\_size[0]-1)-1}{stride[0]}\rfloor+1 $$
>$$W_{out} =\lfloor \frac{W_{in} + 2 \times padding[1]-dilation[1]\times (kernel\_size[1]-1)-1}{stride[1]}\rfloor+1 $$
其中的dilation指的是

这里比较学术比较细,弄得太吓人了，还是看看吴恩达的简洁些，也好记些:
> $\large{Summary\;of\;notation}$
> $If\;layer\;l\;is\;a\;convolution\;layer:$
> $\;\;\;\;\;\;f^{[l]}=filter\;size\qquad \qquad \qquad Input(n^{[l-1]}_H\times n^{[l-1]}_W\times n^{[l-1]}_C)$
>  $\;\;\;\;\;\;p^{[l]}=padding\;\;\qquad \qquad \qquad output(n^{[l]}_H\times n^{[l]}_W\times n^{[l]}_C)$
>  $\;\;\;\;\;\;s^{[l]}=stride$
>  **则有**$$n^{[l]}_H=\lfloor \frac{n^{[l-1]}_H+2p^{[l]}-f^{[l]}}{s^{[l]}}+1\rfloor$$
>  $$n^{[l]}_W=\lfloor \frac{n^{[l-1]}_W+2p^{[l]}-f^{[l]}}{s^{[l]}}+1\rfloor$$


##卷积层叠加的好处
![@卷积层累加特征捕捉|center](./1552098410180.png)

##pytorch卷积实现
![Alt text](./1552098699720.png)

##Pooling层
- 下采样($downsample$):可以每隔步长采样，
- pooling使用最大值或者平均值运算
![@pytorch Pooling示例](./1552108269531.png)

上采样($upsample$)
![@pytorch Upsample示例](./1552108433758.png)

##Batch Normalization
###为什么使用BatchNorm?
有时候我们不得不使用Sigmoid函数，我们知道，当使用Sigmoid函数时，当输入在范围比较大的时候，范围的两端会在Sigmoid中的梯度变化，为了减少这种梯度弥散的现象，将输入用规范缩放到Sigmiod梯度较大效果的范围内即0的附近,方便训练取得比较好的效果。
![Alt text](./1552109242903.png)
另外，将特征缩放到0周围，小方差，这样子最优值搜索快捷方便稳定。
这就是**特征缩放**的思想
特征缩放在图片中有：
- Image Normalization
![@j图片正则|center](./1552109796732.png)

- Batch Norm
	- Batch Norm
	- Layer Norm
	- Instance Norm
	- Group Norm
![Alt text|center](./1552110050377.png)
###pytorch实现BatchNorm
![@pytorch 1D BatchNorm|center](./1552113027754.png)
具体规范化写法
![@BatchNorm规范化表示](./1552113597300.png)
![@pytorch 2D BatchNorm](./1552113854260.png)
其中weight对应的是规范化中提到的$\gamma$,bias对应的是$\beta$
>**Test**的时候只会一个sample,这时的$\gamma$,$\beta$不会改变用之前的训练的值,这时的$\mu$和$\sigma$是全局的$\mu$和$\sigma$,一个epoch的。需要调用函数切换
![@切换到test模式示例|center](./1552572359056.png)


###使用BN的好处
- 收敛更快
- 更好的性能
- 更稳定
	- 使得模型受到学习率等超参数不再那么敏感，参数调整会更方便

##经典卷积神经网络
![@卷积神经网络发展|center](./1552464435071.png)
###LeNet-5
80年代发明，最开始的版本用于手写数字的识别。
当时deeplearning没有得到很好的发展，被SVM统治。
###AlexNet
- Similar framework to LeNet but:
	- Max pooling,ReLU nonlinearity
	- More data and bigger model(7 hidden layers,650K units,60M params)
	- GPU  implementation (50x speedup over CPU)
		- Trained on two GPUs for a week
	- Dropout regularization
###VGG
- VGG11
- VGG16
- VGG19
####1*1 Convolution
- 很少的计算量同样可以完成卷积运算
- 可以实现维度的转换。比如
>$Input = [1,3,28,28]$
>$Weight = [16,3,1,1]$
>$Out = [1,16,28,28]$
### GoogLeNet
探索出来的新的东西:
- 对同一层可以使用不同类型的卷积核，不同的视野可以感受到不同的信息量,例如:
![@同一层不同类型卷积核|center](./1552467650626.png)
### ResNet
$The\;residual\;module:$
- 介绍了$skip\;or\;shortcut$(短路层) $connections$
- 使得网络层数来表达
- 因为某种原因,需要去跳过至少两层
**我们之前把$conv-BN-Pool-ReLU$称为一个单元Unit**，现在一个单元为$短接线-conv-ReLU-conv-ReLU-短接线$,即
![@ResNet一个单元|center](./1552555626086.png)
**使用2~3层卷积层为Unit效果比较好**
之所以叫残差是因为对于短接层:
$$F(x)=H(x)-x$$
学习的是$H(x)$和$x$的残差
###Inception



###DenseNet
中间的任一层都有机会和最开始的每一层有机会接触。



