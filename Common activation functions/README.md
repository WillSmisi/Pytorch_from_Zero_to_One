# 常见激活函数及其变种
@(pytorch与深度学习)[机器学习, 深度学习, 梯度下降, TODO]

---------------------------
[TOC]
## Sigmoid(*)
> $f(x) = sigmoid(x)=\frac{1}{1+e^{-x}}$

> $f'(x)=f(x)(1-f(x))$



##Tanh(*)
> $f(x) = tanh(x)=2sigmoid(2x)-1$

> $f'(x)=1-tanh^2(x)$

 ![@Sigmoid和Tanh|center](./1551703698600.png)
##Rectified Linear Unit(ReLU)(*)
> $f(n)= \begin{cases} 0, & \text {for $x$ <0} \\ x, & \text{for $x \geq$ 0} \end{cases}$
![@ReLU|center](./1551705200837.png)
>ReLu虽然在大于0的区间是线性的，在小于等于0的部分也是线性的，但是它整体不是线性的，因为不是一条直线。多个线性操作的组合也是一个线性操作，没有非线性激活，就相当于只有一个超平面去划分空间。但是ReLu是非线性的，效果类似于划分和折叠空间，组合多个（线性操作 + ReLu）就可以任意的划分空间。

### Leaky ReLU(*)

------
### SELU

### Softplus

## Softmax与logits
- 多分类需求
 - 所有的输出加起来和为1
 - 所有的输出区间[0,1]
- Softmax作用
	- 金字塔效应，放大差距(原来大的会更加大)
	- 提供概率输出的性质
- Derivative
推导
>总结 

下图为总结图
![@Softmax总结图|center|380*220](./1551620054081.png)
>logits: 没有经过sigmoid和softmax叫做logits  [摘自](https://study.163.com/course/courseLearn.htm?courseId=1208894818&share=1&shareId=11055994&from=study#/learn/video?lessonId=1278349767&courseId=1208894818)

**为什么引入Relu,ReLU减缓了Sigmoid函数的梯度弥散和梯度下降？**
>第一，采用sigmoid等函数，反向传播求误差梯度时，求导计算量很大，而Relu求导非常容易。
>第二，对于深层网络，sigmoid函数反向传播时，很容易就会出现梯度消失的情况（在sigmoid接近饱和区时，变换太缓慢，导数趋于0），从而无法完成深层网络的训练。
>第三，Relu会使一部分神经元的输出为0，这样就造成了网络的稀疏性，并且减少了参数的相互依存关系，缓解了过拟合问题的发生。

