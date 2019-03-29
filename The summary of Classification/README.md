# 分类总结
@(pytorch与深度学习)[机器学习, Pytorch, TODO]

---------------------------
[TOC]

##信息论
### 信息量
[剑桥大学信息论视频](https://www.youtube.com/watch?v=BCiZc0n6COY&list=PLruBu5BI5n4aFpG32iMbdWoRVAA-Vcso6)
[信息论百度文库课件](https://wenku.baidu.com/view/4f6aa601ff00bed5b9f31dc9.html)

假设我们听到了两件事，分别如下:

**事件A**：巴西队进入了2018世界杯决赛圈。
**事件B**：中国队进入了2018世界杯决赛圈。

仅凭**直觉**来说，显而易见事件B的信息量比事件A的信息量要大。究其原因，是因为事件A发生的概率很大，事件B发生的概率很小。**所以当越不可能的事件发生了，我们获取到的信息量就越大。越可能发生的事件发生了，我们获取到的信息量就越小。**那么信息量应该和事件发生的概率有关。因此我们想找到一个函数$I(x)$,它是概率$ p(x)$ 的单调函数，表达了信息的内容。怎么寻找呢？如果我们有两个不相关的事件$ x $和$ y$，那么观察两个事件同时发生时获得的信息量应该等于观察到事件各自发生时获得的信息之和，即:$I(x,y)=I(x)+I(y)$
我们知道,因为两个时间是独立不相关的,因此$p(x,y)=p(x)p(y)$。根据这两个关系，**可以看出$I(x)$一定与$p(x)$的对数有关(因为对数的运算法则是$log_a(mn)=log_am+log_an$)。
于是，有人给出这样的**信息量**定义来描述我们这种直观的感受(数学建模)
>假设$X$是一个离散型的随机变量，其取值集合为$\chi$,概率分布函数为$p(x) = Pr(X=x),x\in\chi$,则定义事件$X = x_0$的信息量为:
>$$I(x_0)=-log(p(x_0))$$
>**自信息量的单位**:若这里的对数底为2，则单位为比特bit，由于在计算机上是二进制，信息论一般都采用比特。而机器学习中基常常选择为自然常数，因此单位常常被称为奈特nats。
>信息量代表两种含义：
>一、**事件$x$发生之前**,$I(x)$表示事件x发生的不确定性;
>二、**事件$x$发生之后**,$I(x)$表示事件x所提供的信息量。


由于$p(x_0)$是概率,所以$p(x_0)$的取值范围是[0,1],绘制为图形如下：
![@自信息函数图像 | center | 400x300](./1551419796464.png)
(摘录自简书-蒙奇奇路西)

-----

**自信息量计算的应用**
下面通过一道题目来看看
题目:假设一条电线上串联了8个灯泡$x_1$,$x_2$,...,$x_8$,这8个灯泡损坏的可能性是等概率的，假设有也只有一个灯泡损坏，用万用表去测量,获得足够的信息量，才能获知和确定哪个灯泡$x_i$损坏。下面就来看我们最少需要获得多少信息量才能判断出。
![@测量步骤 | center](./1551422036569.png)

解:
- 第一次测量获得的信息量:
  - $I(p_1(x))-I(p_2(x)) = log\frac{1}{p_1(x)}-log\frac{1}{p_2(x)}=3-2=1(bit)$
  - 本来是8个里面中的一个，变成4个里面的一个，缩小了样本空间;从公式的角度也反映了不确定性的降低，即为**获得的信息量**

- 第二次测量获得的信息量:
	- $I(p_2(x))-I(p_3(x)) = log\frac{1}{p_1(x)}-log\frac{1}{p_2(x)}=2-1=1(bit)$
- 第三次测量获得的信息量:
	- $I(p_3(x))-I(p_4(x)) = log\frac{1}{p_1(x)}-log\frac{1}{p_2(x)}=1-0=1(bit)$
所以，经过3次测量后使得不确定性变成了0，那么也就是需要3bit信息量。
### 熵
前面我们根据概率模型，通过自信息量的计算，能得到信源以及信宿中每个消息的不确定性。然而，事实上，人们往往关注的不仅仅是每个消息的不确定性，而是整个系统的不确定性的统计特性即整个信源自信息量的统计平均值——**熵**。

例1:假设有两个信源X和Y:

| $x_i$      |     0 |   1   |
| :--------: | :--------:| :------: |
| $P(x_i)$   |   0.5 |  0.5  |

| $y_i$      |     0 |   1   |
| :--------: | :--------:| :------: |
| $P(y_i)$   |   0.99 |  0.01  |

我们可以直观感受到，信源X和Y两个系统的稳定程度是不一样的，信源X我们可能收到的信号具有很大的不确定性，信源Y我们收到的信号只有很少的不确定性，大部分认为收到的是0。

所以为了衡量这种直观的感觉，学界给出了**平均自信息量——熵的定义**
> 设$X$是一个集合(即信息系统如信源或信道),其概率模型为$\{x_i,p(x_i)\}$,则定义系统$X$的平均自信息量为:
> $$H(x) = \sum_{x_i\in X}p(x_i)I(x_i)= -\sum_{x_i\in X}p(x_i)log(p(x_i))$$
> 熵的单位是比特/符号.

直观来看，$I(x_i)$是唯一确定$x_i$所需要的信息量,那么$H(X)$就是唯一确定$X$中任意时间所需的平均信息量。它反映了$X$中事件$x_i$出现的平均不确定性。当且仅当信源X中个消息等概率时成立，即个消息等概率分布为$p=\frac{1}{|X|}$时，信源熵最大。`很好理解，当所有信源的事件等概率，我们对可能接受到的信号丝毫把握都没有，这时的不确定性最大。`这也很好衡量了例1中的不稳定性，我们求$H(X)$和$H(Y)$,由定义有:$$H(X)=1$$
$$H(Y)=0.08$$
显然,$H(X)>>H(Y)$,这表示信源$X$的**平均不稳定性**远远大于信源$Y$的平均不稳定性,通过这种**建模定义**也符合我们的**直观感受**。

>**注意点**：
>- 熵只依赖于随机变量的分布,与随机变量取值无关，所以也可以将 $X$ 的熵记作 H(p)。
>- 令$0log0=0$(因为某个取值概率可能为0)。

----

### 熵的应用
我们用过一个例子看看熵的应用:
>考虑一个随机变量$ x$。这个随机变量有4种可能的状态，每个状态都是等可能的。为了把 $x$ 的值传给接收者，我们需要传输2比特的消息。$$H(X)=−4×\frac{1}{4}log_2\frac{1}{4}=2 bits$$
现在考虑一个具有4种可能的状态 {a,b,c,d} 的随机变量，每个状态各自的概率为 $(\frac{1}{2},\frac{1}{4},\frac{1}{8},\frac{1}{8})$
这种情形下的熵为：
$$H(X)=−\frac{1}{2}log_2\frac{1}{2}−\frac{1}{4}log_2\frac{1}{4}−\frac{1}{8}log_2
\frac{1}{8}−\frac{1}{8}log_2\frac{1}{8}=1.75 bits$$
我们可以看到，非均匀分布比均匀分布的熵要小。现在让我们考虑如何把变量状态的类别传递给接收者。我们可以使用一个2比特的数字来完成这件事情。这样
$$平均编码长度=2\times\frac{1}{2}+2\times\frac{1}{4}+2\times\frac{1}{8}+2\times\frac{1}{8}=2bits$$
然而，我们可以利用非均匀分布这个特点，使用更短的编码来描述更可能的事件，使用更长的编码来描述不太可能的事件。我们希望这样做能得到一个更短的平均编码长度。我们可以使用下面的编码串(哈夫曼编码):0、10、110、111来表示状态$\{a,b,c,d\}$。传输的编码的平均长度就是:
$$平均编码长度= 1\times \frac{1}{2}+2\times\frac{1}{4}+3\times\frac{1}{8}+3\times\frac{1}{8}=1.75bits$$
这个值与上方的随机变量的熵相等。熵和最短编码长度的这种关系是一种普遍的情形。[hannon 编码定理](https://baike.baidu.com/item/Shannon%20%E7%BC%96%E7%A0%81%E5%AE%9A%E7%90%86/15585931?fr=aladdin)**表明熵是传输一个随机变量状态值所需的比特位下界（最短平均编码长度）**因此，信息熵可以应用在数据压缩方面。详情[数据压缩与信息熵](http://www.ruanyifeng.com/blog/2014/09/information-entropy.html)

### 相对熵(KL散度)
**相对熵**(Relative entropy),也称**KL散度**(Kullback-Leibler divergence)。
>设$p(x)$、$q(x)$是离散随机变量$X$中取值的两个概率分布,则$p$对$q$的相对熵是:
$$D_{KL}(p||q)=\sum_{x}p(x)log\frac{p(x)}{q(x)}=E_{p(x)}log\frac{p(x)}{q(x)}$$
>**性质**:
1、如果 p(x) 和 q(x) 两个分布相同，那么相对熵等于0
2、$D_{KL}(p||q)≠D_{KL}(q||p)$ ,相对熵具有**不对称性**。
3、$D_{KL}(p||q)\ge 0$ 证明（[Jensen不等式](https://en.wikipedia.org/wiki/Jensen%27s_inequality)）
**总结:相对熵可以用来衡量两个概率分布之间的差异，上面的公式的意义就是求$p$与$q$之间的对数差在$p$上的期望值.**
### 交叉熵
现在有关于样本集的两个概率分布$p(x)$和$q(x)$,其中$p(x)$为真实分布,$q(x)$非真实分布。如果用真实分布$p(x)$来衡量识别一个样本所需要的编码长度的期望(平均编码长度)为:
$$H(p)=\sum_xp(x)log\frac{1}{p(x)}$$
如果使用非真实分布$q(x)$来表示真实分布$p(x)$的平均编码长度,则是:
$$H(p,q)=\sum_xp(x)log\frac{1}{q(x)}$$
**PS:因为用q(x)来编码的样本来自于分布q(x),所以H(p,q)中的概率是p(x)。此时将H(p,q)称之为`交叉熵`**
>举个例子:
>考虑一个随机变量$x$,真实分布$p(x)=(\frac{1}{2},\frac{1}{4},\frac{1}{8},\frac{1}{8})$,非真实分布$q(x)=(\frac{1}{4},\frac{1}{4},\frac{1}{4},\frac{1}{4})$,则$H(p)=1.75bits$(最短平均码长),交叉熵$H(p,q)=\frac{1}{2}log_24+\frac{1}{4}log_24+\frac{1}{8}log_24+\frac{1}{8}log_24=2bits$。由此可以看出根据非真实分布$q(x)$得到的平均码长大于根据真实分布$p(x)$得到的平均码长。


化简**相对熵**的公式我们可以得到$$D_{KL}(p||q)=H(p,q)-H(p)$$**当用非真实分布$q(x)$得到的平均码长比真实分布$p(x)$得到的平均码长多出的比特数就是`相对熵`**,又因为$D_{KL}(p||q)\ge 0$,所以$H(p,q)\ge H(p)$。当$p(x)=q(x)$时取等号,此时**交叉熵等于信息熵**。
>在`机器学习`中，训练数据分布是固定的。当$H(p)$为常量时,最小化相对熵$D_{KL}(p||q)$等价于最小化交叉熵$H(p,q)$也等价于**最大化似然估计**。
在机器学习中，我们希望在训练数据上模型学到的分布 P(model) 和真实数据的分布  P(real) 越接近越好，所以我们可以使其相对熵最小。但是我们没有真实数据的分布，所以只能希望模型学到的分布 P(model) 和训练数据的分布 P(train) 尽量相同。假设训练数据是从总体中独立同分布采样的，那么我们可以通过最小化训练数据的经验误差来降低模型的泛化误差。即：
1.希望学到的模型的分布和真实分布一致，$P_{model}(x)≃P_{real}(x)$
2.但是真实分布不可知，假设训练数据是从真实数据中独立同分布采样的，$P_{train}(x)≃P_{real}(x)$
因此，我们希望学到的模型分布至少和训练数据的分布一致，$P_{train}(x)≃P_{model}(x)$
根据之前的描述，最小化训练数据上的分布$  P_{train}(x) $与最小化模型分布$ P_{model}(x) $的差异等价于最小化相对熵，即 $D_{KL}(P_{train}(x)||P_{model}(x))$。此时，$ P_{train}(x) $就是$D_{KL}(p||q) $中的$ p$，即真实分布，$P_{model}(x) $就是 $q$。又因为训练数据的分布 $p $是给定的，所以求  $D_{KL}(p||q) $ 等价于求 $H(p,q)$。得证，**交叉熵可以用来计算学习模型分布与训练分布之间的差异**。交叉熵广泛用于逻辑回归的$Sigmoid$和$Softmax$函数中作为损失函数使用。

------

**以下都是`联合分布中(同一个分布中)两个变量`相互影响的关系**

### 条件自信息量
之前引入`自信息量`以及`熵`的概念,用以描述信源和信宿,事实上,**信宿收到的消息是与信源发出的消息密切相关。并且`接受信息与发送信息之间的关系往往是判定一个信道的好坏的最佳标准`**
>PS：这不恰好可以等价是说**真实输出是信源发出的数据,经过我们假设函数得到的输出是信宿得到的数据，而我们的假设函数（模型）就是信道**。衡量信道好坏就是衡量模型好坏。
`错误`：可惜结论是错误的,在信息论上面的概念有模糊才会有上面的**“PS”**错误结论偏差,机器学习学习到的分布$P(model)$和真实分布$P(real)$属于**同一随机变量不同分布**，我们通过机器学习的方法，使得针对原数据的分布我们学习到的分布能够更加接近于真实分布。换句话来说，我们在用我们从数据的学习到的分布来估计数据真实分布，表现数据真实分布。即$p(model)\longrightarrow p(real)$

所以，我们需要引入**互信息量**的概念，在学习互信息量之前我们先来了解**条件信息量**的概念。
>设消息$x$发出的先验概率为$p(x)$,接收到的消息$y$是由$x$发出的概率为$p(y \;| \;x)$,则在收到$y$是由$x$发出的条件自信息量$I(x\;|\;y)$定义为:
$$I(x\;|\;y)=-log(p(x\;|\;y))$$

**计算条件自信息量的例子**
例2 在二进制对称信道BSC中，若信道转移概率矩阵为:
$$[p(y|x)]=\begin{matrix}
\vphantom{\vdots} x/y \\
\vphantom{\vdots} 0 \\
\vphantom{\vdots} 1\\
\end{matrix}
\begin{bmatrix}
\vphantom{\vdots} 0.875 &0.125\\
\vphantom{\vdots} 0.125 &0.875  \\
\end{bmatrix}
$$
计算下列条件自信息量(若$p(0)=p(1)=1$):
$I(x=0|y=1),I(y=1|\;x=0),I(y=1|\; x=1)$
由已知条件可得:
$$p(x =0|y=1)=\frac{1}{8},I(x=0|y=1)=log8=3$$
$$p(y=1|x =0)=\frac{1}{8},I(y=1|x=0)=log8=3$$
$$p(y=1|x=1)=\frac{7}{8},I(y=1|x=1)=log8-log7$$

-----
>直观来看,我们知道,在通信之前,消息x具有不确定性$p(x)$,其大小为x的**自信息量**:
$$I(x)=-log p(x)$$
当我们收到信息$y$,它是否由$x$发出也有一定的不确定性$p(x|y)$,其大小为**条件自信息量**:
$$I(x|y)=-log(p(x|y))$$
两者之间的差就是我们通过这一次通信所获得到的信息量的大小。
同样，收到的消息为$y$具有不确定性$p(y)$，其大小为$y$的**自信息量**:
$$I(y)=-log(p(y))$$
当我们发出消息$x$,它是否收到$y$也有一定的不确定性$p(y|x)$,其大小为**条件自信息量**:
$$I(y|x)=-log p(y|x)$$
两者之间的差也是我们通过这一次通信所**获得的信息量**大小。
### 互信息量
很显然,从通信的角度来看,上述两个差值应该相等,即:
$$I(x)-I(x\;|\; y) = I(y)-I(y\; |\; x)$$
事实上,由概率论概率的乘积公式有:
$$p(x,y)=p(x)·p(y\; |\; x)=p(y)·p(x\; |\; y)$$
故:
$$I(x)-I(x\; |\; y)=log\frac{p(x\; |\; y)}{p(x)}=log\frac{p(x\; |\; y)}{p(x)}=I(y)-I(y\;|\;x)$$
这样，用$(x;y)$或$(y;x)$记改差式,称为$x$与$y$之间的**互信息量**,单位也是比特。
这里，有必要概括下**互信息量的性质**。
> - 一、对称性:  $I(x;y)=I(y;x)$,其通信意义表示发出$x$收到$y$所能提供给我们的信息量的大小;
> - 二、当$x$与$y$统计独立时,$I(x;y)=I(y;x)=0$,表示这样一次通信不能为我们提供任何信息。
**上述两条性质与我们实际情况非常吻合。**

----

>熵是信源平均不确定性的度量,一般情况下，他并不等于信宿所获得的平均信息量，只有在无噪的情况下，两者才相等(和之前线性模型提到的思想相通)。为此我们需要学习条件熵。同时我们由条件熵引出平均互信息量的概念，其可以用来衡量一个**信道的好坏**。

### 条件熵
>设$X$是信道的消息集,$Y$是信宿消息集,对条件自信息量I(x|y)取统计平均值得到条件熵$H(X|Y)$,即:
$$H(X|Y)=\sum_x\sum_y p(x,y)I(x|y)=-\sum_x\sum_y p(x,y)log p(x|y)$$
其中$p(x,y)$为联合概率,$p(x|y)$为条件概率
### 平均互信息量
>很显然,信源$X$的熵$H(X)$与条件熵$H(X|Y)$的差值和信宿$Y$的熵$H(Y)$与条件熵$H(Y|X)$的差值相等，我们称为$X$和$Y$的平均互信息量,记为:
$$H(X;Y)=H(X)-H(X|Y)=H(Y)-H(Y|X)$$
$I(X;Y)$是一个用来衡量信道好坏的非常好的工具。

**计算条件熵的例子**
例3  设一个二进制对诚信到BSC:
$$[p(y|x)]=
\begin{bmatrix}
\vphantom{\vdots} 0.9 &0.1\\
\vphantom{\vdots} 0.1 &0.9\\
\end{bmatrix}
$$
其先验概率为p(0)=p(1)=1/2,计算条件熵.
**解答:**我们知道$p(x,y)=p(y|x)·p(x)$
所以有
$p(1,0)=0.005,p(1,1)=0.45,p(0,1)=0.05,p(0,0)=0.45$
所以
$$\begin{split}
H(X|Y)&=  -\sum_x\sum_y p(x,y)log(p(y\;|\;x)) \\
&=-2 \times 0.45log0.9-2\times0.05log0.1=0.469 \\
&=H(Y|X)\end{split}$$
结果表明,虽然每个字符的错误率只有0.1,可导致**整个信宿对信源的平均不确定性**达到了0.496,将近一半。可见通信系统对信道要求非常高。

###基尼指数
$Gini(p)=1-\sum^{K}_{k=1}p_k^2$
对于给定的样本集合$D$,其基尼指数为:
$Gini(D)=1-\sum_{k=1}^{K})(\frac{|C_k|}{|D|})$
这里,$C_k$是$D$中属于第$k$类的样本子集,$K$是类的个数
## 机器学习中的Cross Entropy
###One-hot编码分类问题的损失函数
>在数据处理和特征工程中，经常会遇到类型数据，如性别分为[男，女]，手机运营商分为[移动，联通，电信]等，我们通常将其转为数值带入模型，如[0,1], [-1,0,1]等，但模型往往默认为连续型数值进行处理，这样其实是违背我们最初设计的，也会影响模型效果。

**One-hot**便是解决这个问题，其方法是使用N位状态寄存器来对N个状态进行编码，每个状态都由他独立的寄存器位，并且在任意时候，其中只有一位有效。

如自然编码为：0，1
独热编码为：10，01           

可以理解为对有m个取值的特征，经过独热编码处理后，转为m个二元特征，每次只有一个激活。
如数字字体识别0~9中，6的独热编码为：0000001000
**优点**:
- 能够处理非连续型数值特征
- 在一定程度上也扩充了特征，比如性别本身是一个特征,经过One-hot编码后，就变成了男或女两个特征

当然在特征类别较多时，数据通过One-hot编码会变得过于稀疏。在语言处理用来表示词向量的过程这个问题表示的比较突出。采用**嵌入向量**的方式解决。这里不再过多拓展。[详情](https://blog.csdn.net/dugudaibo/article/details/79071541)

每一个样本都是独立同分布.所以可以针对每个样本计算**熵**，在One-hot编码中,式子$$H(p,q)=D_{KL}(p||q)+H(p)$$
中$H(p)=1\times log1=0$,所以$H(p,q)=D_{KL}(p||q)$,这样优化$H(p,q)$就是优化$D_{KL}(p||q)$使得$p,q$分布接近,当$D_{KL}(p||q)$**越接近0**,也就是说我们的**模型假设数据分布**与**真实分布**越接近。
>针对二分类,
$$\begin{split}H(P,Q)&=-\sum_{i=(cat,dog)}P(i)log Q(i)\\
&=-P(cat)logQ(cat)-P(dog)logQ(dog)\\
令P(cat)=y,得到&=-(ylog(p)+(1-y)log(1-p))
\end{split}$$

-----
**分类不使用MSE？**

- sigmoid+MSE
	- **gradient vanish**
>**梯度弥散**:靠近输出层的隐藏层梯度大，参数更新快，所以很快就会收敛，而靠近输出层的隐藏层,梯度小,参数更新慢,几乎就和初始状态一样,随机分布。
	
- converage slower
- But,sometimes
	- **e.g.** meta-learning
## 多分类问题