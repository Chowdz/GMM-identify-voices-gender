# GMM-identify-voices-gender
EM algorithm based on GMM to extract voice features and identify male and female voices

## 1.理论推导

### 1.1.EM算法的推导

对于样本 $x\sim F(x,\theta)$，有$x=(x^{(1)},x^{(2)},...,x^{(n)})$，若用$MLE$方法求未知参数$\theta$，则

$\hat{\theta}=\mathop{argmax}\limits_{\theta}L(\theta)=\mathop{argmax}\limits_{\theta}\sum_{i=1}^n lnp(x^{(i)},\theta)$，但如果样本来自若干个仅参数不同的相同分布，就无法用$MLE$方法求出，可以设每

个样本$x^{(i)}$有$z^{(k)}$的概率服从于某个分布，$z^{(k)}$也是个随机变量，服从于多项分布，

有$z=(z^{(1)},z^{(2)},...,z^{(k)})$，设$P(z^{(i)})=Q_i(z^{(i)})$，

那么有$\sum_{z^{(i)}}Q_i(z^{(i)})=1$，且对于求解$\theta$来说，$z$是隐变量，那么如何求解未知参数$\theta$呢？有：

$\hat{\theta}=\mathop{argmax}\limits_{\theta}\sum_{i=1}^n lnp(x^{(i)},\theta)=\mathop{argmax}\limits_{\theta}\sum_{i=1}^n ln[\sum_{z^{(i)}}P(z^{(i)})p(x^{(i)},\theta|z^{(i)})] \\ =\mathop{argmax}\limits_{\theta}\sum_{i=1}^n ln[\sum_{z^{(i)}}p(x^{(i)},z^{(i)},\theta)]$ 

那么将$\hat{\theta}$作变换，有$\hat{\theta}=\mathop{argmax}\limits_{\theta}\sum_{i=1}^n ln[\sum_{z^{(i)}}Q_i(z^{(i)})\frac{p(x^{(i)},z^{(i)},\theta)}{Q_i(z^{(i)})}]$ ，由$Jensen$不等式可知，若$f^{''}(x)\geq0$，有$Ef(x)\geq f(Ex),(lnx)^{''}=-\frac{1}{x^2}<0, Elnx \leq ln(Ex)$，可以看出$\sum_{z^{(i)}}Q_i(z^{(i)})\frac{p(x^{(i)},z^{(i)},\theta)}{Q_i(z^{(i)})}$就是$\frac{p(x^{(i)},z^{(i)},\theta)}{Q_i(z^{(i)})}$的关于$z^{(i)}$的期望。



$\sum_{i=1}^n ln[\sum_{z^{(i)}}Q_i(z^{(i)})\frac{p(x^{(i)},z^{(i)},\theta)}{Q_i(z^{(i)})}]=\sum_{i=1}^n ln[E\frac{p(x^{(i)},z^{(i)},\theta)}{Q_i(z^{(i)})}]\geq\sum_{i=1}^n Eln\frac{p(x^{(i)},z^{(i)},\theta)}{Q_i(z^{(i)})} \\ =\sum_{i=1}^n \sum_{z^{(i)}}Q_i(z^{(i)})ln\frac{p(x^{(i)},z^{(i)},\theta)}{Q_i(z^{(i)})} \\ =\sum_{i=1}^n \sum_{z^{(i)}}Q_i(z^{(i)})lnp(x^{(i)},z^{(i)},\theta)-\sum_{i=1}^n \sum_{z^{(i)}}Q_i(z^{(i)})lnQ_i(z^{(i)})$



$ \hat{\theta}=\mathop{argmax}\limits_{\theta} \Big[\sum_{i=1}^n \sum_{z^{(i)}}Q_i(z^{(i)})lnp(x^{(i)},z^{(i)},\theta)-\sum_{i=1}^n \sum_{z^{(i)}}Q_i(z^{(i)})lnQ_i(z^{(i)})\Big]$，可以看出上式后半部分与$\theta$无关

$ \hat{\theta}=\mathop{argmax}\limits_{\theta} \sum_{i=1}^n \sum_{z^{(i)}}Q_i(z^{(i)})lnp(x^{(i)},z^{(i)},\theta)$

但注意，我们这里是当且仅当$Jensen$不等式成立时，也就是$P(X=EX)=1$，即认为$X$为常数时，放到这里也就是$\frac{p(x^{(i)},z^{(i)},\theta)}{Q_i(z^{(i)})}=c$（$c$为常数），$ L(\theta)=\sum_{i=1}^n ln[E\frac{p(x^{(i)},z^{(i)},\theta)}{Q_i(z^{(i)})}]\geq\sum_{i=1}^n Eln\frac{p(x^{(i)},z^{(i)},\theta)}{Q_i(z^{(i)})}$，若我们最大化$L(\theta)$，通过最大化该式右半部分也就是最大化$L(\theta)$的下界，未必可以找到$L(\theta)$的最大值，除非等号成立，也就是$\frac{p(x^{(i)},z^{(i)},\theta)}{Q_i(z^{(i)})}=c$，这其实就是$EM$算法中的$E$步。

若定义$\frac{p(x^{(i)},z^{(i)},\theta)}{Q_i(z^{(i)})}=c$，$ p(x^{(i)},z^{(i)},\theta)=c\ast Q_i(z^{(i)})$，两边对$z^{(i)}$求和，$\therefore \sum_{z^{(i)}}p(x^{(i)},z^{(i)},\theta)=c\ast \sum_{z^{(i)}}Q_i(z^{(i)})=c$

$ \frac{p(x^{(i)},z^{(i)},\theta)}{Q_i(z^{(i)})}=\sum_{z^{(i)}}p(x^{(i)},z^{(i)},\theta) , Q_i(z^{(i)})=\frac{p(x^{(i)},z^{(i)},\theta)}{\sum_{z^{(i)}}p(x^{(i)},z^{(i)},\theta)}=\frac{p(x^{(i)},z^{(i)},\theta)}{\sum_{z^{(i)}}p(z^{(i)})p(x^{(i)},\theta|z^{(i)})}=p(z^{(i)}|x^{(i)},\theta)$

$$当且仅当$\frac{p(x^{(i)},z^{(i)},\theta)}{Q_i(z^{(i)})}=c$，即为定值时，有：

<font color="#dd000"> $\hat{\theta}=\mathop{argmax}\limits_{\theta} \sum_{i=1}^n \sum_{z^{(i)}}p(z^{(i)}|x^{(i)},\theta)*lnp(x^{(i)},z^{(i)},\theta)$</font>

> 可以看出$\sum_{i=1}^n \sum_{z^{(i)}}p(z^{(i)}|x^{(i)},\theta)*lnp(x^{(i)},z^{(i)},\theta)$就是$lnp(x^{(i)},z^{(i)},\theta)$关于$z|x,\theta$的条件期望，也就是当样本和未知参数$\theta$给定时，我们既然不知道隐变量$z$的取值，就用期望去替代它，当然时在样本$x$和参数$\theta$给定的条件下，求完这个条件期望，再去关于$\theta$去极大化$L(\theta)$，这就是$EM$算法的基本思想。
>
> 所以，我们的$E$步就是，先随机定出$\theta^{j}$作为初始值，当然这个初始值也包含隐变量$z$，然后利用样本$x$去求条件概率$Q_i(z^{(i)})=p(z^{(i)}|x^{(i)},\theta^j),j=1,2,...N$为迭代次数，$\theta^j$和$\theta$不一样，$\theta$是上帝知道的最优值，$\theta^j$是不断迭代去靠近$\theta$的值，求出$Q_i(z^{(i)})$后代入到$L(\theta)=\sum_{i=1}^n \sum_{z^{(i)}}Q_i(z^{(i)})*lnp(x^{(i)},z^{(i)},\theta)$当中，然后对$\theta$求偏导，极大化$L(\theta)$，得到$\theta^{j+1}$，这是$M$步。再把$\theta^{j+1}$替换原来的$\theta^j$，再重新执行$E$步，不断迭代，直到$L(\theta)$收敛。



### 1.2.EM算法的收敛性证明

那么如何得知$L(\theta)$按照这样的方法一定会收敛（极大化）呢？

即要证明$L(\theta,\theta^{j+1})\geq L(\theta,\theta^j)$就能说明每次$L(\theta,\theta^j)$都在变大。

$ L(\theta,\theta^{j})=\sum_{i=1}^n \sum_{z^{(i)}}p(z^{(i)}|x^{(i)},\theta^j)*lnp(x^{(i)},z^{(i)},\theta),$ 	①

定义$H(\theta,\theta^{j})=\sum_{i=1}^n \sum_{z^{(i)}}p(z^{(i)}|x^{(i)},\theta^j)*lnp(z^{(i)}|x^{(i)},\theta),$ 	②

①$-$②：

$L(\theta,\theta^{j})-H(\theta,\theta^{j})=\sum_{i=1}^n \sum_{z^{(i)}}p(z^{(i)}|x^{(i)},\theta^j)*ln \frac{p(x^{(i)},z^{(i)},\theta)}{p(z^{(i)}|x^{(i)},\theta)}=\sum_{i=1}^n \sum_{z^{(i)}}p(z^{(i)}|x^{(i)},\theta^j)*lnp(x^{(i)},\theta) \\ =\sum_{i=1}^n lnp(x^{(i)},\theta)\sum_{z^{(i)}}p(z^{(i)}|x^{(i)},\theta^j)=\sum_{i=1}^n lnp(x^{(i)},\theta)$



即$\sum_{i=1}^n lnp(x^{(i)},\theta)=L(\theta,\theta^{j})-H(\theta,\theta^{j})$

$ \sum_{i=1}^n lnp(x^{(i)},\theta^{j+1})-\sum_{i=1}^n lnp(x^{(i)},\theta^j)=[L(\theta^{j+1},\theta^{j})-H(\theta^{j+1},\theta^{j})]-[L(\theta^j,\theta^{j})-H(\theta^j,\theta^{j})] \\ =[L(\theta^{j+1},\theta^{j})-L(\theta^j,\theta^{j})]-[H(\theta^{j+1},\theta^{j})-H(\theta^j,\theta^{j})]$

定义前半部分为$A$，$A=L(\theta^{j+1},\theta^{j})-L(\theta^j,\theta^{j})$，$$从$\theta^j$代入到$L(\theta)$中，到$\theta^{j+1}$代入到$L(\theta)$，就是在不断极大化$L(\theta)$，属于$M$步，所以$L(\theta^{j+1},\theta^{j})$肯定大于$L(\theta^{j},\theta^{j})$，$ A\geq0$

定义后半部分为$B$，有$B=H(\theta^{j+1},\theta^{j})-H(\theta^j,\theta^{j})=\sum_{i=1}^n \sum_{z^{(i)}}p(z^{(i)}|x^{(i)},\theta^j)*lnp(z^{(i)}|x^{(i)},\theta^{j+1})-\sum_{i=1}^n \sum_{z^{(i)}}p(z^{(i)}|x^{(i)},\theta^j)*lnp(z^{(i)}|x^{(i)},\theta^j) \\ =\sum_{i=1}^n \sum_{z^{(i)}}p(z^{(i)}|x^{(i)},\theta^j)*ln \frac{p(z^{(i)}|x^{(i)},\theta^{j+1})}{p(z^{(i)}|x^{(i)},\theta^{j})}=\sum_{i=1}^n Eln \frac{p(z^{(i)}|x^{(i)},\theta^{j+1})}{p(z^{(i)}|x^{(i)},\theta^{j})}\leq\sum_{i=1}^n lnE \frac{p(z^{(i)}|x^{(i)},\theta^{j+1})}{p(z^{(i)}|x^{(i)},\theta^{j})} \\ =\sum_{i=1}^n ln\sum_{z^{(i)}}p(z^{(i)}|x^{(i)},\theta^j) \frac{p(z^{(i)}|x^{(i)},\theta^{j+1})}{p(z^{(i)}|x^{(i)},\theta^{j})}=\sum_{i=1}^n ln\sum_{z^{(i)}} p(z^{(i)}|x^{(i)},\theta^{j+1})=\sum_{i=1}^n ln1=0$

$\big($上式中用到了$Jensen$不等式，$Elnx\leq ln(Ex)$ $\big)$，$ B\leq0$

$ A\geq0,B\leq 0$，大于等于0的数减去小于等于0的数必大于等于0，$ L(\theta,\theta^{j+1})\geq L(\theta,\theta^j)$，$$似然函数收敛



### 1.3.EM算法的步骤

有样本$x=(x^{(1)},x^{(2)},...,x^{(n)})$，隐变量$z=(z^{(1)},z^{(2)},...,z^{(n)})$，设最大迭代次数为$N$，未知参数$\theta=(\theta_1,\theta_2,...,\theta_m,z)$，注意，$z$也是个未知参数

1.取$\theta_{iter}=\theta^0$，初始迭代次数$n_{iter}=0$

2.$\quad while$   $n_{iter} \leq N:$

​		$$E步=\begin{cases}last\theta_{iter}=\theta_{iter} \\ Q_i(z^{(i)})=p(z^{(i)}|x^{(i)},\theta_{iter}) \\ L(\theta,\theta_{iter})=\sum_{i=1}^n \sum_{z^{(i)}}Q_i(z^{(i)})lnp(x^{(i)},z^{(i)},\theta_{iter})\end{cases}$$ 

​		$$M步=\begin{cases}\theta_{iter}=\mathop{argmax}\limits_{\theta}L(\theta,\theta_{iter}) \\ if \quad |L(\theta,\theta_{iter})-L(\theta,last\theta_{iter})|<\epsilon \\ \qquad break\end{cases}$$ 

​		$$n=n+1$$ 

​		<font color="#dd000">$return \quad \theta_{iter}$</font>



### 1.4.GMM混合高斯分布在EM算法中的求解过程

现有样本$x=(x_1,x_2,...,x_n)$，每个单独的样本为一个二维向量，即$x_i=(x_i^{(1)},x_i^{(2)})$

但是样本里混合了两个高斯分布，两个高斯分布的均值方差和相关系数均未知，即未知参数为$\theta=(u,\sigma^2,\rho)$

且每个样本均有$Q_i(z^{(i)})$的概率服从一个二维高斯分布，二维高斯密度为：

$\varphi(x^{(1)},x^{(2)})=\frac{1}{2\pi\sigma_1\sigma_2\sqrt{1-\rho^2}}exp\left\{-\frac{1}{2(1-\rho^2)}\big[\frac{(x^{(1)}-u^{(1)})^2}{\sigma_1^2}-2\rho\frac{(x^{(1)}-u^{(1)})(x^{(2)}-u^{(2)})}{\sigma_1\sigma_2}+\frac{(x^{(2)}-u^{(2)})^2}{\sigma_2^2}\big]\right\}$

$$加入隐变量的最大似然过程为：

<font color="#dd000"> $\hat{\theta}=\mathop{argmax}\limits_{\theta} \sum_{i=1}^n \sum_{j=1}^2p(z_{ji}|x_i,\theta)*lnp(x_i,z_ji,\theta),\quad x_i=(x_i^{(1)},x_i^{(2)})$</font>

其中$p(z_{ji}|x_i,\theta)=\frac{p(z_ji)p(x_i,\theta|z_ji)}{p(z_{1i})p(x_i,\theta|z_1i)+p(z_{2i})p(x_i,\theta|z_2i)},j=1,2;i=1,2,...,n$

隐变量条件分布的矩阵为：

|          |          $x_1$           |         $x_2$          | ...  |         $x_n$          |
| :------: | :----------------------: | :--------------------: | :--: | :--------------------: |
| $p(z_1)$ | $$p(z_{11}|x_1,\theta)$$ | $p(z_{12}|x_2,\theta)$ | ...  | $p(z_{1n}|x_n,\theta)$ |
| $p(z_2)$ |  $p(z_{21}|x_1,\theta)$  | $p(z_{22}|x_2,\theta)$ | ...  | $p(z_{2n}|x_n,\theta)$ |
|  $sum$   |            1             |           1            | ...  |           1            |

每个$x_i=(x_i^{(1)},x_i^{(2)})\sim N(u^{(1)},u^{(2)};\sigma^{2(1)},\sigma^{2(2)};\rho)$，我们首先要设置初始值：

$u^{(0)}=(u_{11},u_{12};u_{21},u_{22}),\quad \sigma^{2(0)}=(\sigma_{11}^2,\sigma_{12}^2;\sigma_{21}^2,\sigma_{22}^2),\quad \rho^{(0)}=(\rho_1,\rho_2),\quad p(z)^{(0)}=(p(z_1),p(z_2))$

经过对似然函数求偏导方法，可得：

$E$步为：

$p(z_1)^{(1)}=\frac{\sum_{i=1}^n p(z_1|x_i,\theta)}{n},\qquad p(z_2)^{(1)}=\frac{\sum_{i=1}^n p(z_2|x_i,\theta)}{n}$

$M$步为：

<font color="#dd000">$u_{11}^{(1)}=\frac{\sum_{i=1}^n p(z_1|x_i,\theta)x_{i}^{(1)}}{\sum_{i=1}^n p(z_1|x_i,\theta)},\quad u_{12}^{(1)}=\frac{\sum_{i=1}^n p(z_1|x_i,\theta)x_{i}^{(2)}}{\sum_{i=1}^n p(z_1|x_i,\theta)},\quad u_{21}^{(1)}=\frac{\sum_{i=1}^n p(z_2|x_i,\theta)x_{i}^{(1)}}{\sum_{i=1}^n p(z_2|x_i,\theta)},\quad u_{22}^{(1)}=\frac{\sum_{i=1}^n p(z_2|x_i,\theta)x_{i}^{(2)}}{\sum_{i=1}^n p(z_2|x_i,\theta)}$</font>

<font color="#dd000">$\sigma_{11}^{2(1)}=\frac{\sum_{i=1}^n p(z_1|x_i,\theta)(x_{i}^{(1)}-u_{11}^{(1)})^2}{\sum_{i=1}^n p(z_1|x_i,\theta)},\quad \sigma_{12}^{2(1)}=\frac{\sum_{i=1}^n p(z_1|x_i,\theta)(x_{i}^{(2)}-u_{12}^{(1)})^2}{\sum_{i=1}^n p(z_1|x_i,\theta)},\quad \sigma_{21}^{2(1)}=\frac{\sum_{i=1}^n p(z_2|x_i,\theta)(x_{i}^{(1)}-u_{21}^{(1)})^2}{\sum_{i=1}^n p(z_2|x_i,\theta)},\quad \sigma_{22}^{2(1)}=\frac{\sum_{i=1}^n p(z_2|x_i,\theta)(x_{i}^{(2)}-u_{22}^{(1)})^2}{\sum_{i=1}^n p(z_2|x_i,\theta)}$</font>

<font color="#dd000">$\rho_1^{(1)}=p(z_1)^{(1)}r,\quad \rho_2^{(1)}=p(z_2)^{(1)}r$</font>

迭代退出条件：$|L(\theta^{j+1})-L(\theta^{j})|<\epsilon$



### 1.5.EM算法的评价方法

若衡量EM算法求解出的位置参数$\theta$所包含的信息量的多少，其实就是求出样本的$Fisher$信息量，即为似然方程的方差，如果方差越大，说明样本收集的信息越多，也就说明用该样本估计出的$\hat {\theta}$越能够代表总体。

但一般的$Fisher$信息量比较好求，而这里的似然方程包含隐变量，其推导如下：

首先，包含隐变量的、且取完对数的、并求导的样本似然函数为$S(x,\theta)= \sum_{i=1}^{n}\frac{\partial \sum_{z}p(z|x_i,\theta^j)*lnp(x_i,z,\theta)}{\partial \theta}$，我们称其为似然方程，<font color="#dd000">现在我们就要求出 $S(x,\theta)$的方差，即可衡量$EM$算法包含$\theta$信息的多少。这里注意，$p(z|x_i,\theta^j)$中的$\theta^j$和$lnp(x_i,z,\theta)$中的$\theta$是不一样的。</font>

$ E\big(S(x,\theta)\big)=\sum_{i=1}^n \sum_z p(z|x_i,\theta^j) \int_{-\infty}^{+\infty} \frac{\partial lnp(x_i,z,\theta)}{\partial \theta}p(x_i,z,\theta)dx=\sum_{i=1}^n \sum_z p(z|x_i,\theta^j)  \int_{-\infty}^{+\infty} \frac {1}{p(x_i,z,\theta)} \frac{\partial p(x_i,\theta)}{\partial \theta} p(x_i,\theta)dx$

$=\sum_{i=1}^n \sum_z p(z|x_i,\theta^j) \frac{\partial}{\partial \theta} \int_{-\infty}^{+\infty} p(x_i,z,\theta)dx$

<font color="#dd000">（上式中用了求导和积分可交换顺序）</font>

$ E\big(S(x,\theta)\big)=\sum_{i=1}^n \sum_z p(z|x_i,\theta^j) \frac{\partial}{\partial \theta} *1 =0$，即$E\big(S(x,\theta)\big)=0$

$ Var\big(S(x,\theta)\big)=E\big(S(x,\theta)\big)^2-[E\big(S(x,\theta)\big)]^2=E\big(S(x,\theta)\big)^2-0=E\big(S(x,\theta)\big)^2$

$ Var\big(S(x,\theta)\big)=E\big(S(x,\theta)\big)^2=\sum_{i=1}^n E\bigg(\frac{\partial \sum_{z}p(z|x_i,\theta^j)*lnp(x_i,z,\theta)}{\partial \theta}\bigg)^2$，记$\sum_{z}p(z|x_i,\theta^j)$为$\alpha_i$，即代表各个分布的权重，则：

<font color="#dd000">$Var\big(S(x,\theta)\big)=nE\bigg(\alpha_1 \frac{\partial lnp(x,z_1,\theta)}{\partial \theta} +\alpha_2 \frac{\partial lnp(x,z_2,\theta)}{\partial \theta} + ...+\alpha_n \frac{\partial lnp(x,z_n,\theta)}{\partial \theta}\bigg)^2=nI(\theta)$ 即为$EM$算法的似然方程的方差，其方差越大，即$I(\theta)$越大，则估计越好。</font>
