---
theme: gaia
_class: lead
paginate: true
color: black
backgroundColor: #fff
# backgroundImage: url('https://marp.app/assets/hero-background.jpg')
marp: true
---

<style>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
</style>


# [MobileNets](https://arxiv.org/abs/1605.07146)
###### : Efficient Convolutional Neural Networks <br>for Mobile Vision Applications

<br>

##### 2021. 08. 03. 
#### 이동건

---
## 0. Contents
1. Introduction
2. Prior Work<!-- Section 2 reviews prior work in building small models.  -->
3. MobileNet Architecture<!-- Section 3 describes the MobileNet architecture and two hyper-parameters width multiplier and resolution multiplier to define smaller and more efficient MobileNets.  -->
3.1. Depthwise Separable Convolution
3.2. Network Structure and Training
3.3. Width Multiplier: Thinner Models
3.4. Resolution Multiplier: Reduced Representation
---
## 0. Contents
4. Experiments<!-- Section 4 describes experiments on ImageNet as well a variety of different applications and use cases.  -->
4.1. Model Choices
4.2. Model Shrinking Hyperparameters
4.3. Fine Grained Recognition
4.4. Large Scale Geolocalizaton
4.5. Face Attributes
4.6. Object Detection
4.7. Face Embeddings
5. Conclusion<!-- Section 5 closes with a summary and conclusion. -->
6. References
---

## 1. Introduction
<br>

- #accuracy
- #efficiency
    - #size
    - #speed
---
## 1. Introduction
- AlexNet 이후로 <!--CNN은 다양한 vision 분야에서 사용되고 있다.--> <!--모델의 -->accuracy를 높이기 위해 layer를 더욱 깊게 쌓아 네트워크의 depth를 늘리는 것이 일반적인 트렌드가 되었다.
- Deep한 네트워크를 설계하여 accuracy를 개선하는 것은 모델의 size, speed와 같은 efficiency 측면에서 볼 때 문제가 될 수 있다.
- Real world에서는 computer resource가 제한적이기 때문에 문제가 된다.
<br>
- 본 논문에서는 small, low latency model을 설계하기 위한 방법을 통해,  efficient network architecture를 제안한다.

---

## 2. Prior Work<!-- Section 2 reviews prior work in building small models.  -->

- 최근에도 small and efficient neural network를 설계하기 위한 연구가 있지만 기존의 방법들은 pretrained network를 압축하거나 small network를 directly하게 학습시켰던 두 분류로 나눌 수 있다.

- 본 논문에서는 제한된 latency, size 안에서 small network를 구체적으로 설계할 수 있는 아키텍처를 제안한다.
---
## 2. Prior Work<!-- Section 2 reviews prior work in building small models.  -->

> *<br>MobileNets primarily focus on optimizing for latency but also yield small networks.
<br>Many papers on small networks focus only on size but do not consider speed.<br>ㅤ*

- 제안하는 MobileNets은 latency를 최적화하면서 small network를 만드는 것에 초점을 맞춘다.

---



---

## References
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- [eremo2002 - Paper Review : MobileNets~](https://eremo2002.github.io/paperreview/MobileNet/)


---
## [코드 실습]()

- References

