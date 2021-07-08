# [WRN Codes](https://drive.google.com/file/d/1hG8Dp5n3lgg4HEsLk8hx3NxQud6hjHHT/view?usp=sharing)
1. [WRN Main Architecture](#1-wrn-main-architecture)
    * [1.1 Main code](#11-main-code)
        + [1.1.1. depth가 6n+4인 이유](#111-depth가-6n4인-이유)
        + [1.1.2. WRN의 전체적인 흐름](#112-wrn의-전체적인-흐름)
        + [1.1.3. conv block](#113-conv-block)
        + [1.1.4. BasickBlock](#114-basickblock)
    - [1.2. WRN Structures](#12-wrn-structures)
    * [1.2.1. WRN-40-2](#121-wrn-40-2)
    * [1.2.2. WRN-16-8](#122-wrn-16-8)
    * [1.2.3. WRN-16-10](#123-wrn-16-10)
    * [1.2.4. WRN-40-4](#124-wrn-40-4)
2. [Graphs](#2-graphs)
    * [2.1 WRN-40-2](#21-wrn-40-2)
    * [2.2 WRN-16-8](#22-wrn-16-8)
    * [2.3 WRN-16-10](#23-wrn-16-10)
    * [2.4 WRN-40-4](#24-wrn-40-4)
3. [Experiment Results](#3-experiment-results)
4. [참고](#4-참고)

* 모든 실험은 CIFAR-10 Dataset을 활용하여 진행하였습니다.

---
<br>

## 1. WRN Main Architecture

### 1.1 Main code

<details>
<summary>Code 보기</summary>
<div markdown="1">

```python
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8, 1, 0)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
```

</div>
</details>
<br>

#### 1.1.1. depth가 6n+4인 이유
![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbEiXsY%2Fbtq3Ih0enEU%2FdvcJRSPncF6xZ6fyHJSgH0%2Fimg.png)

먼저, conv1에서 1개, 그리고 conv2, conv3, conv4에서 shortcut connection 형태로 layer가 들어가게 되어 기본적으로 4개의 layer가 있어야 합니다.

그리고 6n이 되는 이유는, conv2에서 3x3 짜리 2개, conv3에서 3x3짜리 2개, conv4에서 3x3짜리 2개로 총 6개가 기본적인 setting이므로, 이 setting보다 몇 배 더 많은 layer를 구성할 것인지를 결정하기 위해 6n가 되게 됩니다. 

만약 layer의 총 개수가 16이라면, 6*2 + 4가 되며, 이는 conv2 block 내에서의 layer가 4개, conv3 block 내에서의 layer가 4개, conv4 block 내에서의 layer가 4개가 됩니다. 

즉, n이라는 것은 conv block 내에서 3x3 conv layer의 개수에 곱해지는 계수라고 생각할 수 있습니다.

<br>

#### 1.1.2. WRN의 전체적인 흐름
위의 Table 1의 structure가 코드로 구현되어 있음을 확인할 수 있습니다.

```Python
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8, 1, 0)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
```
(WideResNet에서 이쪽 부분을 확인해주시면 됩니다!)

<br>

먼저 self.conv1은 3x3 conv이며, 16 channel이 output이 되도록 연산이 되게 됩니다.

self.block1, self.block2, self.block3는 위 Table 1에서 conv2, conv3, conv4에 대응됩니다.

self.block에 대해서는 조금 있다가 자세히 살펴보도록 할 예정입니다.

self.block3까지 지난 결과를 BN과 ReLU를 통과시키고, average pooling을 적용해줍니다. 

그리고 이를 [batch_size, 64*widen_factor]의 shape로 만들어주고, 이를 마지막 fully connected layer에 연결하여 dataset의 class 각각에 대한 probability를 정답으로 내게 됩니다.

<br>

#### 1.1.3. conv block

```Python
self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], BasicBlock, 1, dropRate)
```

먼저 첫 번째 conv block을 살펴봅니다.

간단한 예시를 들기 위해서, n = 2 / nChannels[0] = 16 / nChannels[1] = 64 / dropRate = 0.3으로 지정한다고 가정하겠습니다.

```python
class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)
```

따라서, 우리가 만들어낸 것은 ```NetworkBlock(nb_layers = 2, in_planes = 16, out_planes = 64, block = BasicBlock, stride = 1, dropRate= 0.3)```이 됩니다.

이를 통해서 만들어지는 것은 다음과 같습니다.

```python
def _make_layer(self, block = BasicBlock, in_planes=16, out_planes=64, nb_layers=2, stride=1, dropRate=0.3):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
```

layer의 개수만큼 for문을 돌게 되고, 이를 통해서 ```layers = []```에 block이 append 되는 구조입니다.

nb_layers = 2이므로, block이 2개 쌓이게 되겠죠?

<br>

여기서 조금 생소한 코드가 등장하는데요, `A and B or C `형태의 문법이 사용되었습니다.

이는 `B if A else C`와 똑같은 의미입니다.

즉, `BasicBlock(in_planes if i == 0 else out_planes, out_planes, stride if i == 0 else 1, dropRate)`가 만들어집니다.

`layers = [BasicBlock(16, 64, 1, 0.3), BasicBlock(64, 64, 1, 0.3)]`가 될 것입니다.

<br>

#### 1.1.4. BasickBlock

```python
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)
```

BasicBlock(16, 64, 1, 0.3)이라고 가정하고 생각해보겠습니다.

여기서 먼저 따져봐야 할 점은 input의 shape와 output의 shape가 같은지입니다. 

이를 확인한 후, `self.equalInOut`이라는 변수로 boolean 값으로 저장합니다.

BasicBlock(16, 64, 1, 0.3)이라면 input의 shape와 output의 shape가 다른 경우이므로, `self.equalInOut`은 `False`가 됩니다.

`self.convShortcut`의 경우는` self.equalInOut`이 `False`이면 `nn.conv2d`를 적용하고, `True`이면 `None`이 됩니다.

<br> 

다음으로는 forward 부분을 보면 됩니다.

`if not self.equalInOut`이므로, 이 if문에 들어가게 됩니다.

따라서 `x = self.relu1(self.bn1(x))`이 됩니다.

그러고 나서 `out = self.relu2(self.bn2(self.conv1(x)))`이 되고

droprate = 0.3이므로, 여기에 dropout을 적용하여 `out = F.dropout(out, p = self.droprate, training = self.training)`을 적용합니다.

`training = self.training`을 적용하는 이유는, 현재 모델이 train모드인지 eval 모드인지에 따라서 dropout을 사용할지 아닐지를 결정하게 됩니다.

왜냐하면 학습이 된 상태에서 추론을 하는 경우에는 dropout을 사용하지 않아야 하기 때문이죠. 

그리고 나서 `out = self.conv2(out)`을 해주고

`torch.add(self.convShortcut(x))`을 해준 것을 `return` 해줍니다.

이를 그림으로 한번 그려보자면, 다음과 같습니다.

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FzyGDo%2Fbtq3JRUgzX4%2FKQwKnwJbYTW0zWWdnBgeOk%2Fimg.png)

k는 kernel size이며 s는 stride, p는 padding을 나타냅니다.

본 그림을 통해서 BasicBlock의 구동 방식을 이해해볼 수 있을 것이라고 생각이 들고요.

만약 stride = 2인 경우는 self.conv1에서 s = 2로 변하고, 그 이후로 가로와 세로의 길이가 /2로 줄어듭니다.

그리고 input과 output의 channel 수가 동일한 경우는 위의 그림에서 self.conv1 = nn.Conv2d(16, 64, ~)을 통과하면서 channel수가 변하는 것이 아닌, 그대로 channel 수가 유지되게 됩니다.

self.convShortcut 또한 필요가 없어지기 때문에, 이는 None이 되고 위의 def forward에서 return 쪽을 확인해보시게 되면, input인 x를 그대로 torch.add에 넣어주는 것을 확인할 수 있습니다. 

여기까지 WRN의 모델에 대한 설명은 마무리가 되었습니다.

<br>

## 1.2. WRN Structures
### 1.2.1. WRN-40-2

![image](https://user-images.githubusercontent.com/57930520/116812848-d5297580-ab8b-11eb-94e6-563e0fdd52f1.png)

### 1.2.2. WRN-16-8

![image](https://user-images.githubusercontent.com/57930520/116812869-fa1de880-ab8b-11eb-8e5a-b7df982fe2fd.png)

### 1.2.3. WRN-16-10

![image](https://user-images.githubusercontent.com/57930520/117536885-9a668800-b038-11eb-8c4b-48557eb417a0.png)

### 1.2.4. WRN-40-4

![image](https://user-images.githubusercontent.com/57930520/117987629-9f4b7480-b375-11eb-9b1c-ab9848d389ed.png)

---
<br>

## 2. Graphs

### 2.1 WRN-40-2

![image](https://user-images.githubusercontent.com/57930520/116696198-8e594580-a9fc-11eb-91f9-426e7ed847c2.png)

![image](https://user-images.githubusercontent.com/57930520/116696244-9ca76180-a9fc-11eb-89ee-6b88450c1b2c.png)

### 2.2 WRN-16-8

![image](https://user-images.githubusercontent.com/57930520/116812802-93003400-ab8b-11eb-960a-a000e33eec5c.png)

### 2.3 WRN-16-10

![image](https://user-images.githubusercontent.com/57930520/117536803-3217a680-b038-11eb-8b47-0acc00455867.png)

### 2.4 WRN-40-4

![image](https://user-images.githubusercontent.com/57930520/117987311-53003480-b375-11eb-8cb2-4a090ed72eb3.png)

---
<br>

## 3. Experiment Results

실험에 사용된 각 Architecture와 Top Test Accuracy, 그리고 이를 도달했을 때의 epoch을 나타냅니다.



| Network Architecture | Top Test Accuracy | Epoch when reach to Top Test Accuracy |
| -------------------- | ----------------- | ------------------------------------- |
| WRN-40-2             | 95.22%            | 185 epoch                             |
| WRN-16-8             | 95.65%            | 199 epoch                             |
| WRN-16-10            | 95.82%            | 175 epoch                             |
| WRN-40-4             | 96.03%            | 176 epoch                             |

---
<br>

## 4. 참고
* [PeterKim1/paper_code_review](https://github.com/PeterKim1/paper_code_review/blob/master/7.%20Wide%20Residual%20Networks(WRN)/README.md)
* [cumulu-s : Wide Residual Networks(WRN) - code review](https://cumulu-s.tistory.com/36?category=933558)
