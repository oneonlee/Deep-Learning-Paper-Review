# Deep-Learning-Paper-Review
딥 러닝 논문 리뷰

**목차**
1. [영어 논문이라고 겁먹지 말자](https://github.com/oneonlee/Deep-Learning-Paper-Review/blob/main/README.md#%EC%98%81%EC%96%B4-%EB%85%BC%EB%AC%B8%EC%9D%B4%EB%9D%BC%EA%B3%A0-%EA%B2%81%EB%A8%B9%EC%A7%80-%EB%A7%90%EC%9E%90)
2. [논문을 효율적으로 읽는 방법 - 3단계 접근법](https://github.com/oneonlee/Deep-Learning-Paper-Review#%EB%85%BC%EB%AC%B8%EC%9D%84-%ED%9A%A8%EC%9C%A8%EC%A0%81%EC%9C%BC%EB%A1%9C-%EC%9D%BD%EB%8A%94-%EB%B0%A9%EB%B2%95---3%EB%8B%A8%EA%B3%84-%EC%A0%91%EA%B7%BC%EB%B2%95)


## 영어 논문이라고 겁먹지 말자
### 논문의 영어 구조는 명확하다.
```
abstract - 나는 이런 문제를 풀거야
introduction - 사실 이 문제는 이런 동기에서 연구가 시작된건데
related works - 관련해서 이런저런 접근들이 있었지
method - 난 이런 새로운 방식으로 접근해보려고 하는데
experiment - 정말 이게 잘 먹히는지 실험도 해봤어
discussion - 이를 통해 이런 사실도 알아냈지만 한계점도 있지
conclusion - 마지막으로 귀찮은 너를 위해 요약
```

논문은 위의 구조에서 ‘이런, 저런, 어떻게’ 등등이 무엇으로 치환 됐는지만 알면 된다. 수식? 그건 정말 이 논문을 재현할만큼 관심이 있을 때 자세히 들여다 보는거고 (혹은 문장보다는 수식으로 확인하는게 더 명확하기에 들여다 보는거고), 결국 논문의 핵심은 ‘내가 주어진 문제에서 이러한 기여(contribution)를 했다’가 내용의 대부분인 것이다.

### 0. 논문 고르기
*생략*
### 1. 초록 읽기 (Abstract)
세상 연구자들 중 99%는 초록(abstract)부터 읽는다. ~~물론 제목부터 읽고…~~ 초록은 마치 ‘출발 비디오여행’에서 보여주는 영화의 하이라이트와 같기 때문이다. 게다가 대부분의 논문들은 ‘초록 읽기’ 단계에서 나머지를 읽느냐 마느냐가 결정된다. 그러니 논문 읽기는 초록의 한문장 한문장을 유심히 뜯어보는 것으로 시작하도록 하자.

```
Human activity recognition (HAR) in ubiquitous computing is beginning to adopt deep learning to substitute for well-established analysis techniques that rely on hand-crafted feature extraction and classification techniques. (한가한 소리로 시작하고 있네.)

From these isolated applications of custom deep architectures it is, however, difficult to gain an overview of their suitability for problems ranging from the recognition of manipulative gestures to the segmentation and identification of physical activities like running or ascending stairs.(어떤 점들이 어려운 점들이라 하는군... 문제 소개)

In this paper (밑줄 쫙, 이제부터 내가 뭘했다는 얘기다. 아마도 위에서 언급한 어려움을 해결하려 했겠지?) we rigorously explore deep, convolutional, and recurrent approaches across three representative datasets that contain movement data captured with wearable sensors. (딥러닝을 세가지 웨어러블 센서 데이터셋을 가지고 탐색했군. 탐색이라... 뭐 이런 애매한 단어를...) 

We describe (1) how to train recurrent approaches in this setting, introduce a (2) novel regularisation approach, and illustrate (3) how they outperform the state-of-the-art on a large benchmark dataset. (이런 것들을 했구나... 앞으로 이런거 찾으며 읽으면 되겠다.)

Across thousands of recognition experiments with randomly sampled model configurations we investigate the suitability of each model for different tasks in HAR, explore the impact of hyperparameters using the fANOVA framework, and provide guidelines for the practitioner who wants to apply deep learning in their problem setting. (실험도 했고 파라메터들의 영향도 조사했고, 실험결과에 따른 가이드라인도 제시했다고 하네.)
```

논문 초록을 다 읽었다면 적어도 이 논문이 ‘무슨 문제’를 풀려고 했고, ‘어떠한 새로운 기여’를 담고 있는지 파악했어야 한다. (만약 이 문제가 내가 관심있는 문제가 아니라면 논문 패스…)

우리가 선택한 이 논문에서는 애플워치 같은 웨어러블 센서를 이용해 사람의 행동을 인식하는 문제(HAR)를 다루고 있는데,  (1) 웨어러블 센서 데이터는 어떻게 학습해야 하는지 소개하고, (2) 거기에 적당한 새로운 regularization 방법을 제시했으며, (3) 이것이 어떤 파라메터 세팅 속에서 잘되는건지 실험을 통해 증명한 것 같다.

이정도면 오케이. 혹시 내가 잘못 이해했을 수 있으니 결론을 미리 한번 보도록 하자.

### 2. 결론 읽기 (Conclusion)
논문은 꼭 순서대로 읽을 필요가 없다. 이 글은 초록을 통해 ‘다루는 문제와 이 논문의 기여’를 파악한 후, 내가 제대로 이해했는지 확인하기 위해 결론을 먼저 읽는 방법을 택하고 있다.

```
In this work we explored the performance of state-of-the-art deep learning approaches for Human Activity Recognition using wearable sensors. (아까 사람행동 인식 문제를 푼다고 그랬었지?) 

We described (1) how to train recurrent approaches in this setting and (2) introduced a novel regularisation approach. In thousands of (3) experiments we evaluated the performance of the models with randomly sampled hyperparameters. We found that bi-directional LSTMs outperform the current state-of-the-art on Opportunity, a large benchmark dataset, by a considerable margin. (얘 아무리 귀찮아도 앱스트랙이랑 똑같이 썼네...)

(중략)

We found that models differ in the spread of recognition performance for different parameter settings. Regular DNNs, a model that is probably the most approachable for a practitioner, requires a significant investment in parameter exploration and shows a substantial spread between the peak and median performance. Practitioners should therefore not discard the model even if a preliminary exploration leads to poor recognition performance. More sophisticated approaches like CNNs or RNNs show a much smaller spread of performance, and it is more likely to find a configuration that works well with only a few iterations. (원래 파라메터에 따라 성능이 많이 달라지는데, DNN이 파라메터 찾는데 제일 개고생이고 CNN이나 RNN은 그나마 좀 낫다네...)
```

초록과 결론을 통해 논문이 무슨 문제를 풀려했고, 결국 어떠한 기여를 했는지 알았으면 이미 논문의 절반은 읽은거다. 마치 드라마의 인물관계도를 파악하고 나중에 엔딩을 스포일 받은 느낌이랄까?

만약 ‘이 드라마는 이쯤이면 됐어. 그만볼래.’ 싶으면 논문을 그만보면 되는 것이고, ‘우아, 재밌겠다. 도대체 어떻게 한거지?’ 궁금하면 서론부터 더 자세히 읽어나가면 될 것이다.

### 3. 서론 읽기 (Introduction)
사실 서론이야말로 초짜 대학원생들에겐 가장 보물과 같은 파트이다. 왜냐하면 논문의 본론은 단지 자신의 지엽적인 문제 해결만을 다루고 있지만 (게다가 이해하기도 어렵다!) 서론에서는 주옥같은 주요 연구들을 한줄 요약들과 함께 너무나도 친절하게 소개해주고 있기 때문이다. 게다가 소개되는 논문들은 대게 이 문제를 풀어야 할 연구자라면 꼭 읽어야 하는 논문들이 많다!

**그러니 이번 논문은 버리더라도 서론을 통해 다음 논문은 꼭 소개받도록 하자!**

한 논문의 서론에선 적게는 한두개, 많게는 대여섯개까지 읽고 싶은 (혹은 읽어야 할)  논문들을 발견할 수 있다. 그리고 그 다음 것을 읽으면 또 주렁주렁 다음에 읽어야 할 논문들이 생긴다. ~~이것이 대학원생들이 논문만 쌓아놓고 안읽는 이유~~ 첫 논문을 읽기가 어렵지 그 다음의 사슬을 따라가는건 그리 어렵지 않다. 그러니 다시 강조하지만, 연구 초짜라면 서론을 통해 주옥같은 논문들을 소개받도록 하자.

서론은 **(1) 내가 어떤 문제를 풀고 있는지, (2) 관련 연구들은 이 문제를 어떻게 풀어왔는지, (3) 마지막으로 나는 그들과 달리 어떤 입장에서 문제를 해결했는지**를 상대 비교와 함께 설명해준다. 큰 그림을 보여준다고나 할까? 그러니 서론을 읽을 때 산만하게 빠져들지 말고 각 연구들을 왜 서론에서 보여주고 있는지 이해하며 읽도록 하자. 모든 내용은 본론을 잘 이해시키기 위해 존재하는 것들이다.

여기까지 읽었으면 논문의 2/3는 읽은거다! 서론에서 다른 흥미로운 논문을 소개받아 그쪽으로 넘어가고 싶다면 여기서 읽기를 멈춰도 좋다. 하지만 그런식으로 논문 소개만 받다가 소개팅만 백번한 사람으로 끝날 수도 있음에 유념하자. 소개팅의 목적은 다른 사람을 소개받는데에만 그치지 않는다. 언젠간 사귀어야 한다. 

### 4. 쉬어가는 페이지: 표/그림 보기
영어 독해를 쉽게하는 방법 중 하나는 ‘앞에 나올 내용을 예상하며 읽는 것’이다. 이제까지 초록, 결론, 서론을 읽었던 것은 모두 본론에 어떤 내용이 나올지 잘 예측할 수 있기 위해서였다. 여기에 또 한가지 본문 이해에 도움을 주는 소재가 있다면 바로 표와 그림들이다. 영어만 남은 사막같은 논문에 한줄기 오아시스와도 같달까?

**논문을 읽기 귀찮다면 초록,서론,결론으로 논문의 개요를 파악한 뒤 표와 그림을 통해 본문을 예측해보도록 하자.**

![](https://i0.wp.com/gradschoolstory.net/wp-content/uploads/2016/12/Capture-1.png?w=1527&ssl=1)

운 없게도 우리가 선택한 논문은 표가 2개, 그림이 2개 밖에 없다. ~~저자한테 따지고 싶은 심정ㅠ~~ Table 1은 실험에 사용된 다섯가지 딥러닝 모델들(행)과 이들의 파라메터값들(열)을 보여주고 있다. 그냥 실험 세팅 이렇다는 것에 대한 디테일. 과감히 패스….

Table 2는 이 논문의 메인 실험결과이다. 5가지 모델을 가지고 3가지 데이터셋에 대해 실험해봤는데 굵게 표시된 성능들이 최고 성능들이었음을 나타낸다. ~~결국 이 논문도 노가다&쇼 논문이구나ㅠ~~

![](https://i2.wp.com/gradschoolstory.net/wp-content/uploads/2016/12/Capture-2.png?w=1517&ssl=1)

Figure 1은 5가지 모델을 설명하기 위한 그림들이다. 처음보면 어려워보이지만 사실 이 분야 사람들에겐 교과서에 적힌 내용을 옮겨온 것과 다를 바가 없다. 그래서 패스… 

마지막으로 Figure 2은 그냥 단순히 Table 2처럼 성능만 보여주면 뭔가 심심하니까 통계적으로 약간의 허세를 부린 것이다;;; 이 그래프를 통해 얻은 특별한 인사이트가 있나해서 본문에서 찾아봤지만 별게 없었으니 이 역시도 과감히 패스.

### 5. 방법과 실험 (Methods & Experiments)
이제까지의 논문읽기가 “무엇을”, “왜”에 대한 내용이었다면, 방법과 실험은 “어떻게”에 대한 본연구의 자세한 설명이다. 이 부분을 읽는데는 왕도가 없다. 수식이 이해가 안되면 글을 뚫어져라 읽고, 글이 이해가 안되면 수식을 뚫어져라 보도록 하자.

> “수식이 이해안되면 어쩌나요? 그냥 넘어가나요?”
만약 그 수식의 역할만 이해한다면 디테일을 모르고 넘어가도 상관없다. 중요한건 그 수식이 인풋으로 무엇을 받아 아웃풋으로 무엇을 내놓는지 이해하는 것이다. 그리고 왜 이 수식이 필요한지, 없으면 어떤 일이 벌어지는지를 이해하는 것 역시 중요하다.

만약 이정도까지 이해했다면, 디테일한 수식이 외계어로 써있어 못읽겠다 하더라도 이해한셈 치고 넘어가도 좋다. 전체 논문을 읽는데엔 큰 지장이 없기 때문이다. 중요한건 수식이 아니라 ‘내가 뭘 읽고 있는지’와 ‘내가 왜 읽고 있는지’의 능동적 이해 자세이다. 혼미해지는 정신 꽉 부여잡고 이 논문의 핵심스토리에 집중하자.

### 6. 마무리 
우리가 읽었던 논문을 요약하자면 다음과 같다.

```
나는 이런 문제를 풀거야 (abstract)
: 웨어러블센서 데이터를 이용해 사람 행동을 인식하는 문제 (HAR)
사실 이 문제는 이런 동기에서 연구가 시작된건데 (introduction)
: 보통 데이터와는 다른 웨어러블 센서 데이터의 특징들, 그리고 딥러닝 적용에서의 특별 고려사항들
관련해서 이런저런 접근들이 있었지 (related works)
: 딥러닝/HAR/딥러닝 모델들에 대한 소개
난 이런 새로운 방식으로 접근해보려고 하는데 (method)
: 새로운 regularization 방법 제시
정말 이게 잘 먹히는지 실험도 해봤어 (experiment)
: 새로 도입한 regularization 포함, 총 5개 딥러닝 모델의 3가지 데이터에 대한 비교실험
이를 통해 이런 사실도 알아냈지만 한계점도 있지 (discussion)
: DNN은 파라메터에 따라 성능이 많이 바뀌지만 CNN/RNN은 그나마 덜 바뀐다는 것. 반복적인 운동 인식에는 CNN이 성능이 좋고, bi-RNN은 레이어 수에 따라 성능변화가 심하다는 것 등등
마지막으로 귀찮은 너를 위해 요약 (conclusion)
: 사실 아주 획기적인 논문은 아니야. 힝 속았지?
```

이런 식으로 논문 읽기와 논문 요약을 반복해 나가신다면 연구동향 파악과 주제 정하기, 본인의 연구 시작하기에 큰 도움이 될 것이라 믿는다.

* 출처 : [대학원생 때 알았더라면 좋았을 것들 :: 영어 못해도 논문 잘 읽는 법](https://gradschoolstory.net/terry/readingpapers/)



## 논문을 효율적으로 읽는 방법 - 3단계 접근법
### 1단계
첫 번째 단계는 빠르게 훑어서 논문의 조감도를 얻는 것이다. 또한, 다음 단계로 진행할지 말지를 결정한다. 이 단계는 대략 5분에서 10분 안에 아래 절차들을 끝낸다.
```
1. 제목, abstract, introduction을 주의해서 읽는다.

2. 각 섹션의 제목을 확인한다. 나머지는 다 무시한다.

3. (만약에 있다면) 수학적인 부분을 대충 읽어서 이론적 배경이 무엇인지 생각해본다.

4. Conclusion을 읽는다.

5. Reference를 쭉 훑어보고, 이전에 읽어본 게 있나 생각해본다.
```

첫 단계를 거치고 다음 다섯 가지 C에 대답할 수 있을 것이다.

```
1. Category: 이 논문은 어떤 타입인가? 측정에 관한 건가? 기존 시스템 분석에 관한건가? 연구 프로토타입인가?

2. Context: 이 연구와 관련된 다른 연구는 뭘까? 어떤 이론적 배경이 문제 해결에 쓰였나?

3. Correctness: 논문의 가정이 유효한가?

4. Contributions: 이 논문의 주요 공헌은 무엇인가?

5. Clarity: 잘 써졌나?
```

이런 정보를 통해 더 읽을지 말지 결정할 것이다. (그리고 프린트할지도, 나무 보호) 더는 읽지 않는다면 그 이유는 논문이 흥미를 끄는 내용이 아니라서 거나, 논문의 배경을 모르거나, 저자가 유효하지 않은 가정을 했기 때문일 수 있다. 첫 번째 단계는 자신의 연구 분야가 아니더라도 괜찮다. 나중에 관련 있는 걸 알아낼 수 있을 것이다.

각설하고, 자신의 논문을 쓰게 될 때, 논평가들(그리고 독자들)이 대부분 첫 번째 단계만 지난다는 걸 알 것이다. 일관성 있는 섹션, 서브섹션의 제목 선택에 심혈을 기울여라. 만약 비평가들이 한 단계 만에 논문의 요지를 이해하지 못한다면, 그 논문은 거의 거절된다고 볼 수 있다. 만약 독자가 5분 후에 어떤 논문의 하이라이트를 이해하지 못한다면, 그 논문은 다시는 읽히지 않을 것이다. 이런 이유로 괜찮은 그림으로 잘 요약된 '그래피컬한 요약'은 매우 좋은 아이디어다. 그리고 과학적 저널에서 더욱 잘 선택될 것이다.

### 2단계
두 번째 단계에서, 논문에 더욱 집중해서 읽어라. 하지만 증명과 같은 세세한 것들은 무시해라. 읽어가면서 핵심을 써내려가거나, 여백에 비평를 써두면 도움이 된다. Uni Augsburg의 Dominik Grusemann이 "이해 안 되는 용어를 써두거나, 저자에게 질문하고 싶은걸 써놔라"고 했다. 만약 논문을 심사하는 사람에게, 이런 비평들은 나중에 논문 비평을 쓸 때 도움이 될 것이다. 그리고 위원회 회의에서 자신의 의견을 뒷받침하는 메모로 쓸 수 있다.

```
1. 그림, 다이어그램, 그리고 다른 삽화들을 주의 깊게 살펴보아라. 특히나 그래프에 신경을 써서 보아라. 그래프의 축이 적절히 라벨링 되었나? 결과물에 오차를 나타내는 바가 표현되었는지, 그래서 통계적으로 유의한가? 이런 실수들이 대단한 논문을 망쳐버린다.

2. 참고문헌 목록에 나중에 읽을만한 논문을 표시해라 (이건 논문의 배경을 배우기에 좋은 방법이다).
```

두 번째 단계는 경험자의 경우 한 시간 정도가 소요될 것이다. 이 단계 후에 논문의 내용을 파악할 수 있을 것이다. 여러 배경과 증거를 통해 논문의 요지를 요약할 수 있을 것이다. 이 단계는 논문 자체에 얼마나 흥미가 있느냐에 달렸지, 반드시 자신의 연구 분야에 달리 진 않다.

아마 가끔 두 번째 단계가 지나서도 논문이 이해되지 않을 수 있다. 아마도 주제가 생소하거나, 익숙지 않은 용어나 약어들 때문일 것이다. 또는 저자의 증명이나 실험 기술이 이해가 어려운 방법이라 그럴 수 있다. 근거 없는 주장으로 쓰이고 수많은 기존 연구를 바탕으로 쓰여서 그럴 수 있다. 또는 그냥 한밤중에 읽느라 매우 피곤해서 그럴 수 있다. 다음 중에 고를 수 있다.

* (a) 논문을 치워두고, 앞으로 읽거나 이해할 필요가 없길 바란다.
* (b) 나중에 다시 읽는다. 아마 배경 지식을 습득한 상태일 것이다.
* (c) 인내심을 발휘해서 세 번째 단계로 간다.

### 3단계
논문을 완벽히 이해하기 위해, 특히 당신이 논문 비평가라면, 세 번째 단계가 필수다. 세 번째 단계에서 가장 중요한 것은 논문을 가상으로 재 실험해보는 것이다. 즉, 저자와 같은 가정을 하고, 그대로 다시 작업 해보아라. 이런 재작업과 실제 논문의 비교를 통해서, 단지 논문 자체만의 아이디어를 확인하는 것뿐 아니고, 숨겨진 결함과 가정을 찾아낼 수 있을 것이다.

이 단계는 세세한 것들을 위해 고도의 집중력이 필요하다. 모든 문장의 가정 하나하나를 확인해보고, 검토해야 한다. 더욱이, 나 자신의 특별한 아이디어를 생각해봐야 한다. 실제 논문과 가상 실험의 비교가 증명과 논문의 서술 기교에 관한 견고한 통찰력을 갖게 해줄 것이다. 그리고 이것들이 나중에 자신의 논문을 위한 레퍼토리가 될 것이다. 이 단계를 거치며 추후 연구를 위한 아이디어를 써놓을 수 있다.

이 단계는 초심자에게는 몇 시간 이상이 걸리고, 경험자는 한 시간, 또는 두 시간이 넘게 걸릴 것이다. 이 단계가 지나서, 기억을 통해 논문의 구조를 다시 작성해볼 수 있다. 거기에 더해 강점, 약점도 분류해낼 수 있을 것이다. 특히, 실험 또는 분석 기술에서 잠재적 가정이나, 빠진 관련 연구, 그리고 잠재적 이슈들을 찾아낼 수 있을 것이다.

### 문헌 조사 방법
논문 읽는 기술들은 문헌 조사를 위해 쓰일 수 있다. 이는 수십 개 이상의 논문을 읽게 될 수도 있고, 아마 익숙지 않은 분야도 포함될 수 있다. 어떤 논문을 읽어야 할까? 여기 세 단계의 도움이 될만한 접근법이 있다.

첫 번째, Google Scholar나 CiteSeer 같은 학문용 검색 엔진을 쓰고, 알맞은 키워드를 써서 최근에 많이 인용된 3-5개의 해당 영역의 논문을 찾자. 각 연구에서 논문 읽기 1단계를 적용해서, 해당 연구의 맥락을 잡는다. 그러고 나서 관련된 연구 부문을 읽어본다. 그러면 최근 연구들의 간략한 스케치를 할 수 있을 것이고, 아마도 운이 좋으면 최근 연구의 중심점을 찾을 수 있을 것이다. 만약 이런 연구를 찾으면, 된 것이다. 연구를 읽고, 운이 좋음을 축하하자.

그렇지않으면, 두 번째 단계로 간다. 공통으로 인용되는 논문을 찾고, 반복되는 저자의 이름을 찾는다. 그게 바로 해당 영역의 키 논문이 되고, 키 연구자들이 된다. 이 논문들을 내려받고, 한 곳에 두자. 그리고 키 연구자들의 웹 사이트에 가서 그들이 최근에 어디에 발표했는지 살펴본다. 최고의 연구자들은 대게 탑 콘퍼런스에 발표하기 때문에 이를 통해 어디가 탑 콘퍼런스인지 알 수 있을 것이다.

세 번째 단계는 탑 콘퍼런스 웹사이트에 가서 최근 기록들을 살펴본다. 빠르게 훑어서 최근 양질의 관련 연구를 알아낼 수 있을 것이다. 앞서 한곳에 모아둔 논문을 포함해서 이 논문들이, 문헌 조사의 첫 버전이 되는 것이다. 두 단계를 거쳐 이렇게 논문들을 모아라. 만약에 이 논문들이 아직 찾지 못한 어떤 키 논문을 언급하면, 그것도 구해두고, 읽어라. 이를 필요한 만큼 반복한다.

* 출처 : [공돌이의 노트정리 :: 논문을 효율적으로 읽는 방법](https://woongheelee.com/entry/%EB%85%BC%EB%AC%B8%EC%9D%84-%ED%9A%A8%EC%9C%A8%EC%A0%81%EC%9C%BC%EB%A1%9C-%EC%9D%BD%EB%8A%94-%EB%B0%A9%EB%B2%95)
