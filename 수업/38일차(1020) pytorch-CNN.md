

## 수업목표
-  **합성곱 신경망(CNN)을 활용하여 손글씨 숫자(MNIST) 데이터를 분류하는 모델을 학습시키고 성능을 평가**하는 것입니다. 더불어, 딥러닝의 핵심 메커니즘인 **순전파, 손실 계산, 역전파(기울기 계산), 가중치 업데이트** 과정을 실제 간단한 수학식($L=(wx-10)^2$$L=(wx-10)^2$)과 코드를 통해 시각적·직관적으로 검증하고 이해


## 실습파일
- cv1020_1_CNN_pytorch_MNIST.ipynb : ( 교재 2장 67p )


### cv1020_1_CNN_pytorch_MNIST.ipynb  

##### 전체소스 

```python
#---------------------------------
# 환경 설정 및 라이브러리 임포트
#---------------------------------
import torch                                 # pytorch 핵심 패키지
import torch.nn as nn                        # 신경망 계층 모듈
import torch.nn.functional as F              # 함수형 연산
import torch.optim as optim                  # 최적화
from torchvision import datasets, transforms # 데이터셋 및 전처리
from torch.autograd import Variable          # 텐서를 자동 미분 그래프에 연결. 역전파에 필요한 미분 계산을 담당. 가중치를 실제로 업데이트하는 작업은 보통 optimizer가 담당
import matplotlib.pyplot as plt              # 시각화
from torch.cuda import is_available

# GPU 사용 체크
is_cuda = False
if torch.cuda.is_available():
  is_cuda = True
print(is_cuda)


#---------------------------------
#  
#---------------------------------
# MNIST 학습 데이터 전체 픽셀의 평균(mean), 표준편차(std)
raw_dataset = datasets.MNIST('data/', train=True, download=True, transform=transforms.ToTensor())
all_images = torch.stack([img for img, _ in raw_dataset], dim=0)
mean = all_images.mean().item()
std = all_images.std().item()
print(f'전체 픽셀 평균:{mean}')
print(f'전체 픽셀 표준편차:{std}')
#ToTensor(): PIL(Python Image Library) 이미지 -> FloatTensor 변환
# 0.1307, 0.3081: MNIST 학습 데이터 전체 픽셀의 평균(mean), 표준편차(std)
# 입력 데이터 분포를 정규화를 적용하여 학습을 안정적이고 빠르게 처리
transformation = transforms.Compose([transforms.ToTensor(),                        # tensor 타입으로 변환
                                     transforms.Normalize((0.1307,),(0.3081,))     # 정규화
                                     ])
  
# 학습용 60,000장 테스트용 10,000장
train_dataset = datasets.MNIST('data/', train=True, download=True, transform=transformation)
test_dataset = datasets.MNIST('data/', train=False, download=True, transform=transformation)
  
# shuffle=True: 1epoch 마다 순서를 섞어 편향과 과적합 방지
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 합성곱 신경망(CNN) 클래스 정의
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    # 첫번째 활성곱층(입력층)
    # 입력 1채널 -> 출력 10채널(피처맵), 커널 사이즈: 5
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)   #   (28 X 28 X 1) 에서 채널크기 1이 입력값
    # 두번째 활성곱층
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    # 과적합 방지
    self.conv2_drop = nn.Dropout2d()
    # 1번째 완전연결층
    self.fc1 = nn.Linear(320, 50)
    # 2번째 완전연결층(출력층)
    self.fc2 = nn.Linear(50, 10)
  
  # 상속받은 메서드를 오버랑딩
  def forward(self, x):
    # 1번째 활성곱층
    x = F.relu(F.max_pool2d(self.conv1(x), 2))     # maxpooling (2 X 2) =>  반으로 축소
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = x.view(-1, 320)
    # 첫번째 완전연결층
    x = F.relu(self.fc1(x))
    x = F.dropout(x, p=0.1, training=self.training)
    # 2번째 완전연결층(출력층)
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)
  
'''
nn.Conv2d(
    in_channels=1,
    out_channels=10,
    kernel_size=5,
    stride=1,
    padding=1
)
'''
  
# forward 레이어 구조
#    conv1 -> maxpooling -> relu
# -> conv2 -> dropout -> maxpooling -> flattern     특징 추출(학습). 선, 모서리..
# -> full connect -> relu -> dropout                특징 조합(분류)
# -> full connect-> log_softmax                     출력
# 훈련 및 검증
# epoch: 반복횟수  model: 학습/평가 모델  data_loader: 데이터 가져오기
# phase: training(가중치 업데이트), validation(평가)
# volatile: 미분 기울기 계산 on/off
# 리턴값: 샘플당 평균 손실값, 정확도
def fit(epoch, model, data_loader, phase='training', volatile=False):
  if phase == 'training':
    model.train()
  
  if phase == 'validation':
    model.eval()
    volatile=True
  
  running_loss = 0.0   # 전체 손실합계
  running_correct = 0  # 전체 정답 개수
  
  # batch_idx: 배치번호   data: 이미지[32,1,28,28]   target:정답[32]    <- dataLoader에서 배치사이즈:32 로 지정했음
  for batch_idx, (data, target) in enumerate(data_loader):
  
    if is_cuda:
      data, target = data.cuda(), target.cuda()
    # 과거 PyTorch에서는 텐서와 자동 미분 기능을 연결하기 위해 Variable을 사용
    # 현재 PyTorch에서는 Tensor 자체가 자동 미분 기능을 지원
    data, target = Variable(data, volatile), Variable(target)  # 입력 데이터를 과거 방식의 Variable로 변환
  
    # 학습단계이면 이번 배치에서 누적된 기울기를 0으로 초기화
    if phase == 'training':
      optimizer.zero_grad()
  
    # 모델을 호출하여 순전파 실행(forward() 호출)
    output = model(data)
    # 배치 단위 손실 계산
    loss = F.nll_loss(output, target)

    # 배치 내 샘플 손실이 합을 구하여 누적. 에폭 전체 손실합 계산
    running_loss += F.nll_loss(output, target, reduction='sum').item()
    # 정답 개수 누적
    # target.data.view_as(preds) : 정답[32]를 preds와 같은 [32,1] 형태로 변환
    # cpu().sum() : cpu로 옮긴뒤 합산
    # 주의: numpy array는 GPU 메모리에서 연산이 불가능
    preds = output.data.max(dim=1, keepdim=True)[1]
    running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
    if phase == 'training':
      loss.backward()
      optimizer.step()
  
  # 손실 평균
  loss = running_loss/len(data_loader.dataset)
  accuracy = 100. * running_correct/len(data_loader.dataset)
  
  print(f'{phase} {epoch} epoch,  loss:{loss}, accuracy:{accuracy}')
  return loss, accuracy
 
# Net 클래스 객체(인스턴스) 생성
model = Net()  #  __init__() 생성자 호출
  
# cpu 메모리와 gpu 메모리 영역이 다름.
#GPU 사용이 가능하면 모델 정보를 GPU 메모리 이동
if is_cuda:
  model.cuda()    
  
# 확률적 경사하강법 옵티마이저
optimizer = optim.SGD(model.parameters(), lr=0.01)
  
# 훈련데이터 확인용
# iter(): DataLoader를 반복자(iterator)로 변환
# next() 첫번째 배치 1개만 꺼냄
data, target = next(iter(train_loader))
print('size=',data.size())
print('shape=',data.shape) 

# 학습훈련
train_losses, train_accuracies = [], []   # 학습 데이터 손실, 정확도
val_losses, val_accuracies = [], []       # 검증 데이터 손실, 정확도
for i in range(20):
  epoch_loss, epoch_accuracy = fit(i, model, train_loader, phase='training')
  val_epoch_loss, val_epoch_accuracy = fit(i, model, test_loader, phase='validation')
  train_losses.append(epoch_loss)
  train_accuracies.append(epoch_accuracy)
  val_losses.append(val_epoch_loss)
  val_accuracies.append(val_epoch_accuracy)
  
# 훈련 데이터와 검증 데이터의 정확도 그래프
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(range(1, len(train_losses)+1), train_losses, 'bo', label='train_losses')
plt.plot(range(1, len(val_losses)+1), val_losses, 'r', label='val_losses')
plt.legend()
plt.subplot(122)
plt.plot(range(1, len(train_accuracies)+1), train_accuracies, 'bo', label='train_accuracies')
plt.plot(range(1, len(val_accuracies)+1), val_accuracies, 'r', label='val_accuracies')
plt.legend()
plt.show() 
```


##### 파이프라인

#### **[파트 A] MNIST 손글씨 이미지 분류 (CNN)**

1. **환경 설정 및 라이브러리 임포트**
    
    - PyTorch 핵심 모듈(`torch.nn`, `optim`), 데이터셋 처리 도구(`torchvision`), 시각화 도구(`matplotlib`)를 불러옵니다.
    - GPU 사용 가능 여부(`is_cuda`)를 체크하여 연산 환경을 구성합니다.
2. **데이터 분석 및 전처리**
    
    - MNIST 데이터셋 전체의 픽셀 평균(약 0.1307)과 표준편차(약 0.3081)를 구합니다.
    - 데이터 분포를 정규화(`transforms.Normalize`)하여 학습 성능과 속도를 높이고, 데이터를 미니배치(size=32) 형태로 공급하는 `DataLoader`를 정의합니다.
3. **CNN 모델 정의**
    
    - 2개의 합성곱 레이어(`Conv2d`)와 맥스풀링(`max_pool2d`), 드롭아웃(`Dropout2d`), 활성화 함수(`ReLU`)를 사용해 이미지의 특징을 추출하는 신경망을 구성합니다.
    - 마지막 단계에서 완전연결층(`Linear`)을 통해 최종 10개 클래스(숫자 0~9)로 분류하고 `log_softmax`를 출력합니다.
4. **훈련 및 검증 루프 구현**
    
    - 학습 모드(`model.train()`)와 평가 모 gateways(`model.eval()`)를 구분하여 동작하는 `fit()` 함수를 정의합니다.
    - 손실 함수로 Negative Log Likelihood Loss(`nll_loss`)를, 옵티마이저로 SGD(확률적 경사하강법)를 정의합니다.
5. **학습 진행 및 결과 시각화**
    
    - 총 20 Epoch 동안 모델을 학습시키고 매 에폭마다 검증 데이터로 성능을 평가합니다.
    - 학습 완료 후 에폭별 손실(Loss)과 정확도(Accuracy)의 변화 추이를 그래프로 그려 시각적으로 확인합니다.

---

#### **[파트 B] 가중치 및 오차역전파(Backpropagation) 원리 검증 

- **수학적 검증**: 예측값 $Prediction = w imes x$$Prediction = w imes x$과 목표값 $Target = 10$$Target = 10$ 사이의 오차를 정의한 뒤, 손실 $L = (wx-10)^2$$L = (wx-10)^2$을 가중치 $w$$w$에 대해 미분(역전파)했을 때의 동작 원리를 구현합니다.
- **기울기(Gradient)의 역할**: 가중치 $w$$w$의 변화에 따라 손실의 미분값(기울기, `w.grad`)이 어떻게 계산되는지 직접 확인하고, 경사 하강법이 예측값을 정답에 가깝게 만들기 위해 가중치를 어떻게 조절(기증/감소)해야 하는지 직관적으로 테스트하고 학습합니다.