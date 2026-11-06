
```sh
nvidia-smi
```

![[cuda02.png]]
``
```bash
# 가상환경 만들기
python -m venv cuda_env

# 가상환경 활성화
.\cuda_env\Scripts\activate

# **최신 그래픽카드(RTX 4000/5000번대)**
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu132

# 구형 그래픽카드
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 업그레이드
python.exe -m pip install --upgrade pip

# 라이브러리 설치
python -m pip install opencv-python Pillow numpy  tqdm matplotlib

# 가상환경 비활성화
deactivate
```

- cuda python interfreter 선택
	- Antigravity에서 프로젝트를 엽니다.
	- `Ctrl + Shift + P`를 누릅니다.
	- `Python: Select Interpreter` 검색합니다.
	- 다음 경로의 Python을 선택합니다.

##### cuda 사용여부 확인
```python
import torch

print("PyTorch 버전:", torch.__version__)
print("CUDA 사용 가능:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU 이름:", torch.cuda.get_device_name(0))
    print("CUDA 버전:", torch.version.cuda)
    print("GPU 개수:", torch.cuda.device_count())

    x = torch.rand(3000, 3000).cuda()
    y = torch.matmul(x, x)

    print("연산 장치:", y.device)
    print("GPU 연산 성공!")
```