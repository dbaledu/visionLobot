


## 수업목표
- OpenCV를 이용한 이미지 데이터 분석
- 

## 실습파일
- ㅇㅇ
- 

## 개발환경 구축

개발툴 설치

- [https://antigravity.google/download#antigravity-ide](https://antigravity.google/download#antigravity-ide)
- **Visual Studio Code**
- **Antigravity IDE (Standalone)**
    - colab 패키지 설치

파이썬 설치

- [test.py](http://test.py) 파일 생성 → 실행
- 
![[0922_1.png]]

![[0922_2.png]]


- python 패키지 설치

단축키

ctrl + L agent 창 열기

환경설정 → setting → terminal font size 변경

OpenCV 라이브러리 설치

pip install opencv-python

```python
현재 path:    C:\Users\User\AppData\Local\Python\bin
패키지 설치 위치   C:\Users\User\AppData\Local\pip

uv  C:\Users\User\AppData\Local\uv

C:\Users\User>where python
C:\Users\User\AppData\Local\Microsoft\WindowsApps\python.exe
C:\Users\User\AppData\Local\Python\bin\python.exe

# 제거하기
C:\Users\User>py uninstall 3.14
Uninstall Python 3.14.7? [Y/n] y
Removed Python 3.14.7

C:\Users\User>uv python uninstall 3.14.7
Searching for Python versions matching: Python 3.14.7
Uninstalled Python 3.14.7 in 243ms
 - cpython-3.14.7-windows-x86_64-none (python3.14.exe)

```

#### 파이썬 인터프리터

1. Python Install Manager
    
    - `python-manager-26.3.msix` 설치파일 이용하여
    - 명령어 py
2. uv
    
    - Rust로 만든 최신 도구
    - 프로젝트별 가상환경/패키지까지 한번에 관리
    - 별도 pyenv, venv, pip 없이 다 됨
    - uv 설치 : `%LOCALAPPDATA%\Programs\uv` 경로에 설치
    
    ```python
    powershell -ExecutionPolicy ByPass -c "irm <https://astral.sh/uv/install.ps1> | iex"
    ```
    
    - 인터프리터 설치
    
    ```python
    uv python install 3.14
    ```
    
    - 패키지 설치
    
    ```python
    uv init              <-  `pyproject.toml`과 가상환경(.venv)이 자동 생성
    uv add 패키지명       <- 패키지 설치
    uv run python script.py  <- uv가 자동으로 .venv를 활성화해서 방금 설치한 패키지를 바로 쓸 수 있게 해줍니다. 별도로 venv를 activate 할 필요가 없습니다.
                                or  또는 `uv run python`으로 실행
    ```
    
3. Anaconda / Miniconda
    
    - 데이터사이언스/ML 쪽에서 전통적으로 많이 씀. CUDA, GDAL 같은 비-Python 바이너리 의존성까지 관리해줌.
4. winget / chocolatey
    
    - `winget install Python.Python.3.14` ← 인터프리터 설치

#### 설치 후 패키지 관리

1. `pip` : 파이썬에 기본 내장
2. `uv` : pip보다 roughly 8x faster
3. `conda` : ML/과학computing 쪽에서 여전히 강세, 특히 CUDA 등 비-PyPI 의존성
4. `poetry` : 라이브러리 배포/패키징에 특화

**이미지 색상 표현 방식(BGR, HSV, YUV)**

1. RGB
    1. 값이 커질수록 해당 색상의 빛이 밝아지는 원리
    2. RGB = (255, 255, 255) 일 때는 흰색
    3. RGB = (0, 0, 0) 일 때는 검은색
    4. **OpenCV는 그 반대의 순서인 BGR로 표현**
2. **RGBA**
    1. **RGB에 A(알파, alpha)가 추가된 색상 표기법**
    2. A 는 배경의 투명도. A값이 255면 흰색, 0이면 검은색

