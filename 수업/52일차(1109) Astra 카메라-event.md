


## 수업목표
- Astra(Orbbec Depth sensor)  연결테스트

## 실습파일
- as1109.py  : Astra 카메라 연결
- as1109_2.py ; 

### 참고사이트
https://www.orbbec.com/resources/
[OrbbecSDK python 기반 manual](https://orbbec.github.io/pyorbbecsdk/)
[github openni python](https://github.com/severin-lemaignan/openni-python/tree/master)
[wiki 드라이브 설치](https://astra-wiki.readthedocs.io/en/latest/downloadDriver.html)
[github docu](https://github.com/orbbec/OrbbecSDK_v2)


### [Orbbec 3D 카메라](https://www.orbbec.com/kr/home/?utm_source=chatgpt.com)
- 색상 정보뿐 아니라 픽셀별 거리(Depth) 정보도 획득 가능
- **Orbbec(오르벡, 오브벡)**은 중국의 3D 비전 및 AI 카메라 전문 기업
- Astra : 구조광 기반 3D 깊이 카메라

#### 자율주행 로봇에서의 활용

현재 공부하고 있는 Jetson Nano 및 AMR/AGV 분야와도 관련이 깊습니다.

|활용 분야|역할|
|---|---|
|거리 측정|카메라와 장애물 사이의 거리 계산|
|장애물 감지|사람, 물체, 벽 등의 3D 위치 파악|
|SLAM|주변 환경의 3D 정보 수집|
|로봇 주행|장애물 회피 및 경로 계획|
|객체 인식|RGB 영상과 Depth 정보 결합|

핵심 차이점

- 일반 RGB 카메라: 물체의 색상과 형태를 촬영
- Orbbec 3D 카메라: 색상 정보뿐 아니라 픽셀별 거리(Depth) 정보도 획득 가능

#### ASTRA  연결

1. USB 로 카메라 연결하고 장치관리자 확인
	- 카메라를 연결한 뒤 [윈도우 장치 관리자(Device Manager)]를 열어 기타 장치에 새로운 하드웨어가 인식되었는지 확인

![[1109_04.png|543]]

2. 이전 버전 호황을 위해서 드라이버 설치
	vcredist_x64.exe 설치
![[astra01.png|297]]

3. 센서 드라이버 설치
	- OpenNI SDK와 Orbbec 드라이버도 별도로 설치
	- SensorDriver_V4.3.0.22.exe 설치
	- 드라이버 다운로드 : https://astra-wiki.readthedocs.io/en/latest/downloadDriver.html

4. OpenNI2 및 Orbbec OpenNI SDK를 설치
	- 제공되는 테스트 도구(**`NiViewer.exe`**)를 통해 깊이(Depth) 및 컬러 영상 출력이 정상 작동하는지 확인
	- C:\jackson\OpenNI_2.3.0.86\Win64-Release\tools\NiViewer\NiViewer.exe
		- 카메라 실행

#### sample

- C:\jackson\OpenNI_2.3.0.86\samples
	- C언어로 되어 있고 초창기 버전
	- 이전버전은 파이썬 샘플은 없음



![[1109_03.png]]

주요 구성 요소는 다음과 같습니다.

| 구성 요소            | 역할                    |               |       |
| ---------------- | --------------------- | ------------- | ----- |
| **RGB Camera**   | 일반 **컬러** 영상 촬영       |               | 칼라    |
| **IR Projector** | **적외선 패턴**을 물체에 투사    | Infrared(적외선) | IR 패턴 |
| **IR Camera**    | 물체에 반사된 적외선 **패턴 촬영** |               | IR 수신 |
| Depth Processor  | 거리(Depth) 계산          |               |       |
| USB Interface    | PC 또는 Jetson으로 데이터 전송 |               |       |
| Microphone       | 마이크 2개 내장. 음성 명령 수집   |               |       |
![[Pasted image 20261109095142.png|607]]


#### python 3.11 가상환경 실행

```shell
# python 3.11 설치
py install 3.11

# 설치 확인
py -0p
  *               C:\work_vision\astra_env\Scripts\python.exe
 -V:3.14[-64]     C:\Users\User\AppData\Local\Python\pythoncore-3.14-64\python.exe
 -V:3.11[-64]     C:\Users\User\AppData\Local\Python\pythoncore-3.11-64\python.exe

cd C:\work_vision\1109_ASTRA

# Python 3.11 가상환경 생성
py -3.11 -m venv astra_env

# 가상환경 활성화
.\astra_env\Scripts\activate

# pip 업데이트
python -m pip install --upgrade pip

# 필요한 패키지 설치
pip install numpy opencv-python openni
```


#### 참고
dll : dynamic linking library
lib : 윈도우 정적 라이브러리
so : shared Object
![[1109_01.png|530]]

## Linux와 Windows의 라이브러리 확장자 정리

| 구분       | Windows                       | Linux                                                       |                 |
| -------- | ----------------------------- | ----------------------------------------------------------- | --------------- |
| 실행 파일    | `.exe`                        | 실행 권한이 있는 바이너리                                              |                 |
| 동적 라이브러리 | `.dll` (Dynamic Link Library) | `.so` (shared object)<br>`/usr/lib`, `/usr/local/lib` 등에 저장 | 프로그램 실행 단계에서 사용 |
| 정적 라이브러리 | `.lib` (Library)              | `.a`                                                        | 개발 및 링크 단계에서 사용 |
| 헤더 파일    | `.h`, `.hpp`                  | `.h`, `.hpp`                                                |                 |


### as1109.py

전체 소스
```python
#-----------------------------
# filename: as1109.py  
#   - (astra_env) 가상환경에서 실행. 
#   - python version: 3.11

# OpenNI2 를 사용하여 ASTRA 카메라를 연결하고 해제하는 프로그램

# OpenNI : Open Natural Interaction(개방형 자연스러운 상호작용)
#          OpenNI는 7메라, 깊이 센서 등의 입력을 이용해 사람과 기계가 자연스럽게 상호작용할 수 있도록 지원하는 미들웨어 및 API
#-----------------------------

import os
from openni import openni2  # pip install openni

OPENNI_PATH = 'C:\\work_jackson\\OpenNI_2.3.0.86\\samples\\bin'

if os.path.exists(OPENNI_PATH):
    print('OpenNI2 모듈 초기화 성공!')
    openni2.initialize(OPENNI_PATH)
else:
    print('OpenNI2.dll 파일을 찾을 수 없습니다.')

try:
    device = openni2.Device.open_any()
    print(f"카메라 연결 Ok:{device.get_device_info().name.decode('utf-8')}")
except Exception as e:
    print('에러 발생', e)

if 'device' in locals() and device is not None:
    print('ASTRA  카메라 해제')
    device.close()

openni2.unload()
print('OpenNI2 모듈 해제')
```


UART  시리얼 통신

#### as1109_2.py

```python
#-----------------------------
# filename : as1109_2.py   
#  
# - 메인쓰레드와 프레임 이벤트 처리 스레드가 있음
#

# 장치제어
# 1 이벤트 핸들러
#   color_stream.register_new_frame_listener(on_new_frame)      # 
#

# 2.폴링 빙식
#    whule True:         # 장치로부터 계속 수신
#        time.sleep(0.1) # 처리 주기 설정(초)     
#  
#------------------------------

import ctypes
import os
import time
from openni import openni2
from openni import _openni2 as c_api
import threading

OPENNI_PATH = 'C:\\work_jackson\\OpenNI_2.3.0.86\\samples\\bin'

PRINT_INTERVAL = 0.5   # 출력 주기(초)
last_print_time = 0.0  # 출력 시각

# 수신된 비디오 프레임의 중앙 픽셀 RGB값 출력
def analyze_frame(frame):
    video_mode = frame.videoMode
    #RGB888(True color) =- 24bit : R(8bit), G(8bit), B(8bit)
    #RG8565(High color) =- 16bit : R(5bit), G(6bit), B(5bit)
    if video_mode.pixelFormat == c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888:
        buffer = frame.get_buffer_as_triplet()
        # 프레임 정중앙에 위치한 픽셀 index
        middle_index = (frame.height +1) * frame.width // 2
        # 중앙 픽셀의 RGB 값 추출
        raw_r, raw_g, raw_b = buffer[middle_index]
        print(f'R:{raw_r} G:{raw_g}  b:{raw_b}')
    else:
        print('알수없는 Format...')

# 프레임 수신 콜백함수
def on_new_frame(stream):
    global last_print_time
    current_time = time.time()
    print(
        "콜백 스레드:",
        threading.current_thread().name,
        threading.get_ident()
    )
    # 설정한 출력 간격만큼 분석 처리 및 출력
    if current_time - last_print_time >= PRINT_INTERVAL:
        last_print_time = current_time
        frame = stream.read_frame()
        analyze_frame(frame)
    else:
        # 데이터가 누락되는 현상 방지
        _ = stream.read_frame()

def main():
    print('OpenNI2 초기화')
    if os.path.exists(OPENNI_PATH):
        os.chdir(OPENNI_PATH)
        openni2.initialize(OPENNI_PATH)
    else:
        openni2.initialize()
    
    try:
        print('Astra 3D 카메라 연결')
        device = openni2.Device.open_any()

        print('Astra 카메라 연결 성공')
        if not device.has_sensor(c_api.OniSensorType.ONI_SENSOR_COLOR):
            print('오류: 연결된 장치에서 컬러 센서를 찾을 수 없습니다.')
            return

        # color 스트림 객체 생성
        color_stream = device.create_color_stream()

        # color video 모드 설정
        color_stream.set_video_mode(c_api.OniVideoMode(
            pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
            resolutionX=640,
            resolutionY=480,
            fps=30
        ))

        # color 스트림 시작
        color_stream.start()

        # on_new_frame 이벤트 핸들러 리스너에 등록
        color_stream.register_new_frame_listener(on_new_frame)   # 메인 스레드가 아닌 별도의 프레임 이벤트 처리 스레드에서 실행

        # main 루프
        while True:                                              # 프로그램이 계속 실행되도록 메인 쓰레드는 무한 루프
            time.sleep(0.1)

    except KeyboardInterrupt:
        print('사용자에 의해 프로그램 종료')
    except Exception as e:
        print('실행 오류:', e)
    finally:
        print('모든 자원은 해제하고 프로그램 종류')
        # 리스너 해제 및 장치 dtop/close
        if 'color_stream' in locals() and color_stream is not None:
            color_stream.unregister_new_frame_listener(on_new_frame)
            color_stream.stop()
            color_stream.close()
        # 카메라 디바이스닫기
        if 'device' in locals() and device is not None:
            device.close()

        #OpenNI2 자원 해제
        openni2.unload()

if __name__ == '__main__':
    main()
```

##### 패키지 
```python
from openni import openni2
from openni import _openni2 as c_api
```



![[Pasted image 20261111153619.png]]
`pyorbbecsdk`와 `openni2`는 카메라를 제어하는 서로 다른 Python 접근 방식입니다.

## 세 가지 비교

| 구분    | pyorbbecsdk    | openni2        | _openni2          |
| ----- | -------------- | -------------- | ----------------- |
| 성격    | Orbbec 전용 SDK  | OpenNI2 래퍼     | 저수준 모듈            |
| 개발 기반 | C++ / pybind11 | Python         | 네이티브 API 바인딩      |
| 역할    | 카메라 제어         | 카메라 제어         | 저수준 기능 제공         |
| 주요 대상 | Orbbec 카메라     | OpenNI2 호환 카메라 | OpenNI2 API       |
| 사용 방식 | 직접 사용          | 직접 사용          | 주로 openni2와 함께 사용 |
특히 pyorbbecsdk v2는 기존 OpenNI 프로토콜 장치를 지원하지 않습니다. 따라서 카메라 모델과 펌웨어에 따라 SDK를 선택해야 합니다

- `pyorbbecsdk` : Orbbec 전용 카메라 제어 도구
- `openni2` : OpenNI2 표준 인터페이스를 Python에서 사용하기 위한 래퍼
- `_openni2` : openni2가 사용하는 저수준 API 모듈