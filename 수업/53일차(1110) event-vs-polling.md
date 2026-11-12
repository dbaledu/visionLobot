


## 수업목표
- ㅇㅇ
- 

## 실습파일
- ㅇㅇ
- 


##### filename: as1110_polling.py

전체소스
```python
# filename: as1110_polling.py
# 
# 폴링 방식으로 처리
# as1109_2.py는 이벤트핸들러 방식이므로 비교해볼 것

import os
import time
from openni import openni2
from openni import _openni2 as c_api

OPENNI_PATH = 'C:\\work_jackson\\OpenNI_2.3.0.86\\samples\\bin'    # 라이브러리(dll)와 드라이버(dll, lib) 파일 위치
SAMPLE_READ_WAIT_TIMEOUT = 2.0
LOOP_DELAY = 0.5

def main():
    #-----------------------------------
    # 1. 라이브러리 초기화
    #-----------------------------------  
    #     - OpenNI2 네이티브 라이브러리(C로 만든 .dll/.so)를 로드하고 초기화
    #    - OpenNI 런타임이 들어 있는 경로. 파이썬이 코어 라이브러리를 어디서 찾을지 알려줌
    #    - 코어가 Drivers 폴더를 찾아서 센서를 인식
    if os.path.exists(OPENNI_PATH):
        os.chdir(OPENNI_PATH)
        openni2.initialize(OPENNI_PATH)   # 센서 인식
    else:
        openni2.initialize()   # 인자 없이 호출하면 시스템 환경변수(OPENNI2_REDIST, 윈도우는 PATH 등)에서 찾으려 시도

    color_stream = None
    device = None

    try:
        #-----------------------------------
        # 2. 디바이스 연결
        #-----------------------------------  
        device = openni2.Device.open_any()
        print(device.get_device_info())

        #-----------------------------------
        # 3. 센서 확인 & 스트림 센서
        #-----------------------------------  
        # color sensor 체크
        if not device.has_sensor(c_api.OniSensorType.ONI_SENSOR_COLOR):    
            print('오류: 연결된 장치에서 컬러 센서를 찾을수 없습니다')
            return
        
        # color 스트림 객체 생성
        color_stream = device.create_color_stream()
        
        #-----------------------------------
        # 4. 모드 설정 & 센서 시작
        #----------------------------------- 
        # color video 모드 설정
        color_stream.set_video_mode(c_api.OniVideoMode(
        pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
        resolutionX=640, resolutionY=480, fps=30
        ))
        
        # color 스트림 시작
        color_stream.start()

        while True:
            #-----------------------------------
            # 5. 프레임 획득 방식. 폴링 방식: wait_for_any_stream
            #----------------------------------- 
            ready_stream = openni2.wait_for_any_stream([color_stream], timeout=SAMPLE_READ_WAIT_TIMEOUT)

            if ready_stream is None:
                print(f'대기시간 초과 {SAMPLE_READ_WAIT_TIMEOUT} 초 동안 새 프레임이 도착하지 않음')
                continue

            frame = color_stream.read_frame()
            if frame.videoMode.pixelFormat != c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888:
                print('RGB888 포맷 에러!')
                continue

            #-----------------------------------
            # 6. 비전 처리 파이프라인
            #----------------------------------- 
            # 버퍼 읽기 -> Numpy/OpenCV 변환 -> 화면 출력/영상 출력
            middle_index = (frame.height+1) * frame.width // 2
            buffer = frame.get_buffer_as_triplet()
            r, g, b = buffer[middle_index]
            print(f'RED:{r}  GREEN:{g}  BLUE:{b}')
            if LOOP_DELAY > 0:
                time.sleep(LOOP_DELAY)   # 500ms 지연.  센서마다 지연시간 확인해볼것

    except KeyboardInterrupt:
        print('사용자에 의해 종료합니다.')
    except Exception as e:
        print('실행중 오류발생:', e)
    finally:
        print('리소스 해제 및 프로그램 종료')
        #-----------------------------------
        # 7. 리소스 해제 및 장치 stop/close
        #----------------------------------- 
        if 'color_stream' in locals() and color_stream is not None:
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


![[Pasted image 20261110121204.png|621]]
![[Pasted image 20261110121223.png]]


### **2. 단계별 파이프라인 상세 설명**

|단계|파이프라인 작업|코드 예시 (`OpenNI2`)|역할 및 주의점|
|---|---|---|---|
|**① 초기화**|드라이버 및 C 코어 라이브러리 로드|`openni2.initialize(OPENNI_PATH)`|센서 드라이버(`OpenNI2/Drivers`) 인식|
|**② 디바이스 오픈**|물리적 카메라와 통신 채널 열기|`device = openni2.Device.open_any()`|USB 포트에 연결된 Astra 카메라 오픈|
|**③ 스트림 생성**|특정 센서(Color, Depth, IR) 스트림 객체 생성|`color_stream = device.create_color_stream()`|컬러 센서 또는 깊이 센서 전용 채널 확보|
|**④ 모드 설정 & 시작**|해상도, FPS, 픽셀 포맷 세팅 후 전송 시작|`color_stream.set_video_mode(...)`  <br>`color_stream.start()`|카메라 하드웨어에 비디오 전송 트리거|
|**⑤ 프레임 수신**|스트림으로부터 이미지 프레임 캡처|**폴링**: `wait_for_any_stream()` + `read_frame()`  <br>**이벤트**: `on_new_frame(stream)`|데이터 도착 감지 및 원시 버퍼 수신|
|**⑥ 데이터 변환 및 처리**|C 버퍼 → 파이썬 데이터/영상 처리|`buffer = frame.get_buffer_as_triplet()` 또는  <br>`np.frombuffer()` → `cv2.imshow()`|OpenCV/NumPy 연동 및 비전 알고리즘 적용|
|**⑦ 리소스 정리**|역순으로 안전하게 해제|`stream.stop()` → `stream.close()` →  <br>`device.close()` → `openni2.unload()`|메모리 누수 및 USB 장치 락(Lock) 방지|

---

### **3. 두 가지 프레임 취득 파이프라인 비교**

작성하신 두 파일(`as1110.py` vs `as1109_2.py`)의 프레임 취득 파이프라인 차이점입니다:

[폴링 방식 (as1110.py)]

메인 루프(While) ──> 새 프레임 도착 대기(wait) ──> 프레임 읽기(read) ──> 처리 ──> 반복

(구조가 직관적이고 동기식 처리나 OpenCV cv2.imshow 루프와 결합하기 가장 적합)

[이벤트 리스너 방식 (as1109_2.py)]

카메라 하드웨어 ──> 프레임 도착 시 자동 신호 ──> OpenNI 내부 콜백 스레드 실행 (비동기)

(메인 스레드가 다른 무거운 연산을 하면서 백그라운드로 프레임을 받아둘 때 적합)

