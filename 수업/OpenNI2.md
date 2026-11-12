
### **1. 모듈 레벨 전역 함수 (Global Functions)**

| 함수 이름                                                                          | 용도 설명                                                                                                        |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| **`initialize(dll_directories=...)`**                                          | OpenNI2 라이브러리(Windows는 `OpenNI2.dll`, Linux는 `libOpenNI2.so`)를 찾아 로드하고 초기화합니다. DLL 폴더 경로를 직접 인자로 전달할 수 있습니다. |
| **`is_initialized()`**                                                         | 현재 OpenNI2 라이브러리가 초기화되어 로드된 상태인지 여부(`True`/`False`)를 반환합니다.                                                  |
| **`unload()`**                                                                 | 등록된 모든 디바이스, 스트림, 프레임, 리코더, 리스너를 닫고 OpenNI2를 안전하게 종료(`oniShutdown`)합니다. 프로그램 종료 시(`atexit`) 자동으로도 호출됩니다.     |
| **`get_version()`**                                                            | 로드된 OpenNI2 C API 라이브러리의 버전 정보(`OniVersion` 구조체)를 반환합니다.                                                     |
| **`wait_for_any_stream(streams, timeout=None)`**                               | 여러 비디오 스트림 목록 중 새로운 프레임 데이터가 도착할 때까지 대기(timeout 설정 가능)한 뒤, 준비가 완료된 스트림 객체를 반환합니다. (여러 스트림 동시 폴링 시 사용)        |
| **`convert_world_to_depth(depthStream, worldX, worldY, worldZ)`**              | 3차원 실세계 월드 좌표(X, Y, Z, mm 단위)를 뎁스 이미지 평면 상의 픽셀 좌표(depthX, depthY, depthZ)로 변환합니다.                            |
| **`convert_depth_to_world(depthStream, depthX, depthY, depthZ)`**              | 뎁스 이미지 좌표(픽셀 X, Y 및 뎁스 거리 Z)를 3차원 실세계 월드 좌표(worldX, worldY, worldZ, mm 단위)로 변환합니다.                           |
| **`convert_depth_to_color(depthStream, colorStream, depthX, depthY, depthZ)`** | 뎁스 카메라 좌표계의 특정 점(depthX, depthY, depthZ)에 매핑되는 컬러 카메라 이미지의 픽셀 좌표(colorX, colorY)를 계산합니다. (좌표 정렬 및 매핑용)       |
| **`get_bytes_per_pixel(format)`**                                              | 특정 픽셀 포맷(`PIXEL_FORMAT_RGB888`, `PIXEL_FORMAT_DEPTH_1_MM` 등)의 픽셀당 바이트 크기를 반환합니다.                             |
| **`get_log_filename()`**                                                       | OpenNI2가 현재 작성하고 있는 로그 파일의 전체 경로와 이름을 반환합니다.                                                                 |
| **`configure_logging(directory=None, severity=None, console=None)`**           | 로그 저장 디렉토리, 로그 심각도(0: Verbose, 1: Info, 2: Warning, 3: Error), 콘솔 출력 여부를 설정합니다.                              |

### **주요 클래스 및 메서드 (Class Methods)**

#### **(1) `Device` 클래스 (카메라/센서 장치 제어)**

- **생성자 및 팩토리 메서드**
    - `Device.enumerate_uris()`: 연결된 모든 OpenNI2 장치의 URI 목록을 문자열 리스트로 반환합니다.
    - ==Device.open_all():== 연결된 모든 장치를 열어서 `Device` 객체 리스트로 반환합니다.
    - ==Device.open_any()==: 연결된 기본(첫 번째) 장치를 엽니다.
    - ==Device.open_file(filename)==: 녹화된 ONI 파일 재생을 위해 장치 객체를 엽니다.

- **스트림 생성 메서드**
    - ==create_stream(sensor_type)==: 특정 센서 타입(`SENSOR_DEPTH`, `SENSOR_COLOR`, `SENSOR_IR`)의 스트림을 생성합니다.
    - ==create_depth_stream()==: 뎁스(Depth) 비디오 스트림(`VideoStream`)을 생성합니다.
    - ==create_color_stream():== 컬러(RGB) 비디오 스트림을 생성합니다.
    - ==create_ir_stream()==: 적외선(IR) 비디오 스트림을 생성합니다.

- **장치 정보 및 설정 메서드**
    - ==get_device_info()==: 장치 정보(`uri`, `vendor`, `name`, `usbVendorId` 등)를 반환합니다. (속성: `device_info`)
    - ==get_sensor_info(sensor_type)==: 특정 센서가 지원하는 비디오 모드(해상도, FPS, 포맷) 정보를 반환합니다.
    - ==has_sensor(sensor_type)==: 특정 센서(Depth/Color/IR)가 장치에 존재하는지 확인합니다.
    - `is_file()`: 현재 장치가 물리 카메라가 아닌 녹화된 파일 재생 모드인지 확인합니다.
    - `is_image_registration_mode_supported(mode)`: 이미지 정렬(Depth-to-Color 맵핑) 지원 여부를 확인합니다.
    - `get_image_registration_mode()` / `set_image_registration_mode(mode)`: 이미지 정렬 모드를 조회/설정합니다.
    - ==get_depth_color_sync_enabled()== / `set_depth_color_sync_enabled(enable)`: 뎁스와 컬러 프레임 간의 하드웨어/소프트웨어 동기화 여부를 설정합니다. (속성: `depth_color_sync`)
    - `get_property(...)` / `set_property(...)` / `invoke(...)`: 하위 C API 속성 및 커맨드를 직접 제어합니다.

---

#### **(2) `VideoStream` 클래스 (비디오 스트림 제어)**

- **스트림 시작 및 종료**
    - ==start()==: 카메라 스트림 데이터 전송을 시작합니다.
    - ==stop()==: 스트림 데이터 전송을 일시 중지합니다.
    - ==close()==: 스트림을 파괴하고 리소스를 해제합니다.

- **프레임 수신 및 이벤트 리스너**
    - ==read_frame()==: 새로운 프레임 하나를 동기식(blocking)으로 읽어 `VideoFrame` 객체로 반환합니다.
    - ==register_new_frame_listener(callback)==: 새 프레임이 도착했을 때 비동기적으로 호출될 이벤트 콜백 함수를 등록합니다.
    - ==unregister_new_frame_listener(callback)==: 등록된 콜백 리스너를 해제합니다.
    - ==unregister_all_new_frame_listeners()==: 모든 프레임 리스너 콜백을 해제합니다.

- **해상도 및 영상 포맷 설정**
    - `get_video_mode()` / ==set_video_mode(mode)==: 비디오 모드(해상도, FPS, 픽셀포맷)를 조회/설정합니다.
    - `configure_mode(width, height, fps, pixel_format)`: 해상도, 프레임레이트, 픽셀 포맷을 한 번에 설정하는 간편 메서드입니다.

- **화각, 반전 및 영역 설정**
    - `get_horizontal_fov()` / `get_vertical_fov()`: 수평 / 수직 화각(FOV, 라디안 단위)을 반환합니다.
    - `get_mirroring_enabled()` / `set_mirroring_enabled(enabled)`: 좌우 반전(미러링) 여부를 조회/설정합니다. (속성: `mirroring_enabled`)
    - `is_cropping_supported()`: 화면 자르기(Cropping/ROI) 기능 지원 여부를 확인합니다.
    - `set_cropping(originX, originY, width, height)` / `reset_cropping()`: 관심 영역(ROI)을 지정하거나 리셋합니다.
    - `get_max_pixel_value()` / `get_min_pixel_value()`: 센서가 표현 가능한 최소/최대 픽셀값(예: 최대 뎁스 거리)을 반환합니다.
    - `get_recoder(filename, allow_lossy_compression=False)`: 이 스트림을 녹화할 `Recorder` 객체를 생성하고 연결합니다.

---

#### **(3) `VideoFrame` 클래스 (캡처된 개별 프레임 데이터)**

- **메모리 버퍼 접근**
    - `get_buffer_as(ctype)`: 프레임 원본 메모리 주소(`self.data`)를 지정한 ctypes 배열 타입으로 변환합니다.
    - `get_buffer_as_uint8()`: 버퍼를 `c_uint8` 배열로 변환합니다. (8비트 흑백/적외선 이미지 등에 사용)
    - ==get_buffer_as_uint16()==: 버퍼를 `c_uint16` 배열로 변환합니다. (일반적인 16비트 밀리미터 단위 뎁스 데이터에 사용)
    - ==get_buffer_as_triplet()==: 버퍼를 3채널 RGB `c_uint8 * 3` 배열로 변환합니다. (RGB888 컬러 이미지에 사용)

- **프레임 메타데이터 (C 구조체 상속 프로퍼티)**
    - `width`, `height`, `frameIndex`, `timestamp`, `videoMode`, `sensorType`, `cropOriginX`, `cropOriginY` 등

---

#### **(4) `CameraSettings` 클래스 (카메라 하드웨어 제어)**

`VideoStream.camera` 속성을 통해 접근합니다.

- ==get_auto_exposure()== / `set_auto_exposure(enabled)`: 자동 노출 On/Off 설정 (속성: `auto_exposure`)
- ==get_auto_white_balance()== / `set_auto_white_balance(enabled)`: 자동 화이트밸런스 On/Off 설정 (속성: `auto_white_balance`)
- ==get_gain()== / `set_gain(gain)`: 센서 아날로그 게인(증폭/감도) 값 조회 및 설정 (속성: `gain`)
- ==get_exposure()== / `set_exposure(exposure)`: 노출 시간(Exposure) 값 조회 및 설정 (속성: `exposure`)

---

#### **(5) `Recorder` 클래스 (ONI 비디오 녹화)**

- `attach(stream, allow_lossy_compression=False)`: 녹화할 `VideoStream`을 리코더에 연결합니다.
- `start()`: 녹화를 시작합니다.
- `stop()`: 녹화를 중지합니다.

---

#### **(6) `DeviceListener` 클래스 (디바이스 연결/해제 이벤트 감지)**

- `on_connected(devinfo)`: 새로운 카메라 장치가 USB 등에 연결되었을 때 호출되는 콜백 (상속하여 재정의)
- `on_disconnected(devinfo)`: 장치 연결이 끊겼을 때 호출되는 콜백
- `on_state_changed(devinfo, state)`: 장치 상태가 변경되었을 때 호출되는 콜백
- `unregister()`: 리스너 등록을 해제합니다.