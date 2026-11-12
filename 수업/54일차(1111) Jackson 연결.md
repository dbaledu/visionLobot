
## 수업목표
-  리눅스 기초 명령어
- _Jetson Nano Developer Kit_ (임베디드 보드)에서 openni 설치하고 카메라 센서 연결 


## 실습파일
- as1109.py - 카메라 센서 연결 확인
- as110_exam1_event.py - 색상판별
- 

사용하는 임베디드 보드 :
 -  ==**Jetson Nano는 64비트 ARM(aarch64)**==
 - 저가 GPU 임베디드 제품
 - 엣지 AI 컴퓨팅 시스템
 - 객체 인식과 같은 연산량이 많은 모델을 팔로잉 기능을 위해 테스트하려면 **AI가속화 엔진**이 필요
	 - AI 가속화 엔진 : 하드웨어와 소프트웨어적으로 **추론을 가속화**. 애플리케이션 대기시간을 줄임
	 - TensorRT
		 - 엔비디아 제품
		 - 고성능 딥러닝 추론 엔진이며 엔비디아의 GPU와 함께 사용
		 - 딥러닝 추론 최적화 런타임 시스템

##### Astra 카메라 연결 실습 순서
1. Jetson 연결

```shell
# cmd 창에서 연결 확인
ssh nvida@ip주소
```

MobaXterm 연결

Session -> SSH 탭 -> Remote host 에 IP 입력 -> specify username 체크하고 입력
 - colab 처럼 필요한 라이브러리 모두 설치된 이미지임

2. openni 패키지 설치

	1. 다운로드 :  https://www.orbbec.com/developers/openni-sdk/
	2. 파일명 : Orbbec_OpenNI_v2.3.0.86-beta6_linux_release.zip
	3. 압축을 풀면
	      - OpenNI_2.3.0.86_202210111146_4c8f5aa4_beta6_rpi_arm.zip
	      - OpenNI_2.3.0.86_202210111154_4c8f5aa4_beta6_linux_x64.zip
	      - OpenNI_2.3.0.86_202210111155_4c8f5aa4_beta6_a311d_arm64.zip
	      - 어떤 CPU/기기용으로 컴파일했는지 구분함. 파일명 끝부분이 대상 플랫폼을 나타냅니다.
	      - a311d_arm64 파일을 보드로 이동
	4. 보드에서 압축해제하고 폴더명 변경

```shell
$ sudo apt install unzip
$ unzip OpenNI_2.3.0.86_202210111155_4c8f5aa4_beta6_a311d_arm64.zip
$ mv  OpenNI_2.3.0.86_202210111155_4c8f5aa4_beta6_a311d_arm64/  OpenNI_2.3.0  
```

3. 카메라 센서 연결
```shell
# 시스템에 연결된 USB 장치 목록
$ lsusb

# 카메라 연결하고 다시 확인
$ lsusb
Bus 002 Device 002: ID 0bda:0411 Realtek Semiconductor Corp.
Bus 002 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
Bus 001 Device 003: ID 8087:0a2b Intel Corp.
Bus 001 Device 004: ID 2bc5:0401    # `제조사ID:제품ID` (2bc5는 Orbbec의 벤더 ID)                                        # <--- 카메라 센서임
Bus 001 Device 002: ID 0bda:5411 Realtek Semiconductor Corp.
Bus 001 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub

```

4. openni 패키지 설치

```shell
$ pip install openni
```

5. 카메라 테스트
		- as1109.py 복사
		- 탐색기창에서 더블클릭하면 편집창이 열림
		- 소스에서 OPENNI_PATH 경로 수정 'OPENNI_2.3.0/samples/bin'

```shell
# 권한문제 발생  -> 실행권한 부여하여 install.sh 실행
$ cd OpenNI_2.3.0/rules    # <-- work 폴더에서 실행함.
$ chmod 755 install.sh 
$ ./install.sh
$ cd ../..
$ sudo python3 as1109.py
```

6. 리눅스 종료

```shell
$ sudo shutdown -h now
```

### Orbbec_OpenNI_v2.3.0.86-beta6_linux_release.zip 파일 구조

| 파일                    | 대상                           | 쓰는 곳                                        |
| --------------------- | ---------------------------- | ------------------------------------------- |
| `..._rpi_arm.zip`     | 라즈베리 파이용 ARM 빌드              | Raspberry Pi (주로 32비트 ARM, Raspberry Pi OS) |
| `..._linux_x64.zip`   | 일반 PC용 64비트 Linux 빌드         | Intel/AMD CPU의 Ubuntu 등 Linux 데스크톱·서버·노트북   |
| `..._a311d_arm64.zip` | Amlogic A311D 칩용 64비트 ARM 빌드 | Khadas VIM3 같은 A311D 기반 보드 (aarch64)        |
- Jetson Nano는 64비트 ARM(aarch64) **..._a311d_arm64.zip** 파일 사용. uname 명령으로 커널과 하드웨어 정보 확인 가능
- **AArch64**(ARM64)는 ==ARM 아키텍처의 **64비트 명령어 집합 및 실행 상태**==


##### 파일 형식

```
 OpenNI_2.3.0.86_202210111155_4c8f5aa4_beta6_a311d_arm64.zip
  - 2.3.0.86          # OpenNI 버전
  - 202210111155      # 빌드 시각
  - 4c8f5aa4          # 소스 커밋 해시
  - beta6             # 베타 6차 릴리스
  - a311d_arm64       # A311D 보드
```

##### 보드 커널과 하드웨어 정보 확인
```shell
# 현재 리눅스 시스템의 커널과 하드웨어 정보를 출력하는데 m옵션은 CPU 아키텍처
$ uname -m
aarch64                      # 
```

`uname`은 현재 리눅스 시스템의 **커널과 하드웨어 정보**를 출력하는 명령입니다.

**자주 쓰는 옵션**

| 명령         | 출력 내용         | 예시                            |
| ---------- | ------------- | ----------------------------- |
| `uname`    | 커널 이름         | `Linux`                       |
| `uname -m` | CPU 아키텍처      | `x86_64`, `aarch64`, `armv7l` |
| `uname -r` | 커널 버전         | `5.15.0-91-generic`           |
| `uname -s` | 커널 이름         | `Linux`                       |
| `uname -n` | 호스트 이름        | `jetson-nano`                 |
| `uname -v` | 커널 빌드 정보      | `#101-Ubuntu SMP ...`         |
| `uname -o` | 운영체제          | `GNU/Linux`                   |
| `uname -a` | 위 정보를 모두 한 줄로 |                               |

```shell
$ uname
#Linux

$ uname -m
#aarch64

$ uname -r
#4.9.253-tegra

$ uname -s
#Linux

$ uname -n
#nvidia

$ uname -v
#1 SMP PREEMPT Sat Feb 19 08:59:22 PST 2022

$ uname -o
#GNU/Linux

$ uname -a
# Linux nvidia 4.9.253-tegra #1 SMP PREEMPT Sat Feb 19 08:59:22 PST 2022 aarch64 aarch64 aarch64 GNU/Linux

# 우분투 버전
$ lsb_release -a   # or   cat /etc/os-release
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 18.04.6 LTS
Release:        18.04
Codename:       bionic

# JetPack 버전
$ cat /etc/nv_tegra_release   # dpkg -l | grep nvidia-jetpack
R32 (release), REVISION: 7.1, GCID: 29818004, BOARD: t210ref, EABI: aarch64, DATE: Sat Feb 19 17:05:08 UTC 2022

```

##### JetPack
- JetPack은 NVIDIA가 Jetson 보드에서 AI와 딥러닝 프로그램을 개발하고 실행할 수 있도록 제공하는 통합 소프트웨어 플랫폼입니다.
- 잿팩에는 딥러닝 프로세싱을 위한 GPU 드라이버와 GPU 사용을 위한 CUDA, 딥러닝 라이브러리를 사용하기 위한 CuDNN이 설치되어 있으며 카메라와 GUI 개발에 용이한 OpenCV 라이브러리까지 설치되어 있음
- 


##### 카메라장치 인식이 안되는 경우
```shell
$ lsusb -v -d 2bc5:0401 2>/dev/null | grep -E "bcdDevice|iProduct|iSerial|iManufacturer|bNumInterfaces"

$ dmesg | tail -30

```

**펌웨어 차이 가능성**

같은 PID라도 구매 시기에 따라 펌웨어가 다를 수 있고, Windows에서는 Orbbec 드라이버가 이를 처리하지만 리눅스 구형 `liborbbec.so`는 못 여는 경우가 있습니다.

- 그때는 Orbbec 공식 저장소의 **최신 OpenNI 드라이버**로 `liborbbec.so`를 교체하거나, Windows에서 Orbbec 도구로 **펌웨어를 업데이트**해 보세요.
- 현재 SDK는 2022년 10월 빌드이므로, 더 최신 릴리스가 있는지 확인하는 것도 방법입니다.

