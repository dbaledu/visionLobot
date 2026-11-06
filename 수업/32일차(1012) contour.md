


## 수업목표
- ㅇㅇ
- 

## 실습파일
- ㅇㅇ
- 


- 허프를 이용한 단일 차선 검출(이미지) : cv1012_1_single_line_detect_houghlinep.py
    - 시간: 0.01337885856628418
- 허프를 이용한 단일 차선 검출(영상) : cv1012_2_single_line_detect_houghlinep_movie.py
- 컨투어를 이용한 윤곽선 추출: cv1012_3_contour.py
- area를 이용하여 중심좌표 찾기. 사각형그리기: cv1012_4_contour_contourArea.py

추가내용

- 모폴로지를 이용한 차선 검출(이미지) : cv1012_2_single_line_detect_moporlogy_movie.py
    - 시간: 0.022859573364257812
- 모폴로지를 이용한 차선 검출(영상) : cv1012_2_single_line_detect_moporlogy_movie.py
- 컨투어를 이용한 차선검출: cv1012_contour_moive.py

#### 방식 비교

- 단일 차선이 뚜렷한 실내 AMR: 색상 마스크 → 모폴로지 → 중심점 계산을 먼저 추천합니다.
- 직선이나 선분의 기울기를 알아야 하는 경우: Canny → HoughLinesP를 고려합니다.
- 차선이 끊어지거나 잡음이 많은 경우: 모폴로지로 마스크를 정리한 뒤 중심점 계산 또는 Hough 검출을 적용할 수 있습니다.