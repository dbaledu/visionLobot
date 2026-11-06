


## 수업목표
- blur
- 사용자 정의 kernel
- 이미지 그레디언드 → edge(가장자리) 찾기

## 실습파일
- ㅇㅇ
- 


#### 블러(노이즈 제거)

cv0930_1_blur.py

- cv2.blur(image, size) : 평균값 필터
- cv2.**medianBlur**(image, ksize) : 미디언(중앙값) 필터
    - _점 노이즈(slat ans pepper noise) 제거에 효과적_
- cv2.GaussianBlur(image, ksize=size, sigmaX=sigmaX, sigmaY=sigmaY) : 가우시안 필터

[Smoothing Images — OpenCV Tutorials](https://docs.opencv.org/5.0/py_tutorials/py_imgproc/py_filtering/py_filtering.html)

가우시안 블러

![[0930_1.png]]


bilateral 블러 : 엣지(차선)는 유지하면서 블러 처리

![[0930_2.png]]


## 이미지 그레디언드(edge 검출)

[Image Gradients — OpenCV Tutorials](https://docs.opencv.org/5.0/py_tutorials/py_imgproc/py_gradients/py_gradients.html)

- 프리윗
- 소벨
- 샤르

## Sobel

- [**OpenCV Tutorials](https://docs.opencv.org/5.0/tutorials/tutorials.html) > [Image Processing (imgproc module)](https://docs.opencv.org/5.0/tutorials/imgproc/table_of_content_imgproc.html) > Sobel Derivatives**

[Sobel Derivatives — OpenCV Tutorials](https://docs.opencv.org/5.0/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html#sobel-operator)

[Image Filtering — OpenCV Tutorials](https://docs.opencv.org/5.0/main_modules/imgproc_filter.html#sobel)
