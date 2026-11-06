


## 수업목표
- ㅇㅇ
- 

## 실습파일
- ㅇㅇ
- 


## canny - edge 검출

[Canny Edge Detection — OpenCV Tutorials](https://docs.opencv.org/5.0/py_tutorials/py_imgproc/py_canny/py_canny.html)

threshold : 50 ~ 150
![[1001_1.png]]


threshold : 100 ~ 200
![[1001_2.png]]


평활화후 canny 하는 것 보다는 **그레이→ 가우시안 블러 → canny** 가 더 도로 검출이 잘 됨

![[1001_3.png]]

## histogram

#### histogram Eqalization(히스토그램 평활화)

[Histogram Equalization — OpenCV Tutorials](https://docs.opencv.org/5.0/tutorials/imgproc/histograms/histogram_equalization/histogram_equalization.html#histogram-equalization)

## Equalization
![[1001_4.png]]


```python
## 다양한 평활화 적용해보기
#   1) 히스토그램 평활화
#   2) 명시적 평활화
#   3) CLAHE(지역 히스토그램 평활화)
#   4)adaptive CLAHE (지역 히스토그램 평활화)
#   5)clahe와 adaptive CLAHE 차이점 비교
#
#   6) clahe의 clippingLimit과 tileGridSize 값 변경에 따른 차이 비교
```

