# 03. SAM2 실시간 분할

`get_segmentation.ipynb`는 SAM2 real-time의 `build_sam2_camera_predictor`로 첫 프레임에 bbox 프롬프트를 주고, 이후 프레임마다 마스크를 추적해 원본에 겹쳐 그립니다.

- 입력 영상: `../01_rtsp_recording/output_stream_11.avi`
- 모델: `sam2_hiera_small.pt` + `sam2_hiera_s.yaml`
- CUDA GPU가 필요합니다(`bfloat16` autocast, Ampere 이상이면 TF32 사용).

## 준비
SAM2 real-time 저장소를 **저장소 루트와 같은 위치**에 둡니다. 노트북은 `../../sam2_test/segment-anything-2-real-time/`을 찾습니다.

```
<상위 폴더>/
├── 3d_mod/03_segmentation/get_segmentation.ipynb
└── sam2_test/segment-anything-2-real-time/checkpoints/sam2_hiera_small.pt
```

## 참고
- 프롬프트 bbox(`[[1750, 950], [1880, 1030]]`)와 해상도(1920×1080)가 코드에 고정되어 있어 다른 영상에는 값을 바꿔야 합니다.
