<img style="padding: 100px;" src="https://user-images.githubusercontent.com/31213226/63397983-1ead4d80-c407-11e9-98bd-906ca01db919.png">

# Common Bound AI Server
커먼바운드는 인공지능 학습 데이터를 생산하는데 필요한 인공지능을 제공합니다.

생산 AI로 사용자는 더 간편하게 생성할 수 있으며, 검수 AI로 높은 품질의 데이터를 빠르게 제공할 수 있습니다.


## How to Install
### Clone this repository
```
git clone https://git.swmgit.org/root/p1021_dal-1.git
cd p1021_dal-1
mkdir recognition_image
mkdir detection_image
mkdir object_image
mkdir object_recognition_image
```
### Create and activate environment
```
conda env create -f ./environment.yml
conda activate server
```
### install requirements
```
pip install -r requirements.txt
```
### run server
```
python manage.py runserver 0.0.0.0:8000
```

## description
### 생산 AI 어시스턴트
- Text 및 object Detection, Recognition을 위한 API를 제공합니다.
- Detection은 `orig_image`을 요구하고, 영역에 대한 위치, 확률, 레이블을 반환합니다.
- Recognition은 `crop_image`을 요구하고, 영역에 대한 확률, 레이블을 반환합니다.

### 검수 AI 어시스턴트
- 문자열 유사도, 검수 예측을 위한 API를 제공합니다.
- 문자열 유사도는 `label`, `ai_label`를 요구하고, 두 문자열 사이의 유사도를 반환합니다..
- 검수 예측은 `region_attributes`를 요구하고, 검수 확률을 반환합니다.


# API describe
## [Text Detection](https://github.com/clovaai/CRAFT-pytorch)
```
POST	/ocr/detection/
```

### Parameter
```
{
  'id' : uuid,
  'orig_image': [image_encode_base64]
}
```

### Response
```
{
  'id' : request.id,
  'meta': [{
    'crop_image': [
      {
        'x': x,
        'y': y,
        'width': width,
        'height': height,
        'ai_size': size,
      },
      'ai_total_size': total_size
    ]
  }]
}
```
<img width="100%" alt="sample1" src="https://user-images.githubusercontent.com/40608930/69129007-d9c1b180-0af0-11ea-940d-7760d9c8be26.gif">

## [Text Recognition](https://github.com/clovaai/deep-text-recognition-benchmark)
```
POST	/ocr/recognition/
```
### Parameter
```
{
  'id' : uuid,
  'crop_image': [image_encode_base64]
}
```

### Response
```
{
  'label': [label],
  'P': [accuracy]
}
```
<img width="640" height="360" alt="sample1" src="https://user-images.githubusercontent.com/40608930/69129044-f6f68000-0af0-11ea-880c-63cc997d4dc5.PNG">


## Compare 2 strings
```
POST	/ocr/compare_string/
```
### Parameter
```
{
  'label': label,
  'ai_label': ai_label,
}
```
### Response
```
{
  'similarity': similarity
}
```

	
## [Object Detection](https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch)
```
POST	/object/detection/
```
### Parameter
```
{
  'id' : uuid,
  'origin_image': [image_encode_base64]
}
```
### Response
```
{
  'id' : uuid,
  'meta':[{
    'crop_image': [{
      'x': x,
      'y': y,
      'width': width,
      'size': size,
      'label': label
    }]
  }]
}
```


## [Object Recoginition](https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch)
```
POST	/object/recoginition/
```
### Parameter
```
{
  'id' : uuid,
  'crop_image': [image_encode_base64]
}
```
### Response
```
{
  'id' : uuid,
  'label': [label]
}
```


## inspection predict
```
POST	/inspection/predict/
```
### Parameter
```
{
  'meta':{
    'crop_image':[{
      'region_attributes':{
        'prob': prob,
        'label': label,
        'ai_label': ai_label,
        'crop_time': crop_time,
        'image_time': image_time,
        'similarity': similarity
      }
    }]
  }
}
```
### Response
```
{
  'meta':{
    'crop_image':[{
      'region_attributes':{
        'prob': prob,
        'label': label,
        'ai_label': ai_label,
        'crop_time': crop_time,
        'image_time': image_time,
        'similarity': similarity,
        'reliability': reliability
      }
    }]
  }
}
```


## Authors
**[김광호](mailto:rhkd865@naver.com) | [김은수](mailto:eunsu.dev@gmail.com) | [최현서](mailto:arenaofjagal@naver.com )**


## Links
- [main repository](https://git.swmgit.org/root/p1021_dal)
- [Service](https://c-bound.io/)