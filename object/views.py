from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import ast
import base64
from PIL import Image
import io
import os
import numpy as np
import random
import json
import random as rd
from threading import Thread
from multiprocessing import Process, Queue
from .yolo.detect import yolo_object


@csrf_exempt
def detection(request):
    """
    이미지를 받으면 word가 있는 영역들을 x,y,width,height형식으로 반환해준다.
    """
    print("request object detection")
    print("request body", str(request.body.decode('utf-8'))[:80])

    try:
        data = ast.literal_eval(request.body.decode('utf-8'))
    except ValueError:
        data = json.loads(request.body.decode('utf-8'))

    response = {
        'id': data['id'],
        'meta': [{
            'crop_image': [
            ]
        }]
    }

    imPath = os.getcwd() + "/object_image/" + str(data["id"]) + "/"
    if not os.path.isdir(imPath):
        os.mkdir(imPath)
    else:
        for fn in os.listdir(imPath):
            os.remove(imPath + fn)

    # request로 부터 온 이미지를 decode하고 numpy형태로 전환합니다.
    if type(data["orig_image"]) is not list:
        data["orig_image"] = [data["orig_image"]]

    for i in range(len(data["orig_image"])):
        try:
            image = base64.b64decode(data["orig_image"][i].split(',')[-1])
            image = Image.open(io.BytesIO(image))
            image = np.array(image)
        except:
            print("error")
            print(str(data['orig_image'][i])[:80])
            print(data['orig_image'][i][:80])
            return JsonResponse({"error": "can't save image file"})

        cv2.imwrite(imPath + str(i) + ".png", image)

    # 이미지로 부터 객체 영역을 추출합니다.
    results, regions = yolo_object(images=imPath)

    for i in range(len(data["orig_image"])):
        os.remove(imPath + str(i) + '.png')
    os.rmdir(imPath)

    for result, region in zip(results, regions):
        response['meta'][0]['crop_image'].append({
            'x': region['x'],
            'y': region['y'],
            'width': region['width'],
            'height': region['height'],
            'size': region['size'],
            'label': result,
        })

    # response를 반환할 수 있는 Json객체로 만들고 헤더에 too many requests error를 해결하기 위한 설정을 합니다.
    response = JsonResponse(response)
    response['Access-Control-Allow-Origin'] = '*'
    response['Access-Control-Allow-Methods'] = 'POST'
    response['Access-Control-Allow-Age'] = '3600'
    response['Access-Control-Allow-Headers'] = 'Origin,Accept,X-Requested-With,Content-Type,Access-Control-Request-Method,Access-Control-Request-Headers,Authorization'

    return response


@csrf_exempt
def recognition(request):
    """
    이미지를 받으면 word가 있는 영역들을 x,y,width,height형식으로 반환해준다.
    """
    print("request object detection")
    print("request body", str(request.body.decode('utf-8'))[:80])

    try:
        data = ast.literal_eval(request.body.decode('utf-8'))
    except ValueError:
        data = json.loads(request.body.decode('utf-8'))

    response = {
        'id': data['id'],
        'label': []
    }

    imPath = os.getcwd() + "/object_recogition_image/" + str(data["id"]) + "/"
    if not os.path.isdir(imPath):
        os.mkdir(imPath)
    else:
        for fn in os.listdir(imPath):
            os.remove(imPath + fn)

    # request로 부터 온 이미지를 decode하고 numpy형태로 전환합니다.
    if type(data["crop_image"]) is not list:
        data["crop_image"] = [data["crop_image"]]

    for i in range(len(data["crop_image"])):
        try:
            image = base64.b64decode(data["crop_image"][i].split(',')[-1])
            image = Image.open(io.BytesIO(image))
            image = np.array(image)
        except:
            print("error")
            print(str(data['crop_image'][i])[:80])
            print(data['crop_image'][i][:80])
            return JsonResponse({"error": "can't save image file"})

        cv2.imwrite(imPath + str(i) + ".png", image)

    # 이미지로 부터 객체 영역을 추출합니다.
    results, regions = yolo_object(images=imPath)

    for i in range(len(data["crop_image"])):
        os.remove(imPath + str(i) + '.png')
    os.rmdir(imPath)

    response['label'] = results

    # response를 반환할 수 있는 Json객체로 만들고 헤더에 too many requests error를 해결하기 위한 설정을 합니다.
    response = JsonResponse(response)
    response['Access-Control-Allow-Origin'] = '*'
    response['Access-Control-Allow-Methods'] = 'POST'
    response['Access-Control-Allow-Age'] = '3600'
    response['Access-Control-Allow-Headers'] = 'Origin,Accept,X-Requested-With,Content-Type,Access-Control-Request-Method,Access-Control-Request-Headers,Authorization'

    return response
