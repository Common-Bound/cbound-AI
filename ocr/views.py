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
from .Detection import test as craft
from .Recognition import demo as demo
#from textblob import TextBlob
import json
import random as rd
from difflib import SequenceMatcher
from threading import Thread
from multiprocessing import Process, Queue


def result2roi(results):
    meta = dict()
    crop_image = []
    total_size = 0

    for location in results:
        x = int(min(location[0, 0], location[3, 0]))
        y = int(min(location[0, 1], location[1, 1]))
        width = int(max(location[1, 0], location[2, 0]) - x)
        height = int(max(location[2, 1], location[3, 1]) - y)
        size = width * height
        crop_image.append({
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'ai_size': size
        })
        total_size += size

    meta = {
        'crop_image': crop_image,
        'ai_total_size': total_size}

    return meta


def result2roi_thread(results):
    meta = dict()
    crop_image = []
    total_size = 0

    for location in results:
        print(location)
        x = int(min(location[0, 0], location[3, 0]))
        y = int(min(location[0, 1], location[1, 1]))
        width = int(max(location[1, 0], location[2, 0]) - x)
        height = int(max(location[2, 1], location[3, 1]) - y)
        size = width * height
        crop_image.append({
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'ai_size': size
        })
        total_size += size

    meta = {
        'crop_image': crop_image,
        'ai_total_size': total_size}

    return meta


def craft_detection(image=None, imPath=None):
    craft.main(image=image, test_folder=imPath, cuda=True)
    results = craft.eval()

    return results


def craft_detection_thread(image=None, imPath=None, results=list()):
    craft.main(image=image, test_folder=imPath, cuda=True)
    result = craft.eval()

    results.append(result)

    return results


def rare_recogition_thread(imPath, results):
    words, scores = demo.recognition(image_folder=imPath)

    results.append([words, scores])

    return results


def rare_recogition_process(imPath, results):
    words, scores = demo.recognition(image_folder=imPath)

    results.put([words, scores])

    return results


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


@csrf_exempt
def detection(request):
    """
    이미지를 받으면 word가 있는 영역들을 x,y,width,height형식으로 반환해준다.
    """
    print("request detection")
    print("request body", str(request.body.decode('utf-8'))[:80])

    try:
        data = ast.literal_eval(request.body.decode('utf-8'))
    except ValueError:
        data = json.loads(request.body.decode('utf-8'))
    #print("data: ", data["orig_image"][:80])
    image = imPath = None
    response = {
        'id': data['id'],
        'meta': [

        ]
    }

    if not os.path.isdir(os.getcwd() + '/detection_image'):
        os.mkdir(os.getcwd() + '/detection_image')

    # request로 부터 온 이미지를 decode하고 numpy형태로 전환합니다.
    if type(data["orig_image"]) is not list:
        image = base64.b64decode(data["orig_image"].split(',')[-1])
        image = Image.open(io.BytesIO(image))
        image = np.array(image)

        # 이미지 grayscale일 경우 rgb 형식으로 전환합니다.
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image.reshape(image.shape[0], image.shape[1], 1)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        results = craft_detection(image=image)
    else:
        imPath = os.getcwd() + "/detection_image/" + data["id"] + '/'
        os.mkdir(imPath)
        for i in range(len(data["orig_image"])):
            try:
                image = base64.b64decode(data["orig_image"][i].split(',')[-1])
                image = Image.open(io.BytesIO(image))
                image = np.array(image)
            except:
                print("error")
                print(str(data['orig_image'][i])[:80])
                print(data['orig_image'][i][:80])
                exit(-1)

            cv2.imwrite(imPath + str(i) + ".png", image)

        results = craft_detection(imPath=imPath)

        for i in range(len(data["orig_image"])):
            os.remove(imPath + str(i) + '.png')
        os.rmdir(imPath)

    # 이미지로 부터 텍스트 영역을 추출합니다.
    for result in results:
        response['meta'].append(result2roi(result))

    # response를 반환할 수 있는 Json객체로 만들고 헤더에 too many requests error를 해결하기 위한 설정을 합니다.
    response = JsonResponse(response)
    response['Access-Control-Allow-Origin'] = '*'
    response['Access-Control-Allow-Methods'] = 'POST'
    response['Access-Control-Allow-Age'] = '3600'
    response['Access-Control-Allow-Headers'] = 'Origin,Accept,X-Requested-With,Content-Type,Access-Control-Request-Method,Access-Control-Request-Headers,Authorization'

    return response


@csrf_exempt
def detection_thread(request):
    """
    이미지를 받으면 word가 있는 영역들을 x,y,width,height형식으로 반환해준다.
    """
    print("request detection thread ")
    print("request body", str(request.body.decode('utf-8'))[:80])

    try:
        data = ast.literal_eval(request.body.decode('utf-8'))
    except:
        data = json.loads(request.body.decode('utf-8'))

    image = imPath = None
    response = {
        'id': data['id'],
        'meta': [

        ]
    }

    if not os.path.isdir(os.getcwd() + '/detection_image'):
        os.mkdir(os.getcwd() + '/detection_image')

    # request로 부터 온 이미지를 decode하고 numpy형태로 전환합니다.
    if type(data["orig_image"]) is not list:
        image = base64.b64decode(data["orig_image"].split(',')[-1])
        image = Image.open(io.BytesIO(image))
        image = np.array(image)

        # 이미지 grayscale일 경우 rgb 형식으로 전환합니다.
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image.reshape(image.shape[0], image.shape[1], 1)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        results = craft_detection(image=image)
    else:
        imPath = os.getcwd() + "/detection_image/" + data["id"] + '/'
        os.mkdir(imPath)

        results = list()

        p_detection = Thread(target=craft_detection_thread,
                             args=(None, imPath, results))
        p_detection.start()

        for i in range(len(data["orig_image"])):
            try:
                image = base64.b64decode(data["orig_image"][i].split(',')[-1])
                image = Image.open(io.BytesIO(image))
                image = np.array(image)
            except:
                print("error")
                print(data['orig_image'][i][:80])
                exit(-1)

            cv2.imwrite(imPath + str(i) + ".png", image)

        #results = craft_detection(imPath=imPath)
        p_detection.join()

        for i in range(len(data["orig_image"])):
            os.remove(imPath + str(i) + '.png')
        os.rmdir(imPath)

    # 이미지로 부터 텍스트 영역을 추출합니다.
    for result in results:
        response['meta'].append(result2roi_thread(result))

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
    Text가 있는 이미지를 받으면 이를 Text로 변환하고 약간의 오타보정을 해주고 반환한다.
    """
    print("request recognition")
    #print("request body", str(request.body.decode('utf-8'))[:90])

    data = ast.literal_eval(request.body.decode('utf-8'))

    if not os.path.isdir(os.getcwd() + 'recognition_image'):
        os.mkdir(os.getcwd() + 'recognition_image')

    # 이미지 저장 절대 경로
    img_path = os.getcwd() + '/recognition_image/' + \
        str(data['id']) + str(rd.randint(1, 1000) * rd.randint(1, 1000)) + '/'
    os.mkdir(img_path)

    if type(data["crop_image"]) is not list:
        image = base64.b64decode(data["crop_image"].split(',')[-1])
        image = Image.open(io.BytesIO(image))
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cv2.imwrite(img_path + '0.png', image)
    else:
        for i in range(len(data["crop_image"])):
            image = base64.b64decode(data["crop_image"][i].split(',')[-1])
            image = Image.open(io.BytesIO(image))
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            cv2.imwrite(img_path + str(i) + '.png', image)

    words, scores = demo.recognition(image_folder=img_path)

    # print(words)
    # print(scores)

    #text_correct = TextBlob(result_orig)
    #result = str(text_correct.correct())

    for i in range(len(words)):
        os.remove(img_path + str(i) + '.png')
    os.rmdir(img_path)

    response = JsonResponse({'label': words, 'prob': scores})
    response['Access-Control-Allow-Origin'] = '*'
    response['Access-Control-Allow-Methods'] = 'POST'
    response['Access-Control-Allow-Age'] = '3600'
    response['Access-Control-Allow-Headers'] = 'Origin,Accept,X-Requested-With,Content-Type,Access-Control-Request-Method,Access-Control-Request-Headers,Authorization'

    return response


@csrf_exempt
def compare_string(request):
    print("request compare_string")
    #print("request body", str(request.body.decode('utf-8'))[:90])

    data = ast.literal_eval(request.body.decode('utf-8'))
    similarity = list()

    for u_label, a_label in zip(data['label'], data['ai_label']):
        prob = similar(u_label, a_label)
        similarity.append(prob)

    response = JsonResponse({"similarity": similarity})
    response['Access-Control-Allow-Origin'] = '*'
    response['Access-Control-Allow-Methods'] = 'POST'
    response['Access-Control-Allow-Age'] = '3600'
    response['Access-Control-Allow-Headers'] = 'Origin,Accept,X-Requested-With,Content-Type,Access-Control-Request-Method,Access-Control-Request-Headers,Authorization'

    return response


@csrf_exempt
def recognition_process(request):
    """
    Text가 있는 이미지를 받으면 이를 Text로 변환하고 약간의 오타보정을 해주고 반환한다.
    """
    print("request recognition")
    #print("request body", str(request.body.decode('utf-8'))[:90])

    data = ast.literal_eval(request.body.decode('utf-8'))

    if not os.path.isdir(os.getcwd() + 'recognition_image'):
        os.mkdir(os.getcwd() + 'recognition_image')

    # 이미지 저장 절대 경로
    imPath = os.getcwd() + '/recognition_image/' + \
        str(data['id']) + str(rd.randint(1, 1000) * rd.randint(1, 1000)) + '/'
    os.mkdir(imPath)

    if type(data["crop_image"]) is not list:
        image = base64.b64decode(data["crop_image"].split(',')[1])
        image = Image.open(io.BytesIO(image))
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cv2.imwrite(imPath + '0.png', image)

        words, scores = demo.recognition(image_folder=imPath)
    else:
        results = Queue()
        p_recognition = Process(
            target=rare_recogition_process, args=(imPath, results))
        p_recognition.start()

        for i in range(len(data["crop_image"])):
            image = base64.b64decode(data["crop_image"][i].split(',')[1])
            image = Image.open(io.BytesIO(image))
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            cv2.imwrite(imPath + str(i) + '.png', image)

        p_recognition.join()
        words, scores = results.get()

    for i in range(len(words)):
        os.remove(imPath + str(i) + '.png')
    os.rmdir(imPath)

    response = JsonResponse({'label': words, 'prob': scores})
    response['Access-Control-Allow-Origin'] = '*'
    response['Access-Control-Allow-Methods'] = 'POST'
    response['Access-Control-Allow-Age'] = '3600'
    response['Access-Control-Allow-Headers'] = 'Origin,Accept,X-Requested-With,Content-Type,Access-Control-Request-Method,Access-Control-Request-Headers,Authorization'

    return response


@csrf_exempt
def recognition_thread(request):
    """
    Text가 있는 이미지를 받으면 이를 Text로 변환하고 약간의 오타보정을 해주고 반환한다.
    """
    print("request recognition_thread")
    #print("request body", str(request.body.decode('utf-8'))[:90])

    data = ast.literal_eval(request.body.decode('utf-8'))

    if not os.path.isdir(os.getcwd() + 'recognition_image'):
        os.mkdir(os.getcwd() + 'recognition_image')

    # 이미지 저장 절대 경로
    imPath = os.getcwd() + '/recognition_image/' + \
        str(data['id']) + str(rd.randint(1, 1000) * rd.randint(1, 1000)) + '/'
    os.mkdir(imPath)

    if type(data["crop_image"]) is not list:
        image = base64.b64decode(data["crop_image"].split(',')[1])
        image = Image.open(io.BytesIO(image))
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cv2.imwrite(imPath + '0.png', image)

        words, scores = demo.recognition(image_folder=imPath)
    else:
        results = list()
        p_recognition = Thread(
            target=rare_recogition_thread, args=(imPath, results))
        p_recognition.start()

        for i in range(len(data["crop_image"])):
            image = base64.b64decode(data["crop_image"][i].split(',')[1])
            image = Image.open(io.BytesIO(image))
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            cv2.imwrite(imPath + str(i) + '.png', image)

        p_recognition.join()
        words, scores = results[0]

    for i in range(len(words)):
        os.remove(imPath + str(i) + '.png')
    os.rmdir(imPath)

    response = JsonResponse({'label': words, 'prob': scores})
    response['Access-Control-Allow-Origin'] = '*'
    response['Access-Control-Allow-Methods'] = 'POST'
    response['Access-Control-Allow-Age'] = '3600'
    response['Access-Control-Allow-Headers'] = 'Origin,Accept,X-Requested-With,Content-Type,Access-Control-Request-Method,Access-Control-Request-Headers,Authorization'

    return response
