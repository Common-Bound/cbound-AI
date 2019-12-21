from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .model import Model, extract_feature
import ast

@csrf_exempt
def predict(request):
	print("request inspection predict")
	print("request body: {}".format(str(request.body.decode('utf-8'))[:80]))

	data = ast.literal_eval(request.body.decode('utf-8'))
	features = extract_feature(data)
	
	model = Model()
	preds = model.predict(features)

	for i, pred in enumerate(preds.T[0]):
		print("{}: {}".format(i, pred))
		data['meta']['crop_image'][i]['region_attributes']['reliability'] = float(pred)

	response = JsonResponse(data)
	response['Access-Control-Allow-Origin'] = '*'
	response['Access-Control-Allow-Methods'] = 'POST'
	response['Access-Control-Allow-Age'] = '3600'
	response['Access-Control-Allow-Headers'] = 'Origin,Accept,X-Requested-With,Content-Type,Access-Control-Request-Method,Access-Control-Request-Headers,Authorization'

	return response