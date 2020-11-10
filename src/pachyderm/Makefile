setup:
	echo "Creating the rawdata initial repository and the flow of pipelines used to train our captioning model"
	pachctl create repo rawdata
	pachctl create pipeline -f images.json
	pachctl create pipeline -f features.json
	pachctl create pipeline -f consolidate.json
	pachctl create pipeline -f model.json
	pachctl create repo savedmodel
	pachctl create branch savedmodel@master

predict:
	echo "Creating the inpredict initial repository and the pipeline used to predict captions out of images"
	pachctl create repo inpredict
	pachctl create pipeline -f predict.json

deploy:
	pachctl copy file -o model@master:/saved/saved_model.h5 savedmodel@master:/saved_model.h5
