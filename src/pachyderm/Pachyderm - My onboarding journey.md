# Pachyderm Hub 101 
From one initial tsv file of raw data to my trained model: A first exploration of Pachyderm's repositories, jobs, and pipelines applied to a simple "Conceptual" captioning project: https://github.com/nadegepepin/capstone-prototype. 

As a quick context, let's just say that:
- I trained a simple ResnNet/LSTM 'inject and merge' captioning model on [google conceptual caption dataset](https://ai.google.com/research/ConceptualCaptions)  (~3.3 M images) for the sake of dealing with a little more data than the traditional COCO or Flickr dataset.
- I then set to reproduce the same data extraction, tranformation, and the training of the model using [Pachyderm Hub](https://docs.pachyderm.com/latest/pachhub/pachhub_getting_started/) 's platform. 
Find [the source code of my journey](https://github.com/nadegepepin/capstone-prototype/tree/master/src/pachyderm) here. // WORK IN PROGRESS


## Training
Before we dive into the few steps that led me to the set up of my Pachyderm pipelines, here is what we have:
![capstone.png](https://www.dropbox.com/s/9q4tj3vyvhwjak7/capstone.png?dl=0&raw=1)
More details about the processing jobs used to extract/transform our data, and train our [model](https://github.com/nadegepepin/capstone-prototype/blob/master/src/model.py) on Sage Maker here :
- run-image-extraction: [Extract the images from the urls in the raw data tsv file](https://github.com/nadegepepin/ml-docker/blob/master/run-image-extraction.sh)  
- run-features-extraction: [Extract the features of those images using a pre-trained ResNet 50 ](https://github.com/nadegepepin/ml-docker/blob/master/run-features-extraction.sh)
- run-fit: [Feed the model with both caption and associated features](https://github.com/nadegepepin/ml-docker/blob/master/run-fit.sh)


Intuitively, we want to: 
- Convert each repositary in our S3 bucket into a Pachyderm [Repo](https://docs.pachyderm.com/latest/concepts/data-concepts/repo/), their content into [Datum](https://docs.pachyderm.com/latest/concepts/pipeline-concepts/datum/), and our jobs into [Pipelines](https://docs.pachyderm.com/latest/concepts/pipeline-concepts/pipeline/)
- Create/understand the flow of data that will allow us to train our model
- Train at scale
- use our predict function to generate the caption of a random image

---
DO NOT READ BEYOND THIS POINT YET
---

### 1. Deploy and configure your pachyderm cluster in Pachyderm Hub
Let's start with deploying and configuring a free 4hrs sandbox cluster by [following those steps](https://docs.pachyderm.com/latest/pachhub/pachhub_getting_started/).
In a nutshell, you will:
- Login to [PachHub](https://hub.pachyderm.com/orgs/1008/workspaces) with your GitHub account
- Create your first workspace
- Install Pachyderm command line tool ([pachctl](https://docs.pachyderm.com/latest/getting_started/local_installation/#install-pachctl))
- Connect to your workspace

You are now ready to create your first [Repository](https://docs.pachyderm.com/latest/concepts/data-concepts/repo/) and [Pipeline](https://docs.pachyderm.com/latest/concepts/pipeline-concepts/). 
### 2. Initial repository and pipeline creation
- Let's start with the [creation of the first Repository](https://docs.pachyderm.com/latest/getting_started/beginner_tutorial/#create-a-repo) '*rawdata*'. It will receive the tsv file from which our datum(s) ([^bignote]) (ie: individual caption and related url) will be extracted in step 4.
```
pachctl create repo rawdata
```   
[^bignote]: Side step here. Yes the plural of '*datum*' is '*data*'. However, '*datum(s)*' conveys the idea of a list of datum (or atomic piece of data) better. 
```
pachctl create pipeline -f ../../src/pachyderm/pipeline.json  
```    
### 3. Image build, tag, and push to DockerHub
    docker-compose build python-cli
    docker-compose build capstone-runtime
    docker image rm 190146978412.dkr.ecr.us-east-2.amazonaws.com/capstone-prototype:latest
    docker tag ml-docker_capstone-runtime:latest npepin/capstone-prototype:0.0.3
    docker push npepin/capstone-prototype:0.0.3



  
### 4. Data injection into first repopachctl
In the data repository:
```
pachctl put file rawdata@master -f test.tsv --split line --target-file-datums 5
```




