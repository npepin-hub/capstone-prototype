# Pachyderm's foundational concepts: tutorial 201
In this tutorial, we will walk you through the main steps of the migration of an [existing ML](https://github.com/nadegepepin/capstone-prototype) project to Pachyderm's platform. We will add a couple of **tips and tricks** along the way. 

It shoud take you 20 minutes to get there.

## Prerequisite
Ideally, you have successfully completed the [OpenCV tutorial](https://docs.pachyderm.com/latest/getting_started/beginner_tutorial/) and are ready to keep exploring Pachyderm.  If not, make sure that you have: 
- a Pachyderm running [locally](https://docs.pachyderm.com/latest/getting_started/local_installation/) or a workspace on [Pachyderm Hub](https://hub.pachyderm.com/orgs/1008/workspaces)
- pachctl command-line installed, and your context created (ie: you are logged in)
- New to pachctl? Install the [autocompletion extention](https://docs.pachyderm.com/latest/getting_started/install-pachctl-completion/) of pachctl. 

## A little bit of context 
- As a starting point, we are using a very simple ResNet/LSTM 'inject and merge' captioning model trainable on [google conceptual caption dataset](https://ai.google.com/research/ConceptualCaptions)  (~3.3 M images) 
- We have reproduced its data extraction/transformation, the training of its model, and its prediction using Pachyderm's platform. 

Just like any ML project, you will find 2 sets of pipelines here:
- A training pipeline that **extracts** and **transforms** our data to **train** our model. In brief:
    - we feed in a big tsv file of 3.3M caption/URL lines  
    - we retrieve each image and extract its features
    - we feed those features along with their caption to our model 
    - we get a final trained *saved_model.h5* as a result
- An inference pipeline that will use our trained model to **predict** a caption out of an image.

Our DAGs look like this :

![target pipelines.png](https://www.dropbox.com/s/az21yz0few18i8d/target%20pipelines.png?dl=0&raw=1)

    
Note that the training pipelines have been given self-explanatory names. The "consolidate" pipeline is nothing more than a [shuffle pipeline](https://github.com/pachyderm/pachyderm/tree/master/examples/shuffle) (an intermediate processing step that aggregates/joins our features and caption data to feed our model).


Optional: More details about the training pipelines (It really is simpler than it looks):
![detailed pipelinespng.png](https://www.dropbox.com/s/cyicllgpg3x086k/detailed%20pipelinespng.png?dl=0&raw=1)
## Training phase run-through
1. Let's start by cloning https://github.com/nadegepepin/capstone-prototype
2. Training pipeline setup: In pachyderm's root directory `src/pachyderm`, let's run the **setup** target of our `Makefile`:

        make setup
    or run the following commands:
    
        pachctl create repo rawdata
        pachctl create pipeline -f images.json
        pachctl create pipeline -f features.json
        pachctl create pipeline -f consolidate.json
        pachctl create pipeline -f model.json

    A quick check at your dashboard will give you a visual representation of your DAG. 
3. You can also check your list of repositories and pipelines:

            pachctl list repo
    ![Screen Shot 2020-10-29 at 5.18.16 PM.png](https://www.dropbox.com/s/8ihrsg9juyx3kyl/Screen%20Shot%202020-10-29%20at%205.18.16%20PM.png?dl=0&raw=1)
    
   Note that besides the rawdata repository, 4 additional output repo were created, one for each pipeline with the same name.

            pachctl list pipeline
    ![Screen Shot 2020-10-29 at 5.21.46 PM.png](https://www.dropbox.com/s/clckt8a29gry6xo/Screen%20Shot%202020-10-29%20at%205.21.46%20PM.png?dl=0&raw=1)

All of our pipelines are running, ready for the execution of the jobs triggered by an input commit in their entry Repo. 

Did you notice the `standby` state of the pipeline 'images'?  [A pipeline in standby will have no pods running and thus will consume no resources](https://docs.pachyderm.com/latest/reference/pipeline_spec/). This pipeline will activate when datum(s) are commited in its input Repo.
It is as simple as a `"stanby": true` line in your pipeline .json file. See [images.json](https://github.com/nadegepepin/capstone-prototype/blob/master/src/pachyderm/images.json).

4. Let's inject our training data into our 'rawdata' Repository and see what happens. In the `./testdata/` directory, run:

        pachctl put file rawdata@master -f test.tsv --split line --target-file-datums 5
    Note that we used the `--split  --target-file-datums` [option](https://docs.pachyderm.com/latest/how-tos/splitting-data/adjusting_data_processing_w_split/). We are dividing our test.tsv file into chunks of 5 lines. Because our file contains 100 lines total, this will output 20 Datum(s) of 5 lines each in our rawdata Repository. The core use of this split is the ability to **parallelize the job processing**.
    You can check your job list by running:
    
        pachctl list job
    ![Screen Shot 2020-10-29 at 8.36.09 PM.png](https://www.dropbox.com/s/uv6g8tug75r508n/Screen%20Shot%202020-10-29%20at%208.36.09%20PM.png?dl=0&raw=1)
    
    Then look into the logs of a specific job:
    
        pachctl logs -j 2bd5a523a9ad4ea38c4c1adaed24a5d1
    
    or the master logs of a given pipeline: 
    
        pachctl logs -p images --master

5. Let's now have a look at the content of our 'rawdata' Repo through the dashboard:

![Screen Shot 2020-10-29 at 5.42.27 PM.png](https://www.dropbox.com/s/2mbsnlv0d9kk68y/Screen%20Shot%202020-10-29%20at%205.42.27%20PM.png?dl=0&raw=1) 

You should see the list of your 20 Datums in `rawdata/test.tsv/`. One click on a file displays its content. 

6. Have a look at the commits in rawdata:

        pachctl list commit rawdata
        
    ![Screen Shot 2020-10-29 at 5.48.09 PM.png](https://www.dropbox.com/s/jmovqhkw3jgyfib/Screen%20Shot%202020-10-29%20at%205.48.09%20PM.png?dl=0&raw=1)
    
Notice that all 20 Datum(s) are in **one** commit. 
This commit will trigger the first job in our first-in-line pipeline (ie: 'images'). 
The Datum(s) it contains (20 in our case) will be processed and an output commit created in the output Repo of the 'images' pipeline (conveniently called... 'images'). 
The output commit will trigger the following job in the next-in-line pipeline (ie:'features'). 

You got the idea... All the way to our final trained 'saved_model.h5' model.

7. Spotlight on [parallelism](https://docs.pachyderm.com/latest/reference/pipeline_spec/#parallelism-spec-optional): It takes a line.
In [images.json](https://github.com/nadegepepin/capstone-prototype/blob/master/src/pachyderm/images.json), we have used the constant field `parallelism_spec` to set the number of workers to 20 for this job.

    ![Screen Shot 2020-10-29 at 10.12.21 PM.png](https://www.dropbox.com/s/ogeesa2n0t6a6y1/Screen%20Shot%202020-10-29%20at%2010.12.21%20PM.png?dl=0&raw=1)
8. Now, let's have a closer look at how our data are consolidated in [consolidate.json](https://github.com/nadegepepin/capstone-prototype/blob/master/src/pachyderm/consolidate.json) pipeline:
To prepare our training set with the tuples (caption, features) required, we are combining files that reside in 2 separate Pachyderm repositories ('rawdata' and 'features') and match a particular naming pattern using a [Join](https://docs.pachyderm.com/latest/concepts/pipeline-concepts/datum/join/). See the "glob" patterns below.

    ![Screen Shot 2020-10-29 at 6.07.44 PM.png](https://www.dropbox.com/s/h3jxoz5ds1zr9eo/Screen%20Shot%202020-10-29%20at%206.07.44%20PM.png?dl=0&raw=1)

    To highlights how the join works, we have broken down the Repos' file hierarchy of each pipeline. 
    
![Glob _ Join explained.png](https://www.dropbox.com/s/60pwocwi5aavkiy/Glob%20_%20Join%20explained.png?dl=0&raw=1)


In our case, we are joining each initial Datum file in 'rawdata' (renamed captions.tsv) with the entire content of the directory of the same name in 'features'. 
For each directory in the 'consolidate' Repo, the generator will read each captions.tsv and couple it with its associated features.
See also the `consolidate` function in [pachyderm.py](https://github.com/nadegepepin/capstone-prototype/blob/master/src/pachyderm/pachydermtest.py) for more details.
    
 **Useful tip** - To test out the output of your join, have a look at the item 2 of this [lifesaving list](https://medium.com/@jimmymwhitaker/5-tips-and-tricks-scaling-ml-with-pachyderm-b6ee045ff800). The added .json file you will need is in your pachydern directory: [pachyderm-inspect-input.json](https://github.com/nadegepepin/capstone-prototype/blob/master/src/pachyderm/pachyderm-inspect-input.json) 
9. Our trained model: Finally...
If all your jobs have run successfully, which they should, you will find a `saved_model.h5` in the Repo 'model' in `/saved/`. This is the last commit of this training pipeline. 

## Predict phase run-through

You have made it through the training phase! 
Let's try a prediction. 
1. Predict pipeline setup: In pachyderm's root directory, let's run the **predict** target of our `Makefile`:

        make predict
    or run the following commands:
    
    	pachctl create repo inpredict
    	pachctl create pipeline -f predict.json
    Your inference pipeline is ready.  Check it out!     

        pachctl list pipeline
        pachctl list repo

2. Time to give your model a random image
In the `./testdata/` dirctory, run:

        pachctl put file inpredict@master -f image.png
        
    The commit of your image triggered a job in your predict pipeline. 
    
        pachctl list job
     Check your dashboard, the 'predict' Repo should contain a `image.png.txt` file. 
     You have your caption! Yes, it looks funky. Remember, we have trained our model on a minimal set of data. The predicted caption produced is logically off. It is ok. That was not the point of this training.
3. Final note: 
    We used a [cross input](https://docs.pachyderm.com/latest/concepts/pipeline-concepts/datum/cross-union/) to access both the image we want to be captioned ('inpredict' Repo) and the latest trained model ('model' Repo). ![Screen Shot 2020-10-29 at 9.38.49 PM.png](https://www.dropbox.com/s/ipjho4lzakr4ao4/Screen%20Shot%202020-10-29%20at%209.38.49%20PM.png?dl=0&raw=1)

## You might find this useful
Enable stats in development:

The `"enable_stats": true` parameter [turns on statistics tracking for a pipeline](https://docs.pachyderm.com/latest/reference/pipeline_spec/#enable-stats-optional). Your pipeline will commit datum processing information to a special branch ("stats") in its output repo. It stores useful information about the datum it processes (timing, size, **logs**...). You can access those stats in your dashboard.
![Screen Shot 2020-10-29 at 10.02.26 PM.png](https://www.dropbox.com/s/6n20ft0lphte3lo/Screen%20Shot%202020-10-29%20at%2010.02.26%20PM.png?dl=0&raw=1)

## After Thoughts
We are done transfering our ML pipelines to Pachyderm. In our case, we let go of:
- our multi-threading code 
- the split of our data and jobs over 30 machines to boost the data extraction and transformation 
- the manual sequencing of our scripts to run all 3 phases (request images, extract features, and train)

In other words, we have kept the DataScience and outsourced the MLOps. Yay.
## What's next?
[This!](https://github.com/pachyderm/pachyderm/tree/master/examples/deferred_processing). Deferred processing is your next stop.


