# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [x] Create a git repository (M5)
* [x] Make sure that all team members have write access to the GitHub repository (M5)
* [x] Create a dedicated environment for you project to keep track of your packages (M2)
* [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [x] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [x] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)
* [x] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [x] Do a bit of code typing and remember to document essential parts of your code (M7)
* [x] Setup version control for your data or part of your data (M8)
* [x] Add command line interfaces and project commands to your code where it makes sense (M9)
* [x] Construct one or multiple docker files for your code (M10)
* [x] Build the docker files locally and make sure they work as intended (M10)
* [x] Write one or multiple configurations files for your experiments (M11)
* [x] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [x] Use profiling to optimize your code (M12)
* [x] Use logging to log important events in your code (M14)
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [x] Consider running a hyperparameter optimization sweep (M14)
* [x] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [x] Write unit tests related to the data part of your code (M16)
* [x] Write unit tests related to model construction and or model training (M16)
* [x] Calculate the code coverage (M16)
* [x] Get some continuous integration running on the GitHub repository (M17)
* [x] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [x] Add a linting step to your continuous integration (M17)
* [x] Add pre-commit hooks to your version control setup (M18)
* [x] Add a continues workflow that triggers when data changes (M19)
* [x] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [x] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [x] Create a trigger workflow for automatically building your docker images (M21)
* [x] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [x] Create a FastAPI application that can do inference using your model (M22)
* [ ] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [x] Write API tests for your application and setup continues integration for these (M24)
* [x] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [ ] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [x] Revisit your initial project description. Did the project turn out as you wanted?
* [x] Create an architectural diagram over your MLOps pipeline
* [x] Make sure all group members have an understanding about all parts of the project
* [x] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:


29


### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:


s242186, s243280, s242906, s241925, s243299


### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

 For this project, we have chosen to use the TIMM framework for Computer Vision. This framework will allow us to construct and experiment with state-of-the-art deep learning models for our task. As part of the project, we will set up the TIMM framework within our environment. We plan to begin by using pre-trained models on our data and then explore ways to further enhance their performance. It was particularly helpful in our project for quickly experimenting with different model backbones and focusing more on pipeline optimization rather than model construction. Overall, timm was a valuable addition to our workflow and contributed to the successful completion of the project.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:


We managed dependencies using a requirements.txt file to list essential libraries and requirements_dev.txt for development tools. To replicate the environment, a new team member would clone the repository, create and activate a virtual environment (conda), and install the dependencies using pip install -r requirements.txt. Alternatively, we can use the Docker setup provided, which contains the entire environment pre-configured, ensuring consistency and preventing dependency conflicts.


### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:


Our project was initialized using the cookiecutter template, which provided a structured framework for organizing our code. The overall structure includes key folders such as src/ for the main source code, tests/ for unit and integration tests, data/ for raw and processed datasets, models/ for saved models, and configs/ for configuration files.

We focused on filling the essential folders: src/, where we implemented data preprocessing, training, and evaluation scripts; configs/, to manage hyperparameters and pipeline settings; and tests/, to ensure code reliability. We have also added some folder like Wandb or Logs. Some folders, like docs/ and notebooks/, were not heavily utilized as they were not critical for the pipeline. This structure ensured modularity, clarity, and ease of collaboration.


### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:


We used extensive docstrings on important functions as well as typing for documentation. We also used ruff for linting to align with the pep8 style guidelines. Besides that, we also implemented formatting rules and a ruff checker as pre-commit hooker. Clearly, this is particulary beneficial in larger projects, since many people with different backgrounds are working on the same code files, so readability and consistency are key factors for having clean project source code. For example, when working on some new features and the developer wants to integrate this new feature in the main codebase, he most likely will do a pull request to let his coworkers peer-review the latest changes. So it is crucial that everyone understands newly written code easily and is thus able to review.


## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:


In total we have implemented 4 tests. Primarily we are testing the data.py and train.py files, those include our whole deep learning pipeline. In particular, we tested if the model has the correct number of classes and the correct output shape, if the training function outputs a model.pth file in the correct directory and if the evaluation function gives out a feasible number as accuracy.


### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

The total code coverage is 70%. In the figure below, we can see the code coverage for all the source code.

![alt text](image.png)

A code coverage of 100% or close is a good indicator but it does not gives us the certainty that the code is free of errors. Code coverage simply counts which lines of code are executed during testing, not if they are working correctly or are capable of handling all situations. Errors like integration problems or specific environment behavior might still arise if they were not fully considered during testing. High coverage is valuable, but it doesn’t account for logic errors and performance issues. To ensure reliability, good test quality, real-world testing and thorough reviews are necessary alongside high coverage.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:


We made use of both branches and pull requests in our project. In our group we decided to not split up the branches member-wise, but rather create branches for the features we are working on. That means, every time someone is working on a new feature, this person creates a new branch to work on and afterwards come up with a pull request to merge and integrate the feature branch to the main branch. The decision to work on different branches came up during the project and was adapted since then. We also investigated that using feature branches instead of member branches is a good habit to keep in mind for future projects.


### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

Yes, we used DVC to manage data in our project. By integrating DVC, we were able to version control our datasets effectively, ensuring that every experiment could be traced back to the exact version of the data it used. This was especially valuable for maintaining consistency across team members and reproducing results. The use of remote storage, such as Google Cloud Storage, allowed us to collaborate seamlessly without manually transferring large files. Additionally, DVC provided a clear history of changes to the data, which helped us identify and debug issues related to dataset modifications. Overall, DVC streamlined data management and significantly improved the reproducibility and efficiency of our workflow.


### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

The primary CI workflow is Python application, focused on code quality, reliability and compatibility. It's implemented in Github Actions and triggered on pushes and pull requests to the main branch. To ensure compatibility, it tests across multiple operating systems as Windows, Ubuntu and MacOs and Pyhton versions (3.10, 3.11 and 3.12). The pipeline includes setting up Python with dependency caching and linting with flake8 to check and upgrade code quality. It also executes tests with pytest and measures code coverage. To run these tests, the data is downloaded from our data bucket and uses the WANDB_API_KEY environment variable through Github secrets to avoid issues with logging in wandb.

Additionally, we have a workflow to handle updates to the model, Model Registry Workflow, and updates to the dataset, Data Change Workflow. The first one is triggered by changes in the models directory by retraining and evaluating the updated model. The second one monitors changes in the data directory and runs the data pipeline (data.py). As the previous workflow, these also test across multiple operating systems and Pyhton versions.

Link to GitHub workflows: https://github.com/mariatmvendas/mlops_project/tree/main/.github/workflows

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We used YAML config files for default hyperparameters and command-line arguments for flexibility. Hydra made it possible to load the hyperparameters from the YAML files and to switch between files. Typer provides a CL interface and CL arguments could override the files, changing the default value of any of the hyperparameters. It also allows us to change the YAML file that we want to use by parsing --config <name_of_config_file>.

By running python src/mlops_project/train.py train --config exp1,
the hyperparameters will be the ones defined in the exp1.yaml file.

By running python src/mlops_project/train.py train --batch-size 12,
the hyperparameters will be the ones defined in the config.yaml file, except for batch_size that will be 12.

For running a hyperparameter optimization sweep, run 'wandb sweep configs/sweep.yaml' followed by 'wandb agent <sweep_id>' with the sweep_id inserted, which was returned by the first command.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

To ensure reproducibility, we first used YAML files to define all hyperparameters, including paths to data, batch size, learning rate, number of epochs, and model path, which Hydra then loaded. Weights and Bias (WB) was then integrated for experiment tracking and logging. It logs key metrics such as training loss and validation accuracy during the train and evaluate functions and the hyperparameters of each run, so we can compare experiments. WB also saves the overrides from the command line, resulting in no information loss and traceable changes. Additionally, trained models were saved as WB artifacts, providing a record of the models and their corresponding results of each experiment version.


### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:


We leveraged wandb to track and log our experiments and to present resulting graphs automatically.

![Figure](figures/WandB_log_train_loss.png)

As seen in the first image, we have tracked the training loss over time in order to verify convergence of the loss curve. This tracking was clearly executed during model training.

![Figure](figures/WandB_log_validation_accuracy.png)

On the second image, the accuracy of the trained model is depicted which is evaluated on the validation data set. This metric is automatically getting logged when executing the evaluation function. The validation accuracy is important to evaluate the actual performance of the model, since it is impractical to evaluate on the training dataset for obvious reasons. The first and the second image were created with the whole dataset.
Moreover, we created an artifact for the validation accuracy and tracked it on wandb. This can be leveraged for further experiments.

![Figure](figures/WandB_hyperparameter_sweep.png)

Last but not least, we performed a hyperparameter optimization sweep on wandb to see which hyperparameter configuration fits the best for our model. This can be seen on the third image 'WandB hyperparameter sweep.png' with all the sweep configurations and the logged training loss as a performance metric. For feasible runtimes on our local machines, we trained with the sweep agent only on approximately one quarter of the data set.


### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

For our project we developed one docker image for both training and evaluation of our model. First we created a dockerfile 'train.dockerfile', from which we can build our docker image with the bash command: 'docker build -f dockerfiles/train.dockerfile . -t train:latest'. Then we are able to build different containers for both training (bash command: 'docker run --name mlops_container_train -v $(pwd)/models:/models train:latest train') and evaluation (bash command: 'docker run --name mlops_container_evaluate -v $(pwd)/data:/data -v $(pwd)/models:/models train:latest evaluate'). As you can see, the containerization of the important functions also includes mounting the data or model to the respective container. The dockerfile for training and evalutation can be found in the 'dockerfiles/train.dockerfile' directory.


### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:


When debugging experiments, we used a combination of standard debugging tools and profiling techniques. Initial debugging was handled using Visual Studio Code, which allowed us to step through code line-by-line and inspect variables. For deeper performance insights, we utilized a dedicated profiling script (profiling.py) that automates profiling with cProfile and analyzes results with pstats or visualizes them using SnakeViz. This allowed us to identify bottlenecks and optimize critical sections of the code.

While profiling revealed opportunities for improvement, most computations in our project were abstracted through frameworks like PyTorch Lightning, making further optimizations challenging. Despite this, our profiling efforts ensured that the code runs efficiently and identified areas that could be improved in the future.


## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

We mainly used the following services: Compute Engine, Cloud Build, Artifact Registry and Cloud Storage. Cloud Storage was used to store our project data. Cloud Build and Artifact Registry were used to manage our containerized workflows. Cloud Build handled the creation of Docker images, while Artifact Registry stored these images. Finally, Compute Engine was used to run our machine learning model.


### Question 18

> **The backbone of GCP is the Compute engine. Explain how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

We used the Compute Engine to run our machine learning model on a virtual machine. We created a e2-medium instance with 2vCPU and 4GB memory. After ensuring that PyTorch was installed, we were able to execute our model in the cloud. This setup allowed us to efficiently train the model while managing the computational resources needed for the task.


### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

We stored both our raw and processed data in our GCP bucket.

![Figure](figures/gcloud_bucket.png)

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

We just stored one docker image in our GCP artifact registry.

![Figure](figures/gcloud_image.png)

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

![Figure](figures/gcloud_buildhistory.png)


### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:


We managed to train our model in the cloud using the Compute Engine. We did this by first creating an appropiate VM that had PyTorch preinstalled. Then, we logged into this VM and checked that Pytorch was indeed installed. After this, we cloned our GitHub repository, we installed the necessary dependencies by running `pip3 install -r requirements.txt` and downloaded the data from our GCS bucket. Finally, we called our train.py file to train our model on the cloud.


## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

We successfully implemented an API for our model using FastAPI. The API is designed to perform a single task: predicting the label of an image. The implementation is located in src.mlops_project.api.py. The API has two endpoints:

POST /inference_satellite/: Accepts an image file and returns the predicted label as a JSON response.
GET /: Provides clear instructions on how to use the API, including an example curl command.

To ensure robustness, the API handles file uploads, converts images to the required format, and executes the classification using a pre-trained model via an external script (inference.py). Temporary files are managed efficiently, and error handling ensures that users receive meaningful feedback if something goes wrong.

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

We successfully deployed our API locally using FastAPI. The API was wrapped into an application located in src.mlops_project.api.py. For local deployment, we used Uvicorn as the ASGI server. The command to start the server is:
uvicorn src.mlops_project.api:app --reload --host 127.0.0.1 --port 8000
Once deployed, the API can be invoked using a POST request to the /inference_satellite/ endpoint. Users can upload an image file for classification and receive the predicted label as a JSON response. For example:
curl -X POST "http://127.0.0.1:8000/inference_satellite/" \
     -H "Content-Type: multipart/form-data" \
     -F "data=@path_to_image.jpg"
The API is designed to provide instructions at the root endpoint (GET /), making it user-friendly and easy to invoke. While we have tested local deployment successfully, further plans include deploying the API to the cloud for wider accessibility.

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

For unit testing, we used the TestClient from FastAPI to test the functionality of the root endpoint (GET /). The unit test verified that the endpoint returned the correct HTTP status code, expected JSON response. The root endpoint returned a 200 status code along with instructions on how to use the API.

For load testing, we used Locust to simulate concurrent users accessing the API. The load tests focused on both endpoints, with the root endpoint handling basic requests and the inference endpoint processing image uploads.

At 6 users it starts failing. At the maximum of 10 usesrs it has 19% failures

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

We did not manage to implement monitoring yet, but we would like to implement it to track the model’s performance over time, particularly its classification accuracy. By continuously monitoring, we could quickly detect any performance degradation and address issues like shifts in image characteristics or changes in data distribution.


## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

We spent 18% of the available credits during development. The most expensive service was the creation of the Docker image. Working in the cloud was easier than we initially expected. It was nice to learn how to upload data to the cloud and train a model there. Once the steps were understood, the process became quite straightforward and efficient.

![Figure](figures/gcloud_credits.png)

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:


We implemented a pre-commit hook that automatically checks for styling and formatting of the provided files before actually committing it, this helps us to comply with good coding style guidelines.
Furthermore, we create loggings of model parameters and the used device and save them in a log file for debugging. This can be found in the 'logs/log_debug.log' directory.


### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:


The diagram illustrates the Machine Learning Operations Pipeline, starting with the user pulling a pre-built Docker image or cloning the project directory from the GitHub repository to their local environment. In the DEV environment, tools like PyTorch, PyTorch Lightning, Weights & Biases, and Hydra are used for model development, configuration management, and experiment tracking. Data is managed and versioned using DVC in combination with a Local Data Storage. Once changes are made, code, data, and models are added, committed, and pushed to GitHub, triggering GitHub Actions workflows to run tests, build containers, and ensure continuous integration.

The CI/CD pipeline leverages Docker for containerization and deploys models to Google Cloud Platform (GCP) using Google Cloud Deploy. The trained model is exposed via a FastAPI application, allowing users to query predictions through a Query Server connected to GCP. This pipeline ensures reproducibility, scalability, and automation, while tracking and updating datasets and models seamlessly for continuous improvement. It integrates local, cloud, and CI/CD environments to maintain an efficient and reliable workflow.

![Figure](figures/MLO_Pipeline.png)


### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

At the beginning we were a bit undecided on what dataset and framework to use. Sadly we came across a a fair dataset to work on and checking previous works we identified Timm as framework to use. Later we tried to quickly implement a training and evaluations script. This was good on one hand because on the first day we already implemented the model part but on the other hand bad because we implemetned everything fast and we needed to refactor the code the day after. The following days didn't have any major problem.
Overall, it was difficult to keep the code always executable, even when different developers were integrating modules over and over again.

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:


Student s242186 was in charge of good coding practices like linting, typing and docstrings. Additionally, the student developed docker files, images and containers. The student also was in charge of logging with log files, wandb logging and artefacts and wandb hyperparameter sweeps. In addition, this student created unit tests for model, training and evaluation and calculated code coverage and was in charge for implementing pre-commit hooks.

Student s243280 was in charge of all the aspects related to cloud computing. Additionally, the student participated in the development of the files needed for data processing as well as the development of the model training and evaluation files. The student also worked on the profiling and monitoring of the project done.

Student s243299 contributed to the development of the data processing pipeline, ensuring the proper handling and transformation of datasets. The student assisted in implementing logging in the code to monitor model performance and track results effectively, promoting transparency and facilitating better analysis of experiments. Additionally, the student played a significant role in designing the project's workflows, as well as creating the project's architectural pipeline and addressing boilerplate code, which established a clean and efficient project structure.

Student s241925 contributed to writing the scripts to train and evaluate the model. Added command line interfaces. Contributed to write unite tests. Created Fast API application. Wrote API tests. Did load testing.

All member contributed to code by 20%.

Furthermore, we have used ChatGPT to help write our code. Additionally, we used GitHub Copilot to help write some of our code.
