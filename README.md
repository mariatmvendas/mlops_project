# mlops_project

mlops project group29

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```

## Project description

### Goal
The overall goal of the project is to apply the concepts and techniques we have learned in the course to a machine learning problem. The chosen problem is the classification of satellite images. We aim to perform a multi-class classification task with these images.

### Framework
For this project, we have chosen to use the TIMM framework for Computer Vision. This framework will allow us to construct and experiment with state-of-the-art deep learning models for our task. As part of the project, we will set up the TIMM framework within our environment. We plan to begin by using pre-trained models on our data and then explore ways to further enhance their performance.

### Data
The dataset we have selected is the Satellite Image Classification dataset from Kaggle. It consists of high-resolution satellite images representing 4 land cover classes, including green areas, water bodies, cloudy skies and desert areas. The dataset contains 5631 labeled images.

### Models
We aim to perform a multi-class image classification task using CNN-based architectures. Using the TIMM framework, we plan to explore models such as EfficientNet, ConvNeXt, and ResNet. By experimenting with different architectures, we intend to evaluate their performance on satellite imagery and identify the most effective model for this task.



Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

---------------------------------------------------
download from kaggle https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification
take the four folders and put them in data/raw
rename green area to forest