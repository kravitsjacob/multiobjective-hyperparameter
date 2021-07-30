# I About Project
This is the accompanying repository to [this blog post](https://kravitsjacob.github.io/multiobjective-hyperparameter/) about taking a multi-objective approach to tuning machine learning model hyperparameters. The accompanying script provides a very simple example of this method applied in Python 3. However, the concepts and methods could easily adapted to many other machine-learning problems and other methods of implementation. As for running this script, I recommmend using Docker to ensure consistent results with the blog post. However, it should run in any environmental with Python 3 and all the proper dependencies installed. 

# II Running this code in Docker
1. Clone this repository using command prompt/terminal $git clone https://github.com/kravitsjacob/multiobjective-hyperparameter 
2. Change to the current working directory using command prompt/terminal $ cd <insert_path_to_data>\multiobjective-hyperparameter
3. Download and Run Docker Desktop. For more information on Docker visit [here](https://docs.docker.com/desktop/). To ensure 
that it is installed correctly go to the command prompt/terminal and enter $ docker --version
4. Build the docker image by running $docker build --tag mohyper .
5. Run the image and mount the folder so you can retrieve generated files $docker run -v <insert_path_to_data>\multiobjective-hyperparameter:/app mohyper 

# III Running this code in a Virtual Environment
1. Clone this repository using command prompt/terminal $git clone https://github.com/kravitsjacob/multiobjective-hyperparameter 
2. Change to the current working directory using command prompt/terminal $ cd <insert_path_to_data>\multiobjective-hyperparameter
3. Make sure you have Python 3 installed on your computer. This can be downloaded [here](https://www.python.org/downloads/). To ensure 
that it is installed correctly go to the command prompt/terminal and enter $ python --version
4. Set up and activate a virtual environment by following instructions [here](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/). 
7. Install dependcies using requirements.txt found [here](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#using-requirements-files)
8. Run script using $python main.py

# IV Citing
To cite the methods employed in this blog post please cite:
```bibtex
@article{doi:10.1061/(ASCE)WR.1943-5452.0001414,
author = {Jacob Kravits  and Joseph Kasprzyk  and Kyri Baker  and Konstantinos Andreadis },
title = {Screening Tool for Dam Hazard Potential Classification Using Machine Learning and Multiobjective Parameter Tuning},
journal = {Journal of Water Resources Planning and Management},
volume = {147},
number = {10},
pages = {04021064},
year = {2021},
doi = {10.1061/(ASCE)WR.1943-5452.0001414},
URL = {https://ascelibrary.org/doi/abs/10.1061/%28ASCE%29WR.1943-5452.0001414},
eprint = {https://ascelibrary.org/doi/pdf/10.1061/%28ASCE%29WR.1943-5452.0001414}
}
```

To cite the blog post or this repository please cite:
```bibtex
@misc{author = {Kravits, Jacob},
      title = {{Multi-Objective Machine Learning Hyperparameter Tuning (Without Explicit Objective Weighting)}},
      year = {2021},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/kravitsjacob/multiobjective-hyperparameter}},
}
```

