# I About Project
This is the accompanying repository to [this blog post](https://kravitsjacob.github.io/multiobjective-hyperparameter/) about taking a multi-objective approach to tuning machine learning model hyperparameters. The accompanying script provides a very simple example of this method applied in Python 3. However, the concepts and methods could easily adapted to many other machine-learning problems and other methods of implementation. As for running this script, I recommmend using Docker to ensure consistent results with the blog post. However, it should run in any environmental with Python 3 and all the proper dependencies installed. 

# II Running this code in Docker
1. Clone this repository using command prompt/terminal $git clone https://github.com/kravitsjacob/multiobjective-hyperparameter 
2. Change to the current working directory using command prompt/terminal $ cd <insert_path_to_\multiobjective-hyperparameter>
3. Download and Run Docker Desktop. For more information on Docker visit [here](https://docs.docker.com/desktop/). To ensure 
that it is installed correctly go to the command prompt/terminal and enter $ docker --version
4. Build the docker image by running $docker build --tag mohyper .
5. Run the image using $docker run mohyper

# Running this code in Virtual Environment
1. Clone this repository using command prompt/terminal $git clone https://github.com/kravitsjacob/multiobjective-hyperparameter 
2. Change to the current working directory using command prompt/terminal $ cd <insert_path_to_\multiobjective-hyperparameter>
3. Make sure you have Python 3 installed on your computer. This can be downloaded [here](https://www.python.org/downloads/). To ensure 
that it is installed correctly go to the command prompt/terminal and enter $ python --version
4. Set up and activate a virtual environment by following instructions [here](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/). 
7. Install dependcies using requirements.txt found [here](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#using-requirements-files)
8. Run script using $python main.py
