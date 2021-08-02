# I About Project
This is the accompanying repository to [this blog post](https://kravitsjacob.github.io/multiobjective-hyperparameter/) about taking a multi-objective approach to tuning machine learning model hyperparameters. The accompanying script provides a very simple example of this method applied in Python. However, the concepts and methods could easily be adapted to many other machine-learning problems and other methods of implementation. As for running this script, Docker is the recommended procedure to ensure consistent results with the blog post. However, the script should run in any environment with Python 3 and all the proper dependencies installed. 

# II Running this code in Docker
1. Clone this repository using command prompt/terminal $```git clone https://github.com/kravitsjacob/multiobjective-hyperparameter``` 
2. Change to the current working directory using command prompt/terminal $```cd <insert_path_to_data>\multiobjective-hyperparameter```
3. Download and run Docker Desktop. For more information on Docker visit [here](https://docs.docker.com/desktop/). To ensure 
that it is installed correctly go to the command prompt/terminal and enter $ docker --version
4. Build the docker image by running $```docker build --tag mohyper .```
5. Run the image and mount the folder so you can retrieve generated files $```docker run -v <insert_path_to_data>\multiobjective-hyperparameter:/app mohyper``` 

# III Running this code in a Virtual Environment
1. Clone this repository using command prompt/terminal $```git clone https://github.com/kravitsjacob/multiobjective-hyperparameter```
2. Change to the current working directory using command prompt/terminal $```cd <insert_path_to_data>\multiobjective-hyperparameter```
3. Make sure you have Python 3 installed on your computer. This can be downloaded [here](https://www.python.org/downloads/). To ensure 
that it is installed correctly go to the command prompt/terminal and enter $```python --version```
4. Set up and activate a virtual environment by following instructions [here](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/). 
7. Install dependencies using requirements.txt found [here](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#using-requirements-files)
8. Run script using $```python main.py```

# IV Citing
To cite the methods employed in this blog post, please cite:
```bibtex
@article{kravits_screening_2021,
         title = {Screening {Tool} for {Dam} {Hazard} {Potential} {Classification} {Using} {Machine} {Learning} and {Multiobjective} {Parameter} {Tuning}},
         volume = {147},
         url = {https://ascelibrary.org/doi/abs/10.1061/%28ASCE%29WR.1943-5452.0001414},
         doi = {10.1061/(ASCE)WR.1943-5452.0001414},
         number = {10},
         journal = {Journal of Water Resources Planning and Management},
         author = {Kravits, Jacob and Kasprzyk, Joseph and Baker, Kyri and Andreadis, Konstantinos},
         year = {2021},
         note = {\_eprint: https://ascelibrary.org/doi/pdf/10.1061/\%28ASCE\%29WR.1943-5452.0001414},
         pages = {04021064},
}
```

To cite this repository or the corresponding blog post, please cite:
```bibtex
@misc{kravits_multi-objective_2021,
      title = {Multi-{Objective} {Machine} {Learning} {Hyperparameter} {Tuning} ({Without} {Explicit} {Objective} {Weighting})},
      url = {https://github.com/kravitsjacob/multiobjective-hyperparameter},
      publisher = {GitHub},
      author = {Kravits, Jacob},
      year = {2021},
      doi = {https://doi.org/10.5281/zenodo.5149306},
}
```

