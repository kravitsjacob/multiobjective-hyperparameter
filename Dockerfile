FROM continuumio/miniconda3
RUN conda install numpy=1.20.2
RUN conda install pandas=1.2.4
RUN conda install scikit-learn=0.24.2
RUN pip install pymoo==0.4.2.2
RUN pip install hiplot==0.1.24
WORKDIR /app
COPY ["main.py", "/app/"]
ENTRYPOINT ["python", "-u", "main.py"]