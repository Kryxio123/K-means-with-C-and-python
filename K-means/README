K-means in C with Python Bindings
Requirements
-Linux
-Python 3.12
-Numpy
--gcc 13.3.0

How to run 

Activate your python venv
Within K-means folder
use Make (An error regarding unused parameter shows up
          ignore it)
cd python
python main.py 

(OPTIONALLY) export OMP_NUM_THREADS = 8
the above will give a slightly faster speed
since it will set your threads to 8 

Working 

Implemented K-means completely in C and used the K-means
in python using Cpython api, attemped multithreading with 
OpenMP

Benchmarked against Unoptimized Numpy baseline

Project Structure
K-means/
-C/
--K-means.h #function declarations
--K-means.c #The core of K-means
--bindings.c # Bindings to connect python and C
-python/
--main.py #main benchmarking script
--wrapper.py #wrapper function
-Makefile

Example output With threads = 8

(venv) sneh2604@DESKTOP-VAO24N9:~/K-means/python$ python main.py
NumPy time: 3.900s
C-extension time: 0.316s
Speedup: 12.34x