# -------- Configuration --------
PYTHON:= python3
PY_CONFIG:= $(PYTHON)-config
NUMPY_INC:= $(shell $(PYTHON) -c "import numpy; print(numpy.get_include())")

CFLAGS:= -O3 -march=native -fopenmp -Wall -Wextra -fPIC
LDFLAGS:= -shared -fopenmp

C_DIR:= C
PY_DIR:= python

SOURCE:= $(C_DIR)/K-means.c $(C_DIR)/bindings.c
TARGET:= $(PY_DIR)/kmeans_c$(shell $(PY_CONFIG) --extension-suffix)

# -------- Rules --------
all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) \
	    $(shell python3-config --includes) \
	    -I$(NUMPY_INC) \
	    $(SOURCE) \
	    $(LDFLAGS) \
	    -o $(TARGET)

clean:
	rm -f $(PY_DIR)/kmeans_c*.so $(PY_DIR)/kmeans_c*.pyd
