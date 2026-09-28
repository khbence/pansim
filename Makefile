.PHONY: buildCPU buildGPU dbuildBaseCPU dbuildCPU dbuildGPU drunCPU drunGPU format

# AppleClang has no built-in OpenMP support, and CMake's FindOpenMP can't locate
# Homebrew's libomp without hints. Point it at the Homebrew prefix on macOS.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	LIBOMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
	ifneq ($(LIBOMP_PREFIX),)
		OPENMP_CMAKE_FLAGS := -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include" \
			-DOpenMP_CXX_LIB_NAMES="omp" \
			-DOpenMP_omp_LIBRARY="$(LIBOMP_PREFIX)/lib/libomp.dylib"
	endif
endif

buildCPU:
	mkdir -p build;
	cd build; cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU=OFF $(OPENMP_CMAKE_FLAGS)
	cd build; make -j

buildGPU:
	mkdir -p build;
	cd build; cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU=ON $(OPENMP_CMAKE_FLAGS)
	cd build; make -j

dbuildBaseCPU:
	docker build . -f Dockerfile.baseCPU -t khbence/covid_ppcu:base_cpu

dbuildCPU:
	docker build . -f Dockerfile.CPU -t khbence/covid_ppcu:cpu

dbuildGPU:
	docker build . -f Dockerfile.GPU -t khbence/covid_ppcu:gpu

drunCPU: dbuildCPU
	docker run --rm khbence/covid_ppcu:cpu

drunGPU: dbuildGPU
	docker run --rm khbence/covid_ppcu:gpu

format:
	docker run --rm --mount src=${CURDIR},target=/app,type=bind khbence/format
