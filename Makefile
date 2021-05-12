.PHONY: buildCPU buildGPU dbuildBaseCPU dbuildCPU dbuildGPU

buildCPU:
	mkdir -p build;
	cd build; cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU=OFF
	cd build; make -j

buildGPU:
	mkdir -p build;
	cd build; cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU=ON
	cd build; make -j

dbuildBaseCPU:
	docker build . -f Dockerfile.baseCPU -t khbence/covid_ppcu:base_cpu

dbuildCPU:
	docker build . -f Dockerfile.CPU -t khbence/covid_ppcu:cpu

dbuildGPU:
	docker build . -f Dockerfile.GPU -t khbence/covid_ppcu:gpu

drunCPU: dbuildCPU
	docker run khbence/covid_ppcu:cpu