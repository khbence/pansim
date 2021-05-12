.PHONY: buildCPU buildGPU buildBaseCPU pushBaseCPU dockerCPU dockerRunCPU dockerGPU dockerRunGPU

buildCPU:
	mkdir -p build;
	cd build; cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU=OFF
	cd build; make -j

buildGPU:
	mkdir -p build;
	cd build; cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU=ON -DTHRUST_INCLUDE_CUB_CMAKE=ON
	cd build; make -j

buildBaseCPU:
	docker build . -f Dockerfile.baseCPU -t khbence/covid_ppcu:base_cpu

pushBaseCPU: buildBaseCPU
	docker push khbence/covid_ppcu:base_cpu

dbuildCPU:
	docker build . -f Dockerfile.CPU -t khbence/covid_ppcu:cpu

drunCPU: dbuildCPU
	docker run khbence/covid_ppcu:cpu

dbuildGPU:
	docker build . -f Dockerfile.GPU -t khbence/covid_ppcu:gpu

drunGPU: dbuildGPU
	docker run khbence/covid_ppcu:gpu

pushGPU: dbuildGPU
	docker push khbence/covid_ppcu:gpu