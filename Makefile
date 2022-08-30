.PHONY: debug release rund runr

debug: 
	cmake -E make_directory $(CURDIR)/debug
	cmake -S $(CURDIR) -B $(CURDIR)/debug -DCMAKE_BUILD_TYPE=Debug -DENABLE_TESTING=ON
	cmake --build $(CURDIR)/debug --parallel

release:
	cmake -E make_directory $(CURDIR)/release
	cmake -S $(CURDIR) -B $(CURDIR)/release -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTING=OFF
	cmake --build $(CURDIR)/release --parallel

rund: debug
	./debug/pansim

runr: release
	./release/pansim

unittest: debug
	./debug/test/testPansim
	