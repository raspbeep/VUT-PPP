make: build
	./build/data_generator && mpiexec -np 2 ./build/ppp_proj01 -m 1 -n 1 -i ppp_input_data.h5 -o output.h5

sync:
	# rsync --exclude 'build' --exclude 'scriptsx' --exclude 'scripts1' -rvu ~/Downloads/ass/ xkrato61@karolina.it4i.cz:~/ppp/projekt
	# rsync --exclude 'build' -rvu ~/ppp/projekt/ xkrato61@merlin.fit.vutbr.cz:~/ppp/projekt
	rsync --exclude 'build' --exclude 'scriptsx' --exclude 'scripts1' -rvu ~/Downloads/ass/ xkrato61@barbora.it4i.cz:~/ppp/projekt

gen:
	rm -rf build && cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Debug -DLOGIN=xkrato61 -Bbuild -S.

build: gen
	cmake --build build --config Debug