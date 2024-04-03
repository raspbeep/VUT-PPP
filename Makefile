make: build
	cp ./build/comm ./comm

sync:
	# rsync --exclude 'build' -rvu ~/ppp/projekt/ xkrato61@karolina.it4i.cz:~/ppp/projekt
	# rsync --exclude 'build' -rvu ~/ppp/projekt/ xkrato61@merlin.fit.vutbr.cz:~/ppp/projekt
	rsync --exclude 'build' -rvu ~/ppp/assignment/ xkrato61@barbora.it4i.cz:~/ppp/projekt

gen:
	cmake -Bbuild -S.

build:
	cmake --build build --config Release