all:
	xbuild /p:Configuration=Release GPUVerify.sln
test:
	cd testsuite && \
	../gvtester.py -w new.pickle . && \
	../gvtester.py -c baseline.pickle new.pickle && \
	cd ..
