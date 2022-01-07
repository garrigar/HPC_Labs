.PHONY = build clean

build: clean
	@echo "----COMPILE CUDA FILES----"
	nvcc --compiler-options -fPIC -lib -odir lib -o lib/libvectorsum.a cuda/vectorsum.cu 
	@echo "--------------------------"
	@echo "---COMPILE CYTHON FILES---"
	python3 compile_cython.py build_ext --inplace
	@echo "--------------------------"

clean:
	@echo "---------CLEANING---------"
	rm -f lib/*.a
	rm -f vectorsum_methods*
	rm -f sourcecode_vectorsum_methods.cpp
	rm -rf build
	@echo "--------------------------"

