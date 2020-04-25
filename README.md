# cublas_tests
There are 4 files described as follows:-<br />
1.) matmul2.cu - To be run with the profiler.(compute type fp32)<br />
2.) matmul2_fp16.cu - To be run withe the profiler.(compute type fp16)<br />
3.) matmul2timimg.cu - Can be run by compiling using nvcc.(compute type fp32)<br />
4.) matmul2timimg_fp16.cu - Can be run by compiling using nvcc.(compute type fp16)<br />

To run files 1 and 2:-<br />
They are to be run using the profiler which can be download from https://developer.nvidia.com/gameworksdownload#?tx=$gameworks,developer_tools <br />
Version 2019.5 of Nsight Systems is to be downloaded.<br />
It can be installed using bash /path/to/run/file <br/>
The GUI for the profiler can be found in /pah/to/root/of/profiler/host-linux-x64 <br/>
The GUI can be fired up using bash nsight-nsys<br/>
The .qdrep files are also present which can be direclty opened using the profiler.<br/>


To run the files 3 and 4 using nvcc use:-<br />
nvcc <filename> && ./a.out<br />
The matmul parameters can be changed from inside the files. The current are set to (4096 4096) X (4096 4096)<br />





