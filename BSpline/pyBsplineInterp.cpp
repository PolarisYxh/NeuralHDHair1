
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "BSpline.h"
#include <iostream>
using namespace std;
namespace py = pybind11;

// #include <io.h>  
// #include <fcntl.h>  

py::array_t<float> GetBsplineInterp(py::array_t<float> strands,py::array_t<int> segments,int final_num,int k=3)
{	
	py::buffer_info buf1 = strands.request();
	py::buffer_info buf2 = segments.request();
    printf("dim:%d \n", buf1.ndim);
	for (int cnt = 0; cnt < buf1.ndim; cnt++)
	{
		printf("dim size %d : %d\n", cnt, buf1.shape[cnt]);
	}
	size_t point_num=buf1.shape[0];
	size_t dims=buf1.shape[1];
	size_t strand_num = buf2.shape[0];
	float *ptr1 = (float *)buf1.ptr;
	int *ptr2 = (int *)buf2.ptr;
	// printf("image data: %f %f %f %f %f %f\n", ptr1[0], ptr1[1], ptr1[2], ptr1[3], ptr1[4], ptr1[5]);
	py::array_t<float> result = py::array_t<float>(strand_num*final_num*3);
	py::buffer_info buf3 = result.request();
	float *ptr3 = (float *)buf3.ptr;
	MakeInterp3d2(ptr1,ptr2,ptr3,point_num,strand_num,final_num,k);
	
	size_t nums = strand_num*final_num;
	result.resize({ nums,  dims});

	printf("dim:%d \n", buf3.ndim);
	for (int cnt = 0; cnt < buf3.ndim; cnt++)
	{
		printf("dim size %d : %d\n", cnt, buf3.shape[cnt]);
	}
	// printf("image data: %f %f %f %f %f %f\n", ptr3[0], ptr3[1], ptr3[2], ptr3[3], ptr3[4], ptr3[5]);
	return result;
}
py::array_t<float> GetBsplineInterpDifnum(py::array_t<float> strands,py::array_t<int> segments,py::array_t<int> final_num,int k=3)
{	
	py::buffer_info buf1 = strands.request();
	py::buffer_info buf2 = segments.request();
	py::buffer_info buf_final = final_num.request();
    printf("dim:%d \n", buf1.ndim);
	for (int cnt = 0; cnt < buf1.ndim; cnt++)
	{
		printf("dim size %d : %d\n", cnt, buf1.shape[cnt]);
	}
	size_t point_num=buf1.shape[0];
	size_t dims=buf1.shape[1];
	size_t strand_num = buf2.shape[0];
	float *ptr1 = (float *)buf1.ptr;
	int *ptr2 = (int *)buf2.ptr;
	int *ptr_final = (int *)buf_final.ptr;
	int res_sum = 0;
	for(int n=0;n<strand_num;n++)//n:控制点个数-1
    {
		res_sum+=ptr_final[n];
	}
	printf("result_num:%d\n",res_sum);
	// printf("image data: %f %f %f %f %f %f\n", ptr1[0], ptr1[1], ptr1[2], ptr1[3], ptr1[4], ptr1[5]);

	py::array_t<float> result = py::array_t<float>(res_sum*3);
	py::buffer_info buf3 = result.request();
	float *ptr3 = (float *)buf3.ptr;
	MakeInterp3d3(ptr1,ptr2,ptr3,point_num,strand_num,ptr_final,k);
	
	size_t nums = res_sum;
	result.resize({ nums,  dims});

	printf("dim:%d \n", buf3.ndim);
	for (int cnt = 0; cnt < buf3.ndim; cnt++)
	{
		printf("dim size %d : %d\n", cnt, buf3.shape[cnt]);
	}
	// printf("image data: %f %f %f %f %f %f\n", ptr3[0], ptr3[1], ptr3[2], ptr3[3], ptr3[4], ptr3[5]);
	return result;
}

PYBIND11_MODULE(pyBsplineInterp, m) {
	m.doc() = "Get hair strand Bspline Interp result"; // optional module docstring

	m.def("GetBsplineInterp", &GetBsplineInterp, "Get hair strand Bspline Interp result");
	m.doc() = "Get hair strand Bspline Interp result in different sample num"; // optional module docstring

	m.def("GetBsplineInterpDifnum", &GetBsplineInterpDifnum, "Get hair strand Bspline Interp result in different sample num");
}