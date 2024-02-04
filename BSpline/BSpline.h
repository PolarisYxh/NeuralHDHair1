//
// Created by chh3213 on 2022/11/25.
//

#ifndef CHHROBOTICS_CPP_BSPLINE_H
#define CHHROBOTICS_CPP_BSPLINE_H
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include<vector>
#include<cmath>
#include<algorithm>
using namespace std;
using namespace Eigen;

double baseFunction(int i, int k, double u, const vector<double>& node_vector);

vector<double> u_quasi_uniform(int n,int k);


vector<double> u_piecewise_B_Spline(int n,int k);
vector<Vector3d> Interp3d(vector<Vector3d> Ps,int k=3,int nums=100);
void MakeInterp3d3(float * stds,int* segs,float * result,int point_num,int strand_num,int* final_nums,int k=3);
void MakeInterp3d2(float * stds,int* segs,float * result,int point_num,int strand_num,int final_num=100,int k=3);
// void MakeInterp3d1(float * stds,int* segs,float * result,int point_num,int strand_num,int final_num=100,int k=3);
// void MakeInterp3d(float * stds,int* segs,float * result,int point_num,int strand_num,int final_num=100,int k=3);

#endif //CHHROBOTICS_CPP_BSPLINE_H
