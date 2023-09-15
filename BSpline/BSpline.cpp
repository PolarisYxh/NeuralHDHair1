//
// Created by chh3213 on 2022/11/25.
//

#include "BSpline.h"
#include "omp.h"
#include "algorithm"

#include <set>
#include <map>
/**
 * 基函数定义
 * @param i
 * @param k B样条阶数k
 * @param u 自变量
 * @param node_vector 节点向量 array([u0,u1,u2,...,u_n+k],shape=[1,n+k+1].
 */
double baseFunction(int i, int k, double u, const vector<double>& node_vector) {
    //0次B样条（1阶B样条）
    double Bik_u;
    if(k==1){
        if(u>=node_vector[i]&&u<node_vector[i+1]){
            Bik_u=1;
        }else{
            Bik_u=0;
        }

    }else{
        //公式中的两个分母
        double denominator_1  =node_vector[i+k-1]-node_vector[i];
        double denominator_2  =node_vector[i+k]-node_vector[i+1];
        //# 如果遇到分母为 0的情况：
        //# 1. 如果此时分子也为0，约定这一项整体为0；
        //# 2. 如果此时分子不为0，则约定分母为1 。
        if(denominator_1==0)denominator_1=1;
        if(denominator_2==0)denominator_2=1;
        Bik_u = (u-node_vector[i])/denominator_1* baseFunction(i,k-1,u,node_vector)+(node_vector[i+k]-u)/denominator_2*
                                                                                                         baseFunction(i+1,k-1,u,node_vector);
    }
    return Bik_u;
}

/**
 * 准均匀B样条的节点向量计算
 * 首末值定义为 0 和 1
 * @param n 控制点个数-1，控制点共n+1个
 * @param k B样条阶数k， k阶B样条，k-1次曲线.
 * @return
 */
vector<double> u_quasi_uniform(int n, int k) {
    vector<double> node_vector(n+k+1); //准均匀B样条的节点向量计算，共n+1个控制顶点，k-1次B样条，k阶
    double  piecewise = n - k + 2; //B样条曲线的段数:控制点个数-次数
    if(piecewise==1){//只有一段曲线时，n = k-1
        for(int i=n+1;i<n+k+1;i++)node_vector[i]=1;
    }else{
        //中间段内节点均匀分布：两端共2k个节点，中间还剩(n+k+1-2k=n-k+1）个节点
        for(int i=0;i<n-k+1;i++){
            node_vector[k+i]=node_vector[k+i-1]+1/piecewise;
        }
        for(int i=n+1;i<n+k+1;i++)node_vector[i]=1;//末尾重复度k
    }
    return node_vector;
}

/**
 * 分段B样条
 * 首末值定义为 0 和 1
 * 分段Bezier曲线的节点向量计算，共n+1个控制顶点，k阶B样条，k-1次曲线
 * 分段Bezier端节点重复度为k，内间节点重复度为k-1,且满足n/(k-1)为正整数
 * @param n 控制点个数-1，控制点共n+1个
 * @param k B样条阶数k， k阶B样条，k-1次曲线
 * @return
 */
vector<double> u_piecewise_B_Spline(int n, int k) {
    vector<double> node_vector(n+k+1);
    if(n%(k-1)==0&&(k-1)>0){//满足n是k-1的整数倍且k-1为正整数
        for(int i=n+1;i<n+k+1;i++)node_vector[i]=1;//末尾n+1到n+k+1的数重复
        int piecewise = n / (k-1);  //设定内节点的值
        if(piecewise>1){
            //内节点重复k-1次
            for(int i=1;i<piecewise;i++){
                for(int j=0;j<k-1;j++)node_vector[(k-1)*i+j+1]=i/piecewise;
            }
            
        }
    }else{
        cout<<"error!需要满足n是k-1的整数倍且k-1为正整数"<<endl;
    }
    return node_vector;
}


vector<Vector3d> Interp3d(vector<Vector3d> Ps,int k,int nums)
{
    vector<Vector3d> result(nums);
    int n =Ps.size()-1; //控制点个数-1
    Vector3d p_u(0,0,0);
    vector<double>bik_u(n+1);

    vector<double>node_vector;
    node_vector = u_piecewise_B_Spline(n,k);
    double step = 1.0/nums;
    for(int j=0;j<nums;j++){
        double u=step*j;
        for(int i=0;i<n+1;i++){
            bik_u[i]= baseFunction(i,k,u,node_vector);
        }
        for(int i=0;i< Ps.size();i++){
            p_u = p_u + Ps[i]*bik_u[i];
            //cout<<p_u<<","<<endl;
        }
        // x_.push_back(p_u[0]);
        // y_.push_back(p_u[1]);
        // z_.push_back(p_u[2]);
        result[j]=p_u;
        p_u=Vector3d(0,0,0);
        std::cout<<result[j];
    }
    return result;
}
void MakeInterp3d1(float * stds,int* segs,float * result,int point_num,int strand_num,int final_num,int k){
    // 预计算bik_u版本 31s 并行后：3.9196135997772217s，反而更慢了，
    //TODO：参考MakeInterp3d2函数，将177行改为eigen矩阵乘法
    map<int,vector<double>> node_vector;
    std::set<int> flag;//segment的无重复版本
    vector<int> starts(strand_num);
    int start = 0;

    Eigen::Matrix<float,Dynamic,3,Eigen::RowMajor> strands1 = Eigen::Map<Eigen::Matrix<float,Dynamic,3,Eigen::RowMajor>>(stds,point_num,3);
    std::vector<Eigen::Matrix<float,Dynamic,3>> strands(strand_num);
    for(int n=0;n<strand_num;n++)//n:控制点个数-1
    {
        if(flag.count(segs[n])>0)
        {
            starts.push_back(start);
            auto x= strands1.block(start,0,segs[n],3);
            strands[n]=x;
            start+=segs[n];
            continue;
        }
        vector<double> node_vector0= u_quasi_uniform(segs[n]-1,k);
        node_vector[segs[n]] = node_vector0;
        flag.insert(segs[n]);
        starts.push_back(start);
        auto x= strands1.block(start,0,segs[n],3);
        strands[n]=x;
        start+=segs[n];
    }
    // map<int,vector<vector<double>>> bik_u;
    map<int,SparseMatrix<double>> bik_u;
    double step = 1.0/final_num;
    for (auto it = flag.begin(); it != flag.end(); it++){
        // vector<vector<double>> b1(final_num);
        SparseMatrix<double> b1(final_num,*it);
        for(int j=0;j<final_num;j++){
            double u=step*j;
            vector<double> b(*it);
            for(int i=0;i<*it;i++){
            //n:控制点个数-1
                double b= baseFunction(i,k,u,node_vector[*it]);
                b1.insert(j, i) = b;
            }
        }
        b1.makeCompressed();
        bik_u[*it]=b1;
    }

    #pragma omp parallel for schedule(static)
    for(int strand_idx=0;strand_idx<strand_num;strand_idx++)
    {
        int start=starts[strand_idx];
        int num=segs[strand_idx];
        int n =num-1; //控制点个数-1
        Vector3d p_u(0,0,0);
        for(int j=0;j<final_num;j++){
            for(int i=0;i< num;i++){
                p_u = p_u + Vector3d(stds[(start+i)*3],stds[(start+i)*3+1],stds[(start+i)*3+2])*bik_u[num].coeff(j, i);
                //cout<<p_u<<","<<endl;
            }
            result[(strand_idx*final_num+j)*3]=p_u[0];
            result[(strand_idx*final_num+j)*3+1]=p_u[1];
            result[(strand_idx*final_num+j)*3+2]=p_u[2];
            // std::cout<<p_u[0]<<" "<<p_u[1]<<" "<<p_u[2]<<std::endl;
            // std::cout.flush();
            p_u=Vector3d(0,0,0);
        }
        // std::cout<<strand_idx<<" ";
    }
    // return result;
}

void MakeInterp3d(float * stds,int* segs,float * result,int point_num,int strand_num,int final_num,int k){
    // 原始版本 25s 并行后:2.40796375274658
    int start=0;
    vector<int> starts(strand_num);
    for(int n=0;n<strand_num;n++)//n:控制点个数-1
    {
        starts.push_back(start);
        start+=segs[n];
    }
    // #pragma omp parallel for schedule(static)
    for(int strand_idx=0;strand_idx<strand_num;strand_idx++)
    {
        int num=segs[strand_idx];
        int n =num-1; //控制点个数-1
        Vector3d p_u(0,0,0);
        vector<double> bik_u(n+1);
        int start=starts[strand_idx];
        
        const vector<double> node_vector= u_quasi_uniform(n,k);
        double step = 1.0/final_num;
        for(int j=0;j<final_num;j++){
            double u=step*j;
            for(int i=0;i<n+1;i++){
                bik_u[i]= baseFunction(i,k,u,node_vector);
            }
            for(int i=0;i< num;i++){
                p_u = p_u + Vector3d(stds[(start+i)*3],stds[(start+i)*3+1],stds[(start+i)*3+2])*bik_u[i];
            }
            // x_.push_back(p_u[0]);
            // y_.push_back(p_u[1]);
            // z_.push_back(p_u[2]);
            result[(strand_idx*final_num+j)*3]=p_u[0];
            result[(strand_idx*final_num+j)*3+1]=p_u[1];
            result[(strand_idx*final_num+j)*3+2]=p_u[2];
            // std::cout<<p_u[0]<<" "<<p_u[1]<<" "<<p_u[2]<<std::endl;
            // std::cout.flush();
            p_u=Vector3d(0,0,0);
        }
    }
    // return result;
}
void MakeInterp3d2(float * stds,int* segs,float * result,int point_num,int strand_num,int final_num,int k){
    // 25s 并行后:2.40796375274658
    int start=0;
    vector<int> starts(strand_num);
    Eigen::Matrix<float,Dynamic,3,Eigen::RowMajor> strands1 = Eigen::Map<Eigen::Matrix<float,Dynamic,3,Eigen::RowMajor>>(stds,point_num,3);
    std::vector<Eigen::Matrix<float,Dynamic,3>> strands(strand_num);
    for(int n=0;n<strand_num;n++)//n:控制点个数-1
    {
        starts.push_back(start);
        auto x= strands1.block(start,0,segs[n],3);
        strands[n]=x;
        start+=segs[n];
    }
    #pragma omp parallel for schedule(static)
    for(int strand_idx=0;strand_idx<strand_num;strand_idx++)
    {
        int num=segs[strand_idx];
        int n =num-1; //控制点个数-1
        Eigen::MatrixXf bik_u(n+1, 1);
        
        const vector<double> node_vector= u_quasi_uniform(n,k);
        double step = 1.0/final_num;
        for(int j=0;j<final_num;j++){
            double u=step*j;
            for(int i=0;i<n+1;i++){
                bik_u(i,0)= baseFunction(i,k,u,node_vector);
            }
            auto y = bik_u.transpose()*strands[strand_idx];
            result[(strand_idx*final_num+j)*3]=y(0);
            result[(strand_idx*final_num+j)*3+1]=y(1);
            result[(strand_idx*final_num+j)*3+2]=y(2);
        }
    }
    // return result;
}