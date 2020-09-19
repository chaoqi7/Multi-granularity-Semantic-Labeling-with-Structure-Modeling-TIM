// Reference By loicland/cut-pursuit(github)
#include <iostream>
#include <cstdio>
#include <vector>
#include <math.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <numpy/ndarrayobject.h>
#include "boost/tuple/tuple.hpp"
#include "boost/python/object.hpp"
#include <boost/tuple/tuple_comparison.hpp>
#include <limits>
#include <map>
#include <string.h>

namespace bp = boost::python;
namespace ei = Eigen;
namespace bpn = boost::python::numpy;

typedef ei::Matrix<float, 3, 3> Matrix3f;
typedef ei::Matrix<float, 3, 1> Vector3f;

typedef boost::tuple< std::vector< std::vector<float> >, std::vector< std::vector<uint8_t> >, std::vector<std::vector<uint32_t> > > Custom_tuple;
typedef boost::tuple< uint32_t, uint32_t, uint32_t > Space_tuple;

std::vector< float > getgeof(ei::MatrixXf position,int k_nn);
ei::MatrixXf positionpart(ei::MatrixXf position,int start,int length);
float distance(std::vector< float > geofup,std::vector< float > geofdown);

struct VecToArray
{//converts a vector<uint32_t> to a numpy array
    static PyObject * convert(const std::vector<uint8_t> & vec) {
    npy_intp dims = vec.size();
    PyObject * obj = PyArray_SimpleNew(1, &dims, NPY_UINT8);
    void * arr_data = PyArray_DATA((PyArrayObject*)obj);
    memcpy(arr_data, &vec[0], dims * sizeof(uint8_t));
    return obj;
    }
};

template <class T>
struct VecvecToArray
{//converts a vector< vector<uint32_t> > to a numpy 2d array
    static PyObject * convert(const std::vector< std::vector<T> > & vecvec)
    {
        npy_intp dims[2];
        dims[0] = vecvec.size();
        dims[1] = vecvec[0].size();
        PyObject * obj;
        if (typeid(T) == typeid(uint8_t))
            obj = PyArray_SimpleNew(2, dims, NPY_UINT8);
        else if (typeid(T) == typeid(float))
            obj = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
        else if (typeid(T) == typeid(uint32_t))
            obj = PyArray_SimpleNew(2, dims, NPY_UINT32);
        void * arr_data = PyArray_DATA((PyArrayObject*)obj);
        std::size_t cell_size = sizeof(T);
        for (std::size_t i = 0; i < dims[0]; i++)
        {
            memcpy(arr_data + i * dims[1] * cell_size, &(vecvec[i][0]), dims[1] * cell_size);
        }
        return obj;
    }
};

struct to_py_tuple
{//converts to a python tuple
    static PyObject* convert(const Custom_tuple & c_tuple){
        bp::list values;

        PyObject * pyo1 = VecvecToArray<float>::convert(c_tuple.get<0>());
        PyObject * pyo2 = VecvecToArray<uint8_t>::convert(c_tuple.get<1>());
        PyObject * pyo3 = VecvecToArray<uint32_t>::convert(c_tuple.get<2>());

        values.append(bp::handle<>(bp::borrowed(pyo1)));
        values.append(bp::handle<>(bp::borrowed(pyo2)));
        values.append(bp::handle<>(bp::borrowed(pyo3)));

        return bp::incref( bp::tuple( values ).ptr() );
    }
};

class AttributeGrid {
//voxelization of the space, allows to accumulate the position, the color and labels
    std::map<Space_tuple, uint64_t> space_tuple_to_index;//associate eeach non-empty voxel to an index
    uint64_t index;
    std::vector<uint32_t> bin_count;//count the number of point in each non-empty voxel
    std::vector< std::vector<float> > acc_xyz;//accumulate the position of the points
    std::vector< std::vector<uint32_t> > acc_rgb;//accumulate the color of the points
    std::vector< std::vector<uint32_t> > acc_labels;//accumulate the label of the points
  public:
    AttributeGrid():
       index(0)
    {}
    //---methods for the occurence grid---
    uint64_t n_nonempty_voxels()
    {
        return this->index;
    }
    uint32_t get_index(uint32_t x_bin, uint32_t  y_bin, uint32_t z_bin)
    {
        return space_tuple_to_index.at(Space_tuple(x_bin, y_bin, z_bin));
    }
    bool add_occurence(uint32_t x_bin, uint32_t  y_bin, uint32_t z_bin)
    {
        Space_tuple st(x_bin, y_bin, z_bin);
        auto inserted = space_tuple_to_index.insert(std::pair<Space_tuple, uint64_t>(st, index));
        if (inserted.second)
        {
            this->index++;
            return true;
        }
        else
        {
            return false;
        }
    }
    std::map<Space_tuple,uint64_t>::iterator begin()
    {
        return this->space_tuple_to_index.begin();
    }
    std::map<Space_tuple,uint64_t>::iterator end()
    {
        return this->space_tuple_to_index.end();
    }
    //---methods for accumulating atributes---
    void initialize(uint8_t n_class)
    {//must be run once space_tuple_to_index is complete and the number of non-empty voxels is known
        bin_count  = std::vector<uint32_t>(this->index, 0);
        acc_xyz    = std::vector< std::vector<float> >(this->index, std::vector <float>(3,0));
        acc_rgb    = std::vector< std::vector<uint32_t> >(this->index, std::vector <uint32_t>(3,0));
        acc_labels = std::vector< std::vector<uint32_t> >(this->index, std::vector <uint32_t>(n_class+1,0));
    }
    uint32_t get_count(uint64_t voxel_index)
    {
        return bin_count.at(voxel_index);
    }
    std::vector<float> get_pos(uint64_t voxel_index)
    {
        return acc_xyz.at(voxel_index);
    }
    std::vector<uint32_t> get_rgb(uint64_t voxel_index)
    {
        return acc_rgb.at(voxel_index);
    }
    std::vector<uint32_t> get_acc_labels(uint64_t voxel_index)
    {
        return acc_labels.at(voxel_index);
    }
    uint8_t get_label(uint64_t voxel_index)
    {//return the majority label from ths voxel
     //ignore the unlabeled points (0), unless all points are unlabeled
        std::vector<uint32_t> label_hist = acc_labels.at(voxel_index);
        std::vector<uint32_t>::iterator chosen_label = std::max_element(label_hist.begin() + 1, label_hist.end());
        if (*chosen_label == 0)
        {
            return 0;
        }
        else
        {
            return (uint8_t)std::distance(label_hist.begin(), chosen_label);
        }
    }
    void add_attribute(uint32_t x_bin, uint32_t  y_bin, uint32_t z_bin, float x, float y, float z, uint8_t r, uint8_t g, uint8_t b)
    {//add a point x y z in voxel x_bin y_bin z_bin
        uint64_t bin = get_index(x_bin, y_bin, z_bin);
        bin_count.at(bin) = bin_count.at(bin) + 1;
        acc_xyz.at(bin).at(0) = acc_xyz.at(bin).at(0) + x;
        acc_xyz.at(bin).at(1) = acc_xyz.at(bin).at(1) + y;
        acc_xyz.at(bin).at(2) = acc_xyz.at(bin).at(2) + z;
        acc_rgb.at(bin).at(0) = acc_rgb.at(bin).at(0) + r;
        acc_rgb.at(bin).at(1) = acc_rgb.at(bin).at(1) + g;
        acc_rgb.at(bin).at(2) = acc_rgb.at(bin).at(2) + b;
    }
    void add_attribute(uint32_t x_bin, uint32_t  y_bin, uint32_t z_bin, float x, float y, float z, uint8_t r, uint8_t g, uint8_t b, uint8_t label)
    {//add a point x y z in voxel x_bin y_bin z_bin - with label
        uint32_t bin =get_index(x_bin, y_bin, z_bin);
        bin_count.at(bin) = bin_count.at(bin) + 1;
        acc_xyz.at(bin).at(0) = acc_xyz.at(bin).at(0) + x;
        acc_xyz.at(bin).at(1) = acc_xyz.at(bin).at(1) + y;
        acc_xyz.at(bin).at(2) = acc_xyz.at(bin).at(2) + z;
        acc_rgb.at(bin).at(0) = acc_rgb.at(bin).at(0) + r;
        acc_rgb.at(bin).at(1) = acc_rgb.at(bin).at(1) + g;
        acc_rgb.at(bin).at(2) = acc_rgb.at(bin).at(2) + b;
        acc_labels.at(bin).at(label) = acc_labels.at(bin).at(label) + 1;//Focus QI,dui yu mei yi ge part li,you yige label jiu jia yi.
    }
};

PyObject *  prune(const bpn::ndarray & xyz ,float voxel_size, const bpn::ndarray & rgb, const bpn::ndarray & labels, const int n_classes)
{//prune the point cloud xyz with a regular voxel grid
    std::cout << "=========================" << std::endl;
    std::cout << "======== pruninging ========" << std::endl;
    std::cout << "=========================" << std::endl;
    uint64_t n_ver = bp::len(xyz);
    bool have_labels = n_classes>0;
    //---read the numpy arrays data---
    const float * xyz_data = reinterpret_cast<float*>(xyz.get_data());
    const uint8_t * rgb_data = reinterpret_cast<uint8_t*>(rgb.get_data());
    const uint8_t * label_data;
    if (have_labels)
        label_data = reinterpret_cast<uint8_t*>(labels.get_data());
    //---find min max of xyz----
    float x_max = std::numeric_limits<float>::lowest(), x_min = std::numeric_limits<float>::max();
    float y_max = std::numeric_limits<float>::lowest(), y_min = std::numeric_limits<float>::max();
    float z_max = std::numeric_limits<float>::lowest(), z_min = std::numeric_limits<float>::max();
    #pragma omp parallel for reduction(max : x_max, y_max, z_max), reduction(min : x_min, y_min, z_min)
    for (std::size_t i_ver = 0; i_ver < n_ver; i_ver ++)
    {
        if (x_max < xyz_data[3 * i_ver]){           x_max = xyz_data[3 * i_ver];}
        if (y_max < xyz_data[3 * i_ver + 1]){       y_max = xyz_data[3 * i_ver + 1];}
        if (z_max < xyz_data[3 * i_ver + 2]){       z_max = xyz_data[3 * i_ver + 2];}
        if (x_min > xyz_data[3 * i_ver]){           x_min = xyz_data[3 * i_ver];}
        if (y_min > xyz_data[3 * i_ver + 1]){       y_min = xyz_data[3 * i_ver + 1];}
        if (z_min > xyz_data[3 * i_ver + 2 ]){      z_min = xyz_data[3 * i_ver + 2];}
    }
    //---compute the voxel grid size---
    uint32_t n_bin_x = std::ceil((x_max - x_min) / voxel_size);
    uint32_t n_bin_y = std::ceil((y_max - y_min) / voxel_size);
    uint32_t n_bin_z = std::ceil((z_max - z_min) / voxel_size);
    std::cout << "Voxelization into " << n_bin_x << " x " << n_bin_y << " x " << n_bin_z << " grid" << std::endl;
    //---detect non-empty voxels----
    AttributeGrid vox_grid;
    for (std::size_t i_ver = 0; i_ver < n_ver; i_ver ++)
    {
        uint32_t bin_x = std::floor((xyz_data[3 * i_ver] - x_min) / voxel_size);
        uint32_t bin_y = std::floor((xyz_data[3 * i_ver + 1] - y_min) / voxel_size);
        uint32_t bin_z = std::floor((xyz_data[3 * i_ver + 2] - z_min) / voxel_size);
        vox_grid.add_occurence(bin_x, bin_y, bin_z);
    }
    std::cout << "Reduced from " << n_ver << " to " << vox_grid.n_nonempty_voxels() << " points ("
              << std::ceil(10000 * vox_grid.n_nonempty_voxels() / n_ver)/100 << "%)" << std::endl;
    vox_grid.initialize(n_classes);
    //---accumulate points in the voxel map----
    for (std::size_t i_ver = 0; i_ver < n_ver; i_ver ++)
    {
        uint32_t bin_x = std::floor((xyz_data[3 * i_ver    ] - x_min) / voxel_size);
        uint32_t bin_y = std::floor((xyz_data[3 * i_ver + 1] - y_min) / voxel_size);
        uint32_t bin_z = std::floor((xyz_data[3 * i_ver + 2] - z_min) / voxel_size);
        if (have_labels)
            vox_grid.add_attribute(bin_x, bin_y, bin_z
                    , xyz_data[3 * i_ver], xyz_data[3 * i_ver + 1], xyz_data[3 * i_ver + 2]
                    , rgb_data[3 * i_ver], rgb_data[3 * i_ver + 1], rgb_data[3 * i_ver + 2], label_data[i_ver]);
        else
            vox_grid.add_attribute(bin_x, bin_y, bin_z
                    , xyz_data[3 * i_ver], xyz_data[3 * i_ver + 1], xyz_data[3 * i_ver + 2]
                    , rgb_data[3 * i_ver], rgb_data[3 * i_ver + 1], rgb_data[3 * i_ver + 2]);
    }
    //---compute pruned cloud----
    std::vector< std::vector< float > > pruned_xyz(vox_grid.n_nonempty_voxels(), std::vector< float >(3, 0.f));
    std::vector< std::vector< uint8_t > > pruned_rgb(vox_grid.n_nonempty_voxels(), std::vector< uint8_t >(3, 0));
    std::vector< std::vector< uint32_t > > pruned_labels(vox_grid.n_nonempty_voxels(), std::vector< uint32_t >(n_classes + 1, 0));
    for (std::map<Space_tuple,uint64_t>::iterator it_vox=vox_grid.begin(); it_vox!=vox_grid.end(); ++it_vox)
    {//loop over the non-empty voxels and compute the average posiition/color + majority label
        uint64_t voxel_index = it_vox->second; //
        float count = (float)vox_grid.get_count(voxel_index);
        std::vector<float> pos = vox_grid.get_pos(voxel_index);
        pos.at(0) = pos.at(0) / count;
        pos.at(1) = pos.at(1) / count;
        pos.at(2) = pos.at(2) / count;
        pruned_xyz.at(voxel_index) = pos;
        std::vector<uint32_t> col = vox_grid.get_rgb(voxel_index);
        std::vector<uint8_t> col_uint8_t(3);
        col_uint8_t.at(0) = (uint8_t)((float) col.at(0) / count);
        col_uint8_t.at(1) = (uint8_t)((float) col.at(1) / count);
        col_uint8_t.at(2) = (uint8_t)((float) col.at(2) / count);
        pruned_rgb.at(voxel_index) = col_uint8_t;
        pruned_labels.at(voxel_index) = vox_grid.get_acc_labels(voxel_index);
    }
    return to_py_tuple::convert(Custom_tuple(pruned_xyz,pruned_rgb, pruned_labels));
}


PyObject * compute_geof_ori(const bpn::ndarray & xyz ,const bpn::ndarray & target, int min_k_nn,int max_k_nn)
{//compute the following geometric features (geof) features of a point cloud:
 //linearity planarity scattering verticality
    std::cout << "========shiyan========" << std::endl;
    std::size_t n_ver = bp::len(xyz);
    std::vector< std::vector< float > > geof(n_ver, std::vector< float >(8,0));
    //--- read numpy array data---
    const uint32_t * target_data = reinterpret_cast<uint32_t*>(target.get_data());
    const float * xyz_data = reinterpret_cast<float*>(xyz.get_data());
    std::size_t s_ver = 0;
    #pragma omp parallel for schedule(static)
    for (std::size_t i_ver = 0; i_ver < n_ver; i_ver++)
    {//each point can be treated in parallell independently
        //--- compute 3d covariance matrix of neighborhood ---
        ei::MatrixXf position(max_k_nn+1,3);
        ei::ArrayXf weight(max_k_nn+1);
        std::size_t i_edg = max_k_nn * i_ver;
        std::size_t ind_nei;
        position(0,0) = xyz_data[3 * i_ver];
        position(0,1) = xyz_data[3 * i_ver + 1];
        position(0,2) = xyz_data[3 * i_ver + 2];
        weight(0)=0;
        for (std::size_t i_nei = 0; i_nei < max_k_nn; i_nei++)
        {
                //add the neighbors to the position matrix
            ind_nei = target_data[i_edg];
            position(i_nei+1,0) = xyz_data[3 * ind_nei];
            position(i_nei+1,1) = xyz_data[3 * ind_nei + 1];
            position(i_nei+1,2) = xyz_data[3 * ind_nei + 2];
            float temp=1/sqrt(pow(position(i_nei+1,0)-position(0,0),2)+pow(position(i_nei+1,1)-position(0,1),2)+pow(position(i_nei+1,2)-position(0,2),2));
            weight(i_nei+1)=temp;
            i_edg++;
        }
        int k_nn_suit=0;
        float res_suit=0;
        for(std::size_t i_n = min_k_nn; i_n < max_k_nn; i_n++)
        {
           ei::MatrixXf positionup=positionpart(position,0,i_n);
           ei::MatrixXf positiondown=positionpart(position,i_n,max_k_nn+1-i_n);
           float SUM = weight.sum();
           float N1 = weight.head(i_n).sum();
           float N2 =weight.tail(max_k_nn+1-i_n).sum();
           std::vector< float > geoftempt = getgeof(positionup,i_n);
           std::vector< float > geofup = geoftempt;
           std::vector< float > geofdown = getgeof(positiondown,max_k_nn+1-i_n);
           float res = (N1/SUM)*(N2/SUM)*distance(geofup,geofdown);//(SUM/N1)*(SUM/N2)*
           if(res>res_suit)
           {
              res_suit=res;
              k_nn_suit=i_n;
              geof[i_ver] = geoftempt;
           }
           /*float N2 = weight.block(i_n,0,k_nn+1-i_n,0).sum();
           float res = (N1/sum)*(N2/sum)*(positionup-positiondown).norm();*/
        }
        if(i_ver%5000==0)
            {
             std::cout << "========"+std::to_string(i_ver)+"--:"+std::to_string(k_nn_suit)+"========" << std::endl;
             //std::cout << "========geofup--:"+std::to_string(geofup[0])+","+std::to_string(geofup[1])+","+std::to_string(geofup[2])+","+std::to_string(geofup[3])+","+std::to_string(geofup[4])+","+std::to_string(geofup[5])+","+std::to_string(geofup[6])+","+std::to_string(geofup[7])+"========" << std::endl;
             //std::cout << "========geofdown--:"+std::to_string(geofdown[0])+","+std::to_string(geofdown[1])+","+std::to_string(geofdown[2])+","+std::to_string(geofdown[3])+","+std::to_string(geofdown[4])+","+std::to_string(geofdown[5])+","+std::to_string(geofdown[6])+","+std::to_string(geofdown[7])+"========" << std::endl;
            }
        
        //---progression---
        s_ver++;//if run in parellel s_ver behavior is udnefined, but gives a good indication of progress
        if (s_ver % 10000 == 0)
        {
            std::cout << s_ver << "% done          \r" << std::flush;
            std::cout << ceil(s_ver*100/n_ver) << "% done          \r" << std::flush;
        }
    }
    std::cout <<  std::endl;
    return VecvecToArray<float>::convert(geof);
}

ei::MatrixXf positionpart(ei::MatrixXf position,int start,int length)
{
     ei::MatrixXf positionpart(length,3);
     for(int i =0;i<length;i++)
     {
        positionpart(i,0)=position(start+i,0);
        positionpart(i,1)=position(start+i,1);
        positionpart(i,2)=position(start+i,2);
     }
     return positionpart;
}

float distance(std::vector< float > geofup,std::vector< float > geofdown)
{
   float distance=0;
   for(int i=0;i<geofup.size();i++)
   {
      distance =distance+(geofup[i]-geofdown[i])*(geofup[i]-geofdown[i]);
   }
   //return pow(distance,0.5);
   return distance;
}

std::vector< float > getgeof(ei::MatrixXf position,int k_nn)
{
// compute the covariance matrix
        //std::vector< std::vector< float > > geof(1, std::vector< float >(8,0));
        std::vector< float > geof(8,0);
        ei::MatrixXf centered_position = position.rowwise() - position.colwise().mean();
        ei::Matrix3f cov = (centered_position.adjoint() * centered_position) / float(k_nn + 1);
        ei::EigenSolver<Matrix3f> es(cov);
        //--- compute the eigen values and vectors---
        std::vector<float> ev = {es.eigenvalues()[0].real(),es.eigenvalues()[1].real(),es.eigenvalues()[2].real()};
        std::vector<int> indices(3);
        std::size_t n(0);
        std::generate(std::begin(indices), std::end(indices), [&]{ return n++; });
        std::sort(std::begin(indices),std::end(indices),
                       [&](int i1, int i2) { return ev[i1] > ev[i2]; } );
        std::vector<float> lambda = {(std::max(ev[indices[0]],0.f)),
                                    (std::max(ev[indices[1]],0.f)),
                                    (std::max(ev[indices[2]],0.f))};
        std::vector<float> v1 = {es.eigenvectors().col(0)(0).real()
                               , es.eigenvectors().col(0)(1).real()
                               , es.eigenvectors().col(0)(2).real()};
        float linearity  = (sqrtf(lambda[0]) - sqrtf(lambda[1])) / sqrtf(lambda[0]);
        float planarity  = (sqrtf(lambda[1]) - sqrtf(lambda[2])) / sqrtf(lambda[0]);
        float scattering =  sqrtf(lambda[2]) / sqrtf(lambda[0]);
        //---fill the geof vector---

        geof[0] = linearity;
        geof[1] = planarity;
        geof[2] = scattering;
        geof[3] = 1.f/(exp(-1*v1[0]/v1[1])+1);
        geof[4] = sqrtf(lambda[2])/(sqrtf(lambda[0])+sqrtf(lambda[1])+sqrtf(lambda[2]));
        geof[5] = 1.f/(exp(-1*v1[0]/v1[2])+1);
        geof[6] = 0;//pow(sqrtf(lambda[0])*sqrtf(lambda[1])*sqrtf(lambda[2]),1/3)/(sqrtf(lambda[0])+sqrtf(lambda[1])+sqrtf(lambda[2]));
        geof[7] = 0;//-sqrtf(lambda[0])*log(sqrtf(lambda[0]))-sqrtf(lambda[1])*log(sqrtf(lambda[1]))-sqrtf(lambda[2])*log(sqrtf(lambda[2]));
        return geof;
}


using namespace boost::python;
BOOST_PYTHON_MODULE(libply_c)
{
    _import_array();
    bp::to_python_converter<std::vector<std::vector<float>, std::allocator<std::vector<float> > >, VecvecToArray<float> >();
    bp::to_python_converter< Custom_tuple, to_py_tuple>();
    Py_Initialize();
    bpn::initialize();
    def("compute_geof_ori", compute_geof_ori);
    def("prune", prune);
}
