#ifndef u_short
#define u_short unsigned short
#endif

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <iostream>

#include "BlobFindingAlg.h"
#include "BlobFindingAlg.cpp"

boost::python::list getBlobs(boost::python::dict param, boost::python::numpy::ndarray data)
{
    //setup output
    boost::python::list listlist;
    boost::python::list listx;
    boost::python::list listy;
    boost::python::list listintegral;
    boost::python::list listarea;
    
    //read configuration from dictionary
    int threshold = boost::python::extract<int>(param["threshold"]);
    int clustersize = boost::python::extract<int>(param["clustersize"]);
    int bitdepth = boost::python::extract<int>(param["bitdepth"]);
    boost::python::tuple shape = boost::python::extract<boost::python::tuple>(data.attr("shape"));
    if (boost::python::len(shape) != 2)
    {
        std::cout<<"Need 2-dimensional image."<<std::endl;
        return listlist;
    }
    if ((bitdepth != 8) && (bitdepth != 16))
    {
        std::cout<<"Can only process 8 or 16 bit images"<<std::endl;
        return listlist;
    }
    int width = boost::python::extract<int>(shape[1]);
    int height = boost::python::extract<int>(shape[0]);
    int size = width * height;
    BlobConfig * config = new BlobConfig(width, height, size, threshold, clustersize);
    
    //copy image to C++
    int k = 0;
    if (bitdepth == 16)
    {
        u_short * dat = reinterpret_cast<u_short *>(data.get_data());
        for (int i = 0; i < height; ++i)
            for (int j = 0; j < width; ++j)
            {
                config->spectrum[k] = dat[k];
                ++k;
            }
    }
    if (bitdepth == 8)
    {
        unsigned char * dat = reinterpret_cast<unsigned char *>(data.get_data());
        for (int i = 0; i < height; ++i)
            for (int j = 0; j < width; ++j)
            {
                config->spectrum[k] = dat[k];
                ++k;
            }
    }
    LinkedList<Position> * bloblist = new LinkedList<Position>;
    BlobFindingMain * calculator = new BlobFindingMain(config);
    int numberofblobs = calculator->GetBlobPositions(bloblist);

    if (numberofblobs)
    {
        for (int sz = 0; sz < numberofblobs; ++sz)
        {
            listx.append((bloblist->GetThing(sz))->GetXPosition());
            listy.append((bloblist->GetThing(sz))->GetYPosition());
            listintegral.append((bloblist->GetThing(sz))->GetIntegral());
            listarea.append((bloblist->GetThing(sz))->GetArea());
        }
    }
    listlist.append(listx);
    listlist.append(listy);
    listlist.append(listintegral);
    listlist.append(listarea);
    
    delete config;
    return listlist;
}

BOOST_PYTHON_MODULE(Blobfinder)
{
    //init
    Py_Initialize();
    boost::python::numpy::initialize();
    // Expose functions
    def("getBlobs", &getBlobs);
}

