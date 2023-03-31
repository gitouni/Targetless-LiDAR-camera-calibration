#include <fstream>
#include <stdio.h>
#include <dirent.h>
#include <sys/types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>  
#include <pcl/filters/filter.h>
#include <pcl/filters/passthrough.h>
#include <omp.h>
#include <utils.h>

typedef pcl::PointXYZI PointType;
void preprocess(pcl::PointCloud<PointType>::Ptr laserCloudIn, pcl::PointCloud<PointType>::Ptr laserCloudOut);
inline void checkpath(std::string &path);
void makedir(char* folder);

int main(int argc,char** argv){
    // omp_set_num_threads(4);
    DIR *dirp = nullptr;
    struct dirent *dir_entry = nullptr;
    if((dirp=opendir(argv[1]))==nullptr){
        pcl::console::print_error("Open %s failed!\n",argv[1]);
        return -1;
    }
    utils::makedir(argv[2]);
    std::string open_path=argv[1], write_path=argv[2];
    utils::checkpath(open_path);
    utils::checkpath(write_path);
    pcl::PointCloud<PointType>::Ptr laserCloudRead(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr laserCloudWrite(new pcl::PointCloud<PointType>);
    std::vector<std::string> read_files;
    while((dir_entry=readdir(dirp))!=nullptr){
        if(dir_entry->d_type!=DT_REG)
            continue;
        read_files.push_back(dir_entry->d_name);
    }
    #pragma omp parallel for schedule(static) num_threads(4)
    for(auto pcd_filename:read_files){
        if(pcl::io::loadPCDFile(open_path+pcd_filename, *laserCloudRead)==-1){
            pcl::console::print_error("read pcd file %s failed\n",pcd_filename.c_str());
            continue;
        }
        laserCloudWrite->clear();
        preprocess(laserCloudRead,laserCloudWrite);
        pcl::console::print_highlight("pcd file %s processed, points %ld -> %ld\n",pcd_filename.c_str(),laserCloudRead->size(),laserCloudWrite->size());
        pcl::io::savePCDFileBinary(write_path+pcd_filename, *laserCloudWrite);
    }
    closedir(dirp);
    pcl::console::print_info("directory %s closed.\n",open_path.c_str());
    return 0;
}

void preprocess(pcl::PointCloud<PointType>::Ptr laserCloudIn, pcl::PointCloud<PointType>::Ptr laserCloudOut){
    std::vector<int> mapping;
    pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudOut, mapping);
    pcl::PassThrough<PointType> pass;//设置滤波器对象
    pass.setInputCloud(laserCloudOut);
    pass.setFilterFieldName("x");			
    pass.setFilterLimits(0,100);  // x:[0,50]
    pass.setNegative(false);
    pass.filter(*laserCloudOut);
}

