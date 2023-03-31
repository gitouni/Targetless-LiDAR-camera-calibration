#include <iostream>
#include <pcl/visualization/pcl_visualizer.h>
#include <chrono>
#include <thread>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/io.h>
 
using namespace pcl;
using namespace std;

int main(int argc, char** argv) {
	PointCloud<PointXYZI>::Ptr cloud(new PointCloud<PointXYZI>);
 
	if (io::loadPCDFile(argv[1], *cloud) == -1) { // 
		cerr << "can't read file" << argv[1] << endl;
		return -1;
	}
 
	pcl::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
 
	pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> fildColor(cloud,"intensity");
 
	viewer->addPointCloud<pcl::PointXYZI>(cloud, fildColor, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud"); // 
 
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		std::this_thread::sleep_for(std::chrono::microseconds(100000));

	}
 
	return 0;
}