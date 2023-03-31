#include <utils.h>
#include <fstream>
#include <stdio.h>
#include <dirent.h>
#include <sys/types.h>
#include <unistd.h>  // access
#include <sys/stat.h> // mkdir
#include <omp.h>

namespace utils{
    inline void checkpath(std::string &path){
    if(path[path.size()-1] != '/')
        path = path + "/";
    }   

    void makedir(char* folder){
        if(access(folder,F_OK)){
            if(mkdir(folder,0755)!=0)
                printf("Directory %s created Failed with unknown error!\n",folder);
            else
                printf("Directory %s created successfully!\n",folder);
        }else
            printf("Directory %s not accessible or alreadly exists!\n",folder);
    }
    int getNearestElement(double arr[], int n, int target) {
    if (target <= arr[0])
        return arr[0];
    if (target >= arr[n - 1])
        return arr[n - 1];
    int left = 0, right = n, mid = 0;
    while (left < right) {
        mid = (left + right) / 2;
        if (arr[mid] == target)
            return arr[mid];
        if (target < arr[mid]) {
            if (mid > 0 && target > arr[mid - 1])
                return target-arr[mid-1] <= arr[mid]-target ? mid-1 : mid;
            right = mid;
        } 
        else {
            if (mid < n - 1 && target < arr[mid + 1])
            return target-arr[mid] <= arr[mid+1]-target ? mid : mid+1;
            left = mid + 1;
        }
    }
    return mid;
    }
    /**
     * @brief QuickSort
     * @ref https://blog.csdn.net/weixin_59566851/article/details/122097368
     * @param array 
     * @param low 
     * @param high 
     */
    void QuickSort(std::vector<double> array,int low,int high){	//快排 
        if(low>=high){	//若待排序序列只有一个元素，返回空 
            return ;
        }
        int i=low;	//i作为指针从左向右扫描 
        int j=high;	//j作为指针从右向左扫描
        int key=array[low];//第一个数作为基准数 
        while(i<j){
            while(array[j]>=key&&i<j){	//从右边找小于基准数的元素 （此处由于j值可能会变，所以仍需判断i是否小于j） 
                j--;	//找不到则j减一 
            }
            array[i]=array[j];	//找到则赋值 
            while(array[i]<=key&&i<j){	//从左边找大于基准数的元素 
                i++;	//找不到则i加一 
            }
            array[j]=array[i];	//找到则赋值 
        }
        array[i]=key;	//当i和j相遇，将基准元素赋值到指针i处 
        QuickSort(array,low,i-1);	//i左边的序列继续递归调用快排 
        QuickSort(array,i+1,high);	//i右边的序列继续递归调用快排 
    }

}
