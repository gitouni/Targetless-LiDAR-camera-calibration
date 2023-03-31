#include <fstream>
#include <stdio.h>
#include <dirent.h>
#include <sys/types.h>
#include <unistd.h>  // access
#include <sys/stat.h> // mkdir
#include <vector>

namespace utils{
    inline void checkpath(std::string &path){
    if(path[path.size()-1] != '/')
        path = path + "/";
    }   

    void makedir(char* folder){
        if(access(folder,F_OK)){
            if(mkdir(folder,0755)!=0)
                printf("\033[1;33mDirectory %s created Failed with unknown error!\033[0m\n",folder);
            else
                printf("\033[1;34mDirectory %s created successfully!\033[0m\n",folder);
        }else
            printf("\033[1;35mDirectory %s not accessible or alreadly exists!\033[0m\n",folder);
    }
    template <typename T>
    int getNearestElement(std::vector<T> &arr, T target) {
    int n = arr.size();
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
    template <typename T, typename idxType>
    void _QuickSort(std::vector<T> &array,idxType low,idxType high){	//快排 
        if(low >= high){	//若待排序序列只有一个元素，返回空 
            return ;
        }
        idxType i = low;	//i作为指针从左向右扫描 
        idxType j = high;	//j作为指针从右向左扫描
        T key = array[low];//第一个数作为基准数 
        while(i < j){
            while(array[j]>=key && i<j){	//从右边找小于基准数的元素 （此处由于j值可能会变，所以仍需判断i是否小于j） 
                j--;	//找不到则j减一 
            }
            array[i] = array[j];	//找到则赋值 
            while(array[i]<=key && i<j){	//从左边找大于基准数的元素 
                i++;	//找不到则i加一 
            }
            array[j] = array[i];	//找到则赋值 
        }
        array[i] = key;	//当i和j相遇，将基准元素赋值到指针i处 
        _QuickSort(array,low,i-1);	//i左边的序列继续递归调用快排 
        _QuickSort(array,i+1,high);	//i右边的序列继续递归调用快排 
    }
    /**
     * @brief QuickSort
     * @ref https://blog.csdn.net/weixin_59566851/article/details/122097368
     * @param array 
     * @param low 
     * @param high 
     */
    template <typename T, typename idxType>
    void _QuickArgsort(std::vector<T> &array, std::vector<idxType> &indice ,idxType low,idxType high){	//快排 
        if(low >= high){	//若待排序序列只有一个元素，返回空 
            return ;
        }
        idxType i = low;	//i作为指针从左向右扫描 
        idxType j = high;	//j作为指针从右向左扫描
        T key = array[low];//第一个数作为基准数 
        while(i<j){
            while(array[j]>=key && i<j){	//从右边找小于基准数的元素 （此处由于j值可能会变，所以仍需判断i是否小于j） 
                j--;	//找不到则j减一 
            }
            array[i] = array[j];	//找到则赋值 
            indice[i] = indice[j];
            while(array[i]<=key && i<j){	//从左边找大于基准数的元素 
                i++;	//找不到则i加一 
            }
            array[j]=array[i];	//找到则赋值 
            indice[j] = indice[i];
        }
        array[i] = key;	//当i和j相遇，将基准元素赋值到指针i处 
        indice[i] = low;
        _QuickArgsort(array,low,i-1);	//i左边的序列继续递归调用快排 
        _QuickArgsort(array,i+1,high);	//i右边的序列继续递归调用快排 
    }
    template <typename T>
    void QuickSort(std::vector<T> &arr){
        std::size_t low = 0;
        std::size_t high = arr.size()-1;
        _QuickSort(arr,low,high);
    }
    template <typename T>
    std::vector<int> QuickArgsort(std::vector<T> &arr){
        std::size_t low = 0;
        std::size_t high = arr.size() - 1;
        std::vector<int> indice;
        for(std::size_t i=low;i<=high;++i){
            indice.push_back(i);
        }
        _QuickArgsort(arr,indice,low,high);
        return indice;
    }


}
