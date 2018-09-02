#pragma once
//STL headers.
#include <string>
#include <vector>
//DLIB headers.
#include <dlib/matrix.h>


/*
初始化系统。
*/
int init(
	const std::string& faceDbPath,
	const std::string& faceModelPath
);

/*
建库。
*/
int create(
	const std::string& faceDbPath,
	const std::string& faceModelPath,
	const std::string& faceImagesPath,
	bool append = true
);

/*
搜索。
*/
int search(
	const std::string& faceImageFile,
	const std::string& faceDbPath,
	const std::string& faceModelPath,
	const float distanceThreshold,
	std::vector<std::string>& matchedImgFileVec);

/*
查询脸库数据记录数。

返回值：
脸库数据记录数。
*/
size_t count();

namespace face {
class Helper {
	public:
		//计算特征矩阵src和dst数据之间的最小距离。
		static float minDistance(dlib::matrix<float> src, std::vector<dlib::matrix<float,0,1>> dst);
};
}