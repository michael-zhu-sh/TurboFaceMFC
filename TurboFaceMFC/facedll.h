#pragma once
#include <string>
#include <vector>

/*
初始化系统。
*/
__declspec(dllexport) int init(
	const std::string& faceDbPath,
	const std::string& faceModelPath
);

/*
搜索。
*/
__declspec(dllexport) int create(
	const std::string& faceDbPath,
	const std::string& faceModelPath,
	const std::string& faceImagesPath,
	bool append = true
);

/*
建库。
*/
__declspec(dllexport) size_t search(
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
__declspec(dllexport) size_t count();
