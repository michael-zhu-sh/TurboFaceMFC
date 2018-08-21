#pragma once
#include <string>
#include <vector>

/*
��ʼ��ϵͳ��
*/
__declspec(dllexport) int init(
	const std::string& faceDbPath,
	const std::string& faceModelPath
);

/*
������
*/
__declspec(dllexport) int create(
	const std::string& faceDbPath,
	const std::string& faceModelPath,
	const std::string& faceImagesPath,
	bool append = true
);

/*
���⡣
*/
__declspec(dllexport) size_t search(
	const std::string& faceImageFile,
	const std::string& faceDbPath,
	const std::string& faceModelPath,
	const float distanceThreshold,
	std::vector<std::string>& matchedImgFileVec);
