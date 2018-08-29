#pragma once
#include <string>
#include <vector>

/*
��ʼ��ϵͳ��
*/
int init(
	const std::string& faceDbPath,
	const std::string& faceModelPath
);

/*
���⡣
*/
int create(
	const std::string& faceDbPath,
	const std::string& faceModelPath,
	const std::string& faceImagesPath,
	bool append = true
);

/*
������
*/
int search(
	const std::string& faceImageFile,
	const std::string& faceDbPath,
	const std::string& faceModelPath,
	const float distanceThreshold,
	std::vector<std::string>& matchedImgFileVec);

/*
��ѯ�������ݼ�¼����

����ֵ��
�������ݼ�¼����
*/
size_t count();
