#pragma once
//STL headers.
#include <string>
#include <vector>
//DLIB headers.
#include <dlib/matrix.h>


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

namespace face {
class Helper {
	public:
		//������������src��dst����֮�����С���롣
		static float minDistance(dlib::matrix<float> src, std::vector<dlib::matrix<float,0,1>> dst);
};
}