#pragma once
#include <string>
#include <vector>

__declspec(dllexport) size_t search(
	const std::string& faceImgFile,
	const std::string& faceDbCsv,
	const float distanceThreshold,
	std::vector<std::pair<std::string, float>>& matchedImageVec
);