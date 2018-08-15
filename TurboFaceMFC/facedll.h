#pragma once
#include <string>
#include <vector>

__declspec(dllexport) size_t search(
	const std::string& faceImageFile,
	const std::string& faceDbPath,
	const std::string& faceModelPath,
	const float distanceThreshold,
	std::vector<std::string>& matchedImgFileVec);
