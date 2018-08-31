/*
author: Michael Zhu.
email:michael.ai@foxmail.com
*/
#include "stdafx.h"

//STL headers.
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <io.h>

//DLib headers.
#include <dlib/cmd_line_parser.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/dnn.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/logger.h>

//project headers.
#include "facelib.h"

using namespace std;
using namespace dlib;


template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
	alevel0<
	alevel1<
	alevel2<
	alevel3<
	alevel4<
	max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
	input_rgb_image_sized<150>
	>>>>>>>>>>>>;

/*������$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
frontal_face_detector gDetector = get_frontal_face_detector();	//���������������
shape_predictor gSp;//�궨������
anet_type gNet;		//����ResNet׼��������������ץȡ��
bool gNNFlag = false;	//�������Ƿ��ѳ�ʼ����
/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

/*����$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
std::map< std::string, std::vector<matrix<float,0,1>> > gFaceCache;	//���⻺�档
bool gDBFlag = false;	//�����Ƿ��Ѽ��ص����档
/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

/*��־$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
class LoggerHook	{
private:
	ofstream fout;

public:
	LoggerHook() {
		fout.open("c:/turboface/logs/face.log.txt", ios::app);
	}

	void log(
		const string& logger_name,
		const log_level& ll,
		const uint64 thread_id,
		const char* message_to_log
	)
	{
		CTime now = CTime::GetCurrentTime();
		CString tStr = now.Format("%Y-%m-%d %H:%M:%S");
		fout << "[" << CT2A(tStr) << "]["<<ll<<"] " << logger_name << ": " << message_to_log << endl;
		// Log all messages from any logger to our log file.
		//fout << ll << " [" << thread_id << "] " << logger_name << ": " << message_to_log << endl;
	}

};
LoggerHook hook;
logger flog("face");
bool gLogFlag = false;
/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

/*class Helper=============================================================================*/
float face::Helper::minDistance(dlib::matrix<float> src, std::vector<dlib::matrix<float,0,1>> dst) {
	float minDist = 1.0f,dist;
	for (size_t i = 0; i != dst.size(); i++) {
		dist = length(src - dst[i]);
		if ( dist < minDist) {
			minDist = dist;
		}
	}

	return minDist;
}
/*=============================================================================*/

/*=============================================================================*/
void splitString(const std::string& s, std::vector<std::string>& v, const std::string& c)
{
	std::string::size_type pos1, pos2;
	pos2 = s.find(c);
	pos1 = 0;
	while (std::string::npos != pos2)
	{
		v.push_back(s.substr(pos1, pos2 - pos1));

		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}
	if (pos1 != s.length()) {
		v.push_back(s.substr(pos1));
	}
}

void initLog() {
	if (!gLogFlag) {
		set_all_logging_output_hooks(hook);
		flog.set_level(LALL);
		gLogFlag = true;
		flog << LDEBUG << "Succeed to initialize log file c:/turboface/logs/face.log.txt.";
	}
}

/*
��ʼ��������ϵͳ��
return: 0��ʾ��ʼ���ɹ���-1��ʾʧ�ܡ�
*/
int initModel(const std::string& faceModelPath) {
	if (gNNFlag)	return 0;

	const string nnLandmarkFile = faceModelPath + "/turboface_landmark.dat";
	const string nnModelFile = faceModelPath + "/turboface_model.dat";
	ifstream nnLandmark(nnLandmarkFile);
	if (!nnLandmark) {
		nnLandmark.close();
		flog << LERROR << "FAIL to open and read turboface_landmark.dat by file " + nnLandmarkFile + " in function initModel().";
		return -1;
	}
	else {
		nnLandmark.close();
	}

	ifstream nnModel(nnModelFile);
	if (!nnModel) {
		nnModel.close();
		flog << LERROR << "FAIL to open and read turboface_model.dat by file " + nnModelFile + " in function initModel().";
		return -1;
	}
	else {
		nnModel.close();
	}
	deserialize(nnLandmarkFile) >> gSp;
	deserialize(nnModelFile) >> gNet;
	gNNFlag = true;
	flog << LDEBUG << "Succeed to initialize 2 model files [" + nnLandmarkFile + "] and [" + nnModelFile + "] in function initModel().";

	return 0;
}

/*
�������⵽���档
*/
int initDB(const std::string& faceDbPath) {
	if (gDBFlag) {
		//�ѳ�ʼ�����⣬SKIP.
		return 0;
	}
	CTime t1 = CTime::GetCurrentTime();
	const string faceDbFile = faceDbPath + "/turbofacedb.dat";
	ifstream facedb(faceDbFile);
	if (!facedb) {
		facedb.close();
		flog << LERROR << "FAIL to open facedb " + faceDbFile + " in function initDB()";
		return -1;
	}
	else {
		flog << LINFO << "Succeed to open facedb " + faceDbFile + " in function initDB()";
	}

	std::string line,fvStr;
	std::vector<std::string> strVec,fvVec;
	size_t featureNum;
	float fv;
	stringstream ss;
	while (!facedb.eof()) {
		facedb >> line;
		if (line.empty())	continue;

		strVec.clear();
		splitString(line, strVec, "|");
		ss << strVec[1];//�м����������������1��ͼ�����ж��faces������ж�����������������ˡ�
		ss >> featureNum;
		ss.clear();
		std::vector<matrix<float,0,1>> features;
		for (size_t i = 0; i != featureNum; i++) {
			fvStr = strVec[2+i];//����������ֵ����
			splitString(fvStr, fvVec, ",");//������ÿ��ֵ��
			matrix<float, 128, 1> feature;
			for (size_t r = 0; r != 128; r++) {
				ss << fvVec[r];
				ss >> fv;
				feature(r, 0) = fv;
				ss.clear();
			}
			fvVec.clear();
			features.push_back(feature);
		}
		gFaceCache.insert(std::pair<std::string, std::vector<matrix<float,0,1>>>(strVec[0], features));
	}
	facedb.close();
	gDBFlag = true;
	CTimeSpan ts = CTime::GetCurrentTime() - t1;
	flog << LINFO << "In fuction initDB(), it takes " << ts.GetTotalSeconds() << " seconds to load " << gFaceCache.size() << " faces from FaceDB file to memory cache.";

	return 0;
}

inline bool CmpByValue(const pair<std::string, float>& left, const pair<std::string, float>& right) {
	//distanceС������ǰ�棬�������С�
	return left.second < right.second;
}

/*
�ݹ�Ѱ��ָ��Ŀ¼�����е�jpg��png�ļ���
*/
void findFilesRecursively(const string& path, std::vector<string>& fileVec)
{
	//�ļ����  
	intptr_t hFile = 0;
	//�ļ���Ϣ  
	struct _finddata_t fileinfo;  //�����õ��ļ���Ϣ��ȡ�ṹ
	string p;  //string�������˼��һ����ֵ����:assign()���кܶ����ذ汾
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib &  _A_SUBDIR))  //�Ƚ��ļ������Ƿ����ļ���
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				{
					findFilesRecursively(p.assign(path).append("/").append(fileinfo.name), fileVec);
				}
			}
			else
			{
				if (strstr(fileinfo.name, "jpg") || strstr(fileinfo.name, "png")) {
					fileVec.push_back(p.assign(path).append("/").append(fileinfo.name));
				}
				else {
					std::string str(fileinfo.name);
					flog << LWARN << str + " is NOT a jpg or png, we SKIP it in function findFilesRecursively()!";
				}
			}
		} while (_findnext(hFile, &fileinfo) == 0);  //Ѱ����һ�����ɹ�����0������-1

		_findclose(hFile);
	}
}

/*=============================================================================*/
/*
��ʼ��ϵͳ��
*/
int init(
	const std::string& faceDbPath,
	const std::string& faceModelPath
) {
	int ret = 0;
	CTime t1 = CTime::GetCurrentTime();

	initLog();

	ret	= initModel(faceModelPath);

	ret	= initDB(faceDbPath);

	CTimeSpan ts = CTime::GetCurrentTime() - t1;
	flog << LDEBUG << "In function init()_external, it takes " << ts.GetTotalSeconds() << " seconds to finish.";

	return ret;
}

/*
���⡣
*/
int create(
	const std::string& faceDbPath,
	const std::string& faceModelPath,
	const std::string& faceImagesPath,
	bool append
) {
	if (faceDbPath.find('\\') != string::npos || faceModelPath.find('\\') != string::npos || faceImagesPath.find('\\') != string::npos) {
		flog << LERROR << "parameter path seperator should be char '/' in dllexport create(), so function return -1 now!";
		return -1;
	}
	if (0 != initModel(faceModelPath)) {
		flog << LERROR << "Because call initModel() return not 0 in dllexport create(), so function return -1 now!";
		return -1;
	}

	std::string msgStr;
	const string faceDbFile = faceDbPath + "/turbofacedb.dat";
	ofstream facedb(faceDbFile, append ? ofstream::app : ofstream::trunc);
	if (facedb.eof() || !facedb) {
		flog << LERROR << "FAIL to open or create facedb [" << faceDbFile << "] in function create(), so function return -2 now!";
		return -2;
	}

	CTime t1 = CTime::GetCurrentTime();
	std::vector<string> faceFileVecSource;	//�������е�ͼ���ļ�����
	findFilesRecursively(faceImagesPath, faceFileVecSource);	//�ݹ�����faceImagesPathĿ¼�µ������ļ���

	if (!append) {
		//����Ǹ���ģʽ���⣬��������⻺�档
		gFaceCache.clear();
	}
	else {
		//�����׷��ģʽ���������档
	}

	std::vector<matrix<rgb_pixel>> faces;
	size_t cnt = 0;
	stringstream ss;
	ss << faceFileVecSource.size();
	ss >> msgStr;
	flog << LINFO << "In function create()_external we find " + msgStr + " image files to process, so wait a long time......";
	for (size_t i = 0; i != faceFileVecSource.size(); i++) {
		//�������е�ͼ���ļ����������ֻ��1����������д�����⡣
		if (append && 0 != gFaceCache.count(faceFileVecSource[i])) {
			//��ǰ�������Ѵ������������SKIP it.
			continue;
		}

		matrix<rgb_pixel> img;
		load_image(img, faceFileVecSource[i]);

		faces.clear();
		for (auto face : gDetector(img))	{
			auto shape = gSp(img, face);
			matrix<rgb_pixel> face_chip;
			extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
			faces.push_back(move(face_chip));
		}
		if (0 == faces.size()) {
			flog << LWARN << "In function create()_external, we can NOT find any faces in [" << faceFileVecSource[i] << "], so continue to next image file.";
			continue;
		}
		else if (1 != faces.size()) {
			ss.clear();
			ss << faces.size();
			msgStr.clear();
			ss >> msgStr;
			flog << LDEBUG << "In function create()_external, " << msgStr << " faces are found in [" << faceFileVecSource[i] << "].";
		}
		std::vector<matrix<float,0,1>> features = gNet(faces);
		facedb << faceFileVecSource[i] << "|" << features.size() << "|";
		for (size_t j = 0; j != features.size(); j++) {
			matrix<float> feature = features[j];
			for (size_t r = 0; r != feature.nr(); r++) {
				facedb << feature(r, 0) << ",";
			}
			facedb << "|";
		}
		facedb << endl;
		cnt++;

		//writeback to memory cache.
		gFaceCache.insert(std::pair< std::string, std::vector<matrix<float,0,1>> >(faceFileVecSource[i], features));

		if (i % 1000 == 0) {
			ss.clear();
			msgStr.clear();
			ss << i;
			ss >> msgStr;
			flog << LINFO << "In function create()_external, we already proceed " << msgStr << "image files.";
			ss.clear();
			msgStr.clear();
			ss << (faceFileVecSource.size() - i);
			ss >> msgStr;
			flog << LINFO << "But total " << msgStr << " remainder image files need to be proceed, please drink a cup of coffee to wait......";
		}
		if (i % 100 == 0) {
			ss.clear();
			msgStr.clear();
			ss << i;
			ss >> msgStr;
			flog << LDEBUG << "In function create()_external, we already proceed " << msgStr << " image files, please drink a cup of coffee to wait......";
		}
	}
	ss.clear();
	msgStr.clear();
	ss << cnt;
	ss >> msgStr;
	if (append) {
		flog << LINFO << msgStr << " face row records are APPEND to FaceDB.";
	}
	else {
		flog << LINFO << msgStr << " face row records are OVERWRITE to FaceDB.";
	}
	facedb.close();
	CTimeSpan ts = CTime::GetCurrentTime() - t1;
	flog << LINFO << "In create()_external, it takes " << ts.GetTotalSeconds() << " seconds to proceed all of image files.";

	return 0;
}

/*
���������
faceImgFile:��Ҫ��������ͼ���ļ���ȫ·���������п���û��������Ҳ�����ж��������
faceDbPath:�������ݿ�Ŀ¼��
distanceThreshold:2���������ƶ���ֵ��ֻ��С�ڴ���ֵ������ͼ��Żᱻѡ�񲢷��ء�

���������
matchedImgFileVec:������������ֵҪ�������ͼ��ȫ·������

�����룺
0��δ�ҵ�ƥ���������
-1����ʼ������ʧ�ܣ��޷�����������
-2�������ļ���ȡ����
-3������ͼ���ļ���û�м�⵽������
-4������ͼ���ļ��м�⵽���������
*/
int search(
	const std::string& faceImageFile,
	const std::string& faceDbPath,
	const std::string& faceModelPath,
	const float distanceThreshold,
	std::vector<std::string>& matchedImgFileVec) {

	if (0 != init(faceDbPath, faceModelPath)) {
		//��ʼ��ϵͳʧ�ܣ�
		flog << LERROR << "In function search()_external, FAIL to call init() so return -1!";
		return -1;
	}
	ifstream imageFile(faceImageFile);
	if (!imageFile) {
		imageFile.close();
		flog << LERROR << "In function search()_external, FAIL to open faceImageFile so return -2!";
		return -2;
	}
	else {
		imageFile.close();
	}

	CTime t1 = CTime::GetCurrentTime();
	matrix<rgb_pixel> img;
	load_image(img, faceImageFile);
	std::vector<dlib::rectangle> dets = gDetector(img);
	if (0 == dets.size()) {
		flog << LWARN << "In function search()_external, Can NOT find any faces in input file:" << faceImageFile << ", so return 0.";
		return -3;
	}
	else if (1 != dets.size()) {
		flog << LWARN << "In function search()_external, more than 1 faces are found in input file:" << faceImageFile << ", so return 0.";
		return -4;
	}
	else {
		flog << LDEBUG << "In function search()_external, 1 faces are found in input file:" << faceImageFile << " normally.";
	}

	auto shape = gSp(img, dets[0]);
	matrix<rgb_pixel> face_chip;
	extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
	std::vector<matrix<rgb_pixel>> faces;
	faces.push_back(move(face_chip));
	std::vector<matrix<float, 0, 1>> features = gNet(faces);
	matrix<float> feature = features[0];
	float distance;
	std::vector<pair<std::string, float>> matchedImageVec;

	for (std::map<std::string, std::vector<matrix<float,0,1>>>::iterator it = gFaceCache.begin(); it != gFaceCache.end(); ++it) {
		distance = face::Helper::minDistance(feature, (it->second));
		if (distance < distanceThreshold) {
			matchedImageVec.push_back(make_pair(it->first, distance));
		}
	}
	sort(matchedImageVec.begin(), matchedImageVec.end(), CmpByValue);
	if (0 != matchedImgFileVec.size()) {
		matchedImgFileVec.clear();
	}
	for (size_t i = 0; i != matchedImageVec.size(); i++) {
		if (i > 7)	break;	//��෵��8������ͼ���ļ���
		matchedImgFileVec.push_back(matchedImageVec[i].first);
	}
	CTimeSpan ts = CTime::GetCurrentTime() - t1;
	flog << LDEBUG << "In function search()_external, it takes " << ts.GetTotalSeconds() << " seconds to finish.";

	return matchedImgFileVec.size();
}

/*
��ѯ�������ݼ�¼����

����ֵ��
�������ݼ�¼����
*/
size_t count() {
	size_t rows = 0;
	if (gDBFlag) {
		rows = gFaceCache.size();
	}
	else {
		flog << LWARN << "In function count()_external, FaceDB is NOT initialized, please call init() first THEN call count().";
		rows = -1;
	}

	return rows;
}

/*
void MatToCImage(cv::Mat& mat, CImage& cimage) {
if (mat.empty() || 0 == mat.total()) {
return;
}
int nChannels = mat.channels();
if ((1 != nChannels) && (3 != nChannels)) {
return;
}
int64 nWidth = mat.cols;
int64 nHeight = mat.rows;
//�ؽ�cimage
cimage.Destroy();
cimage.Create(nWidth, nHeight, 8 * nChannels);
//��������
uchar* pucRow;									//ָ������������ָ��
uchar* pucImage = (uchar*)cimage.GetBits();		//ָ����������ָ��
int nStep = cimage.GetPitch();					//ÿ�е��ֽ���,ע���������ֵ�����и�
if (1 == nChannels)								//���ڵ�ͨ����ͼ����Ҫ��ʼ����ɫ��
{
RGBQUAD* rgbquadColorTable;
int nMaxColors = 256;
rgbquadColorTable = new RGBQUAD[nMaxColors];
cimage.GetColorTable(0, nMaxColors, rgbquadColorTable);
for (int nColor = 0; nColor < nMaxColors; nColor++) {
rgbquadColorTable[nColor].rgbBlue = (uchar)nColor;
rgbquadColorTable[nColor].rgbGreen = (uchar)nColor;
rgbquadColorTable[nColor].rgbRed = (uchar)nColor;
}
cimage.SetColorTable(0, nMaxColors, rgbquadColorTable);
delete[] rgbquadColorTable;
}
for (int nRow = 0; nRow < nHeight; nRow++)
{
pucRow = (mat.ptr<uchar>(nRow));
for (int nCol = 0; nCol < nWidth; nCol++)
{
if (1 == nChannels)
{
*(pucImage + nRow * nStep + nCol) = pucRow[nCol];
}
else if (3 == nChannels)
{
for (int nCha = 0; nCha < 3; nCha++)
{
*(pucImage + nRow * nStep + nCol * 3 + nCha) = pucRow[nCol * 3 + nCha];
}
}
}
}
}

bool turbo::Video::open(const std::string& videoFile) {
this->capture.open(videoFile);
if (this->capture.isOpened())	return true;

return false;
}
bool turbo::Video::read(CImage& cimage) {
cv::Mat frame;
bool b = this->capture.read(frame);
MatToCImage(frame, cimage);

return b;
}
void turbo::Video::release() {
this->capture.release();
}
*/
