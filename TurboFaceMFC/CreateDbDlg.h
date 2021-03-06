#pragma once
#include <string>

using namespace std;


// CreateDbDlg 对话框

class CreateDbDlg : public CDialogEx
{
	DECLARE_DYNAMIC(CreateDbDlg)

public:
	CreateDbDlg(CWnd* pParent = nullptr);   // 标准构造函数
	virtual ~CreateDbDlg();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_CREATEDB_DIALOG };
#endif

protected:
	virtual BOOL OnInitDialog();
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedSelectDbfileButton();
	afx_msg void OnBnClickedSelectModelButton();
	afx_msg void OnBnClickedSelectImageDirButton();


public:
	CString registrySection		= _T("turboface");
	CString regFaceImgPath	= _T("FaceImagePath");	//人脸图像文件目录。
	CString regFaceDBPath	= _T("FaceDBPath");	//脸库数据文件目录。
	CString regFaceModelPath = _T("FaceModelPath");	//脸库模型文件目录。

	afx_msg void OnBnClickedDoCreatedbButton();

private:
	static UINT childThread(LPVOID param);
	void threadEntry();

private:
	std::string dbPathStr;
	std::string modelPathStr;
	std::string imgPathStr;
	bool append;
	CString m_log;
};
