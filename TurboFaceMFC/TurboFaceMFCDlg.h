
// TurboFaceMFCDlg.h: 头文件
//

#pragma once
#include <string>
#include <vector>

using namespace std;


// CTurboFaceMFCDlg 对话框
class CTurboFaceMFCDlg : public CDialogEx
{
// 构造
public:
	CTurboFaceMFCDlg(CWnd* pParent = nullptr);	// 标准构造函数

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_TURBOFACEMFC_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持


// 实现
protected:
	HICON m_hIcon;

	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()

private:
	CString registrySection = _T("turboface");
	CString regFaceDBPath	= _T("FaceDBPath");	//脸库数据文件目录。
	CString regFaceModelPath= _T("FaceModelPath");	//脸库模型文件目录。

	CFont m_Font;
	std::string inputFaceFile;
	std::string dbPathStr;
	std::string modelPathStr;
	std::vector<std::string> matchedImageVec;

	//在controlId指定的控件上，把imgFile指定的图像文件画上去。
	void drawOutputFace(UINT controlId, std::string imgFile);

	//在controlId指定的控件上，输出text文本。
	void writeTextOnControl(UINT controlId, CString text);

	//初始化显示8个图像输出控件。
	void displayOutputFaceControl();

	//清空8个图像控件。
	void clearOutputFaceControl();

public:
	afx_msg void OnBnClickedOpenButton();
	afx_msg void OnBnClickedSearchButton();
	afx_msg void OnStnClickedOutputFace1();
	afx_msg void OnStnClickedOutputFace2();
	afx_msg void OnStnClickedOutputFace3();
	afx_msg void OnStnClickedOutputFace4();
	afx_msg void OnStnClickedOutputFace5();
	afx_msg void OnStnClickedOutputFace6();
	afx_msg void OnStnClickedOutputFace7();
	afx_msg void OnStnClickedOutputFace8();
	afx_msg void On32771();
	CSpinButtonCtrl m_spin;

	CString defaultRegFaceDBPath	= _T("c:/turboface/db");
	CString defaultRegFaceModelPath = _T("c:/turboface/model");

	const int BUTTON_HEIGHT = 32;
};
