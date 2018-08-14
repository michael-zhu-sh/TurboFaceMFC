
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

public:
	afx_msg void OnMenuCreatedb();

private:
	CFont m_Font;
	std::string inputFaceFile;
	float distanceThreshold = 0.6f;
	std::string faceDbCsv = "C:/turboface/db/facedb.csv";
	std::vector<std::pair<std::string, float>> matchedImageVec;

	void drawOutputFace(UINT controlId, std::string imgFile);
	void writeTextOnControl(UINT controlId, CString text);

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
};
