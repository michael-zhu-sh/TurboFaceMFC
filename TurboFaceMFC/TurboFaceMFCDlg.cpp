
// TurboFaceMFCDlg.cpp: 实现文件
//

#include "stdafx.h"
#include "TurboFaceMFC.h"
#include "TurboFaceMFCDlg.h"
#include "afxdialogex.h"
#include "facedll.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

#define INPUT_PICTURE_WIDTH 512
#define INPUT_PICTURE_HEIGHT 512
#define MARGIN_BTWEEN_INPUT_OUTPUT 128
#define OUTPUT_PICTURE_WIDTH 256
#define OUTPUT_PICTURE_HEIGHT 256

using namespace std;

// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
	ON_BN_CLICKED(IDC_SEARCH_BUTTON, &CTurboFaceMFCDlg::OnBnClickedSearchButton)
END_MESSAGE_MAP()


// CTurboFaceMFCDlg 对话框



CTurboFaceMFCDlg::CTurboFaceMFCDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_TURBOFACEMFC_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CTurboFaceMFCDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CTurboFaceMFCDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_OPEN_BUTTON, &CTurboFaceMFCDlg::OnBnClickedOpenButton)
	ON_BN_CLICKED(IDC_SEARCH_BUTTON, &CTurboFaceMFCDlg::OnBnClickedSearchButton)
	ON_STN_CLICKED(IDC_OUTPUT_FACE1, &CTurboFaceMFCDlg::OnStnClickedOutputFace1)
	ON_STN_CLICKED(IDC_OUTPUT_FACE2, &CTurboFaceMFCDlg::OnStnClickedOutputFace2)
	ON_STN_CLICKED(IDC_OUTPUT_FACE3, &CTurboFaceMFCDlg::OnStnClickedOutputFace3)
	ON_STN_CLICKED(IDC_OUTPUT_FACE4, &CTurboFaceMFCDlg::OnStnClickedOutputFace4)
	ON_STN_CLICKED(IDC_OUTPUT_FACE5, &CTurboFaceMFCDlg::OnStnClickedOutputFace5)
	ON_STN_CLICKED(IDC_OUTPUT_FACE6, &CTurboFaceMFCDlg::OnStnClickedOutputFace6)
	ON_STN_CLICKED(IDC_OUTPUT_FACE7, &CTurboFaceMFCDlg::OnStnClickedOutputFace7)
	ON_STN_CLICKED(IDC_OUTPUT_FACE8, &CTurboFaceMFCDlg::OnStnClickedOutputFace8)
	ON_COMMAND(ID_32771, &CTurboFaceMFCDlg::On32771)
END_MESSAGE_MAP()


// CTurboFaceMFCDlg 消息处理程序

BOOL CTurboFaceMFCDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}
	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标
	ShowWindow(SW_MAXIMIZE);

	/*自定义控件的初始化============================================================================*/
	CMenu menu;
	menu.LoadMenuW(IDR_MENU1);
	SetMenu(&menu);

	//固定Picture Control控件IDC_INPUT_FACE的大小为512x512。  
	CRect rect;
	GetDlgItem(IDC_INPUT_FACE)->GetWindowRect(&rect);
	ScreenToClient(&rect);
	int x = rect.left;
	int y = rect.top;
	GetDlgItem(IDC_INPUT_FACE)->MoveWindow(x, y, INPUT_PICTURE_WIDTH, INPUT_PICTURE_HEIGHT, true);

	//固定Picture Control控件IDC_OUTPUT_FACE1的大小为256x256。  
	const int x1 = x + INPUT_PICTURE_WIDTH + MARGIN_BTWEEN_INPUT_OUTPUT;
	int y1 = y;
	GetDlgItem(IDC_OUTPUT_FACE1)->MoveWindow(x1, y1, OUTPUT_PICTURE_WIDTH, OUTPUT_PICTURE_HEIGHT, true);
	int x2 = x1 + OUTPUT_PICTURE_WIDTH;
	int y2 = y1;
	GetDlgItem(IDC_OUTPUT_FACE2)->MoveWindow(x2, y2, OUTPUT_PICTURE_WIDTH, OUTPUT_PICTURE_HEIGHT, true);
	int x3 = x2 + OUTPUT_PICTURE_WIDTH;
	int y3 = y1;
	GetDlgItem(IDC_OUTPUT_FACE3)->MoveWindow(x3, y3, OUTPUT_PICTURE_WIDTH, OUTPUT_PICTURE_HEIGHT, true);
	int x4 = x3 + OUTPUT_PICTURE_WIDTH;
	int y4 = y1;
	GetDlgItem(IDC_OUTPUT_FACE4)->MoveWindow(x4, y4, OUTPUT_PICTURE_WIDTH, OUTPUT_PICTURE_HEIGHT, true);

	int x5 = x1;
	int y5 = y1 + OUTPUT_PICTURE_HEIGHT;
	GetDlgItem(IDC_OUTPUT_FACE5)->MoveWindow(x5, y5, OUTPUT_PICTURE_WIDTH, OUTPUT_PICTURE_HEIGHT, true);
	int x6 = x5 + OUTPUT_PICTURE_WIDTH;
	int y6 = y5;
	GetDlgItem(IDC_OUTPUT_FACE6)->MoveWindow(x6, y6, OUTPUT_PICTURE_WIDTH, OUTPUT_PICTURE_HEIGHT, true);
	int x7 = x6 + OUTPUT_PICTURE_WIDTH;
	int y7 = y5;
	GetDlgItem(IDC_OUTPUT_FACE7)->MoveWindow(x7, y7, OUTPUT_PICTURE_WIDTH, OUTPUT_PICTURE_HEIGHT, true);
	int x8 = x7 + OUTPUT_PICTURE_WIDTH;
	int y8 = y5;
	GetDlgItem(IDC_OUTPUT_FACE8)->MoveWindow(x8, y8, OUTPUT_PICTURE_WIDTH, OUTPUT_PICTURE_HEIGHT, true);

	//设置输入人脸图像文件全路径的编辑文本框。
	m_Font.CreatePointFont(80, _T("Arial"), NULL);
	GetDlgItem(IDC_INPUT_FACE_DIR)->SetFont(&m_Font, true);
	GetDlgItem(IDC_INPUT_FACE_DIR)->MoveWindow(x, y + INPUT_PICTURE_HEIGHT + 8, INPUT_PICTURE_WIDTH, 48, true);

	//设置输出人脸图像文件全路径的编辑文本框。
	GetDlgItem(IDC_OUTPUT_FACE_DIR)->SetFont(&m_Font, true);
	GetDlgItem(IDC_OUTPUT_FACE_DIR)->MoveWindow(x1, y8 + OUTPUT_PICTURE_HEIGHT + 8, OUTPUT_PICTURE_WIDTH * 4, 48, true);
	
	/*============================================================================*/

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CTurboFaceMFCDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CTurboFaceMFCDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CTurboFaceMFCDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

void CTurboFaceMFCDlg::drawOutputFace(UINT item, std::string imgFile) {
	CRect   rect;
	CImage  image;
	CWnd* pWnd = NULL;
	CDC* pDc = NULL;
	pWnd = GetDlgItem(item);
	pWnd->GetClientRect(&rect);
	pDc = pWnd->GetDC();
	CString pszFileName(imgFile.c_str());
	HRESULT result = image.Load(pszFileName);
	result = image.Draw(pDc->m_hDC, rect);//将图片绘制到picture表示的区域内  

	ReleaseDC(pDc);
}

void CTurboFaceMFCDlg::writeTextOnControl(UINT controlId, CString text) {
	GetDlgItem(controlId)->SetWindowTextW(text);
}

/*
点击【选择人脸】按钮后弹出文件选择对话框，选择人脸图像文件。
*/
void CTurboFaceMFCDlg::OnBnClickedOpenButton()
{
	// 设置过滤器   
	TCHAR szFilter[] = _T("jpg文件(*.jpg)|*.jpg|png文件(*.png)|*.png|");
	// 构造打开文件对话框   
	CFileDialog fileDlg(TRUE, _T("jpg"), NULL, 0, szFilter, this);
	CString imgFilePath;

	// 显示打开文件对话框   
	if (IDOK == fileDlg.DoModal())
	{
		// 如果点击了文件对话框上的“打开”按钮，则将选择的文件路径显示到编辑框里   
		imgFilePath = fileDlg.GetPathName();
		GetDlgItem(IDC_INPUT_FACE_DIR)->SetWindowTextW(imgFilePath);
		this->inputFaceFile = CT2A(imgFilePath);	//传给searchFace算法函数的文件名称参数值。
	}
	if (this->inputFaceFile.empty())	return;

	int cx, cy;
	CImage  image;
	CRect   rect;
	image.Load(imgFilePath);
	//获取图片的宽 高度  
	cx = image.GetWidth();
	cy = image.GetHeight();

	//获取Picture Control控件的大小  
	GetDlgItem(IDC_INPUT_FACE)->GetWindowRect(&rect);
	//将客户区选中到控件表示的矩形区域内  
	ScreenToClient(&rect);
	//窗口移动到控件表示的区域  
	GetDlgItem(IDC_INPUT_FACE)->MoveWindow(rect.left, rect.top, INPUT_PICTURE_WIDTH, INPUT_PICTURE_HEIGHT, TRUE);
	CWnd *pWnd = NULL;
	pWnd = GetDlgItem(IDC_INPUT_FACE);//获取控件句柄  
	pWnd->GetClientRect(&rect);//获取句柄指向控件区域的大小  

	CDC *pDc = NULL;
	pDc = pWnd->GetDC();//获取picture的DC  

	image.Draw(pDc->m_hDC, rect);//将图片绘制到picture表示的区域内  

	ReleaseDC(pDc);
}

/*
点击【搜索脸库】按钮后，调用DLL中的搜索算法。
*/
void CTurboFaceMFCDlg::OnBnClickedSearchButton()
{
	//如果未选择输入图像文件，则do nothing。
	if (this->inputFaceFile.empty()) {
		MessageBoxA(this->m_hWnd, "请先选择一个包含人脸的图像文件！", "警告", MB_ICONEXCLAMATION);
		return;
	}

	//调用DLL中的搜索函数，函数声明在facedll.h头文件中。
	size_t found = search(
		this->inputFaceFile,
		this->faceDbCsv,
		this->distanceThreshold,
		this->matchedImageVec
	);

	if (0 == found) {
		MessageBoxA(this->m_hWnd, "未在脸库中找到匹配人脸！", "警告", MB_ICONEXCLAMATION);
	}
	else {
		switch (found) {
		case 1:
			drawOutputFace(IDC_OUTPUT_FACE1, matchedImageVec[0].first);
			break;
		case 2:
			drawOutputFace(IDC_OUTPUT_FACE1, matchedImageVec[0].first);
			drawOutputFace(IDC_OUTPUT_FACE2, matchedImageVec[1].first);
			break;
		case 3:
			drawOutputFace(IDC_OUTPUT_FACE1, matchedImageVec[0].first);
			drawOutputFace(IDC_OUTPUT_FACE2, matchedImageVec[1].first);
			drawOutputFace(IDC_OUTPUT_FACE3, matchedImageVec[2].first);
			break;
		case 4:
			drawOutputFace(IDC_OUTPUT_FACE1, matchedImageVec[0].first);
			drawOutputFace(IDC_OUTPUT_FACE2, matchedImageVec[1].first);
			drawOutputFace(IDC_OUTPUT_FACE3, matchedImageVec[2].first);
			drawOutputFace(IDC_OUTPUT_FACE3, matchedImageVec[3].first);
			break;
		case 5:
			drawOutputFace(IDC_OUTPUT_FACE1, matchedImageVec[0].first);
			drawOutputFace(IDC_OUTPUT_FACE2, matchedImageVec[1].first);
			drawOutputFace(IDC_OUTPUT_FACE3, matchedImageVec[2].first);
			drawOutputFace(IDC_OUTPUT_FACE3, matchedImageVec[3].first);
			drawOutputFace(IDC_OUTPUT_FACE5, matchedImageVec[4].first);
			break;
		case 6:
			drawOutputFace(IDC_OUTPUT_FACE1, matchedImageVec[0].first);
			drawOutputFace(IDC_OUTPUT_FACE2, matchedImageVec[1].first);
			drawOutputFace(IDC_OUTPUT_FACE3, matchedImageVec[2].first);
			drawOutputFace(IDC_OUTPUT_FACE3, matchedImageVec[3].first);
			drawOutputFace(IDC_OUTPUT_FACE5, matchedImageVec[4].first);
			drawOutputFace(IDC_OUTPUT_FACE6, matchedImageVec[5].first);
			break;
		case 7:
			drawOutputFace(IDC_OUTPUT_FACE1, matchedImageVec[0].first);
			drawOutputFace(IDC_OUTPUT_FACE2, matchedImageVec[1].first);
			drawOutputFace(IDC_OUTPUT_FACE3, matchedImageVec[2].first);
			drawOutputFace(IDC_OUTPUT_FACE3, matchedImageVec[3].first);
			drawOutputFace(IDC_OUTPUT_FACE5, matchedImageVec[4].first);
			drawOutputFace(IDC_OUTPUT_FACE6, matchedImageVec[5].first);
			drawOutputFace(IDC_OUTPUT_FACE7, matchedImageVec[6].first);
			break;
		case 8:
			drawOutputFace(IDC_OUTPUT_FACE1, matchedImageVec[0].first);
			drawOutputFace(IDC_OUTPUT_FACE2, matchedImageVec[1].first);
			drawOutputFace(IDC_OUTPUT_FACE3, matchedImageVec[2].first);
			drawOutputFace(IDC_OUTPUT_FACE3, matchedImageVec[3].first);
			drawOutputFace(IDC_OUTPUT_FACE5, matchedImageVec[4].first);
			drawOutputFace(IDC_OUTPUT_FACE6, matchedImageVec[5].first);
			drawOutputFace(IDC_OUTPUT_FACE7, matchedImageVec[6].first);
			drawOutputFace(IDC_OUTPUT_FACE8, matchedImageVec[7].first);
			break;
		}
	}
}


void CTurboFaceMFCDlg::OnStnClickedOutputFace1()
{
	if (0 != this->matchedImageVec.size()) {
		CString outputDir((this->matchedImageVec[0]).first.c_str());
		writeTextOnControl(IDC_OUTPUT_FACE_DIR, outputDir);
	}
}


void CTurboFaceMFCDlg::OnStnClickedOutputFace2()
{
	const int idx = 1;
	if (this->matchedImageVec.size()>idx) {
		CString outputDir((this->matchedImageVec[idx]).first.c_str());
		writeTextOnControl(IDC_OUTPUT_FACE_DIR, outputDir);
	}
}


void CTurboFaceMFCDlg::OnStnClickedOutputFace3()
{
	const int idx = 2;
	if (this->matchedImageVec.size()>idx) {
		CString outputDir((this->matchedImageVec[idx]).first.c_str());
		writeTextOnControl(IDC_OUTPUT_FACE_DIR, outputDir);
	}
}


void CTurboFaceMFCDlg::OnStnClickedOutputFace4()
{
	const int idx = 3;
	if (this->matchedImageVec.size()>idx) {
		CString outputDir((this->matchedImageVec[idx]).first.c_str());
		writeTextOnControl(IDC_OUTPUT_FACE_DIR, outputDir);
	}
}


void CTurboFaceMFCDlg::OnStnClickedOutputFace5()
{
	const int idx = 4;
	if (this->matchedImageVec.size()>idx) {
		CString outputDir((this->matchedImageVec[idx]).first.c_str());
		writeTextOnControl(IDC_OUTPUT_FACE_DIR, outputDir);
	}
}


void CTurboFaceMFCDlg::OnStnClickedOutputFace6()
{
	const int idx = 5;
	if (this->matchedImageVec.size()>idx) {
		CString outputDir((this->matchedImageVec[idx]).first.c_str());
		writeTextOnControl(IDC_OUTPUT_FACE_DIR, outputDir);
	}
}


void CTurboFaceMFCDlg::OnStnClickedOutputFace7()
{
	const int idx = 6;
	if (this->matchedImageVec.size()>idx) {
		CString outputDir((this->matchedImageVec[idx]).first.c_str());
		writeTextOnControl(IDC_OUTPUT_FACE_DIR, outputDir);
	}
}


void CTurboFaceMFCDlg::OnStnClickedOutputFace8()
{
	const int idx = 7;
	if (this->matchedImageVec.size()>idx) {
		CString outputDir((this->matchedImageVec[idx]).first.c_str());
		writeTextOnControl(IDC_OUTPUT_FACE_DIR, outputDir);
	}
}

/*
建库。
*/
void CTurboFaceMFCDlg::On32771()
{
	MessageBoxA(this->m_hWnd, "开始建库！", "警告", MB_ICONEXCLAMATION);
}
