/*
脸库维护对话框，可以重新建库或者追加。
注册表：
turboface->FaceImageFilesPath,	人脸图像文件目录。
turboface->FaceDBFilePath,		脸库数据库文件目录。
turboface->FaceModelFilesPath,	脸库模型文件目录。

std::string stdstr = CT2A(mfc_str);
CString mfc_str(stdstr.c_str());
*/
#include "stdafx.h"
#include "afxdialogex.h"

#include "TurboFaceMFC.h"
#include "CreateDbDlg.h"
#include "facedll.h"

// CreateDbDlg 对话框

IMPLEMENT_DYNAMIC(CreateDbDlg, CDialogEx)

CreateDbDlg::CreateDbDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_CREATEDB_DIALOG, pParent)
{

}

CreateDbDlg::~CreateDbDlg()
{
}

void CreateDbDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(CreateDbDlg, CDialogEx)
	ON_BN_CLICKED(IDC_SELECT_DBFILE_BUTTON, &CreateDbDlg::OnBnClickedSelectDbfileButton)
	ON_BN_CLICKED(IDC_SELECT_MODEL_BUTTON, &CreateDbDlg::OnBnClickedSelectModelButton)
	ON_BN_CLICKED(IDC_SELECT_IMAGE_DIR_BUTTON, &CreateDbDlg::OnBnClickedSelectImageDirButton)
	ON_BN_CLICKED(IDC_DO_CREATEDB_BUTTON, &CreateDbDlg::OnBnClickedDoCreatedbButton)
END_MESSAGE_MAP()


// CreateDbDlg 消息处理程序
BOOL CreateDbDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
	CString dbPath = AfxGetApp()->GetProfileString(CreateDbDlg::registrySection, CreateDbDlg::regFaceDBPath);
	GetDlgItem(IDC_FACEDB_FILE)->SetWindowTextW(dbPath);
	this->dbPathStr = CT2A(dbPath);	//脸库目录。

	CString modelPath = AfxGetApp()->GetProfileString(CreateDbDlg::registrySection, CreateDbDlg::regFaceModelPath);
	GetDlgItem(IDC_MODEL_FILE)->SetWindowTextW(modelPath);
	this->modelPathStr = CT2A(modelPath);	//模型目录。

	CString imgPath = AfxGetApp()->GetProfileString(CreateDbDlg::registrySection, CreateDbDlg::regFaceImgPath);
	GetDlgItem(IDC_FACEIMG_DIR)->SetWindowTextW(imgPath);
	this->imgPathStr = CT2A(imgPath);	//人脸图像文件目录。

	//设置缺省建库模式为追加模式。
	CheckDlgButton(IDC_APPEND_RADIO, 1);
	/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

	return TRUE;
}

/*
点击【选择脸库数据目录】按钮。
*/
void CreateDbDlg::OnBnClickedSelectDbfileButton()
{
	TCHAR folder[512];     //存放选择的目录路径 
	BROWSEINFO bi;
	bi.hwndOwner = m_hWnd;
	bi.pidlRoot = NULL;
	bi.pszDisplayName = NULL;
	bi.lpszTitle = _T("请选择脸库数据所在的目录：");
	bi.ulFlags = 0;
	bi.lpfn = NULL;
	bi.lParam = 0;
	bi.iImage = 0;
	//弹出选择目录对话框
	LPCITEMIDLIST lpc = SHBrowseForFolder(&bi);
	if (!lpc) {
		MessageBoxA(this->m_hWnd, "请正确选择脸库数据文件所在的目录！", "警告", MB_ICONEXCLAMATION);
	}
	else {
		SHGetPathFromIDList(lpc, folder);
		GetDlgItem(IDC_FACEDB_FILE)->SetWindowTextW(folder);
		//保存选定的路径到注册表中
		BOOL b = AfxGetApp()->WriteProfileString(CreateDbDlg::registrySection, CreateDbDlg::regFaceDBPath, folder);
		if (b) {
			this->dbPathStr = CT2A(folder);
		}
		else {
			MessageBoxA(this->m_hWnd, "脸库配置写入注册表失败，请电联公司解决！", "警告", MB_ICONEXCLAMATION);
		}
	}
	CoTaskMemFree((LPVOID)lpc); //释放pIDL所指向内存空间
}

void CreateDbDlg::OnBnClickedSelectModelButton()
{
//GetDlgItem(IDC_MODEL_FILE)->SetWindowTextW(modelFileDir);
	TCHAR folder[512];     //存放选择的目录路径 
	BROWSEINFO bi;
	bi.hwndOwner = m_hWnd;
	bi.pidlRoot = NULL;
	bi.pszDisplayName = NULL;
	bi.lpszTitle = _T("请选择脸库数据所在的目录：");
	bi.ulFlags = 0;
	bi.lpfn = NULL;
	bi.lParam = 0;
	bi.iImage = 0;
	//弹出选择目录对话框
	LPCITEMIDLIST lpc = SHBrowseForFolder(&bi);
	if (!lpc) {
		MessageBoxA(this->m_hWnd, "请正确选择脸库数据文件所在的目录！", "警告", MB_ICONEXCLAMATION);
	}
	else {
		SHGetPathFromIDList(lpc, folder);
		GetDlgItem(IDC_MODEL_FILE)->SetWindowTextW(folder);
		//保存选定的路径到注册表中
		BOOL b = AfxGetApp()->WriteProfileString(CreateDbDlg::registrySection, CreateDbDlg::regFaceModelPath, folder);
		if (b) {
			this->modelPathStr = CT2A(folder);
		}
		else {
			MessageBoxA(this->m_hWnd, "脸库配置写入注册表失败，请电联公司解决！", "警告", MB_ICONEXCLAMATION);
		}
	}
	CoTaskMemFree((LPVOID)lpc); //释放pIDL所指向内存空间
}


/*
点击【选择人脸图像目录】按钮。
typedef struct_browseinfo
{
	HWND hwndOwner;
	LPCITEMIDLIST pidlRoot;
	LPSTR pszDisplayName;
	LPCSTR lpszTitle;
	UINT ulFlags;
	BFFCALLBACK lpfn;
	LPARAM lParam;
	int iImage;
}

BOOL WriteProfileString(
	LPCTSTR lpAppName,	// 节的名字，是一个以0结束的字符串
	LPCTSTR lpKeyName,	// 键的名字，是一个以0结束的字符串。若为NULL，则删除整个节
	LPCTSTR lpString	// 键的值，是一个以0结束的字符串。若为NULL，则删除对应的键
)

DWORD GetProfileString(
	LPCTSTR lpAppName,            // 节名
	LPCTSTR lpKeyName,            // 键名，读取该键的值
	LPCTSTR lpDefault,            // 若指定的键不存在，该值作为读取的默认值
	LPTSTR lpReturnedString,      // 一个指向缓冲区的指针，接收读取的字符串
	DWORD nSize                   // 指定lpReturnedString指向的缓冲区的大小
)
*/
void CreateDbDlg::OnBnClickedSelectImageDirButton()
{
	TCHAR folder[512];     //存放选择的目录路径 
	BROWSEINFO bi;
	bi.hwndOwner = m_hWnd;
	bi.pidlRoot = NULL;
	bi.pszDisplayName = NULL;
	bi.lpszTitle = _T("请选择人脸图像文件所在的目录：");
	bi.ulFlags = 0;
	bi.lpfn = NULL;
	bi.lParam = 0;
	bi.iImage = 0;
	//弹出选择目录对话框
	LPCITEMIDLIST lpc = SHBrowseForFolder(&bi);
	if (!lpc) {
		MessageBoxA(this->m_hWnd, "请正确选择人脸图像文件所在的目录！", "警告", MB_ICONEXCLAMATION);
	}
	else {
		SHGetPathFromIDList(lpc, folder);
		GetDlgItem(IDC_FACEIMG_DIR)->SetWindowTextW(folder);
		//保存选定的路径到注册表中
		BOOL b = AfxGetApp()->WriteProfileString(CreateDbDlg::registrySection, CreateDbDlg::regFaceImgPath, folder);
		if (b) {
			this->imgPathStr = CT2A(folder);
		}
		else {
			MessageBoxA(this->m_hWnd, "脸库配置写入注册表失败，请电联公司解决！", "警告", MB_ICONEXCLAMATION);
		}
	}
	CoTaskMemFree((LPVOID)lpc); //释放pIDL所指向内存空间

}

/*
执行建库流程。
*/
void CreateDbDlg::OnBnClickedDoCreatedbButton()
{
	bool append;
	if (IDC_APPEND_RADIO ==
		GetDlgItem(IDC_APPEND_RADIO)->GetCheckedRadioButton(IDC_APPEND_RADIO, IDC_OVERWRITE_RADIO)
		) {
		append = true;
	}
	else {
		append = false;
	}

	UpdateData(TRUE);
	m_log += "开始建库，模式：";
	if (append) {
		m_log += "追加";
	}
	else {
		m_log += "覆盖";
	}
	m_log += "，请等待......";
	m_log += "\r\n";
	GetDlgItem(IDC_LOG_OUTPUT)->SetWindowText(m_log);
	//GetDlgItem(IDC_LOG_OUTPUT)->LineScroll(GetDlgItem(IDC_LOG_OUTPUT)->GetLineCount());
	UpdateData(FALSE);

	int ret = create(this->dbPathStr, this->modelPathStr, this->imgPathStr, append);
	if (0 == ret) {
		UpdateData(TRUE);
		m_log += "建库成功。";
		m_log += "\r\n";
		GetDlgItem(IDC_LOG_OUTPUT)->SetWindowText(m_log);
		UpdateData(FALSE);
	}
	else {
		UpdateData(TRUE);
		m_log += "建库失败，错误码：";
		CString str;
		str.Format(_T("%d"), ret);
		m_log += str;
		m_log += "\r\n";
		GetDlgItem(IDC_LOG_OUTPUT)->SetWindowText(m_log);
		UpdateData(FALSE);
	}
}
