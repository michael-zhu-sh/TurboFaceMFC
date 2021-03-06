/*
脸库维护对话框，可以重新建库或者追加。
注册表：
turboface->FaceImageFilesPath,	人脸图像文件目录。
turboface->FaceDBFilePath,		脸库数据库文件目录。
turboface->FaceModelFilesPath,	脸库模型文件目录。

std::string stdstr = CT2A(mfc_str);
CString msg(stdstr.c_str());
MessageBoxA(this->m_hWnd, (LPCSTR)(LPCTSTR)msg, "警告", MB_ICONEXCLAMATION);

size_t rows;
CString str;
str.Format(_T("%d"), rows);
GetDlgItem(IDC_DB_ROWS)->SetWindowTextW(str);
*/
#include "stdafx.h"
#include "afxdialogex.h"

#include "TurboFaceMFC.h"
#include "CreateDbDlg.h"

#include "facelib.h"

CEvent threadDone;

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
	dbPath.Replace(L"\\",L"/");
	GetDlgItem(IDC_FACEDB_FILE)->SetWindowTextW(dbPath);
	this->dbPathStr = CT2A(dbPath);	//脸库目录。

	CString modelPath = AfxGetApp()->GetProfileString(CreateDbDlg::registrySection, CreateDbDlg::regFaceModelPath);
	modelPath.Replace(L"\\", L"/");
	GetDlgItem(IDC_MODEL_FILE)->SetWindowTextW(modelPath);
	this->modelPathStr = CT2A(modelPath);	//模型目录。

	CString imgPath = AfxGetApp()->GetProfileString(CreateDbDlg::registrySection, CreateDbDlg::regFaceImgPath);
	imgPath.Replace(L"\\", L"/");
	GetDlgItem(IDC_FACEIMG_DIR)->SetWindowTextW(imgPath);
	this->imgPathStr = CT2A(imgPath);	//人脸图像文件目录。

	//显示脸库记录数。
	size_t rows = count();
	CString str;
	str.Format(_T("%d"), rows);
	GetDlgItem(IDC_DB_ROWS)->SetWindowTextW(str);

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
			CString str;
			str.Format(_T("%s"), folder);
			str.Replace(L"\\", L"/");
			this->dbPathStr = CT2A(str);
		}
		else {
			MessageBoxA(this->m_hWnd, "脸库配置写入注册表失败，请电联公司解决！", "警告", MB_ICONEXCLAMATION);
		}
	}
	CoTaskMemFree((LPVOID)lpc); //释放pIDL所指向内存空间
}

void CreateDbDlg::OnBnClickedSelectModelButton()
{
	TCHAR folder[512];     //存放选择的目录路径 
	BROWSEINFO bi;
	bi.hwndOwner = m_hWnd;
	bi.pidlRoot = NULL;
	bi.pszDisplayName = NULL;
	bi.lpszTitle = _T("请选择脸库模型所在的目录：");
	bi.ulFlags = 0;
	bi.lpfn = NULL;
	bi.lParam = 0;
	bi.iImage = 0;
	//弹出选择目录对话框
	LPCITEMIDLIST lpc = SHBrowseForFolder(&bi);
	if (!lpc) {
		MessageBoxA(this->m_hWnd, "请正确选择脸库模型文件所在的目录！", "警告", MB_ICONEXCLAMATION);
	}
	else {
		SHGetPathFromIDList(lpc, folder);
		GetDlgItem(IDC_MODEL_FILE)->SetWindowTextW(folder);
		//保存选定的路径到注册表中
		BOOL b = AfxGetApp()->WriteProfileString(CreateDbDlg::registrySection, CreateDbDlg::regFaceModelPath, folder);
		if (b) {
			CString str;
			str.Format(_T("%s"), folder);
			str.Replace(L"\\", L"/");
			this->modelPathStr = CT2A(str);
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
			CString str;
			str.Format(_T("%s"), folder);
			str.Replace(L"\\", L"/");
			this->imgPathStr = CT2A(str);
		}
		else {
			MessageBoxA(this->m_hWnd, "脸库配置写入注册表失败，请电联公司解决！", "警告", MB_ICONEXCLAMATION);
		}
	}
	CoTaskMemFree((LPVOID)lpc); //释放pIDL所指向内存空间

}
/*
建库线程的功能函数。
*/
void CreateDbDlg::threadEntry() {
	this->m_log += "开始建库，模式：";
	if (this->append) {
		this->m_log += "追加";
	}
	else {
		this->m_log += "重建";
	}
	this->m_log += "，请等待......";
	this->m_log += "\r\n";
	GetDlgItem(IDC_LOG_OUTPUT)->SetWindowText(m_log);

	CTime startTime = CTime::GetCurrentTime();
	int ret = create(this->dbPathStr, this->modelPathStr, this->imgPathStr, append);//调用DLL。
	CTime curTime = CTime::GetCurrentTime();
	CTimeSpan timeSpan = curTime - startTime;
	if (0 == ret) {
		CString minStr,hourStr,secStr,rowStr;
		hourStr.Format(_T("%d"),	timeSpan.GetHours());
		minStr.Format(_T("%d"),		timeSpan.GetMinutes());
		secStr.Format(_T("%d"),		timeSpan.GetSeconds());
		m_log += "建库成功，耗时";
		m_log += hourStr;
		m_log += "小时";
		m_log += minStr;
		m_log += "分钟";
		m_log += secStr;
		m_log += "秒。\r\n";
		GetDlgItem(IDC_LOG_OUTPUT)->SetWindowText(m_log);

		size_t rows = count();
		rowStr.Format(_T("%d"), rows);
		GetDlgItem(IDC_DB_ROWS)->SetWindowTextW(rowStr);
	}
	else {
		m_log += "建库失败，错误码：";
		CString str;
		str.Format(_T("%d"), ret);
		m_log += str;
		m_log += "\r\n";
		GetDlgItem(IDC_LOG_OUTPUT)->SetWindowText(m_log);
	}
	//建库过程已结束。
	GetDlgItem(IDC_DO_CREATEDB_BUTTON)->EnableWindow(true);
	GetDlgItem(IDCANCEL)->EnableWindow(true);

	threadDone.ResetEvent();
}

UINT CreateDbDlg::childThread(LPVOID param) {
	threadDone.SetEvent();	//表示线程已开始工作。

	CreateDbDlg* wnd = (CreateDbDlg*)param;
	wnd->threadEntry();

	return 0;
}

/*
执行建库流程。
*/
void CreateDbDlg::OnBnClickedDoCreatedbButton()
{
	if (IDC_APPEND_RADIO == GetCheckedRadioButton(IDC_APPEND_RADIO, IDC_OVERWRITE_RADIO)) {
		this->append = true;
	}
	else {
		this->append = false;
	}
	UINT ok;
	if (this->append) {
		ok = MessageBoxA(this->m_hWnd, "即将使用【追加】模式建库，需等待较长时间，你确定吗？", "警告", MB_OKCANCEL | MB_ICONQUESTION);
	}
	else {
		ok = MessageBoxA(this->m_hWnd, "即将使用【覆盖】模式建库（原始数据全部清除），需等待较长时间，你确定吗？", "警告", MB_OKCANCEL | MB_ICONQUESTION);
	}
	if (ok == IDOK) {
		//开始建库。
		GetDlgItem(IDC_DO_CREATEDB_BUTTON)->EnableWindow(false);
		GetDlgItem(IDCANCEL)->EnableWindow(false);

		AfxBeginThread((AFX_THREADPROC)childThread, (LPVOID)this, THREAD_PRIORITY_IDLE);
	}
	else {
		//不进行建库操作。
		m_log += "你已放弃建库，可以点击【返回】按钮后返回主界面。";
		m_log += "\r\n";
		GetDlgItem(IDC_LOG_OUTPUT)->SetWindowText(m_log);
	}
}
