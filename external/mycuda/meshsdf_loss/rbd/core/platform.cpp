// This code contains NVIDIA Confidential Information and is disclosed to you
// under a form of NVIDIA software license agreement provided separately to you.
//
// Notice
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software and related documentation and
// any modifications thereto. Any use, reproduction, disclosure, or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA Corporation is strictly prohibited.
//
// ALL NVIDIA DESIGN SPECIFICATIONS, CODE ARE PROVIDED "AS IS.". NVIDIA MAKES
// NO WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ALL IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, AND FITNESS FOR A PARTICULAR PURPOSE.
//
// Information and code furnished is believed to be accurate and reliable.
// However, NVIDIA Corporation assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA Corporation. Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA Corporation products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA Corporation.
//
// Copyright (c) 2013-2016 NVIDIA Corporation. All rights reserved.

#include "core.h"
#include "platform.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <ctype.h>
#include <stdio.h>
#include <string.h>

using namespace std;

#if defined(_WIN32)

#include <windows.h>
#include <commdlg.h>
#include <mmsystem.h>

double GetSeconds()
{
	static LARGE_INTEGER lastTime;
	static LARGE_INTEGER freq;
	static bool first = true;
	
	if (first)
	{	
		QueryPerformanceCounter(&lastTime);
		QueryPerformanceFrequency(&freq);

		first = false;
	}
	
	static double time = 0.0;
	
	LARGE_INTEGER t;
	QueryPerformanceCounter(&t);
	
	__int64 delta = t.QuadPart-lastTime.QuadPart;
	double deltaSeconds = double(delta) / double(freq.QuadPart);
	
	time += deltaSeconds;

	lastTime = t;

	return time;

}

void Sleep(double seconds)
{
	::Sleep(DWORD(seconds*1000));
}


#include <iostream>
#include <vector>
#include <bitset>
#include <array>
#include <string>
#include <intrin.h>

namespace 
{

class InstructionSet
{
    // forward declarations
    class InstructionSet_Internal;

public:
    // getters
    static std::string Vendor(void) { return CPU_Rep.vendor_; }
    static std::string Brand(void) { return CPU_Rep.brand_; }

    static bool SSE3(void) { return CPU_Rep.f_1_ECX_[0]; }
    static bool PCLMULQDQ(void) { return CPU_Rep.f_1_ECX_[1]; }
    static bool MONITOR(void) { return CPU_Rep.f_1_ECX_[3]; }
    static bool SSSE3(void) { return CPU_Rep.f_1_ECX_[9]; }
    static bool FMA(void) { return CPU_Rep.f_1_ECX_[12]; }
    static bool CMPXCHG16B(void) { return CPU_Rep.f_1_ECX_[13]; }
    static bool SSE41(void) { return CPU_Rep.f_1_ECX_[19]; }
    static bool SSE42(void) { return CPU_Rep.f_1_ECX_[20]; }
    static bool MOVBE(void) { return CPU_Rep.f_1_ECX_[22]; }
    static bool POPCNT(void) { return CPU_Rep.f_1_ECX_[23]; }
    static bool AES(void) { return CPU_Rep.f_1_ECX_[25]; }
    static bool XSAVE(void) { return CPU_Rep.f_1_ECX_[26]; }
    static bool OSXSAVE(void) { return CPU_Rep.f_1_ECX_[27]; }
    static bool AVX(void) { return CPU_Rep.f_1_ECX_[28]; }
    static bool F16C(void) { return CPU_Rep.f_1_ECX_[29]; }
    static bool RDRAND(void) { return CPU_Rep.f_1_ECX_[30]; }

    static bool MSR(void) { return CPU_Rep.f_1_EDX_[5]; }
    static bool CX8(void) { return CPU_Rep.f_1_EDX_[8]; }
    static bool SEP(void) { return CPU_Rep.f_1_EDX_[11]; }
    static bool CMOV(void) { return CPU_Rep.f_1_EDX_[15]; }
    static bool CLFSH(void) { return CPU_Rep.f_1_EDX_[19]; }
    static bool MMX(void) { return CPU_Rep.f_1_EDX_[23]; }
    static bool FXSR(void) { return CPU_Rep.f_1_EDX_[24]; }
    static bool SSE(void) { return CPU_Rep.f_1_EDX_[25]; }
    static bool SSE2(void) { return CPU_Rep.f_1_EDX_[26]; }

    static bool FSGSBASE(void) { return CPU_Rep.f_7_EBX_[0]; }
    static bool BMI1(void) { return CPU_Rep.f_7_EBX_[3]; }
    static bool HLE(void) { return CPU_Rep.isIntel_ && CPU_Rep.f_7_EBX_[4]; }
    static bool AVX2(void) { return CPU_Rep.f_7_EBX_[5]; }
    static bool BMI2(void) { return CPU_Rep.f_7_EBX_[8]; }
    static bool ERMS(void) { return CPU_Rep.f_7_EBX_[9]; }
    static bool INVPCID(void) { return CPU_Rep.f_7_EBX_[10]; }
    static bool RTM(void) { return CPU_Rep.isIntel_ && CPU_Rep.f_7_EBX_[11]; }
    static bool AVX512F(void) { return CPU_Rep.f_7_EBX_[16]; }
    static bool RDSEED(void) { return CPU_Rep.f_7_EBX_[18]; }
    static bool ADX(void) { return CPU_Rep.f_7_EBX_[19]; }
    static bool AVX512PF(void) { return CPU_Rep.f_7_EBX_[26]; }
    static bool AVX512ER(void) { return CPU_Rep.f_7_EBX_[27]; }
    static bool AVX512CD(void) { return CPU_Rep.f_7_EBX_[28]; }
    static bool SHA(void) { return CPU_Rep.f_7_EBX_[29]; }

    static bool PREFETCHWT1(void) { return CPU_Rep.f_7_ECX_[0]; }

    static bool LAHF(void) { return CPU_Rep.f_81_ECX_[0]; }
    static bool LZCNT(void) { return CPU_Rep.isIntel_ && CPU_Rep.f_81_ECX_[5]; }
    static bool ABM(void) { return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[5]; }
    static bool SSE4a(void) { return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[6]; }
    static bool XOP(void) { return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[11]; }
    static bool TBM(void) { return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[21]; }

    static bool SYSCALL(void) { return CPU_Rep.isIntel_ && CPU_Rep.f_81_EDX_[11]; }
    static bool MMXEXT(void) { return CPU_Rep.isAMD_ && CPU_Rep.f_81_EDX_[22]; }
    static bool RDTSCP(void) { return CPU_Rep.isIntel_ && CPU_Rep.f_81_EDX_[27]; }
    static bool _3DNOWEXT(void) { return CPU_Rep.isAMD_ && CPU_Rep.f_81_EDX_[30]; }
    static bool _3DNOW(void) { return CPU_Rep.isAMD_ && CPU_Rep.f_81_EDX_[31]; }

private:
    static const InstructionSet_Internal CPU_Rep;

    class InstructionSet_Internal
    {
    public:
        InstructionSet_Internal()
            : nIds_{ 0 },
            nExIds_{ 0 },
            isIntel_{ false },
            isAMD_{ false },
            f_1_ECX_{ 0 },
            f_1_EDX_{ 0 },
            f_7_EBX_{ 0 },
            f_7_ECX_{ 0 },
            f_81_ECX_{ 0 },
            f_81_EDX_{ 0 },
            data_{},
            extdata_{}
        {
            //int cpuInfo[4] = {-1};
            std::array<int, 4> cpui;

            // Calling __cpuid with 0x0 as the function_id argument
            // gets the number of the highest valid function ID.
            __cpuid(cpui.data(), 0);
            nIds_ = cpui[0];

            for (int i = 0; i <= nIds_; ++i)
            {
                __cpuidex(cpui.data(), i, 0);
                data_.push_back(cpui);
            }

            // Capture vendor string
            char vendor[0x20];
            memset(vendor, 0, sizeof(vendor));
            *reinterpret_cast<int*>(vendor) = data_[0][1];
            *reinterpret_cast<int*>(vendor + 4) = data_[0][3];
            *reinterpret_cast<int*>(vendor + 8) = data_[0][2];
            vendor_ = vendor;
            if (vendor_ == "GenuineIntel")
            {
                isIntel_ = true;
            }
            else if (vendor_ == "AuthenticAMD")
            {
                isAMD_ = true;
            }

            // load bitset with flags for function 0x00000001
            if (nIds_ >= 1)
            {
                f_1_ECX_ = data_[1][2];
                f_1_EDX_ = data_[1][3];
            }

            // load bitset with flags for function 0x00000007
            if (nIds_ >= 7)
            {
                f_7_EBX_ = data_[7][1];
                f_7_ECX_ = data_[7][2];
            }

            // Calling __cpuid with 0x80000000 as the function_id argument
            // gets the number of the highest valid extended ID.
            __cpuid(cpui.data(), 0x80000000);
            nExIds_ = cpui[0];

            char brand[0x40];
            memset(brand, 0, sizeof(brand));

            for (int i = 0x80000000; i <= nExIds_; ++i)
            {
                __cpuidex(cpui.data(), i, 0);
                extdata_.push_back(cpui);
            }

            // load bitset with flags for function 0x80000001
            if (nExIds_ >= 0x80000001)
            {
                f_81_ECX_ = extdata_[1][2];
                f_81_EDX_ = extdata_[1][3];
            }

            // Interpret CPU brand string if reported
            if (nExIds_ >= 0x80000004)
            {
                memcpy(brand, extdata_[2].data(), sizeof(cpui));
                memcpy(brand + 16, extdata_[3].data(), sizeof(cpui));
                memcpy(brand + 32, extdata_[4].data(), sizeof(cpui));
                brand_ = brand;
            }
        };

        int nIds_;
        int nExIds_;
        std::string vendor_;
        std::string brand_;
        bool isIntel_;
        bool isAMD_;
        std::bitset<32> f_1_ECX_;
        std::bitset<32> f_1_EDX_;
        std::bitset<32> f_7_EBX_;
        std::bitset<32> f_7_ECX_;
        std::bitset<32> f_81_ECX_;
        std::bitset<32> f_81_EDX_;
        std::vector<std::array<int, 4>> data_;
        std::vector<std::array<int, 4>> extdata_;
    };
};

// Initialize static member data
const InstructionSet::InstructionSet_Internal InstructionSet::CPU_Rep;

} // anonymous namespace

std::string GetCPUString()
{
	return InstructionSet::Brand();
}

//// helper function to get exe path
//string GetExePath()
//{
//	const uint32_t kMaxPathLength = 2048;
//
//	char exepath[kMaxPathLength];
//
//	// get exe path for file system
//	uint32_t i = GetModuleFileName(NULL, exepath, kMaxPathLength);
//
//	// rfind first slash
//	while (i && exepath[i] != '\\' && exepath[i] != '//')
//		i--;
//
//	// insert null terminater to cut off exe name
//	return string(&exepath[0], &exepath[i+1]);
//}
//
//string FileOpenDialog(char *filter)
//{
//	HWND owner=NULL;
//
//	OPENFILENAME ofn;
//	char fileName[MAX_PATH] = "";
//	ZeroMemory(&ofn, sizeof(ofn));
//	ofn.lStructSize = sizeof(OPENFILENAME);
//	ofn.hwndOwner = owner;
//	ofn.lpstrFilter = filter;
//	ofn.lpstrFile = fileName;
//	ofn.nMaxFile = MAX_PATH;
//	ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_HIDEREADONLY | OFN_NOCHANGEDIR;
//	ofn.lpstrDefExt = "";
//
//	string fileNameStr;
//
//	if ( GetOpenFileName(&ofn) )
//		fileNameStr = fileName;
//
//	return fileNameStr;
//}
//
//bool FileMove(const char* src, const char* dest)
//{
//	BOOL b = MoveFileEx(src, dest, MOVEFILE_REPLACE_EXISTING);
//	return b == TRUE;
//}
//
//bool FileScan(const char* pattern, vector<string>& files)
//{
//	HANDLE          h;
//	WIN32_FIND_DATA info;
//
//	// build a list of files
//	h = FindFirstFile(pattern, &info);
//
//	if (h != INVALID_HANDLE_VALUE)
//	{
//		do
//		{
//			if (!(strcmp(info.cFileName, ".") == 0 || strcmp(info.cFileName, "..") == 0))
//			{
//				files.push_back(info.cFileName);
//			}
//		} 
//		while (FindNextFile(h, &info));
//
//		if (GetLastError() != ERROR_NO_MORE_FILES)
//		{
//			return false;
//		}
//
//		FindClose(h);
//	}
//	else
//	{
//		return false;
//	}
//
//	return true;
//}


#else

// linux, mac platforms
#include <sys/time.h>

double GetSeconds()
{
	// Figure out time elapsed since last call to idle function
	static struct timeval last_idle_time;
	static double time = 0.0;	

	struct timeval time_now;
	gettimeofday(&time_now, NULL);

	if (last_idle_time.tv_usec == 0)
		last_idle_time = time_now;

	float dt = (float)(time_now.tv_sec - last_idle_time.tv_sec) + 1.0e-6*(time_now.tv_usec - last_idle_time.tv_usec);

	time += dt;
	last_idle_time = time_now;

	return time;
}

std::string GetCPUString()
{
	return "CPU";
}

#endif


uint8_t* LoadFileToBuffer(const char* filename, uint32_t* sizeRead)
{
	FILE* file = fopen(filename, "rb");
	if (file)
	{
		fseek(file, 0, SEEK_END);
		long p = ftell(file);
		
		uint8_t* buf = new uint8_t[p];
		fseek(file, 0, SEEK_SET);
		uint32_t len = uint32_t(fread(buf, 1, p, file));
	
		fclose(file);
		
		if (sizeRead)
		{
			*sizeRead = len;
		}
		
		return buf;
	}
	else
	{
		std::cout << "Could not open file for reading: " << filename << std::endl;
		return NULL;
	}
}

string LoadFileToString(const char* filename)
{
	//std::ifstream file(filename);
	//return string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	uint32_t size;
	uint8_t* buf = LoadFileToBuffer(filename, &size);
	
	if (buf)
	{
		string s(buf, buf+size);
		delete[] buf;
		
		return s;
	}
	else
	{
		return "";
	}	
}

bool SaveStringToFile(const char* filename, const char* s)
{
	FILE* f = fopen(filename, "w");
	if (!f)
	{
		std::cout << "Could not open file for writing: " << filename << std::endl;		
		return false;
	}
	else 
	{
		fputs(s, f);
		fclose(f);
		
		return true;
	}
}


string StripFilename(const char* path)
{
	// simply find the last 
	const char* iter=path;
	const char* last=NULL;
	while (*iter)
	{
		if (*iter == '\\' || *iter== '/')
			last = iter;
		
		++iter;
	}
	
	if (last)
	{
		return string(path, last+1);
	}
	else
		return string();
	
}

string GetExtension(const char* path)
{
	const char* s = strrchr(path, '.');
	if (s)
	{
		return string(s+1);
	}
	else
	{
		return "";
	}
}

string StripExtension(const char* path)
{
	const char* s = strrchr(path, '.');
	if (s)
	{
		return string(path, s);
	}
	else
	{
		return string(path);
	}
}

string NormalizePath(const char* path)
{
	string p(path);
	replace(p.begin(), p.end(), '\\', '/');
	transform(p.begin(), p.end(), p.begin(), ::tolower);
	
	return p;
}

// strips the path from a file name
string StripPath(const char* path)
{
	// simply find the last 
	const char* iter=path;
	const char* last=NULL;
	while (*iter)
	{
		if (*iter == '\\' || *iter== '/')
			last = iter;
		
		++iter;
	}
	
	if (!last)
	{
		return string(path);
	}
	
	// eat the last slash
	++last;
	
	if (*last)
	{
		return string(last);
	}
	else
	{
		return string();	
	}
}

#if _WIN32

#include <fstream>
#include <iostream>
#include <filesystem>

namespace fs = std::experimental::filesystem::v1;

void ListDirectory(const char* dir, const char* filter, std::vector<std::string>& filenames)
{    
    for (const auto & entry : fs::directory_iterator(dir))
	{
		if (GetExtension(entry.path().string().c_str()) == filter)
		{
			filenames.push_back(entry.path().string());
		}
	}
}

#else

void ListDirectory(const char* dir, const char* filter, std::vector<std::string>& filenames)
{
}

#endif