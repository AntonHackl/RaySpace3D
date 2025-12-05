#pragma once

#include <string>
#include <fstream>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <limits.h>
#endif

inline std::string detectPTXPath()
{
    std::string dir;

#ifdef _WIN32
    char exePath[MAX_PATH];
    DWORD len = GetModuleFileNameA(nullptr, exePath, MAX_PATH);
    if (len > 0 && len < MAX_PATH) {
        std::string path(exePath, len);
        size_t pos = path.find_last_of("\\/");
        if (pos != std::string::npos) {
            dir = path.substr(0, pos + 1);
        }
    }
#else
    char exePath[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
    if (len != -1) {
        exePath[len] = '\0';
        std::string path(exePath);
        size_t pos = path.find_last_of('/');
        if (pos != std::string::npos) {
            dir = path.substr(0, pos + 1);
        }
    }
#endif

    if (!dir.empty()) {
        {
            std::string candidate = dir + "raytracing.ptx";
            std::ifstream f(candidate, std::ios::binary);
            if (f.good()) return candidate;
        }

#ifdef _WIN32
        {
            std::string candidate = dir + "..\\raytracing.ptx";
            std::ifstream f(candidate, std::ios::binary);
            if (f.good()) return candidate;
        }
#else
        {
            std::string candidate = dir + "../raytracing.ptx";
            std::ifstream f(candidate, std::ios::binary);
            if (f.good()) return candidate;
        }
#endif
    }

    {
        std::string candidate = "raytracing.ptx";
        std::ifstream f(candidate, std::ios::binary);
        if (f.good()) return candidate;
    }

    return "raytracing.ptx";
}
