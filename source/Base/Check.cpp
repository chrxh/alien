#include "Check.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <Windows.h>

#include <DbgHelp.h>
#else
#include <execinfo.h>
#endif

namespace
{
#ifdef _WIN32
    std::string createCallstack()
    {
        constexpr USHORT maxFrames = 64;
        void* stack[maxFrames] = {};
        auto const process = GetCurrentProcess();

        static std::mutex mutex;
        static bool symbolsInitialized = false;

        std::lock_guard lock(mutex);
        if (!symbolsInitialized) {
            SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_UNDNAME);
            if (!SymInitialize(process, nullptr, TRUE)) {
                return "unavailable";
            }
            symbolsInitialized = true;
        }

        auto const numFrames = CaptureStackBackTrace(0, maxFrames, stack, nullptr);
        if (numFrames <= 2) {
            return "unavailable";
        }

        std::ostringstream out;
        for (USHORT i = 2; i < numFrames; ++i) {
            std::array<char, sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(TCHAR)> symbolBuffer = {};
            auto* symbol = reinterpret_cast<SYMBOL_INFO*>(symbolBuffer.data());
            symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
            symbol->MaxNameLen = MAX_SYM_NAME;

            DWORD64 displacement = 0;
            if (!SymFromAddr(process, reinterpret_cast<DWORD64>(stack[i]), &displacement, symbol)) {
                out << "#" << (i - 2) << " 0x" << std::hex << reinterpret_cast<std::uintptr_t>(stack[i]) << std::dec << "\n";
                continue;
            }

            IMAGEHLP_LINE64 line = {};
            line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
            DWORD lineDisplacement = 0;

            out << "#" << (i - 2) << " " << symbol->Name;
            if (SymGetLineFromAddr64(process, symbol->Address, &lineDisplacement, &line)) {
                out << " (" << std::filesystem::path(line.FileName).filename().string() << ":" << line.LineNumber << ")";
            }
            out << "\n";
        }

        return out.str();
    }
#else
    std::string createCallstack()
    {
        constexpr int maxFrames = 64;
        void* stack[maxFrames] = {};

        auto const numFrames = backtrace(stack, maxFrames);
        if (numFrames <= 2) {
            return "unavailable";
        }

        auto* symbols = backtrace_symbols(stack, numFrames);
        if (nullptr == symbols) {
            return "unavailable";
        }

        std::unique_ptr<char*, decltype(&std::free)> cleanup(symbols, &std::free);

        std::ostringstream out;
        for (int i = 2; i < numFrames; ++i) {
            out << "#" << (i - 2) << " " << symbols[i] << "\n";
        }

        return out.str();
    }
#endif
}

[[noreturn]] void throwCheckException(char const* expression, char const* file, int line)
{
    std::ostringstream out;
    out << "check failed: " << expression << " (" << std::filesystem::path(file).filename().string() << ":" << line << ")\n";
    out << "Callstack:\n" << createCallstack();
    throw std::runtime_error(out.str());
}
