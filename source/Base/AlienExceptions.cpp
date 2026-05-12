#include "AlienExceptions.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <sstream>

#ifdef _WIN32
#include <Windows.h>

#include <DbgHelp.h>
#else
#include <execinfo.h>
#endif

namespace
{
#ifdef _WIN32
    bool isDebugSymbol(IMAGEHLP_MODULE64 const& module)
    {
        return module.SymType == SymCoff || module.SymType == SymCv || module.SymType == SymPdb || module.SymType == SymSym || module.SymType == SymDia;
    }

    std::string getModuleName(IMAGEHLP_MODULE64 const& module)
    {
        if (module.LoadedImageName[0] != '\0') {
            return std::filesystem::path(module.LoadedImageName).filename().string();
        }
        if (module.ImageName[0] != '\0') {
            return std::filesystem::path(module.ImageName).filename().string();
        }
        if (module.ModuleName[0] != '\0') {
            return module.ModuleName;
        }
        return "unknown";
    }

    void printFallbackFrame(std::ostringstream& out, USHORT frameIndex, std::uintptr_t address, IMAGEHLP_MODULE64 const* module)
    {
        out << "#" << frameIndex << " ";
        if (module != nullptr && module->BaseOfImage != 0 && address >= module->BaseOfImage) {
            out << getModuleName(*module) << "+0x" << std::hex << (address - module->BaseOfImage) << " [0x" << address << std::dec << "]";
        } else {
            out << "0x" << std::hex << address << std::dec;
        }
        out << "\n";
    }

    std::string createCallstack()
    {
        constexpr USHORT maxFrames = 64;
        void* stack[maxFrames] = {};
        auto const process = GetCurrentProcess();
        auto const applicationModuleBase = reinterpret_cast<DWORD64>(GetModuleHandle(nullptr));

        auto const numFrames = CaptureStackBackTrace(2, maxFrames, stack, nullptr);
        if (numFrames == 0) {
            return "unavailable";
        }

        static std::mutex mutex;
        static bool symbolInitializationAttempted = false;
        static bool symbolsAvailable = false;

        std::lock_guard lock(mutex);
        if (!symbolInitializationAttempted) {
            SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_UNDNAME);
            symbolsAvailable = SymInitialize(process, nullptr, TRUE);
            symbolInitializationAttempted = true;
        }

        std::ostringstream out;
        for (USHORT i = 0; i < numFrames; ++i) {
            std::array<char, sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(TCHAR)> symbolBuffer = {};
            auto* symbol = reinterpret_cast<SYMBOL_INFO*>(symbolBuffer.data());
            symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
            symbol->MaxNameLen = MAX_SYM_NAME;

            IMAGEHLP_MODULE64 module = {};
            module.SizeOfStruct = sizeof(IMAGEHLP_MODULE64);
            auto const address = reinterpret_cast<std::uintptr_t>(stack[i]);
            auto const moduleAvailable = SymGetModuleInfo64(process, static_cast<DWORD64>(address), &module);

            DWORD64 displacement = 0;
            if (!symbolsAvailable || !SymFromAddr(process, static_cast<DWORD64>(address), &displacement, symbol)) {
                printFallbackFrame(out, i, address, moduleAvailable ? &module : nullptr);
                continue;
            }

            IMAGEHLP_LINE64 line = {};
            line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
            DWORD lineDisplacement = 0;
            auto const lineAvailable = SymGetLineFromAddr64(process, static_cast<DWORD64>(address), &lineDisplacement, &line);
            auto const isApplicationFrame = moduleAvailable && module.BaseOfImage == applicationModuleBase;
            auto const hasDebugSymbols = moduleAvailable && isDebugSymbol(module);
            auto const hasExactExportSymbol = moduleAvailable && module.SymType == SymExport && displacement == 0;

            if (!lineAvailable && !hasDebugSymbols && (isApplicationFrame || !hasExactExportSymbol)) {
                printFallbackFrame(out, i, address, moduleAvailable ? &module : nullptr);
                continue;
            }

            out << "#" << i << " " << symbol->Name << " [0x" << std::hex << address << std::dec << "]";
            if (lineAvailable) {
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

AlienException::AlienException(std::string const& what)
    : std::runtime_error(what)
    , _callstack(createCallstack())
{}

std::string const& AlienException::getCallstack() const
{
    return _callstack;
}
