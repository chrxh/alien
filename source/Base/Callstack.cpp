#include "Callstack.h"

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
    auto constexpr MaxFrames = 64;
}

#ifdef _WIN32
namespace
{
    std::mutex symbolMutex;
    bool symbolInitializationAttempted = false;
    bool symbolsAvailable = false;

    void initializeSymbols(HANDLE process)
    {
        if (!symbolInitializationAttempted) {
            SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_UNDNAME);
            symbolsAvailable = SymInitialize(process, nullptr, TRUE);
            symbolInitializationAttempted = true;
        } else if (symbolsAvailable) {
            SymRefreshModuleList(process);
        }
    }

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
            out << getModuleName(*module) << "+0x" << std::hex << (address - module->BaseOfImage) << std::dec;
        } else {
            out << "0x" << std::hex << address << std::dec;
        }
        out << "\n";
    }

    std::string formatFrames(HANDLE process, void* const* stack, USHORT numFrames)
    {
        auto const applicationModuleBase = reinterpret_cast<DWORD64>(GetModuleHandle(nullptr));

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

            out << "#" << i << " " << symbol->Name;
            if (lineAvailable) {
                out << " (" << std::filesystem::path(line.FileName).filename().string() << ":" << line.LineNumber << ")";
            }
            out << "\n";
        }

        return out.str();
    }
}

std::string Callstack::capture(int numSkippedFrames)
{
    void* stack[MaxFrames] = {};
    auto numFrames = CaptureStackBackTrace(static_cast<DWORD>(1 + numSkippedFrames), MaxFrames, stack, nullptr);
    if (numFrames == 0) {
        return "unavailable";
    }

    auto process = GetCurrentProcess();
    std::lock_guard lock(symbolMutex);
    initializeSymbols(process);
    return formatFrames(process, stack, numFrames);
}

std::string Callstack::capture(_CONTEXT const& context)
{
    auto process = GetCurrentProcess();
    std::lock_guard lock(symbolMutex);
    initializeSymbols(process);

    auto walkedContext = context;
    STACKFRAME64 frame = {};
    frame.AddrPC = {.Offset = walkedContext.Rip, .Mode = AddrModeFlat};
    frame.AddrFrame = {.Offset = walkedContext.Rbp, .Mode = AddrModeFlat};
    frame.AddrStack = {.Offset = walkedContext.Rsp, .Mode = AddrModeFlat};

    void* stack[MaxFrames] = {};
    USHORT numFrames = 0;
    while (numFrames < MaxFrames
           && StackWalk64(
               IMAGE_FILE_MACHINE_AMD64, process, GetCurrentThread(), &frame, &walkedContext, nullptr, SymFunctionTableAccess64, SymGetModuleBase64, nullptr)
           && frame.AddrPC.Offset != 0) {
        stack[numFrames++] = reinterpret_cast<void*>(frame.AddrPC.Offset);
    }
    if (numFrames == 0) {
        return "unavailable";
    }
    return formatFrames(process, stack, numFrames);
}
#else
std::string Callstack::capture(int numSkippedFrames)
{
    void* stack[MaxFrames] = {};

    auto numFrames = backtrace(stack, MaxFrames);
    auto firstFrame = 1 + numSkippedFrames;
    if (numFrames <= firstFrame) {
        return "unavailable";
    }

    auto* symbols = backtrace_symbols(stack, numFrames);
    if (nullptr == symbols) {
        return "unavailable";
    }

    std::unique_ptr<char*, decltype(&std::free)> cleanup(symbols, &std::free);

    std::ostringstream out;
    for (int i = firstFrame; i < numFrames; ++i) {
        out << "#" << (i - firstFrame) << " " << symbols[i] << "\n";
    }

    return out.str();
}
#endif
