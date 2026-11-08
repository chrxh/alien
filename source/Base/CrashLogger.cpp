#include "CrashLogger.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <functional>
#include <mutex>
#include <ranges>
#include <string>
#include <csignal>

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include "Callstack.h"
#include "LoggingService.h"
#include "StringHelper.h"

namespace
{
    using SignalHandler = void (*)(int);

    auto constexpr MaxCrashLoggingDuration = std::chrono::seconds(10);
#ifndef _WIN32
    auto constexpr FatalSignalNumbers = std::array{SIGSEGV, SIGBUS, SIGILL, SIGFPE};
#endif

    std::recursive_timed_mutex crashLoggingMutex;
    bool isCrashLogged = false;

    void logCrash(std::string const& reason, std::function<std::string()> const& captureCallstack)
    {
        // Threads crashing at the same time wait for the first one to finish logging
        std::unique_lock lock(crashLoggingMutex, MaxCrashLoggingDuration);
        if (!lock.owns_lock() || isCrashLogged) {
            return;
        }
        isCrashLogged = true;
        try {
            log(Priority::Important, "ALIEN crashed: " + reason);
            log(Priority::Important, "Callstack:\n" + captureCallstack());
        } catch (...) {
        }
    }

#ifdef _WIN32
    std::string getExceptionDescription(DWORD code)
    {
        auto result = StringHelper::formatInHex(code);
        switch (code) {
        case EXCEPTION_ACCESS_VIOLATION:
            return result + " (access violation)";
        case EXCEPTION_ILLEGAL_INSTRUCTION:
            return result + " (illegal instruction)";
        case EXCEPTION_INT_DIVIDE_BY_ZERO:
            return result + " (integer division by zero)";
        case EXCEPTION_STACK_OVERFLOW:
            return result + " (stack overflow)";
        case 0xE06D7363:
            return result + " (C++ exception)";
        default:
            return result;
        }
    }

    std::string getCodeLocation(void const* address)
    {
        HMODULE module = nullptr;
        char modulePath[MAX_PATH] = {};
        auto flags = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
        if (!GetModuleHandleExA(flags, static_cast<LPCSTR>(address), &module) || GetModuleFileNameA(module, modulePath, MAX_PATH) == 0) {
            return StringHelper::formatInHex(reinterpret_cast<std::uintptr_t>(address));
        }
        auto offset = reinterpret_cast<std::uintptr_t>(address) - reinterpret_cast<std::uintptr_t>(module);
        return std::filesystem::path(modulePath).filename().string() + "+" + StringHelper::formatInHex(offset);
    }

    struct CrashReport
    {
        std::string reason;
        EXCEPTION_RECORD const* exceptionRecord = nullptr;
        CONTEXT const* context = nullptr;
    };

    DWORD WINAPI logCrashReport(LPVOID parameter)
    {
        auto const& report = *static_cast<CrashReport const*>(parameter);
        auto reason = report.reason;
        if (report.exceptionRecord != nullptr) {
            auto const& record = *report.exceptionRecord;
            reason = "unhandled exception " + getExceptionDescription(record.ExceptionCode) + " at " + getCodeLocation(record.ExceptionAddress);
        }
        logCrash(reason, [&report] { return Callstack::capture(*report.context); });
        return 0;
    }

    // Logs on a fresh stack, since the crashed thread may have exhausted its own, and gives up if logging deadlocks
    void logCrashOnHelperThread(CrashReport report)
    {
        auto timeout = std::chrono::duration_cast<std::chrono::milliseconds>(MaxCrashLoggingDuration).count();
        if (auto thread = CreateThread(nullptr, 0, logCrashReport, &report, 0, nullptr)) {
            WaitForSingleObject(thread, static_cast<DWORD>(timeout));
            CloseHandle(thread);
        }
    }

    void logCrashAtCurrentPosition(std::string const& reason)
    {
        CONTEXT context;
        RtlCaptureContext(&context);
        logCrashOnHelperThread({.reason = reason, .context = &context});
    }
#else
    volatile std::sig_atomic_t crashSignalNumber = 0;

    void raiseCrashSignal(int)
    {
        std::signal(crashSignalNumber, SIG_DFL);
        std::raise(crashSignalNumber);
    }

    // Ends the process with the crash signal if logging deadlocks, e.g. after a crash inside malloc
    void startWatchdog(int signalNumber)
    {
        crashSignalNumber = signalNumber;
        std::signal(SIGALRM, raiseCrashSignal);
        alarm(static_cast<unsigned int>(MaxCrashLoggingDuration.count()));
    }
#endif

    SignalHandler previousAbortHandler = SIG_DFL;

    void logAbort(int)
    {
#ifdef _WIN32
        // The CRT has reset the handler, so a second abort() would end the process before this one is logged
        std::signal(SIGABRT, logAbort);
        logCrashAtCurrentPosition("abort() was called");
#else
        startWatchdog(SIGABRT);
        logCrash("abort() was called", [] { return Callstack::capture(); });
        std::signal(SIGABRT, SIG_DFL);
        std::raise(SIGABRT);
#endif
    }

    std::terminate_handler previousTerminateHandler = nullptr;

    void logTerminate()
    {
#ifndef _WIN32
        startWatchdog(SIGABRT);
#endif
        auto reason = std::string("std::terminate() was called");
        if (auto exception = std::current_exception()) {
            try {
                std::rethrow_exception(exception);
            } catch (std::exception const& e) {
                reason = std::string("uncaught exception: ") + e.what();
            } catch (...) {
                reason = "uncaught exception of unknown type";
            }
        }
#ifdef _WIN32
        logCrashAtCurrentPosition(reason);
#else
        logCrash(reason, [] { return Callstack::capture(); });
#endif
        if (previousTerminateHandler != nullptr) {
            previousTerminateHandler();
        }
        std::abort();
    }

#ifdef _WIN32
    LPTOP_LEVEL_EXCEPTION_FILTER previousExceptionFilter = nullptr;

    LONG WINAPI logUnhandledException(EXCEPTION_POINTERS* exceptionPointers)
    {
        logCrashOnHelperThread({.exceptionRecord = exceptionPointers->ExceptionRecord, .context = exceptionPointers->ContextRecord});
        return previousExceptionFilter != nullptr ? previousExceptionFilter(exceptionPointers) : EXCEPTION_CONTINUE_SEARCH;
    }
#else
    std::array<SignalHandler, FatalSignalNumbers.size()> previousFatalSignalHandlers = {};

    void logFatalSignal(int signalNumber)
    {
        startWatchdog(signalNumber);
        logCrash(std::string("signal ") + strsignal(signalNumber), [] { return Callstack::capture(); });
        std::signal(signalNumber, SIG_DFL);
        std::raise(signalNumber);
    }
#endif
}

_CrashLogger::_CrashLogger()
{
    previousTerminateHandler = std::set_terminate(logTerminate);
    previousAbortHandler = std::signal(SIGABRT, logAbort);
#ifdef _WIN32
    previousExceptionFilter = SetUnhandledExceptionFilter(logUnhandledException);
#else
    for (auto&& [signalNumber, previousHandler] : std::views::zip(FatalSignalNumbers, previousFatalSignalHandlers)) {
        previousHandler = std::signal(signalNumber, logFatalSignal);
    }
#endif
}

_CrashLogger::~_CrashLogger()
{
#ifdef _WIN32
    SetUnhandledExceptionFilter(previousExceptionFilter);
#else
    for (auto&& [signalNumber, previousHandler] : std::views::zip(FatalSignalNumbers, previousFatalSignalHandlers)) {
        std::signal(signalNumber, previousHandler);
    }
#endif
    std::signal(SIGABRT, previousAbortHandler);
    std::set_terminate(previousTerminateHandler);
}
