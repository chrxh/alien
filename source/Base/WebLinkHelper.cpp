#include "WebLinkHelper.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <thread>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

void WebLinkHelper::openInBrowser(std::string const& url)
{
#ifdef _WIN32
    ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#else
    std::string program = "xdg-open";
    auto argument = url;
    char* arguments[] = {program.data(), argument.data(), nullptr};
    pid_t processId = 0;
    if (posix_spawnp(&processId, program.c_str(), nullptr, nullptr, arguments, environ) == 0) {
        std::thread([processId] { waitpid(processId, nullptr, 0); }).detach();
    }
#endif
}
