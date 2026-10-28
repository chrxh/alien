#include "ServerSession.h"

#include <chrono>
#include <condition_variable>
#include <mutex>

#include <Base/GlobalSettings.h>

#include <Network/NetworkService.h>

namespace
{
    auto constexpr RefreshInterval = std::chrono::minutes(5);

    UserInfo createUserInfo(std::string const& gpuName)
    {
        UserInfo result;
        if (GlobalSettings::get().getValue(Const::ShareGpuInfoSettingsKey, Const::ShareGpuInfoDefault)) {
            result.gpu = gpuName;
        }
        return result;
    }
}

ServerSession::ServerSession(std::string const& userName, std::string const& password, std::string const& gpuName, bool countAsActive)
{
    NetworkService::get().setup();

    if (!userName.empty()) {
        LoginErrorCode errorCode = LoginErrorCode_Other;
        if (!NetworkService::get().login(errorCode, userName, password, createUserInfo(gpuName))) {
            _errorMessage =
                errorCode == LoginErrorCode_UnknownUser ? "The user '" + userName + "' is not activated yet." : "Could not log in the user '" + userName + "'.";
            return;
        }
        _loggedIn = true;
    }
    if (!_loggedIn && !countAsActive) {
        return;
    }
    _refreshThread = std::jthread([countAsActive](std::stop_token const& stopToken) {
        std::mutex mutex;
        std::condition_variable_any conditionVariable;
        std::unique_lock lock(mutex);
        while (!conditionVariable.wait_for(lock, stopToken, RefreshInterval, [&] { return stopToken.stop_requested(); })) {
            NetworkService::get().refreshLogin();
            if (countAsActive) {
                NetworkService::get().reportActiveSimulator();
            }
        }
    });
}

ServerSession::~ServerSession()
{
    if (_refreshThread.joinable()) {
        _refreshThread.request_stop();
        _refreshThread.join();
    }
    if (_loggedIn) {
        NetworkService::get().logout();
    }
}

bool ServerSession::isLoggedIn() const
{
    return _loggedIn;
}

std::optional<std::string> const& ServerSession::getErrorMessage() const
{
    return _errorMessage;
}
