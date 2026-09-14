#include "LoginSession.h"

#include <chrono>
#include <condition_variable>
#include <mutex>

#include <Base/GlobalSettings.h>

#include <Network/NetworkService.h>

namespace
{
    auto constexpr LoginRefreshInterval = std::chrono::minutes(5);

    UserInfo createUserInfo(std::string const& gpuName)
    {
        UserInfo result;
        if (GlobalSettings::get().getValue(Const::ShareGpuInfoSettingsKey, Const::ShareGpuInfoDefault)) {
            result.gpu = gpuName;
        }
        return result;
    }
}

LoginSession::LoginSession(std::string const& userName, std::string const& password, std::string const& gpuName)
{
    NetworkService::get().setup();

    LoginErrorCode errorCode = LoginErrorCode_Other;
    if (!NetworkService::get().login(errorCode, userName, password, createUserInfo(gpuName))) {
        _errorMessage =
            errorCode == LoginErrorCode_UnknownUser ? "The user '" + userName + "' is not activated yet." : "Could not log in the user '" + userName + "'.";
        return;
    }
    _refreshThread = std::jthread([](std::stop_token const& stopToken) {
        std::mutex mutex;
        std::condition_variable_any conditionVariable;
        std::unique_lock lock(mutex);
        while (!conditionVariable.wait_for(lock, stopToken, LoginRefreshInterval, [&] { return stopToken.stop_requested(); })) {
            NetworkService::get().refreshLogin();
        }
    });
}

LoginSession::~LoginSession()
{
    if (_errorMessage.has_value()) {
        return;
    }
    _refreshThread.request_stop();
    _refreshThread.join();
    NetworkService::get().logout();
}

std::optional<std::string> const& LoginSession::getErrorMessage() const
{
    return _errorMessage;
}
