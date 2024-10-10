#include "LoginController.h"

#include "Base/GlobalSettings.h"
#include "Network/NetworkService.h"
#include "EngineInterface/SimulationController.h"
#include "PersisterInterface/LoginRequestData.h"
#include "PersisterInterface/SenderInfo.h"

#include "MessageDialog.h"
#include "ActivateUserDialog.h"
#include "BrowserWindow.h"

namespace
{
    auto constexpr LoginSenderId = "Login";
}

void LoginController::init(
    SimulationController const& simController,
    PersisterController const& persisterController,
    ActivateUserDialog const& activateUserDialog,
    BrowserWindow const& browserWindow)
{
    _simController = simController;
    _persisterController = persisterController;
    _activateUserDialog = activateUserDialog;
    _browserWindow = browserWindow;

    auto& settings = GlobalSettings::getInstance();
    _remember = settings.getBool("controller.login.remember", _remember);
    _shareGpuInfo = settings.getBool("controller.login.share gpu info", _shareGpuInfo);

    if (_remember) {
        _userName = settings.getString("dialogs.login.user name", "");
        _password = settings.getString("dialogs.login.password", "");
        onLogin();
    }
}

void LoginController::shutdown()
{
    saveSettings();
}

void LoginController::onLogin()
{
    if (!_userName.empty()) {
        _pendingLoginRequestIds.emplace_back(_persisterController->scheduleLogin(
            SenderInfo{.senderId = SenderId{LoginSenderId}, .wishResultData = true, .wishErrorInfo = true},
            LoginRequestData{.userName = _userName, .password = _password, .userInfo = getUserInfo()}));
        if (!_remember) {
            _userName.clear();
            _password.clear();
        }
    }
}

void LoginController::process()
{
    std::vector<PersisterRequestId> newLoginRequestIds;
    for (auto const& requestId : _pendingLoginRequestIds) {
        auto state = _persisterController->getRequestState(requestId);
        if (state == PersisterRequestState::Finished) {
            auto const& data = _persisterController->fetchLoginData(requestId);
            if (data.unknownUser) {
                auto& settings = GlobalSettings::getInstance();
                auto userName = settings.getString("dialogs.login.user name", "");
                auto password = settings.getString("dialogs.login.password", "");
                _activateUserDialog->open(userName, password, getUserInfo());
            }
            saveSettings();
            _browserWindow->onRefresh();
        }
        if (state == PersisterRequestState::InQueue || state == PersisterRequestState::InProgress) {
            newLoginRequestIds.emplace_back(requestId);
        }
    }
    _pendingLoginRequestIds = newLoginRequestIds;

    auto criticalErrors = _persisterController->fetchAllErrorInfos(SenderId{LoginSenderId});
    if (!criticalErrors.empty()) {
        MessageDialog::get().information("Error", criticalErrors);
    }
}

void LoginController::saveSettings()
{
    auto& settings = GlobalSettings::getInstance();
    settings.setBool("controller.login.remember", _remember);
    settings.setBool("controller.login.share gpu info", _shareGpuInfo);
    if (_remember) {
        settings.setString("dialogs.login.user name", _userName);
        settings.setString("dialogs.login.password", _password);
    }
}

bool LoginController::shareGpuInfo() const
{
    return _shareGpuInfo;
}

void LoginController::setShareGpuInfo(bool value)
{
    _shareGpuInfo = value;
}

bool LoginController::isRemember() const
{
    return _remember;
}

void LoginController::setRemember(bool value)
{
    _remember = value;
}

std::string const& LoginController::getUserName() const
{
    return _userName;
}

void LoginController::setUserName(std::string const& value)
{
    _userName = value;
}

std::string const& LoginController::getPassword() const
{
    return _password;
}

void LoginController::setPassword(std::string const& value)
{
    _password = value;
}

UserInfo LoginController::getUserInfo()
{
    UserInfo result;
    if (_shareGpuInfo) {
        result.gpu = _simController->getGpuName();
    }
    return result;
}
