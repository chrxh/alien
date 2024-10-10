#include "LoginController.h"

#include "Base/GlobalSettings.h"
#include "Network/NetworkService.h"
#include "EngineInterface/SimulationController.h"
#include "PersisterInterface/LoginRequestData.h"
#include "PersisterInterface/SenderInfo.h"

namespace
{
    auto constexpr LoginSenderId = "Login";
}

LoginController& LoginController::get()
{
    static LoginController instance;
    return instance;
}

void LoginController::init(SimulationController const& simController, PersisterController const& persisterController)
{
    _simController = simController;
    _persisterController = persisterController;

    auto& settings = GlobalSettings::getInstance();
    _remember = settings.getBool("dialogs.login.remember", _remember);
    _shareGpuInfo = settings.getBool("dialogs.login.share gpu info", _shareGpuInfo);

    if (_remember) {
        auto userName = settings.getString("dialogs.login.user name", "");
        auto password = settings.getString("dialogs.login.password", "");

        if (!userName.empty()) {
            persisterController->scheduleLogin(
                SenderInfo{.senderId = LoginSenderId, .wishResultData = false, .wishErrorInfo = true},
                LoginRequestData{.userName = userName, .password = password, .userInfo = getUserInfo()});
            //LoginErrorCode errorCode;
            //if (!NetworkService::login(errorCode, _userName, _password, getUserInfo())) {
            //    if (errorCode != LoginErrorCode_UnknownUser) {
            //        MessageDialog::getInstance().information("Error", "Login failed.");
            //    }
            //}
        }
    }
}

void LoginController::shutdown()
{
    saveSettings();
}

void LoginController::saveSettings()
{
    auto& settings = GlobalSettings::getInstance();
    settings.setBool("dialogs.login.remember", _remember);
    settings.setBool("dialogs.login.share gpu info", _shareGpuInfo);
    if (_remember) {
        auto userName = NetworkService::getLoggedInUserName();
        auto password = NetworkService::getPassword();
        if (userName.has_value() && password.has_value()) {
            settings.setString("dialogs.login.user name", *userName);
            settings.setString("dialogs.login.password", *password);
        }
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

UserInfo LoginController::getUserInfo()
{
    UserInfo result;
    if (_shareGpuInfo) {
        result.gpu = _simController->getGpuName();
    }
    return result;
}
