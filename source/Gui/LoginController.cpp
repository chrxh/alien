#include "LoginController.h"

#include "Base/GlobalSettings.h"
#include "Network/NetworkService.h"
#include "EngineInterface/SimulationFacade.h"
#include "PersisterInterface/LoginRequestData.h"
#include "PersisterInterface/SenderInfo.h"
#include "PersisterInterface/TaskProcessor.h"

#include "GenericMessageDialog.h"
#include "ActivateUserDialog.h"
#include "BrowserWindow.h"
#include "MainLoopEntityController.h"

void LoginController::init(SimulationFacade simulationFacade, PersisterFacade persisterFacade)
{
    _simulationFacade = simulationFacade;
    _persisterFacade = persisterFacade;
    _taskProcessor = _TaskProcessor::createTaskProcessor(_persisterFacade);

    auto& settings = GlobalSettings::get();
    _remember = settings.getValue("controller.login.remember", _remember);
    _shareGpuInfo = settings.getValue("controller.login.share gpu info", _shareGpuInfo);

    if (_remember) {
        _userName = settings.getValue("dialogs.login.user name", "");
        _password = settings.getValue("dialogs.login.password", "");
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
        _taskProcessor->executeTask(
            [&](auto const& senderId) {
                auto result = _persisterFacade->scheduleLogin(
                    SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true},
                    LoginRequestData{.userName = _userName, .password = _password, .userInfo = getUserInfo()});
                if (!_remember) {
                    _userName.clear();
                    _password.clear();
                }
                return result;
            },
            [&](auto const& requestId) {
                auto const& data = _persisterFacade->fetchLoginData(requestId);
                if (data.unknownUser) {
                    auto& settings = GlobalSettings::get();
                    auto userName = settings.getValue("dialogs.login.user name", "");
                    auto password = settings.getValue("dialogs.login.password", "");
                    ActivateUserDialog::get().open(userName, password, getUserInfo());
                }
                saveSettings();
                BrowserWindow::get().onRefresh();
            },
            [&](auto const& criticalErrors) { GenericMessageDialog::get().information("Error", criticalErrors); });
    }
}

void LoginController::process()
{
    _taskProcessor->process();
}

void LoginController::saveSettings()
{
    auto& settings = GlobalSettings::get();
    settings.setValue("controller.login.remember", _remember);
    settings.setValue("controller.login.share gpu info", _shareGpuInfo);
    if (_remember) {
        settings.setValue("dialogs.login.user name", _userName);
        settings.setValue("dialogs.login.password", _password);
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
        result.gpu = _simulationFacade->getGpuName();
    }
    return result;
}
