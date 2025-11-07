#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/SimulationFacade.h>

#include <PersisterInterface/PersisterFacade.h>

#include "Definitions.h"
#include "MainLoopEntity.h"

class LoginController : public MainLoopEntity<>
{
    MAKE_SINGLETON(LoginController);

public:
    void onLogin();

    void saveSettings();

    bool shareGpuInfo() const;
    void setShareGpuInfo(bool value);

    bool isRemember() const;
    void setRemember(bool value);

    std::string const& getUserName() const;
    void setUserName(std::string const& value);

    std::string const& getPassword() const;
    void setPassword(std::string const& value);

    UserInfo getUserInfo();

private:
    void init() override;
    void process() override;
    void shutdown() override;

    TaskProcessor _taskProcessor;

    bool _shareGpuInfo = true;
    bool _remember = true;
    std::string _userName;
    std::string _password;
};
