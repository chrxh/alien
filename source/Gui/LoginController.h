#pragma once

#include "Definitions.h"
#include "PersisterInterface/PersisterController.h"

class LoginController
{
public:
    static LoginController& get();

    void init(SimulationController const& simController, PersisterController const& persisterController);
    void shutdown();

    void saveSettings();

    bool shareGpuInfo() const;
    void setShareGpuInfo(bool value);

    bool isRemember() const;
    void setRemember(bool value);

    UserInfo getUserInfo();

private:
    SimulationController _simController; 
    PersisterController _persisterController;

    bool _shareGpuInfo = true;
    bool _remember = true;
};
