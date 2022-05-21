#pragma once

#include "RemoteSimulationData.h"
#include "Definitions.h"

class _NetworkController
{
public:
    _NetworkController();
    ~_NetworkController();

    std::string getServerAddress() const;
    bool isLoggedIn() const;

    std::vector<RemoteSimulationData> getRemoteSimulationDataList() const;

private:
    std::string _server;
};
