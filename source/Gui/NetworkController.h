#pragma once

#include "RemoteSimulationData.h"
#include "Definitions.h"

class _NetworkController
{
public:
    _NetworkController();
    ~_NetworkController();

    std::string getServerAddress() const;
    std::optional<std::string> getLoggedInUserName() const;
    bool login(std::string const& userName, std::string const& passwordHash);

    std::vector<RemoteSimulationData> getRemoteSimulationDataList() const;

private:
    std::string _serverAddress;
    std::optional<std::string> _loggedInUserName;
};
