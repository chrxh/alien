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
    void logout();

    std::vector<RemoteSimulationData> getRemoteSimulationDataList() const;
    void uploadSimulation(
        std::string const& simulationName,
        std::string const& description,
        IntVector2D const& size,
        int particles,
        std::string const& content,
        std::string const& settings,
        std::string const& symbolMap);
    void downloadSimulation(std::string& content, std::string& settings, std::string& symbolMap, std::string const& id);

private:
    std::string _serverAddress;
    std::optional<std::string> _loggedInUserName;
    std::optional<std::string> _passwordHash;
};
