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

    bool createUser(std::string const& userName, std::string const& password, std::string const& email);
    bool activateUser(std::string const& userName, std::string const& password, std::string const& activationCode);
    bool login(std::string const& userName, std::string const& password);
    void logout();

    bool getRemoteSimulationDataList(std::vector<RemoteSimulationData>& result) const;
    bool uploadSimulation(
        std::string const& simulationName,
        std::string const& description,
        IntVector2D const& size,
        int particles,
        std::string const& content,
        std::string const& settings,
        std::string const& symbolMap);
    bool downloadSimulation(std::string& content, std::string& settings, std::string& symbolMap, std::string const& id);

private:
    std::string _serverAddress;
    std::optional<std::string> _loggedInUserName;
    std::optional<std::string> _password;
};
