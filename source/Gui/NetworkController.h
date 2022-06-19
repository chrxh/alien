#pragma once

#include "RemoteSimulationData.h"
#include "Definitions.h"

class _NetworkController
{
public:
    _NetworkController();
    ~_NetworkController();

    std::string getServerAddress() const;
    void setServerAddress(std::string const& value);
    std::optional<std::string> getLoggedInUserName() const;
    std::optional<std::string> getPassword() const;

    bool createUser(std::string const& userName, std::string const& password, std::string const& email);
    bool activateUser(std::string const& userName, std::string const& password, std::string const& confirmationCode);
    bool login(std::string const& userName, std::string const& password);
    void logout();
    bool deleteUser();
    bool resetPassword(std::string const& userName, std::string const& email);
    bool setNewPassword(std::string const& userName, std::string const& newPassword, std::string const& confirmationCode);

    bool getRemoteSimulationDataList(std::vector<RemoteSimulationData>& result, bool withRetry) const;
    bool getLikedSimulationIdList(std::vector<std::string>& result) const;
    bool getUserLikesForSimulation(std::set<std::string>& result, std::string const& simId);
    bool toggleLikeSimulation(std::string const& simId);

    bool uploadSimulation(
        std::string const& simulationName,
        std::string const& description,
        IntVector2D const& size,
        int particles,
        std::string const& content,
        std::string const& settings,
        std::string const& symbolMap);
    bool downloadSimulation(std::string& content, std::string& settings, std::string& symbolMap, std::string const& simId);
    bool deleteSimulation(std::string const& simId);

private:
    std::string _serverAddress;
    std::optional<std::string> _loggedInUserName;
    std::optional<std::string> _password;
};
