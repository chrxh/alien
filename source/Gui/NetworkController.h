#pragma once

#include <chrono>

#include "NetworkDataTO.h"
#include "UserData.h"
#include "Definitions.h"

using LoginErrorCode = int;
enum LoginErrorCode_
{
    LoginErrorCode_UnconfirmedUser,
    LoginErrorCode_Other
};

struct UserInfo
{
    std::optional<std::string> gpu;
};

class _NetworkController
{
public:
    _NetworkController();
    ~_NetworkController();

    void process();

    std::string getServerAddress() const;
    void setServerAddress(std::string const& value);
    std::optional<std::string> getLoggedInUserName() const;
    std::optional<std::string> getPassword() const;

    bool createUser(std::string const& userName, std::string const& password, std::string const& email);
    bool activateUser(std::string const& userName, std::string const& password, UserInfo const& userInfo, std::string const& confirmationCode);

    bool login(LoginErrorCode& errorCode, std::string const& userName, std::string const& password, UserInfo const& userInfo);
    bool logout();
    bool deleteUser();
    bool resetPassword(std::string const& userName, std::string const& email);
    bool setNewPassword(std::string const& userName, std::string const& newPassword, std::string const& confirmationCode);

    bool getRemoteSimulationList(std::vector<NetworkDataTO>& result, bool withRetry) const;
    bool getUserList(std::vector<UserData>& result, bool withRetry) const;
    bool getEmojiTypeBySimId(std::unordered_map<std::string, int>& result) const;
    bool getUserNamesForSimulationAndEmojiType(std::set<std::string>& result, std::string const& simId, int likeType);
    bool toggleLikeSimulation(std::string const& simId, int likeType);

    bool uploadSimulation(
        std::string const& simulationName,
        std::string const& description,
        IntVector2D const& size,
        int particles,
        std::string const& data,
        std::string const& settings,
        std::string const& statistics,
        NetworkDataType type);
    bool downloadSimulation(std::string& mainData, std::string& auxiliaryData, std::string& statistics, std::string const& simId);
    bool deleteSimulation(std::string const& simId);

private:
    void refreshLogin();

    std::string _serverAddress;
    std::optional<std::string> _loggedInUserName;
    std::optional<std::string> _password;
    std::optional<std::chrono::steady_clock::time_point> _lastRefreshTime;
};
