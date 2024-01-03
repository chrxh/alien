#pragma once

#include <chrono>

#include "NetworkResourceRawTO.h"
#include "UserTO.h"
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

class NetworkService
{
public:
    NetworkService(NetworkService const&) = delete;
    NetworkService() = delete;

    static void init();

    static std::string getServerAddress();
    static void setServerAddress(std::string const& value);
    static std::optional<std::string> getLoggedInUserName();
    static std::optional<std::string> getPassword();

    static bool createUser(std::string const& userName, std::string const& password, std::string const& email);
    static bool activateUser(std::string const& userName, std::string const& password, UserInfo const& userInfo, std::string const& confirmationCode);

    static bool login(LoginErrorCode& errorCode, std::string const& userName, std::string const& password, UserInfo const& userInfo);
    static bool logout();
    static void shutdown();
    static void refreshLogin();
    static bool deleteUser();
    static bool resetPassword(std::string const& userName, std::string const& email);
    static bool setNewPassword(std::string const& userName, std::string const& newPassword, std::string const& confirmationCode);

    static bool getNetworkResources(std::vector<NetworkResourceRawTO>& result, bool withRetry);
    static bool getUserList(std::vector<UserTO>& result, bool withRetry);
    static bool getEmojiTypeByResourceId(std::unordered_map<std::string, int>& result);
    static bool getUserNamesForResourceAndEmojiType(std::set<std::string>& result, std::string const& simId, int likeType);
    static bool toggleReactToResource(std::string const& simId, int likeType);

    static bool uploadSimulation(
        std::string const& simulationName,
        std::string const& description,
        IntVector2D const& size,
        int particles,
        std::string const& data,
        std::string const& settings,
        std::string const& statistics,
        NetworkResourceType resourceType,
        WorkspaceType workspaceType);
    static bool downloadSimulation(std::string& mainData, std::string& auxiliaryData, std::string& statistics, std::string const& simId);
    static bool deleteResource(std::string const& simId);

private:
    static std::string _serverAddress;
    static std::optional<std::string> _loggedInUserName;
    static std::optional<std::string> _password;
    static std::optional<std::chrono::steady_clock::time_point> _lastRefreshTime;
};
