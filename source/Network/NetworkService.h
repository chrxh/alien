#pragma once

#include <chrono>

#include "Base/Cache.h"
#include "NetworkResourceRawTO.h"
#include "UserTO.h"
#include "Definitions.h"
#include "Base/Singleton.h"

using LoginErrorCode = int;
enum LoginErrorCode_
{
    LoginErrorCode_UnknownUser,
    LoginErrorCode_Other
};

struct UserInfo
{
    std::optional<std::string> gpu;
};

class NetworkService
{
    MAKE_SINGLETON(NetworkService);

public:
    void setup();
    void shutdown();

    std::string getServerAddress();
    void setServerAddress(std::string const& value);
    bool isLoggedIn();
    std::optional<std::string> getLoggedInUserName();
    std::optional<std::string> getPassword();

    bool createUser(std::string const& userName, std::string const& password, std::string const& email);
    bool activateUser(std::string const& userName, std::string const& password, UserInfo const& userInfo, std::string const& confirmationCode);

    bool login(LoginErrorCode& errorCode, std::string const& userName, std::string const& password, UserInfo const& userInfo);
    bool logout();
    void refreshLogin();
    bool deleteUser();
    bool resetPassword(std::string const& userName, std::string const& email);
    bool setNewPassword(std::string const& userName, std::string const& newPassword, std::string const& confirmationCode);

    bool getNetworkResources(std::vector<NetworkResourceRawTO>& result, bool withRetry);
    bool getUserList(std::vector<UserTO>& result, bool withRetry);
    bool getEmojiTypeByResourceId(std::unordered_map<std::string, int>& result);
    bool getUserNamesForResourceAndEmojiType(std::set<std::string>& result, std::string const& simId, int likeType);
    bool toggleEmojiToResource(std::string const& simId, int likeType);

    bool uploadResource(
        std::string& resourceId,
        std::string const& resourceName,
        std::string const& description,
        IntVector2D const& worldSize,
        int numParticles,
        std::string const& data,
        std::string const& settings,
        std::string const& statistics,
        NetworkResourceType resourceType,
        WorkspaceType workspaceType);
    bool replaceResource(
        std::string const& resourceId,
        IntVector2D const& worldSize,
        int numParticles,
        std::string const& data,
        std::string const& settings,
        std::string const& statistics);
    bool downloadResource(std::string& mainData, std::string& auxiliaryData, std::string& statistics, std::string const& simId);
        void incDownloadCounter(std::string const& simId);
    bool editResource(std::string const& simId, std::string const& newName, std::string const& newDescription);
    bool moveResource(std::string const& simId, WorkspaceType targetWorkspace);
    bool deleteResource(std::string const& simId);

private:
    bool appendResourceData(std::string const& resourceId, std::string const& data, int chunkIndex);

    std::string _serverAddress;
    std::optional<std::string> _loggedInUserName;
    std::optional<std::string> _password;
    std::optional<std::chrono::steady_clock::time_point> _lastRefreshTime;

    struct ResourceData
    {
        std::string content;
        std::string auxiliaryData;
        std::string statistics;
    };
    Cache<std::string, ResourceData, 20> _downloadCache;
};
