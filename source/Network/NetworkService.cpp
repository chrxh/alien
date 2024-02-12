#include "NetworkService.h"

#include <ranges>
#include <boost/property_tree/json_parser.hpp>

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <cpp-httplib/httplib.h>

#include "Base/GlobalSettings.h"
#include "Base/LoggingService.h"
#include "Base/Resources.h"

#include "NetworkResourceParserService.h"

namespace
{
    auto constexpr RefreshInterval = 20;  //in minutes
    auto constexpr MaxChunkSize = 24 * 1024 * 1024;

    void configureClient(httplib::SSLClient& client)
    {
        client.set_ca_cert_path("./resources/ca-bundle.crt");
        client.enable_server_certificate_verification(true);
        if (auto result = client.get_openssl_verify_result()) {
            throw std::runtime_error("OpenSSL verify error: " + std::string(X509_verify_cert_error_string(result)));
        }
    }

    httplib::Result executeRequest(std::function<httplib::Result()> const& func, bool withRetry = true)
    {
        auto attempt = 0;
        while (true) {
            auto result = func();
            if (result) {
                return result;
            }
            if (++attempt == 5 || !withRetry) {
                throw std::runtime_error("Error connecting to the server.");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    void logNetworkError()
    {
        log(Priority::Important, "network: an error occurred");
    }

    template<typename T>
    T parseValueFromKey(std::string const& jsonString, std::string const& key)
    {
        std::stringstream stream(jsonString);
        boost::property_tree::ptree tree;
        boost::property_tree::read_json(stream, tree);
        return tree.get<T>(key);
    }

    bool parseBoolResult(std::string const& serverResponse)
    {
        auto result = parseValueFromKey<bool>(serverResponse, "result");
        if (!result) {
            log(Priority::Important, "network: negative response received from server");
        }
        return result;
    }
}

std::string NetworkService::_serverAddress;
std::optional<std::string> NetworkService::_loggedInUserName;
std::optional<std::string> NetworkService::_password;
std::optional<std::chrono::steady_clock::time_point> NetworkService::_lastRefreshTime;
Cache<std::string, NetworkService::ResourceData, 20> NetworkService::_downloadCache;

void NetworkService::init()
{
    _serverAddress = GlobalSettings::getInstance().getStringState("settings.server", "alien-project.org");
}

void NetworkService::shutdown()
{
    GlobalSettings::getInstance().setStringState("settings.server", _serverAddress);
    logout();
}

std::string NetworkService::getServerAddress()
{
    return _serverAddress;
}

void NetworkService::setServerAddress(std::string const& value)
{
    _serverAddress = value;
    logout();
}

std::optional<std::string> NetworkService::getLoggedInUserName()
{
    return _loggedInUserName;
}

std::optional<std::string> NetworkService::getPassword()
{
    return _password;
}

bool NetworkService::createUser(std::string const& userName, std::string const& password, std::string const& email)
{
    log(Priority::Important, "network: create user '" + userName + "'");

    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::Params params;
    params.emplace("userName", userName);
    params.emplace("password", password);
    params.emplace("email", email);

    try {
        auto result = executeRequest([&] { return client.Post("/alien-server/createuser.php", params); });
        return parseBoolResult(result->body);
    } catch (...) {
        logNetworkError();
        return false;
    }
}

bool NetworkService::activateUser(std::string const& userName, std::string const& password, UserInfo const& userInfo, std::string const& confirmationCode)
{
    log(Priority::Important, "network: activate user '" + userName + "'");

    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::Params params;
    params.emplace("userName", userName);
    params.emplace("password", password);
    params.emplace("activationCode", confirmationCode);
    if (userInfo.gpu) {
        params.emplace("gpu", *userInfo.gpu);
    }

    try {
        auto result = executeRequest([&] { return client.Post("/alien-server/activateuser.php", params); });
        return parseBoolResult(result->body);
    } catch (...) {
        logNetworkError();
        return false;
    }
}

bool NetworkService::login(LoginErrorCode& errorCode, std::string const& userName, std::string const& password, UserInfo const& userInfo)
{
    log(Priority::Important, "network: login user '" + userName + "'");

    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::Params params;
    params.emplace("userName", userName);
    params.emplace("password", password);
    if (userInfo.gpu) {
        params.emplace("gpu", *userInfo.gpu);
    }

    try {
        auto result = executeRequest([&] { return client.Post("/alien-server/login.php", params); });

        auto boolResult = parseBoolResult(result->body);
        if (boolResult) {
            _loggedInUserName = userName;
            _password = password;
        }

        errorCode = false;
        std::stringstream stream(result->body);
        boost::property_tree::ptree tree;
        boost::property_tree::read_json(stream, tree);
        errorCode = tree.get<LoginErrorCode>("errorCode");

        return boolResult;
    } catch (...) {
        logNetworkError();
        return false;
    }
}

bool NetworkService::logout()
{
    log(Priority::Important, "network: logout");
    bool result = true;

    if (_loggedInUserName && _password) {
        httplib::SSLClient client(_serverAddress);
        configureClient(client);

        httplib::Params params;
        params.emplace("userName", *_loggedInUserName);
        params.emplace("password", *_password);

        try {
            result = executeRequest([&] { return client.Post("/alien-server/logout.php", params); });
        } catch (...) {
            logNetworkError();
            result = false;
        }
    }

    _loggedInUserName = std::nullopt;
    _password = std::nullopt;
    return result;
}

void NetworkService::refreshLogin()
{
    if (_loggedInUserName && _password) {
        log(Priority::Important, "network: refresh login");

        httplib::SSLClient client(_serverAddress);
        configureClient(client);

        httplib::Params params;
        params.emplace("userName", *_loggedInUserName);
        params.emplace("password", *_password);

        try {
            executeRequest([&] { return client.Post("/alien-server/refreshlogin.php", params); });
        } catch (...) {
        }
    }
}

bool NetworkService::deleteUser()
{
    log(Priority::Important, "network: delete user '" + *_loggedInUserName + "'");

    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::Params params;
    params.emplace("userName", *_loggedInUserName);
    params.emplace("password", *_password);

    try {
        auto postResult = executeRequest([&] { return client.Post("/alien-server/deleteuser.php", params); });

        auto result = parseBoolResult(postResult->body);
        if (result) {
            return logout();
        }
        return result;
    } catch (...) {
        logNetworkError();
        return false;
    }
}

bool NetworkService::resetPassword(std::string const& userName, std::string const& email)
{
    log(Priority::Important, "network: reset password of user '" + userName + "'");

    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::Params params;
    params.emplace("userName", userName);
    params.emplace("email", email);

    try {
        auto result = executeRequest([&] { return client.Post("/alien-server/resetpw.php", params); });
        return parseBoolResult(result->body);
    } catch (...) {
        logNetworkError();
        return false;
    }
}

bool NetworkService::setNewPassword(std::string const& userName, std::string const& newPassword, std::string const& confirmationCode)
{
    log(Priority::Important, "network: set new password for user '" + userName + "'");

    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::Params params;
    params.emplace("userName", userName);
    params.emplace("newPassword", newPassword);
    params.emplace("activationCode", confirmationCode);

    try {
        auto result = executeRequest([&] { return client.Post("/alien-server/setnewpw.php", params); });
        return parseBoolResult(result->body);
    } catch (...) {
        logNetworkError();
        return false;
    }
}

bool NetworkService::getNetworkResources(std::vector<NetworkResourceRawTO>& result, bool withRetry)
{
    log(Priority::Important, "network: get resource list");

    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::Params params;
    params.emplace("version", Const::ProgramVersion);
    if (_loggedInUserName && _password) {
        params.emplace("userName", *_loggedInUserName);
        params.emplace("password", *_password);
    }

    try {
        auto postResult = executeRequest([&] { return client.Post("/alien-server/getversionedsimulationlist.php", params); }, withRetry);

        std::stringstream stream(postResult->body);
        boost::property_tree::ptree tree;
        boost::property_tree::read_json(stream, tree);
        result = NetworkResourceParserService::decodeRemoteSimulationData(tree);
        return true;
    } catch (...) {
        logNetworkError();
        return false;
    }
}

bool NetworkService::getUserList(std::vector<UserTO>& result, bool withRetry)
{
    log(Priority::Important, "network: get user list");

    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    try {
        httplib::Params params;
        auto postResult = executeRequest([&] { return client.Post("/alien-server/getuserlist.php", params); }, withRetry);

        std::stringstream stream(postResult->body);
        boost::property_tree::ptree tree;
        boost::property_tree::read_json(stream, tree);
        result.clear();
        result = NetworkResourceParserService::decodeUserData(tree);
        for (UserTO& userData : result) {
            userData.timeSpent = userData.timeSpent * RefreshInterval / 60;
        }
        return true;
    } catch (...) {
        logNetworkError();
        return false;
    }
}

bool NetworkService::getEmojiTypeByResourceId(std::unordered_map<std::string, int>& result)
{
    log(Priority::Important, "network: get liked resources");

    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::Params params;
    params.emplace("userName", *_loggedInUserName);
    params.emplace("password", *_password);

    try {
        auto postResult = executeRequest([&] { return client.Post("/alien-server/getlikedsimulations.php", params); });

        std::stringstream stream(postResult->body);
        boost::property_tree::ptree tree;
        boost::property_tree::read_json(stream, tree);

        result.clear();
        for (auto const& [key, subTree] : tree) {
            result.emplace(subTree.get<std::string>("id"), subTree.get<int>("likeType"));
        }
        return true;
    } catch (...) {
        logNetworkError();
        return false;
    }
}

bool NetworkService::getUserNamesForResourceAndEmojiType(std::set<std::string>& result, std::string const& simId, int likeType)
{
    log(Priority::Important, "network: get user reactions for resource with id=" + simId + " and reaction type=" + std::to_string(likeType));

    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::Params params;
    params.emplace("simId", simId);
    params.emplace("likeType", std::to_string(likeType));

    try {
        auto postResult = executeRequest([&] { return client.Post("/alien-server/getuserlikes.php", params); });

        std::stringstream stream(postResult->body);
        boost::property_tree::ptree tree;
        boost::property_tree::read_json(stream, tree);

        result.clear();
        for (auto const& [key, subTree] : tree) {
            result.insert(subTree.get<std::string>("userName"));
        }
        return true;
    } catch (...) {
        logNetworkError();
        return false;
    }
}

bool NetworkService::toggleReactToResource(std::string const& simId, int likeType)
{
    log(Priority::Important, "network: toggle like for resource with id=" + simId);

    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::Params params;
    params.emplace("userName", *_loggedInUserName);
    params.emplace("password", *_password);
    params.emplace("simId", simId);
    params.emplace("likeType", std::to_string(likeType));


    try {
        auto result = executeRequest([&] { return client.Post("/alien-server/togglelikesimulation.php", params); });
        return parseBoolResult(result->body);
    } catch (...) {
        logNetworkError();
        return false;
    }
}

bool NetworkService::uploadResource(
    std::string& resourceId,
    std::string const& resourceName,
    std::string const& description,
    IntVector2D const& size,
    int particles,
    std::string const& mainData,
    std::string const& settings,
    std::string const& statistics,
    NetworkResourceType resourceType,
    WorkspaceType workspaceType)
{
    log(Priority::Important, "network: upload resource with name='" + resourceName + "'");

    std::vector<std::string> chunks;

    for (size_t i = 0; i < mainData.length(); i += MaxChunkSize) {
        std::string chunk = mainData.substr(i, MaxChunkSize);
        chunks.emplace_back(chunk);
    }

    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::MultipartFormDataItems items = {
        {"userName", *_loggedInUserName, "", ""},
        {"password", *_password, "", ""},
        {"simName", resourceName, "", ""},
        {"simDesc", description, "", ""},
        {"width", std::to_string(size.x), "", ""},
        {"height", std::to_string(size.y), "", ""},
        {"particles", std::to_string(particles), "", ""},
        {"version", Const::ProgramVersion, "", ""},
        {"content", chunks.front(), "", "application/octet-stream"},
        {"settings", settings, "", ""},
        {"symbolMap", "", "", ""},
        {"type", std::to_string(resourceType), "", ""},
        {"workspace", std::to_string(workspaceType), "", ""},
        {"statistics", statistics, "", ""},
    };

    try {
        auto result = executeRequest([&] { return client.Post("/alien-server/uploadsimulation.php", items); });
        if (parseBoolResult(result->body)) {
            resourceId = parseValueFromKey<std::string>(result->body, "simId");
        } else {
            return false;
        }
    } catch (...) {
        logNetworkError();
        return false;
    }

    for (auto const& chunk: chunks | std::views::drop(1)) {
        if (!appendResourceData(resourceId, chunk)) {
            deleteResource(resourceId);
            return false;
        }
    }
    _downloadCache.insert(resourceId, ResourceData{mainData, settings, statistics});

    return true;
}

bool NetworkService::downloadResource(std::string& mainData, std::string& auxiliaryData, std::string& statistics, std::string const& simId)
{
    try {
        if (auto cachedEntry = _downloadCache.find(simId)) {
            log(Priority::Important, "network: get resource with id=" + simId + " from download cache");
            mainData = cachedEntry->content;
            auxiliaryData = cachedEntry->auxiliaryData;
            statistics = cachedEntry->statistics;
            incDownloadCounter(simId);
            return true;
        } else {
            log(Priority::Important, "network: download resource with id=" + simId);

            httplib::SSLClient client(_serverAddress);
            configureClient(client);

            httplib::Params params;
            params.emplace("id", simId);
            {
                auto result = executeRequest([&] { return client.Get("/alien-server/downloadcontent.php", params, {}); });
                mainData = result->body;
            }
            {
                auto result = executeRequest([&] { return client.Get("/alien-server/downloadsettings.php", params, {}); });
                auxiliaryData = result->body;
            }
            {
                auto result = executeRequest([&] { return client.Get("/alien-server/downloadstatistics.php", params, {}); });
                statistics = result->body;
            }
            _downloadCache.insert(simId, ResourceData{mainData, auxiliaryData, statistics});
            return true;
        }
    } catch (...) {
        logNetworkError();
        return false;
    }
}

void NetworkService::incDownloadCounter(std::string const& simId)
{
    try {
        log(Priority::Important, "network: increment download counter for resource with id=" + simId);

        httplib::SSLClient client(_serverAddress);
        configureClient(client);

        httplib::Params params;
        params.emplace("id", simId);
        executeRequest([&] { return client.Get("/alien-server/incdownloadcount.php", params, {}); });
    }
    catch(...) {
       //do nothing 
    }
}

bool NetworkService::editResource(std::string const& simId, std::string const& newName, std::string const& newDescription)
{
    log(Priority::Important, "network: edit resource with id=" + simId);

    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::Params params;
    params.emplace("userName", *_loggedInUserName);
    params.emplace("password", *_password);
    params.emplace("simId", simId);
    params.emplace("newName", newName);
    params.emplace("newDescription", newDescription);

    try {
        auto result = executeRequest([&] { return client.Post("/alien-server/editsimulation.php", params); });
        return parseBoolResult(result->body);
    } catch (...) {
        logNetworkError();
        return false;
    }
}

bool NetworkService::moveResource(std::string const& simId, WorkspaceType targetWorkspace)
{
    log(Priority::Important, "network: move resource with id=" + simId + " to other workspace");

    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::Params params;
    params.emplace("userName", *_loggedInUserName);
    params.emplace("password", *_password);
    params.emplace("simId", simId);
    params.emplace("targetWorkspace", std::to_string(targetWorkspace));

    try {
        auto result = executeRequest([&] { return client.Post("/alien-server/movesimulation.php", params); });
        return parseBoolResult(result->body);
    } catch (...) {
        logNetworkError();
        return false;
    }
}

bool NetworkService::deleteResource(std::string const& simId)
{
    log(Priority::Important, "network: delete resource with id=" + simId);

    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::Params params;
    params.emplace("userName", *_loggedInUserName);
    params.emplace("password", *_password);
    params.emplace("simId", simId);

    try {
        auto result = executeRequest([&] { return client.Post("/alien-server/deletesimulation.php", params); });
        return parseBoolResult(result->body);
    } catch (...) {
        logNetworkError();
        return false;
    }
}

bool NetworkService::appendResourceData(std::string& resourceId, std::string const& data)
{
    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::MultipartFormDataItems items = {
        {"userName", *_loggedInUserName, "", ""},
        {"password", *_password, "", ""},
        {"simId", resourceId, "", ""},
        {"content", data, "", "application/octet-stream"},
    };

    try {
        auto result = executeRequest([&] { return client.Post("/alien-server/appendsimulationdata.php", items); });
        if (!parseBoolResult(result->body)) {
            return false;
        }
    } catch (...) {
        logNetworkError();
        return false;
    }
    return true;
}
