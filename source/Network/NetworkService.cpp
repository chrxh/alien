#include "NetworkService.h"

#include <boost/property_tree/json_parser.hpp>

#define CPPHTTPLIB_OPENSSL_SUPPORT

#include <Base/GlobalSettings.h>
#include <Base/LoggingService.h>
#include <Base/Resources.h>

#include "NetworkResourceParserService.h"

#include <cpp-httplib/httplib.h>

namespace
{
    void configureClient(httplib::Client& client)
    {
        client.set_ca_cert_path("./resources/ca-bundle.crt");
        client.enable_server_certificate_verification(true);

        // Transparently follow 301/302 redirects so that an nginx vhost
        // adding/removing a trailing slash does not surface to callers as an
        // empty/HTML response body.
        client.set_follow_location(true);
        client.set_connection_timeout(std::chrono::seconds(10));
        client.set_read_timeout(std::chrono::seconds(120));
        client.set_write_timeout(std::chrono::seconds(120));
    }

    std::string toHttpsServerAddress(std::string const& serverAddress)
    {
        if (serverAddress.starts_with("https://")) {
            return serverAddress;
        }
        if (serverAddress.starts_with("http://")) {
            return "https://" + serverAddress.substr(7);
        }
        return "https://" + serverAddress;
    }

    httplib::Client createClient(std::string const& serverAddress)
    {
        httplib::Client client(toHttpsServerAddress(serverAddress));
        configureClient(client);
        return client;
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

    template <typename T>
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

void NetworkService::setup()
{
    _serverAddress = GlobalSettings::get().getValue("settings.server url", std::string(Const::AlienServerURL));
}

void NetworkService::shutdown()
{
    GlobalSettings::get().setValue("settings.server url", _serverAddress);
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

bool NetworkService::isLoggedIn()
{
    return static_cast<bool>(_loggedInUserName);
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

    auto client = createClient(_serverAddress);

    httplib::Params params;
    params.emplace("userName", userName);
    params.emplace("password", password);
    params.emplace("email", email);

    try {
        auto result = executeRequest([&] { return client.Post("/createuser", params); });
        return parseBoolResult(result->body);
    } catch (...) {
        logNetworkError();
        return false;
    }
}

bool NetworkService::activateUser(std::string const& userName, std::string const& password, UserInfo const& userInfo, std::string const& confirmationCode)
{
    log(Priority::Important, "network: activate user '" + userName + "'");

    auto client = createClient(_serverAddress);

    httplib::Params params;
    params.emplace("userName", userName);
    params.emplace("password", password);
    params.emplace("activationCode", confirmationCode);
    if (userInfo.gpu) {
        params.emplace("gpu", *userInfo.gpu);
    }

    try {
        auto result = executeRequest([&] { return client.Post("/activateuser", params); });
        return parseBoolResult(result->body);
    } catch (...) {
        logNetworkError();
        return false;
    }
}

bool NetworkService::login(LoginErrorCode& errorCode, std::string const& userName, std::string const& password, UserInfo const& userInfo)
{
    log(Priority::Important, "network: login user '" + userName + "'");

    auto client = createClient(_serverAddress);

    httplib::Params params;
    params.emplace("userName", userName);
    params.emplace("password", password);
    if (userInfo.gpu) {
        params.emplace("gpu", *userInfo.gpu);
    }

    try {
        auto result = executeRequest([&] { return client.Post("/login", params); });

        auto boolResult = parseBoolResult(result->body);
        if (boolResult) {
            _loggedInUserName = userName;
            _password = password;
        }

        errorCode = LoginErrorCode_UnknownUser;
        std::stringstream stream(result->body);
        boost::property_tree::ptree tree;
        boost::property_tree::read_json(stream, tree);
        errorCode = tree.get<LoginErrorCode>("errorCode");

        return boolResult;
    } catch (...) {
        logNetworkError();
        errorCode = LoginErrorCode_Other;
        return false;
    }
}

bool NetworkService::logout()
{
    log(Priority::Important, "network: logout");
    bool result = true;

    if (_loggedInUserName && _password) {
        auto client = createClient(_serverAddress);

        httplib::Params params;
        params.emplace("userName", *_loggedInUserName);
        params.emplace("password", *_password);

        try {
            result = executeRequest([&] { return client.Post("/logout", params); });
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

        auto client = createClient(_serverAddress);

        httplib::Params params;
        params.emplace("userName", *_loggedInUserName);
        params.emplace("password", *_password);

        try {
            executeRequest([&] { return client.Post("/refreshlogin", params); });
        } catch (...) {
        }
    }
}

bool NetworkService::deleteUser()
{
    log(Priority::Important, "network: delete user '" + *_loggedInUserName + "'");

    auto client = createClient(_serverAddress);

    httplib::Params params;
    params.emplace("userName", *_loggedInUserName);
    params.emplace("password", *_password);

    try {
        auto postResult = executeRequest([&] { return client.Post("/deleteuser", params); });

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

    auto client = createClient(_serverAddress);

    httplib::Params params;
    params.emplace("userName", userName);
    params.emplace("email", email);

    try {
        auto result = executeRequest([&] { return client.Post("/resetpw", params); });
        return parseBoolResult(result->body);
    } catch (...) {
        logNetworkError();
        return false;
    }
}

bool NetworkService::setNewPassword(std::string const& userName, std::string const& newPassword, std::string const& confirmationCode)
{
    log(Priority::Important, "network: set new password for user '" + userName + "'");

    auto client = createClient(_serverAddress);

    httplib::Params params;
    params.emplace("userName", userName);
    params.emplace("newPassword", newPassword);
    params.emplace("activationCode", confirmationCode);

    try {
        auto result = executeRequest([&] { return client.Post("/setnewpw", params); });
        return parseBoolResult(result->body);
    } catch (...) {
        logNetworkError();
        return false;
    }
}

bool NetworkService::getNetworkResources(std::vector<NetworkResourceRawTO>& result, bool withRetry)
{
    log(Priority::Important, "network: get resource list");

    auto client = createClient(_serverAddress);

    httplib::Params params;
    params.emplace("version", Const::ProgramVersion);
    if (_loggedInUserName && _password) {
        params.emplace("userName", *_loggedInUserName);
        params.emplace("password", *_password);
    }

    try {
        auto postResult = executeRequest([&] { return client.Post("/getversionedsimulationlist", params); }, withRetry);

        std::stringstream stream(postResult->body);
        boost::property_tree::ptree tree;
        boost::property_tree::read_json(stream, tree);
        result = NetworkResourceParserService::get().decodeRemoteSimulationData(tree);
        return true;
    } catch (...) {
        logNetworkError();
        return false;
    }
}

bool NetworkService::getUserList(std::vector<UserTO>& result, bool withRetry)
{
    log(Priority::Important, "network: get user list");

    auto client = createClient(_serverAddress);

    try {
        httplib::Params params;
        auto postResult = executeRequest([&] { return client.Post("/getuserlist", params); }, withRetry);

        std::stringstream stream(postResult->body);
        boost::property_tree::ptree tree;
        boost::property_tree::read_json(stream, tree);
        result.clear();
        result = NetworkResourceParserService::get().decodeUserData(tree);
        // ``timeSpent`` is reported by the server in seconds (cumulative
        // online time). The UI converts to a human-readable form on display;
        // no transformation is applied here.
        return true;
    } catch (...) {
        logNetworkError();
        return false;
    }
}

bool NetworkService::getEmojiTypeByResourceId(std::unordered_map<std::string, int>& result)
{
    log(Priority::Important, "network: get liked resources");

    auto client = createClient(_serverAddress);

    httplib::Params params;
    params.emplace("userName", *_loggedInUserName);
    params.emplace("password", *_password);

    try {
        auto postResult = executeRequest([&] { return client.Post("/getlikedsimulations", params); });

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

    auto client = createClient(_serverAddress);

    httplib::Params params;
    params.emplace("simId", simId);
    params.emplace("likeType", std::to_string(likeType));

    try {
        auto postResult = executeRequest([&] { return client.Post("/getuserlikes", params); });

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

bool NetworkService::toggleReactionForResource(std::string const& simId, int likeType)
{
    log(Priority::Important, "network: toggle like for resource with id=" + simId);

    auto client = createClient(_serverAddress);

    httplib::Params params;
    params.emplace("userName", *_loggedInUserName);
    params.emplace("password", *_password);
    params.emplace("simId", simId);
    params.emplace("likeType", std::to_string(likeType));


    try {
        auto result = executeRequest([&] { return client.Post("/togglelikesimulation", params); });
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
    IntVector2D const& worldSize,
    int numObjects,
    std::string const& mainData,
    std::string const& settings,
    std::string const& statistics,
    NetworkResourceType resourceType,
    WorkspaceType workspaceType)
{
    log(Priority::Important, "network: upload resource with name='" + resourceName + "'");

    auto client = createClient(_serverAddress);

    httplib::MultipartFormDataItems items = {
        {"userName", *_loggedInUserName, "", ""},
        {"password", *_password, "", ""},
        {"simName", resourceName, "", ""},
        {"simDesc", description, "", ""},
        {"width", std::to_string(worldSize.x), "", ""},
        {"height", std::to_string(worldSize.y), "", ""},
        {"particles", std::to_string(numObjects), "", ""},
        {"version", Const::ProgramVersion, "", ""},
        {"content", mainData, "content.bin", "application/octet-stream"},
        {"settings", settings, "", ""},
        {"type", std::to_string(resourceType), "", ""},
        {"workspace", std::to_string(workspaceType), "", ""},
        {"statistics", statistics, "", ""},
    };

    try {
        auto result = executeRequest([&] { return client.Post("/uploadsimulation", items); });
        if (parseBoolResult(result->body)) {
            resourceId = parseValueFromKey<std::string>(result->body, "simId");
        } else {
            return false;
        }
    } catch (...) {
        logNetworkError();
        return false;
    }
    _downloadCache.insertOrAssign(resourceId, ResourceData{mainData, settings, statistics});

    return true;
}

bool NetworkService::replaceResource(
    std::string const& resourceId,
    IntVector2D const& worldSize,
    int numObjects,
    std::string const& mainData,
    std::string const& settings,
    std::string const& statistics)
{
    log(Priority::Important, "network: replace resource with id='" + resourceId + "'");

    auto client = createClient(_serverAddress);

    httplib::MultipartFormDataItems items = {
        {"userName", *_loggedInUserName, "", ""},
        {"password", *_password, "", ""},
        {"simId", resourceId, "", ""},
        {"width", std::to_string(worldSize.x), "", ""},
        {"height", std::to_string(worldSize.y), "", ""},
        {"particles", std::to_string(numObjects), "", ""},
        {"version", Const::ProgramVersion, "", ""},
        {"content", mainData, "content.bin", "application/octet-stream"},
        {"settings", settings, "", ""},
        {"statistics", statistics, "", ""},
    };

    try {
        auto result = executeRequest([&] { return client.Post("/replacesimulation", items); });
        if (!parseBoolResult(result->body)) {
            return false;
        }
    } catch (...) {
        logNetworkError();
        return false;
    }
    _downloadCache.insertOrAssign(resourceId, ResourceData{mainData, settings, statistics});

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

            auto client = createClient(_serverAddress);

            httplib::Params params;
            params.emplace("id", simId);
            {
                auto result = executeRequest([&] { return client.Get("/downloadcontent", params, {}); });
                mainData = result->body;
            }
            {
                auto result = executeRequest([&] { return client.Get("/downloadsettings", params, {}); });
                auxiliaryData = result->body;
            }
            {
                auto result = executeRequest([&] { return client.Get("/downloadstatistics", params, {}); });
                statistics = result->body;
            }
            _downloadCache.insertOrAssign(simId, ResourceData{mainData, auxiliaryData, statistics});
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

        auto client = createClient(_serverAddress);

        httplib::Params params;
        params.emplace("id", simId);
        executeRequest([&] { return client.Get("/incdownloadcount", params, {}); });
    } catch (...) {
        //do nothing
    }
}

bool NetworkService::editResource(std::string const& simId, std::string const& newName, std::string const& newDescription)
{
    log(Priority::Important, "network: edit resource with id=" + simId);

    auto client = createClient(_serverAddress);

    httplib::Params params;
    params.emplace("userName", *_loggedInUserName);
    params.emplace("password", *_password);
    params.emplace("simId", simId);
    params.emplace("newName", newName);
    params.emplace("newDescription", newDescription);

    try {
        auto result = executeRequest([&] { return client.Post("/editsimulation", params); });
        return parseBoolResult(result->body);
    } catch (...) {
        logNetworkError();
        return false;
    }
}

bool NetworkService::moveResource(std::string const& simId, WorkspaceType targetWorkspace)
{
    log(Priority::Important, "network: move resource with id=" + simId + " to other workspace");

    auto client = createClient(_serverAddress);

    httplib::Params params;
    params.emplace("userName", *_loggedInUserName);
    params.emplace("password", *_password);
    params.emplace("simId", simId);
    params.emplace("targetWorkspace", std::to_string(targetWorkspace));

    try {
        auto result = executeRequest([&] { return client.Post("/movesimulation", params); });
        return parseBoolResult(result->body);
    } catch (...) {
        logNetworkError();
        return false;
    }
}

bool NetworkService::deleteResource(std::string const& simId)
{
    log(Priority::Important, "network: delete resource with id=" + simId);

    auto client = createClient(_serverAddress);

    httplib::Params params;
    params.emplace("userName", *_loggedInUserName);
    params.emplace("password", *_password);
    params.emplace("simId", simId);

    try {
        auto result = executeRequest([&] { return client.Post("/deletesimulation", params); });
        return parseBoolResult(result->body);
    } catch (...) {
        logNetworkError();
        return false;
    }
}
