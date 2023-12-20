#include "NetworkService.h"

#include <boost/property_tree/json_parser.hpp>

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <cpp-httplib/httplib.h>

#include "Base/GlobalSettings.h"
#include "Base/LoggingService.h"
#include "Base/Resources.h"

#include "NetworkDataParserService.h"

namespace
{
    auto RefreshInterval = 20;  //in minutes

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

    bool parseBoolResult(std::string const& serverResponse)
    {
        std::stringstream stream(serverResponse);
        boost::property_tree::ptree tree;
        boost::property_tree::read_json(stream, tree);
        auto result = tree.get<bool>("result");
        if (!result) {
            log(Priority::Important, "network: negative response received from server");
        }
        return result;
    }
}

_NetworkService::_NetworkService()
{
    _serverAddress = GlobalSettings::getInstance().getStringState("settings.server", "alien-project.org");
}

_NetworkService::~_NetworkService()
{
    GlobalSettings::getInstance().setStringState("settings.server", _serverAddress);
    logout();
}

std::string _NetworkService::getServerAddress() const
{
    return _serverAddress;
}

void _NetworkService::setServerAddress(std::string const& value)
{
    _serverAddress = value;
    logout();
}

std::optional<std::string> _NetworkService::getLoggedInUserName() const
{
    return _loggedInUserName;
}

std::optional<std::string> _NetworkService::getPassword() const
{
    return _password;
}

bool _NetworkService::createUser(std::string const& userName, std::string const& password, std::string const& email)
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

bool _NetworkService::activateUser(std::string const& userName, std::string const& password, UserInfo const& userInfo, std::string const& confirmationCode)
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

bool _NetworkService::login(LoginErrorCode& errorCode, std::string const& userName, std::string const& password, UserInfo const& userInfo)
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

bool _NetworkService::logout()
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

void _NetworkService::refreshLogin()
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

bool _NetworkService::deleteUser()
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

bool _NetworkService::resetPassword(std::string const& userName, std::string const& email)
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

bool _NetworkService::setNewPassword(std::string const& userName, std::string const& newPassword, std::string const& confirmationCode)
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

bool _NetworkService::getRemoteSimulationList(std::vector<NetworkDataTO>& result, bool withRetry) const
{
    log(Priority::Important, "network: get simulation list");

    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::Params params;
    params.emplace("version", Const::ProgramVersion);

    try {
        auto postResult = executeRequest([&] { return client.Post("/alien-server/getversionedsimulationlist.php", params); }, withRetry);

        std::stringstream stream(postResult->body);
        boost::property_tree::ptree tree;
        boost::property_tree::read_json(stream, tree);
        result = NetworkDataParserService::decodeRemoteSimulationData(tree);
        return true;
    } catch (...) {
        logNetworkError();
        return false;
    }
}

bool _NetworkService::getUserList(std::vector<UserTO>& result, bool withRetry) const
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
        result = NetworkDataParserService::decodeUserData(tree);
        for (UserTO& userData : result) {
            userData.timeSpent = userData.timeSpent * RefreshInterval / 60;
        }
        return true;
    } catch (...) {
        logNetworkError();
        return false;
    }
}

bool _NetworkService::getEmojiTypeBySimId(std::unordered_map<std::string, int>& result) const
{
    log(Priority::Important, "network: get liked simulations");

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

bool _NetworkService::getUserNamesForSimulationAndEmojiType(std::set<std::string>& result, std::string const& simId, int likeType)
{
    log(Priority::Important, "network: get user likes for simulation with id=" + simId + " and likeType=" + std::to_string(likeType));

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

bool _NetworkService::toggleLikeSimulation(std::string const& simId, int likeType)
{
    log(Priority::Important, "network: toggle like for simulation with id=" + simId);

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

bool _NetworkService::uploadSimulation(
    std::string const& simulationName,
    std::string const& description,
    IntVector2D const& size,
    int particles,
    std::string const& mainData,
    std::string const& settings,
    std::string const& statistics,
    NetworkDataType type)
{
    log(Priority::Important, "network: upload simulation with name='" + simulationName + "'");

    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::MultipartFormDataItems items = {
        {"userName", *_loggedInUserName, "", ""},
        {"password", *_password, "", ""},
        {"simName", simulationName, "", ""},
        {"simDesc", description, "", ""},
        {"width", std::to_string(size.x), "", ""},
        {"height", std::to_string(size.y), "", ""},
        {"particles", std::to_string(particles), "", ""},
        {"version", Const::ProgramVersion, "", ""},
        {"content", mainData, "", "application/octet-stream"},
        {"settings", settings, "", ""},
        {"symbolMap", "", "", ""},
        {"type", std::to_string(type), "", ""},
        {"statistics", statistics, "", ""},
    };

    try {
        auto result = executeRequest([&] { return client.Post("/alien-server/uploadsimulation.php", items); });
        return parseBoolResult(result->body);
    } catch (...) {
        logNetworkError();
        return false;
    }
}

bool _NetworkService::downloadSimulation(std::string& mainData, std::string& auxiliaryData, std::string& statistics, std::string const& simId)
{
    log(Priority::Important, "network: download simulation with id=" + simId);

    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::Params params;
    params.emplace("id", simId);

    try {
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
        return true;
    } catch (...) {
        logNetworkError();
        return false;
    }
}

bool _NetworkService::deleteSimulation(std::string const& simId)
{
    log(Priority::Important, "network: delete simulation with id=" + simId);

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
