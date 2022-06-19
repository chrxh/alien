#include "NetworkController.h"

#include <boost/property_tree/json_parser.hpp>

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <cpp-httplib/httplib.h>

#include "Base/Resources.h"
#include "Base/LoggingService.h"

#include "GlobalSettings.h"
#include "RemoteSimulationDataParser.h"

_NetworkController::_NetworkController()
{
    _serverAddress = GlobalSettings::getInstance().getStringState("settings.server", "alien-project.org");
}

_NetworkController::~_NetworkController()
{
    GlobalSettings::getInstance().setStringState("settings.server", _serverAddress);
}

std::string _NetworkController::getServerAddress() const
{
    return _serverAddress;
}

void _NetworkController::setServerAddress(std::string const& value)
{
    _serverAddress = value;
    logout();
}

std::optional<std::string> _NetworkController::getLoggedInUserName() const
{
    return _loggedInUserName;
}

std::optional<std::string> _NetworkController::getPassword() const
{
    return _password;
}

namespace
{
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
        while(true) {
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

    void logNetworkError(std::string const& serverResponse)
    {
        log(Priority::Important, "network: an error occurred while parsing the server response: " + serverResponse);
    }

    bool parseBoolResult(std::string const& serverResponse)
    {
        try {
            std::stringstream stream(serverResponse);
            boost::property_tree::ptree tree;
            boost::property_tree::read_json(stream, tree);
            auto result = tree.get<bool>("result");
            if (!result) {
                log(Priority::Important, "network: negative response received from server");
            }
            return result;
        }
        catch(...) {
            logNetworkError(serverResponse);
            return false;
        }
    }

}

bool _NetworkController::createUser(std::string const& userName, std::string const& password, std::string const& email)
{
    log(Priority::Important, "network: create user '" + userName + "'");

    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::Params params;
    params.emplace("userName", userName);
    params.emplace("password", password);
    params.emplace("email", email);

    auto result = executeRequest([&] { return client.Post("/alien-server/createuser.php", params); });

    return parseBoolResult(result->body);
}

bool _NetworkController::activateUser(std::string const& userName, std::string const& password, std::string const& confirmationCode)
{
    log(Priority::Important, "network: activate user '" + userName + "'");

    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::Params params;
    params.emplace("userName", userName);
    params.emplace("password", password);
    params.emplace("activationCode", confirmationCode);

    auto result = executeRequest([&] { return client.Post("/alien-server/activateuser.php", params); });

    return parseBoolResult(result->body);
}

bool _NetworkController::login(std::string const& userName, std::string const& password)
{
    log(Priority::Important, "network: login user '" + userName + "'");

    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::Params params;
    params.emplace("userName", userName);
    params.emplace("password", password);

    auto result = executeRequest([&] { return client.Post("/alien-server/login.php", params); });

    auto boolResult = parseBoolResult(result->body);
    if (boolResult) {
        _loggedInUserName = userName;
        _password = password;
    }
    return boolResult;
}

void _NetworkController::logout()
{
    log(Priority::Important, "network: logout");

    _loggedInUserName = std::nullopt;
    _password = std::nullopt;
}

bool _NetworkController::deleteUser()
{
    log(Priority::Important, "network: delete user '" + *_loggedInUserName + "'");

    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::Params params;
    params.emplace("userName", *_loggedInUserName);
    params.emplace("password", *_password);

    auto postResult = executeRequest([&] { return client.Post("/alien-server/deleteuser.php", params); });

    auto result = parseBoolResult(postResult->body);
    if (result) {
        logout();
    }

    return result;
}

bool _NetworkController::resetPassword(std::string const& userName, std::string const& email)
{
    log(Priority::Important, "network: reset password of user '" + userName + "'");

    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::Params params;
    params.emplace("userName", userName);
    params.emplace("email", email);

    auto result = executeRequest([&] { return client.Post("/alien-server/resetpw.php", params); });

    return parseBoolResult(result->body);
}

bool _NetworkController::setNewPassword(std::string const& userName, std::string const& newPassword, std::string const& confirmationCode)
{
    log(Priority::Important, "network: set new password for user '" + userName + "'");

    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::Params params;
    params.emplace("userName", userName);
    params.emplace("newPassword", newPassword);
    params.emplace("activationCode", confirmationCode);

    auto result = executeRequest([&] { return client.Post("/alien-server/setnewpw.php", params); });

    return parseBoolResult(result->body);
}

bool _NetworkController::getRemoteSimulationDataList(std::vector<RemoteSimulationData>& result, bool withRetry) const
{
    log(Priority::Important, "network: get simulation list");

    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    auto postResult = executeRequest([&] { return client.Get("/alien-server/getsimulationinfo.php"); }, withRetry);

    try {
        std::stringstream stream(postResult->body);
        boost::property_tree::ptree tree;
        boost::property_tree::read_json(stream, tree);
        result.clear();
        result = RemoteSimulationDataParser::decode(tree);
        return true;
    } catch (...) {
        logNetworkError(postResult->body);
        return false;
    }
}

bool _NetworkController::getLikedSimulationIdList(std::vector<std::string>& result) const
{
    log(Priority::Important, "network: get liked simulations");

    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::Params params;
    params.emplace("userName", *_loggedInUserName);
    params.emplace("password", *_password);

    auto postResult = executeRequest([&] { return client.Post("/alien-server/getlikedsimulations.php", params); });

    try {
        std::stringstream stream(postResult->body);
        boost::property_tree::ptree tree;
        boost::property_tree::read_json(stream, tree);

        result.clear();
        for (auto const& [key, subTree] : tree) {
            result.emplace_back(subTree.get<std::string>("id"));
        }
        return true;
    } catch (...) {
        logNetworkError(postResult->body);
        return false;
    }
}

bool _NetworkController::getUserLikesForSimulation(std::set<std::string>& result, std::string const& simId)
{
    log(Priority::Important, "network: get user likes for simulation with id=" + simId);

    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::Params params;
    params.emplace("simId", simId);

    auto postResult = executeRequest([&] { return client.Post("/alien-server/getuserlikes.php", params); });

    try {
        std::stringstream stream(postResult->body);
        boost::property_tree::ptree tree;
        boost::property_tree::read_json(stream, tree);

        result.clear();
        for (auto const& [key, subTree] : tree) {
            result.insert(subTree.get<std::string>("userName"));
        }
        return true;
    } catch (...) {
        logNetworkError(postResult->body);
        return false;
    }
}

bool _NetworkController::toggleLikeSimulation(std::string const& simId)
{
    log(Priority::Important, "network: toggle like for simulation with id=" + simId);

    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::Params params;
    params.emplace("userName", *_loggedInUserName);
    params.emplace("password", *_password);
    params.emplace("simId", simId);

    auto result = executeRequest([&] { return client.Post("/alien-server/togglelikesimulation.php", params); });

    return parseBoolResult(result->body);
}

bool _NetworkController::uploadSimulation(
    std::string const& simulationName,
    std::string const& description,
    IntVector2D const& size,
    int particles,
    std::string const& content,
    std::string const& settings,
    std::string const& symbolMap)
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
        {"content", content, "", "application/octet-stream"},
        {"settings", settings, "", ""},
        {"symbolMap", symbolMap, "", ""},
    };

    auto result = executeRequest([&] { return client.Post("/alien-server/uploadsimulation.php", items); });

    return parseBoolResult(result->body);
}

bool _NetworkController::downloadSimulation(std::string& content, std::string& settings, std::string& symbolMap, std::string const& simId)
{
    log(Priority::Important, "network: download simulation with id=" + simId);

    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::Params params;
    params.emplace("id", simId);

    try {
        {
            auto result = executeRequest([&] { return client.Get("/alien-server/downloadcontent.php", params, {}); });
            content = result->body;
        }
        {
            auto result = executeRequest([&] { return client.Get("/alien-server/downloadsettings.php", params, {}); });
            settings = result->body;
        }
        {
            auto result = executeRequest([&] { return client.Get("/alien-server/downloadsymbolmap.php", params, {}); });
            symbolMap = result->body;
        }
        return true;
    } catch (...) {
        log(Priority::Important, "network: an error occurred");
        return false;
    }
}

bool _NetworkController::deleteSimulation(std::string const& simId)
{
    log(Priority::Important, "network: delete simulation with id=" + simId);

    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::Params params;
    params.emplace("userName", *_loggedInUserName);
    params.emplace("password", *_password);
    params.emplace("simId", simId);

    auto result = executeRequest([&] { return client.Post("/alien-server/deletesimulation.php", params); });

    return parseBoolResult(result->body);
}
