#include "NetworkController.h"

#include <boost/property_tree/json_parser.hpp>

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <cpp-httplib/httplib.h>

#include "Base/Resources.h"

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

std::optional<std::string> _NetworkController::getLoggedInUserName() const
{
    return _loggedInUserName;
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

    httplib::Result executeRequest(std::function<httplib::Result()> const& func)
    {
        auto attempt = 0;
        while(true) {
            auto result = func();
            if (result) {
                return result;
            }
            if (++attempt == 5) {
                throw std::runtime_error("Error connecting to the server.");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    bool parseBoolResult(std::string const& result)
    {
        try {
            std::stringstream stream(result);
            boost::property_tree::ptree tree;
            boost::property_tree::read_json(stream, tree);
            return tree.get<bool>("result");
        }
        catch(...) {
            return false;
        }
    }

}

bool _NetworkController::createUser(std::string const& userName, std::string const& password, std::string const& email)
{
    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::Params params;
    params.emplace("userName", userName);
    params.emplace("password", password);
    params.emplace("email", email);

    auto result = executeRequest([&] { return client.Post("/alien-server/createuser.php", params); });

    return parseBoolResult(result->body);
}

bool _NetworkController::activateUser(std::string const& userName, std::string const& password, std::string const& activationCode)
{
    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::Params params;
    params.emplace("userName", userName);
    params.emplace("password", password);
    params.emplace("activationCode", activationCode);

    auto result = executeRequest([&] { return client.Post("/alien-server/activateuser.php", params); });

    return parseBoolResult(result->body);
}

bool _NetworkController::login(std::string const& userName, std::string const& password)
{
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
    _loggedInUserName = std::nullopt;
}

bool _NetworkController::getRemoteSimulationDataList(std::vector<RemoteSimulationData>& result) const
{
    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    auto postResult = executeRequest([&] { return client.Get("/alien-server/getsimulationinfo.php"); });

    try {
        std::stringstream stream(postResult->body);
        boost::property_tree::ptree tree;
        boost::property_tree::read_json(stream, tree);
        result.clear();
        result = RemoteSimulationDataParser::decode(tree);
        return true;
    } catch (...) {
        return false;
    }
}

bool _NetworkController::getLikedSimulationIdList(std::vector<std::string>& result) const
{
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
        return false;
    }
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

bool _NetworkController::downloadSimulation(std::string& content, std::string& settings, std::string& symbolMap, std::string const& id)
{
    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::Params params;
    params.emplace("id", id);

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
        return false;
    }
}

bool _NetworkController::toggleLikeSimulation(std::string const& id)
{
    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::Params params;
    params.emplace("userName", *_loggedInUserName);
    params.emplace("password", *_password);
    params.emplace("simId", id);

    auto result = executeRequest([&] { return client.Post("/alien-server/togglelikesimulation.php", params); });

    return parseBoolResult(result->body);
}
