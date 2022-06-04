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
            throw std::runtime_error("verify error: " + std::string(X509_verify_cert_error_string(result)));
        }
    }

    void checkResult(httplib::Result const& result)
    {
        if (!result) {
            throw std::runtime_error("Error connecting to the server.");
        }
    }
}

bool _NetworkController::login(std::string const& userName, std::string const& passwordHash)
{
    httplib::SSLClient client(_serverAddress);
    configureClient(client);
    httplib::Params params;
    params.emplace("userName", userName);
    params.emplace("passwordHash", passwordHash);

    auto postResult = client.Post("/alien-server/login.php", params);
    checkResult(postResult);

    std::stringstream stream(postResult->body);
    boost::property_tree::ptree tree;
    boost::property_tree::read_json(stream, tree);
    auto result = tree.get<bool>("result");
    if (result) {
        _loggedInUserName = userName;
        _passwordHash = passwordHash;
    }
    return result;
}

void _NetworkController::logout()
{
    _loggedInUserName = std::nullopt;
}

std::vector<RemoteSimulationData> _NetworkController::getRemoteSimulationDataList() const
{
    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    auto result = client.Get("/alien-server/getsimulationinfo.php");
    checkResult(result);

    std::stringstream stream(result->body);
    boost::property_tree::ptree tree;
    boost::property_tree::read_json(stream, tree);
    return RemoteSimulationDataParser::decode(tree);
}

void _NetworkController::uploadSimulation(
    std::string const& simulationName,
    std::string const& description,
    IntVector2D const& size,
    std::string const& content,
    std::string const& settings,
    std::string const& symbolMap)
{
    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::MultipartFormDataItems items = {
        {"userName", *_loggedInUserName, "", ""},
        {"passwordHash", *_passwordHash, "", ""},
        {"simName", simulationName, "", ""},
        {"description", description, "", ""},
        {"width", std::to_string(size.x), "", ""},
        {"height", std::to_string(size.y), "", ""},
        {"version", Const::ProgramVersion, "", ""},
        {"content", content, "", "application/octet-stream"},
        {"settings", settings, "", ""},
        {"symbolMap", symbolMap, "", ""},
    };

    auto postResult = client.Post("/alien-server/uploadsimulation.php", items);
    checkResult(postResult);
}

void _NetworkController::downloadSimulation(std::string& content, std::string& settings, std::string& symbolMap, std::string const& id)
{
    httplib::SSLClient client(_serverAddress);
    configureClient(client);

    httplib::Params params;
    params.emplace("id", id);

    {
        auto result = client.Get("/alien-server/downloadcontent.php", params, {});
        checkResult(result);
        content = result->body;
    }
    {
        auto result = client.Get("/alien-server/downloadsettings.php", params, {});
        checkResult(result);
        settings = result->body;
    }
    {
        auto result = client.Get("/alien-server/downloadsymbolmap.php", params, {});
        checkResult(result);
        symbolMap = result->body;
    }
}
