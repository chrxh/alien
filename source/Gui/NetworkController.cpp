#include "NetworkController.h"

#include <boost/property_tree/json_parser.hpp>

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>

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
    }
    return result;
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
