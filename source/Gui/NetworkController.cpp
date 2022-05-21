#include "NetworkController.h"

#include <boost/property_tree/json_parser.hpp>

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>

#include "GlobalSettings.h"
#include "RemoteSimulationDataParser.h"

_NetworkController::_NetworkController()
{
    _server = GlobalSettings::getInstance().getStringState("settings.server", "alien-project.org");
}

_NetworkController::~_NetworkController()
{
    GlobalSettings::getInstance().setStringState("settings.server", _server);
}

std::string _NetworkController::getServerAddress() const
{
    return _server;
}

bool _NetworkController::isLoggedIn() const
{
    return false;
}

std::vector<RemoteSimulationData> _NetworkController::getRemoteSimulationDataList() const
{
    httplib::SSLClient cli(_server);
    cli.set_ca_cert_path("./resources/ca-bundle.crt");
    cli.enable_server_certificate_verification(true);

    if (auto result = cli.get_openssl_verify_result()) {
        throw std::runtime_error("verify error: " + std::string(X509_verify_cert_error_string(result)));
    }

    if (auto result = cli.Get("/alien-server/getsimulationinfo.php")) {
        std::stringstream stream(result->body);
        boost::property_tree::ptree tree;
        boost::property_tree::read_json(stream, tree);
        return RemoteSimulationDataParser::decode(tree);
    }
    throw std::runtime_error("Error connecting to the server.");
}
