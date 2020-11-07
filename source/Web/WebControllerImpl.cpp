#include "HttpClient.h"
#include "Parser.h"

#include "WebControllerImpl.h"

using namespace std::string_literals;

namespace
{
    auto const host = "http://localhost/api/"s;
}

WebControllerImpl::WebControllerImpl()
{
    _http = new HttpClient(this);
    connect(_http, &HttpClient::dataReceived, this, &WebControllerImpl::dataReceived);
    connect(_http, &HttpClient::error, this, &WebController::error);
}

void WebControllerImpl::requestSimulationInfos()
{
    auto const apiMethodeName = "getsimulation"s;

    if (_requesting.find(RequestType::SimulationInfo) != _requesting.end()) {
        return;
    }
    _http->get(QUrl(QString::fromStdString(host + apiMethodeName)), static_cast<int>(RequestType::SimulationInfo));
}

void WebControllerImpl::requestConnectToSimulation(std::string const & simulationId, std::string const & password)
{
    auto const apiMethodeName = "connect"s;

    if (_requesting.find(RequestType::Connect) != _requesting.end()) {
        return;
    }
    _http->post(QUrl(QString::fromStdString(host + apiMethodeName)), static_cast<int>(RequestType::Connect), QByteArray::fromStdString(password));

}

void WebControllerImpl::requestTask(std::string const & simulationId)
{
}

void WebControllerImpl::requestDisconnect(std::string const & simulationId)
{
}

void WebControllerImpl::dataReceived(int handler, QByteArray data)
{
    auto requestType = static_cast<RequestType>(handler);
    _requesting.erase(requestType);

    if (RequestType::SimulationInfo == requestType) {
        try {
            auto simulationInfos = Parser::parse(data);
            Q_EMIT simulationInfosReceived(simulationInfos);
        }
        catch (std::exception const& exception) {
            Q_EMIT error(exception.what());
        }
    }
}
