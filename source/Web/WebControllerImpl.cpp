#include <QJsonDocument>

#include "HttpClient.h"

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
}

void WebControllerImpl::requestSimulationInfos()
{
    auto const apiMethodeName = "getsimulation"s;

    if (_requesting.find(RequestType::SimulationInfo) == _requesting.end()) {
        _http->get(QUrl(QString::fromStdString(host + apiMethodeName)), static_cast<int>(RequestType::SimulationInfo));
    }
}

void WebControllerImpl::requestConnectToSimulation(std::string const & simulationId, std::string const & password)
{
}

void WebControllerImpl::requestTask(std::string const & simulationId)
{
}

void WebControllerImpl::requestDisconnect(std::string const & simulationId)
{
}

void WebControllerImpl::dataReceived(int handler, QByteArray data)
{
    QJsonDocument::fromJson(data);

    auto requestType = static_cast<RequestType>(handler);
    _requesting.erase(requestType);

    if (RequestType::SimulationInfo == requestType) {
        auto dataString = QString::fromStdString(data.toStdString());
    }
}
