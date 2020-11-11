#include <QUrlQuery>

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

void WebControllerImpl::requestConnectToSimulation(int simulationId, std::string const & password)
{
    auto const apiMethodeName = "connect"s;

    if (_requesting.find(RequestType::Connect) != _requesting.end()) {
        return;
    }

    QUrlQuery params;
    params.addQueryItem("simulationId", QString("%1").arg(simulationId));
    params.addQueryItem("password", QString::fromStdString(password));

    _http->post(
        QUrl(QString::fromStdString(host + apiMethodeName)), 
        static_cast<int>(RequestType::Connect), 
        params.query().toUtf8());

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

    switch (requestType) {
    case RequestType::SimulationInfo : {
        try {
            auto simulationInfos = Parser::parse(data);
            Q_EMIT simulationInfosReceived(simulationInfos);
        }
        catch (std::exception const& exception) {
            Q_EMIT error(exception.what());
        }
    }
    break;
    case RequestType::Connect: {
        auto const token = !data.isEmpty() ? optional<string>(data.toStdString()) : optional<string>();
        Q_EMIT connectToSimulationReceived(token);
    }
    break;
    }
}
