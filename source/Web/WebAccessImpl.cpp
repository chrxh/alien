#include <QUrlQuery>

#include "HttpClient.h"
#include "Parser.h"

#include "WebAccessImpl.h"

using namespace std::string_literals;

namespace
{
    auto const host = "http://localhost/api/"s;

    auto const apiGetSimulation = "getsimulation"s;
    auto const apiConnect = "connect"s;
    auto const apiDisconnect = "disconnect"s;

}

WebAccessImpl::WebAccessImpl()
{
    _http = new HttpClient(this);
    connect(_http, &HttpClient::dataReceived, this, &WebAccessImpl::dataReceived);
    connect(_http, &HttpClient::error, this, &WebAccess::error);
}

void WebAccessImpl::requestSimulationInfos()
{
    get(apiGetSimulation, RequestType::SimulationInfo);
}

void WebAccessImpl::requestConnectToSimulation(string const& simulationId, string const& password)
{
    post(apiConnect, RequestType::Connect, { { "simulationId", simulationId },{ "password", password } });
}

void WebAccessImpl::requestTask(std::string const & simulationId)
{
}

void WebAccessImpl::requestDisconnect(std::string const & simulationId, string const& token)
{
    post(apiDisconnect, RequestType::Disconnect, {{"simulationId", simulationId}, {"token", token}});
}

void WebAccessImpl::dataReceived(int handler, QByteArray data)
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

void WebAccessImpl::get(string const & apiMethodName, RequestType requestType)
{
    if (_requesting.find(requestType) != _requesting.end()) {
        return;
    }
    _requesting.insert(requestType);

    _http->get(QUrl(QString::fromStdString(host + apiMethodName)), static_cast<int>(requestType));
}

void WebAccessImpl::post(string const & apiMethodName, RequestType requestType, std::map<string, string> keyValues)
{
    if (_requesting.find(requestType) != _requesting.end()) {
        return;
    }
    _requesting.insert(requestType);

    QUrlQuery params;
    for (auto const& keyValue : keyValues) {
        params.addQueryItem(QString::fromStdString(keyValue.first), QString::fromStdString(keyValue.second));
    }

    _http->post(
        QUrl(QString::fromStdString(host + apiMethodName)),
        static_cast<int>(requestType),
        params.query().toUtf8());
}
