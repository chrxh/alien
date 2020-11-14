#include <QUrlQuery>
#include <QHttpMultiPart>

#include "HttpClient.h"
#include "Parser.h"

#include "WebAccessImpl.h"
#include "Task.h"

using namespace std::string_literals;

namespace
{
    auto const HostAddress = "http://localhost/api/backend/"s;

    auto const ApiGetSimulation = "getsimulation"s;
    auto const ApiConnect = "connect"s;
    auto const ApiDisconnect = "disconnect"s;
    auto const ApiGetUnprocessedTasks = "getunprocessedtasks"s;
    auto const ApiSendProcessedTask = "sendprocessedtask"s;
}

WebAccessImpl::WebAccessImpl()
{
    init();
}

void WebAccessImpl::init()
{
    for (auto const& connection : _connections) {
        disconnect(connection);
    }
    _connections.clear();

    delete _http;
    _http = new HttpClient(this);

    _connections.emplace_back(connect(_http, &HttpClient::dataReceived, this, &WebAccessImpl::dataReceived));
    _connections.emplace_back(connect(_http, &HttpClient::error, this, &WebAccess::error));
}

void WebAccessImpl::requestSimulationInfos()
{
    get(ApiGetSimulation, RequestType::SimulationInfo);
}

void WebAccessImpl::requestConnectToSimulation(string const& simulationId, string const& password)
{
    post(ApiConnect, RequestType::Connect, { { "simulationId", simulationId },{ "password", password } });
}

void WebAccessImpl::requestUnprocessedTasks(std::string const & simulationId, string const& token)
{
    post(ApiGetUnprocessedTasks, RequestType::UnprocessedTasks, { { "simulationId", simulationId }, {"token", token} });
}

void WebAccessImpl::sendProcessedTask(string const & simulationId, string const & token, QBuffer* data)
{
    post(ApiSendProcessedTask, RequestType::ProcessedTask, { { "simulationId", simulationId },{ "token", token } }, data);
}

void WebAccessImpl::requestDisconnect(std::string const & simulationId, string const& token)
{
    post(ApiDisconnect, RequestType::Disconnect, {{"simulationId", simulationId}, {"token", token}});
}

void WebAccessImpl::dataReceived(int handler, QByteArray data)
{
    auto requestType = static_cast<RequestType>(handler);
    _requesting.erase(requestType);

    switch (requestType) {
    case RequestType::SimulationInfo : {
        try {
            auto simulationInfos = Parser::parseForSimulationInfos(data);
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
    case RequestType::UnprocessedTasks: {
        auto tasks = Parser::parseForUnprocessedTasks(data);
        Q_EMIT unprocessedTasksReceived(tasks);
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

    _http->get(QUrl(QString::fromStdString(HostAddress + apiMethodName)), static_cast<int>(requestType));
}

void WebAccessImpl::post(string const & apiMethodName, RequestType requestType, std::map<string, string> const& keyValues)
{
    if (_requesting.find(requestType) != _requesting.end()) {
        return;
    }
    _requesting.insert(requestType);

    QUrlQuery params;
    for (auto const& keyValue : keyValues) {
        params.addQueryItem(QString::fromStdString(keyValue.first), QString::fromStdString(keyValue.second));
    }

    _http->postText(
        QUrl(QString::fromStdString(HostAddress + apiMethodName)),
        static_cast<int>(requestType),
        params.query().toUtf8());
}

void WebAccessImpl::post(
    string const & apiMethodName, 
    RequestType requestType, 
    std::map<string, string> const& keyValues, 
    QBuffer* data)
{
    if (_requesting.find(requestType) != _requesting.end()) {
        return;
    }
    _requesting.insert(requestType);


    QHttpMultiPart *multiPart = new QHttpMultiPart(QHttpMultiPart::FormDataType);

    QHttpPart textPart;
    textPart.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant("form-data; name=\"text\""));
    textPart.setBody("my text");

    QHttpPart imagePart;
    imagePart.setHeader(QNetworkRequest::ContentTypeHeader, QVariant("image/png"));
    imagePart.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant("form-data; name=\"image\""));
    imagePart.setBodyDevice(data);

    multiPart->append(textPart);
    multiPart->append(imagePart);

    _http->postBinary(
        QUrl(QString::fromStdString(HostAddress + apiMethodName)),
        static_cast<int>(requestType),
        multiPart);
}
