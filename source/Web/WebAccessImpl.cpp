#include <QUrlQuery>
#include <QHttpMultiPart>

#include "HttpClient.h"
#include "Parser.h"

#include "WebAccessImpl.h"
#include "Task.h"

using namespace std::string_literals;

namespace
{
//    auto const ServerAddress = "https://alien-project.org/world-explorer/api/"s;
    auto const ServerAddress = "http://localhost/api/"s;

    auto const ApiGetSimulation = "getsimulationinfos"s;
    auto const ApiConnect = "connect"s;
    auto const ApiDisconnect = "disconnect"s;
    auto const ApiGetUnprocessedTasks = "getunprocessedtasks"s;
    auto const ApiSendProcessedTask = "sendprocessedtask"s;
    auto const ApiSendStatistics = "sendstatistics"s;
    auto const ApiSendLastImage = "sendlastimage"s;
    auto const ApiSendBugReport = "sendBugReport"s;
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

void WebAccessImpl::sendProcessedTask(
    string const & simulationId, 
    string const & token, 
    string const& taskId, 
    QBuffer* data)
{
    postImage(
        ApiSendProcessedTask, 
        RequestType::ProcessedTask,
        taskId,
        {{"simulationId", simulationId}, {"token", token}, {"taskId", taskId}}, 
        data);
}

void WebAccessImpl::requestDisconnect(std::string const & simulationId, string const& token)
{
    post(ApiDisconnect, RequestType::Disconnect, {{"simulationId", simulationId}, {"token", token}});
}

void WebAccessImpl::sendStatistics(
    string const & simulationId, 
    string const & token, 
    map<string, string> monitorData)
{
    monitorData.emplace("simulationId", simulationId);
    monitorData.emplace("token", token);
    post(ApiSendStatistics, RequestType::SendStatistics, monitorData);
}

void WebAccessImpl::sendLastImage(string const & simulationId, string const & token, QBuffer * data)
{
    postImage(
        ApiSendLastImage,
        RequestType::LastImage,
        "",
        { { "simulationId", simulationId },{ "token", token } },
        data);
}

void WebAccessImpl::sendBugReport(
    string const& protocol,
    string const& email,
    string const& userMessage)
{
    map<string, string> data;
    data.emplace("protocol", protocol);
    data.emplace("email", email);
    data.emplace("userMessage", userMessage);
    post(ApiSendBugReport, RequestType::SendBugReport, data);
}

void WebAccessImpl::dataReceived(string handler, QByteArray data)
{
    QStringList const handlerParts = QString::fromStdString(handler).split(QChar(':'));
    auto requestType = static_cast<RequestType>(handlerParts.first().toUInt());
    auto id = handlerParts.last().toStdString();

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
    case RequestType::ProcessedTask: {
        Q_EMIT sendProcessedTaskReceived(id);
    }
    break;
    case RequestType::LastImage: {
        Q_EMIT sendLastImageReceived();
    }
    break;
    case RequestType::SendBugReport: {
        Q_EMIT sendBugReportReceived();
    } break;
    }
}

void WebAccessImpl::get(string const & apiMethodName, RequestType requestType)
{
    if (_requesting.find(requestType) != _requesting.end()) {
        return;
    }
    _requesting.insert(requestType);

    auto const handler = std::to_string(static_cast<int>(requestType)) + ":";
    _http->get(QUrl(QString::fromStdString(ServerAddress + apiMethodName)), handler);
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

    auto const handler = std::to_string(static_cast<int>(requestType)) + ":";
    _http->postText(
        QUrl(QString::fromStdString(ServerAddress + apiMethodName)),
        handler,
        params.query().toUtf8());
}

void WebAccessImpl::postImage(
    string const & apiMethodName, 
    RequestType requestType, 
    string const& id,
    std::map<string, string> const& keyValues, 
    QBuffer* data)
{
    if (_requesting.find(requestType) != _requesting.end()) {
        return;
    }
    _requesting.insert(requestType);


    QHttpMultiPart *multiPart = new QHttpMultiPart(QHttpMultiPart::FormDataType);

    for (auto const& keyValue : keyValues) {
        QHttpPart textPart;
        textPart.setHeader(QNetworkRequest::ContentDispositionHeader,
            QVariant("form-data; name=\""+ QString::fromStdString(keyValue.first) + "\""));
        textPart.setBody(QByteArray::fromStdString(keyValue.second));
        multiPart->append(textPart);
    }

    QHttpPart imagePart;
    imagePart.setHeader(QNetworkRequest::ContentTypeHeader, QVariant("image/png"));
    imagePart.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant("form-data; name=\"image\""));
    imagePart.setBodyDevice(data);

    multiPart->append(imagePart);

    auto const handler = std::to_string(static_cast<int>(requestType)) + ":" + id;
    _http->postBinary(
        QUrl(QString::fromStdString(ServerAddress + apiMethodName)),
        handler,
        multiPart);
}
