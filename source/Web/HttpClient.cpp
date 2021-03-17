#include <QNetworkRequest>
#include <QNetworkReply>
#include <QHttpMultiPart>

#include "Base/ServiceLocator.h"
#include "Base/LoggingService.h"

#include "HttpClient.h"

HttpClient::HttpClient(QObject* parent /*= nullptr*/)
    : QObject(parent)
{
    connect(&_networkManager, &QNetworkAccessManager::finished, this, &HttpClient::finished);

    //TODO: QNetworkAccessManager::authenticationRequired
}

void HttpClient::get(QUrl const& url, string const& handler)
{
    QNetworkRequest request(url);
    request.setSslConfiguration(QSslConfiguration::defaultConfiguration());
    request.setHeader(QNetworkRequest::UserAgentHeader, "alien");

    auto reply = _networkManager.get(request);
    _handlerByReply.insert_or_assign(reply, handler);
}

void HttpClient::postText(QUrl const & url, string const& handler, QByteArray const & data)
{
    QNetworkRequest request(url);
    request.setSslConfiguration(QSslConfiguration::defaultConfiguration());
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/x-www-form-urlencoded");
    request.setHeader(QNetworkRequest::UserAgentHeader, "alien");

    auto reply = _networkManager.post(request, data);
    _handlerByReply.insert_or_assign(reply, handler);
    _postDataByReply.insert_or_assign(reply, data);
}

void HttpClient::postBinary(QUrl const & url, string const& handler, QHttpMultiPart* data)
{
    QNetworkRequest request(url);
    request.setSslConfiguration(QSslConfiguration::defaultConfiguration());
    request.setHeader(QNetworkRequest::UserAgentHeader, "alien");

    auto reply = _networkManager.post(request, data);
    data->setParent(reply);
    _handlerByReply.insert_or_assign(reply, handler);
    _postDataByReply.insert_or_assign(reply, data);
}

void HttpClient::finished(QNetworkReply * reply)
{
    auto cleanupOnExit = [&]() {
        reply->deleteLater();
        _handlerByReply.erase(reply);
        _postDataByReply.erase(reply);
    };

    auto errorCode = reply->error();
    if (QNetworkReply::NetworkError::NoError != errorCode) {

        auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
        loggingService->logMessage(
            Priority::Important,
            QString("A network error occurred. Error code: %1.").arg(reply->error()).toStdString());

        //        if (QNetworkReply::NetworkError::InternalServerError == errorCode) {
        retry(reply);
        cleanupOnExit();
        return;
        //        }

        auto raw = reply->rawHeaderList();
        Q_EMIT error(QString("Could not read data from server. %1").arg(reply->error()).toStdString());

        reply->deleteLater();
        cleanupOnExit();
        return;
    }
    auto data = reply->readAll();

    auto handler = _handlerByReply.at(reply);
    Q_EMIT dataReceived(handler, data);

    cleanupOnExit();
}

void HttpClient::retry(QNetworkReply* reply)
{
    auto operation = reply->operation();
    if (QNetworkAccessManager::GetOperation == reply->operation()) {
        get(reply->url(), _handlerByReply.at(reply));
    }
    if (QNetworkAccessManager::PostOperation == reply->operation()) {
        auto postData = _postDataByReply.at(reply);
        if (std::holds_alternative<QByteArray>(postData)) {
            postText(reply->url(), _handlerByReply.at(reply), std::get<QByteArray>(postData));
        } else {
            postBinary(reply->url(), _handlerByReply.at(reply), std::get<QHttpMultiPart*>(postData));
        }
    }
}
