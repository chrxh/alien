#include <QNetworkRequest>
#include <QNetworkReply>
#include <QHttpMultiPart>

#include "HttpClient.h"

HttpClient::HttpClient(QObject* parent /*= nullptr*/)
    : QObject(parent)
{
    connect(&_networkManager, &QNetworkAccessManager::finished, this, &HttpClient::finished);

    //TODO: QNetworkAccessManager::authenticationRequired
}

void HttpClient::get(QUrl const& url, int handler)
{
    QNetworkRequest request(url);
    request.setHeader(QNetworkRequest::UserAgentHeader, "alien");

    auto reply = _networkManager.get(request);
    _handlerByReply.insert_or_assign(reply, handler);
}

void HttpClient::postText(QUrl const & url, int handler, QByteArray const & data)
{
    QNetworkRequest request(url);
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/x-www-form-urlencoded");
    request.setHeader(QNetworkRequest::UserAgentHeader, "alien");

    auto reply = _networkManager.post(request, data);
    _handlerByReply.insert_or_assign(reply, handler);
}

void HttpClient::postBinary(QUrl const & url, int handler, QHttpMultiPart* data)
{
    QNetworkRequest request(url);
    request.setHeader(QNetworkRequest::UserAgentHeader, "alien");

    auto reply = _networkManager.post(request, data);
    data->setParent(reply);
    _handlerByReply.insert_or_assign(reply, handler);
}

void HttpClient::finished(QNetworkReply * reply)
{
    if (QNetworkReply::NetworkError::NoError != reply->error()) {
        Q_EMIT error("Could not read data from server.");
        return;
    }
    auto data = reply->readAll();

    auto handler = _handlerByReply.at(reply);
    Q_EMIT dataReceived(handler, data);

    reply->deleteLater();
}
