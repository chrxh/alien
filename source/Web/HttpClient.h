#pragma once

#include <QUrl>
#include <QNetworkAccessManager>
#include "Definitions.h"

class HttpClient : public QObject
{
    Q_OBJECT
public:
    HttpClient(QObject* parent = nullptr);

    void get(QUrl const& url, int handler);
    void postText(QUrl const& url, int handler, QByteArray const& data);
    void postBinary(QUrl const& url, int handler, QHttpMultiPart* data);

    Q_SIGNAL void dataReceived(int handler, QByteArray data);

    Q_SIGNAL void error(string message);

private:
    Q_SLOT void finished(QNetworkReply* reply);

    QNetworkAccessManager _networkManager;
    std::unordered_map<QNetworkReply*, int> _handlerByReply;
};