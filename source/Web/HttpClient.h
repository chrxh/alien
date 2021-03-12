#pragma once

#include <QUrl>
#include <QNetworkAccessManager>
#include "Definitions.h"

class HttpClient : public QObject
{
    Q_OBJECT
public:
    HttpClient(QObject* parent = nullptr);

    void get(QUrl const& url, string const& handler);
    void postText(QUrl const& url, string const& handler, QByteArray const& data);
    void postBinary(QUrl const& url, string const& handler, QHttpMultiPart* data);

    Q_SIGNAL void dataReceived(string handler, QByteArray data);

    Q_SIGNAL void error(string message);

private:
    Q_SLOT void finished(QNetworkReply* reply);

    void retry(QNetworkReply* reply);

    QNetworkAccessManager _networkManager;
    std::unordered_map<QNetworkReply*, string> _handlerByReply;

    using PostData = std::variant<QHttpMultiPart*, QByteArray>;
    std::unordered_map<QNetworkReply*, PostData> _postDataByReply;
};