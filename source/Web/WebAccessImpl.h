#pragma once

#include "WebAccess.h"

class WebAccessImpl : public WebAccess
{
public:
    WebAccessImpl();
    virtual ~WebAccessImpl() = default;

    void init() override;

    void requestCurrentVersion() override;
    void requestSimulationInfos() override;
    void requestConnectToSimulation(string const& simulationId, string const& password) override;
    void requestUnprocessedTasks(string const& simulationId, string const& token) override;
    void sendProcessedTask(string const& simulationId, string const& token, string const& taskId, QBuffer* data) override;
    void requestDisconnect(string const& simulationId, string const& token) override;
    void sendStatistics(string const& simulationId, string const& token, map<string, string> monitorData) override;
    void sendLastImage(string const& simulationId, string const& token, QBuffer* data) override;
    void sendBugReport(string const& protocol, string const& email, string const& userMessage) override;

private:
    Q_SLOT void dataReceived(string handler, QByteArray data);

    enum class RequestType {
        CurrentVersion,
        SimulationInfo,
        Connect,
        Disconnect,
        UnprocessedTasks,
        ProcessedTask,
        SendStatistics,
        LastImage,
        SendBugReport
    };

    void get(string const& apiMethodName, RequestType requestType, bool omitErrorResponse = false);
    void post(string const& apiMethodName, RequestType requestType, std::map<string, string> const& keyValues);
    void postImage(string const& apiMethodName, RequestType requestType, string const& id, 
        std::map<string, string> const& keyValues, QBuffer* data);

private:

    HttpClient* _http = nullptr;

    set<RequestType> _requesting;
    std::vector<QMetaObject::Connection> _connections;
};