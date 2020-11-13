#pragma once

#include "WebAccess.h"

class WebAccessImpl : public WebAccess
{
public:
    WebAccessImpl();
    virtual ~WebAccessImpl() = default;

    void requestSimulationInfos() override;
    void requestConnectToSimulation(string const& simulationId, string const& password) override;
    void requestUnprocessedTasks(string const& simulationId, string const& token) override;
    void requestDisconnect(string const& simulationId, string const& token) override;

private:
    Q_SLOT void dataReceived(int handler, QByteArray data);

    enum class RequestType {
        SimulationInfo,
        Connect,
        Disconnect,
        UnprocessedTasks
    };

    void get(string const& apiMethodName, RequestType requestType);
    void post(string const& apiMethodName, RequestType requestType, std::map<string, string> keyValues);

private:

    HttpClient* _http = nullptr;

    set<RequestType> _requesting;
};