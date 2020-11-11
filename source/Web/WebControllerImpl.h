#pragma once

#include "WebController.h"

class WebControllerImpl : public WebController
{
public:
    WebControllerImpl();
    virtual ~WebControllerImpl() = default;

    void requestSimulationInfos() override;
    void requestConnectToSimulation(string const& simulationId, string const& password) override;
    void requestTask(string const& simulationId) override;
    void requestDisconnect(string const& simulationId, string const& token) override;

private:
    Q_SLOT void dataReceived(int handler, QByteArray data);

    enum class RequestType {
        SimulationInfo,
        Connect,
        Disconnect
    };

    void get(string const& apiMethodName, RequestType requestType);
    void post(string const& apiMethodName, RequestType requestType, std::map<string, string> keyValues);

private:

    HttpClient* _http = nullptr;

    set<RequestType> _requesting;
};