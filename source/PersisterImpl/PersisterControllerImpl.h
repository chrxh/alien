#pragma once

#include <thread>

#include "PersisterInterface/PersisterController.h"
#include "EngineInterface/Definitions.h"

#include "Definitions.h"
#include "PersisterWorker.h"

class _PersisterControllerImpl : public _PersisterController
{
public:
    ~_PersisterControllerImpl() override;

    void init(SimulationController const& simController) override;
    void shutdown() override;
    void restart() override;

    bool isBusy() const override;
    PersisterRequestState getRequestState(PersisterRequestId const& id) const override;
    std::vector<PersisterErrorInfo> fetchAllErrorInfos(SenderId const& senderId) override;
    PersisterErrorInfo fetchError(PersisterRequestId const& id) override;

    PersisterRequestId scheduleSaveSimulationToFile(SenderInfo const& senderInfo, SaveSimulationRequestData const& data) override;
    SavedSimulationResultData fetchSavedSimulationData(PersisterRequestId const& id) override;

    PersisterRequestId scheduleReadSimulationFromFile(SenderInfo const& senderInfo, ReadSimulationRequestData const& data) override;
    ReadSimulationResultData fetchReadSimulationData(PersisterRequestId const& id) override;

    PersisterRequestId scheduleLogin(SenderInfo const& senderInfo, LoginRequestData const& data) override;
    LoginResultData fetchLoginData(PersisterRequestId const& id) override;

    PersisterRequestId scheduleGetNetworkResources(SenderInfo const& senderInfo, GetNetworkResourcesRequestData const& data) override;
    GetNetworkResourcesResultData fetchGetNetworkResourcesData(PersisterRequestId const& id) override;

    PersisterRequestId scheduleDownloadNetworkResource(SenderInfo const& senderInfo, DownloadNetworkResourceRequestData const& data) override;
    DownloadNetworkResourceResultData fetchDownloadNetworkResourcesData(PersisterRequestId const& id) override;

private:
    static auto constexpr MaxWorkerThreads = 4;

    template<typename Request, typename RequestData>
    PersisterRequestId scheduleRequest(SenderInfo const& senderInfo, RequestData const& data);

    template <typename RequestResult, typename ResultData>
    ResultData fetchData(PersisterRequestId const& id);

    PersisterRequestId generateNewJobId();

    PersisterWorker _worker;
    std::thread* _thread[MaxWorkerThreads];
    int _latestJobId = 0;
};


/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

template <typename Request, typename RequestData>
PersisterRequestId _PersisterControllerImpl::scheduleRequest(SenderInfo const& senderInfo, RequestData const& data)
{
    auto requestId = generateNewJobId();
    auto request = std::make_shared<Request>(requestId, senderInfo, data);

    _worker->addRequest(request);

    return requestId;
}

template <typename RequestResult, typename ResultData>
ResultData _PersisterControllerImpl::fetchData(PersisterRequestId const& id)
{
    auto requestResult = std::dynamic_pointer_cast<RequestResult>(_worker->fetchRequestResult(id));
    return requestResult->getData();
}
