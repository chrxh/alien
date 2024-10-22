#pragma once

#include <thread>

#include "PersisterInterface/PersisterFacade.h"
#include "EngineInterface/Definitions.h"

#include "Definitions.h"
#include "PersisterWorker.h"

class _PersisterFacadeImpl : public _PersisterFacade
{
public:
    ~_PersisterFacadeImpl() override;

    void setup(SimulationFacade const& simulationFacade) override;
    void shutdown() override;
    void restart() override;

    bool isBusy() const override;
    std::optional<PersisterRequestState> getRequestState(PersisterRequestId const& id) const override;
    std::vector<PersisterErrorInfo> fetchAllErrorInfos(SenderId const& senderId) override;
    PersisterErrorInfo fetchError(PersisterRequestId const& id) override;

    PersisterRequestId scheduleSaveSimulationToFile(SenderInfo const& senderInfo, SaveSimulationRequestData const& data) override;
    SaveSimulationResultData fetchSaveSimulationData(PersisterRequestId const& id) override;

    PersisterRequestId scheduleReadSimulationFromFile(SenderInfo const& senderInfo, ReadSimulationRequestData const& data) override;
    ReadSimulationResultData fetchReadSimulationData(PersisterRequestId const& id) override;

    PersisterRequestId scheduleLogin(SenderInfo const& senderInfo, LoginRequestData const& data) override;
    LoginResultData fetchLoginData(PersisterRequestId const& id) override;

    PersisterRequestId scheduleGetNetworkResources(SenderInfo const& senderInfo, GetNetworkResourcesRequestData const& data) override;
    GetNetworkResourcesResultData fetchGetNetworkResourcesData(PersisterRequestId const& id) override;

    PersisterRequestId scheduleDownloadNetworkResource(SenderInfo const& senderInfo, DownloadNetworkResourceRequestData const& data) override;
    DownloadNetworkResourceResultData fetchDownloadNetworkResourcesData(PersisterRequestId const& id) override;

    PersisterRequestId scheduleUploadNetworkResource(SenderInfo const& senderInfo, UploadNetworkResourceRequestData const& data) override;
    UploadNetworkResourceResultData fetchUploadNetworkResourcesData(PersisterRequestId const& id) override;

    PersisterRequestId scheduleReplaceNetworkResource(SenderInfo const& senderInfo, ReplaceNetworkResourceRequestData const& data) override;
    ReplaceNetworkResourceResultData fetchReplaceNetworkResourcesData(PersisterRequestId const& id) override;

    PersisterRequestId scheduleGetUserNamesForEmoji(SenderInfo const& senderInfo, GetUserNamesForEmojiRequestData const& data) override;
    GetUserNamesForEmojiResultData fetchGetUserNamesForEmojiData(PersisterRequestId const& id) override;

    PersisterRequestId scheduleDeleteNetworkResource(SenderInfo const& senderInfo, DeleteNetworkResourceRequestData const& data) override;
    DeleteNetworkResourceResultData fetchDeleteNetworkResourcesData(PersisterRequestId const& id) override;

    PersisterRequestId scheduleEditNetworkResource(SenderInfo const& senderInfo, EditNetworkResourceRequestData const& data) override;
    EditNetworkResourceResultData fetchEditNetworkResourcesData(PersisterRequestId const& id) override;

    PersisterRequestId scheduleMoveNetworkResource(SenderInfo const& senderInfo, MoveNetworkResourceRequestData const& data) override;
    MoveNetworkResourceResultData fetchMoveNetworkResourcesData(PersisterRequestId const& id) override;

    PersisterRequestId scheduleToggleEmojiNetworkResource(SenderInfo const& senderInfo, ToggleEmojiNetworkResourceRequestData const& data) override;
    ToggleEmojiNetworkResourceResultData fetchToggleEmojiNetworkResourcesData(PersisterRequestId const& id) override;

private:
    static auto constexpr MaxWorkerThreads = 4;

    template<typename Request, typename RequestData>
    PersisterRequestId scheduleRequest(SenderInfo const& senderInfo, RequestData const& data);

    template <typename RequestResult, typename ResultData>
    ResultData fetchData(PersisterRequestId const& id);

    PersisterRequestId generateNewRequestId();

    PersisterWorker _worker;
    std::thread* _thread[MaxWorkerThreads];
    int _latestRequestId = 0;
};


/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

template <typename Request, typename RequestData>
PersisterRequestId _PersisterFacadeImpl::scheduleRequest(SenderInfo const& senderInfo, RequestData const& data)
{
    auto requestId = generateNewRequestId();
    auto request = std::make_shared<Request>(requestId, senderInfo, data);

    _worker->addRequest(request);

    return requestId;
}

template <typename RequestResult, typename ResultData>
ResultData _PersisterFacadeImpl::fetchData(PersisterRequestId const& id)
{
    auto requestResult = std::dynamic_pointer_cast<RequestResult>(_worker->fetchRequestResult(id));
    return requestResult->getData();
}
