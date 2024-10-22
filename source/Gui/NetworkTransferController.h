#pragma once

#include "Base/Singleton.h"

#include "EngineInterface/Definitions.h"
#include "PersisterInterface/Definitions.h"
#include "PersisterInterface/PersisterRequestId.h"
#include "PersisterInterface/DownloadNetworkResourceRequestData.h"
#include "PersisterInterface/UploadNetworkResourceRequestData.h"
#include "PersisterInterface/ReplaceNetworkResourceRequestData.h"
#include "PersisterInterface/DeleteNetworkResourceRequestData.h"

#include "Definitions.h"
#include "MainLoopEntity.h"

class NetworkTransferController : public MainLoopEntity
{
    MAKE_SINGLETON(NetworkTransferController);

public:
    void init(SimulationFacade const& simulationFacade, PersisterFacade const& persisterFacade);

    void onDownload(DownloadNetworkResourceRequestData const& requestData);
    void onUpload(UploadNetworkResourceRequestData const& requestData);
    void onReplace(ReplaceNetworkResourceRequestData const& requestData);
    void onDelete(DeleteNetworkResourceRequestData const& requestData);

private:
    void process() override;
    void shutdown() override {}

    SimulationFacade _simulationFacade;
    PersisterFacade _persisterFacade;

    TaskProcessor _downloadProcessor;
    TaskProcessor _uploadProcessor;
    TaskProcessor _replaceProcessor;
    TaskProcessor _deleteProcessor;
};
