#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/Definitions.h>
#include <EngineInterface/SimulationFacade.h>

#include <PersisterInterface/Definitions.h>
#include <PersisterInterface/DeleteNetworkResourceRequestData.h>
#include <PersisterInterface/DownloadNetworkResourceRequestData.h>
#include <PersisterInterface/PersisterFacade.h>
#include <PersisterInterface/PersisterRequestId.h>
#include <PersisterInterface/ReplaceNetworkResourceRequestData.h>
#include <PersisterInterface/UploadNetworkResourceRequestData.h>

#include "Definitions.h"
#include "MainLoopEntity.h"

class NetworkTransferController : public MainLoopEntity<>
{
    MAKE_SINGLETON(NetworkTransferController);

public:
    void onDownload(DownloadNetworkResourceRequestData const& requestData);
    void onUpload(UploadNetworkResourceRequestData const& requestData);
    void onReplace(ReplaceNetworkResourceRequestData const& requestData);
    void onDelete(DeleteNetworkResourceRequestData const& requestData);
    void onEdit(EditNetworkResourceRequestData const& requestData);
    void onMove(MoveNetworkResourceRequestData const& requestData);

private:
    void init() override;
    void process() override;
    void shutdown() override {}

    TaskProcessor _downloadProcessor;
    TaskProcessor _uploadProcessor;
    TaskProcessor _replaceProcessor;
    TaskProcessor _deleteProcessor;
    TaskProcessor _editProcessor;
    TaskProcessor _moveProcessor;
};
