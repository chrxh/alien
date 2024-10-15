#pragma once

#include "Base/Singleton.h"

#include "EngineInterface/Definitions.h"
#include "PersisterInterface/Definitions.h"
#include "PersisterInterface/PersisterRequestId.h"
#include "PersisterInterface/DownloadNetworkResourceRequestData.h"
#include "PersisterInterface/UploadNetworkResourceRequestData.h"
#include "PersisterInterface/ReplaceNetworkResourceRequestData.h"

#include "Definitions.h"

class NetworkTransferController
{
    MAKE_SINGLETON(NetworkTransferController);

public:
    void init(
        SimulationController const& simController,
        PersisterController const& persisterController,
        TemporalControlWindow const& temporalControlWindow,
        EditorController const& editorController,
        BrowserWindow const& browserWindow);

    void onDownload(DownloadNetworkResourceRequestData const& requestData);
    void onUpload(UploadNetworkResourceRequestData const& requestData);
    void onReplace(ReplaceNetworkResourceRequestData const& requestData);

    void process();

private:
    SimulationController _simController;
    PersisterController _persisterController;
    TemporalControlWindow _temporalControlWindow;
    EditorController _editorController;
    BrowserWindow _browserWindow;

    TaskProcessor _downloadProcessor;
    TaskProcessor _uploadProcessor;
    TaskProcessor _replaceProcessor;
};
