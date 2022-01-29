#include "OpenPatternDialog.h"

#include <imgui.h>

#include "EngineInterface/SimulationController.h"
#include "EngineInterface/Serializer.h"
#include "ImFileDialog.h"
#include "StatisticsWindow.h"
#include "Viewport.h"
#include "EditorModel.h"

_OpenPatternDialog::_OpenPatternDialog(
    EditorModel const& editorModel,
    SimulationController const& simController,
    Viewport const& viewport)
    : _editorModel(editorModel)
    , _simController(simController)
    , _viewport(viewport)
{}

void _OpenPatternDialog::process()
{
    if (!ifd::FileDialog::Instance().IsDone("OpenPatternDialog")) {
        return;
    }
    if (ifd::FileDialog::Instance().HasResult()) {
        const std::vector<std::filesystem::path>& res = ifd::FileDialog::Instance().GetResults();
        auto firstFilename = res.front();

        Serializer serializer = std::make_shared<_Serializer>();

        ClusteredDataDescription content;
        if (serializer->deserializeContentFromFile(firstFilename.string(), content)) {
            auto center = _viewport->getCenterInWorldPos();
            content.setCenter(center);
            _simController->addAndSelectSimulationData(DataDescription(content));
            _editorModel->update();
        }
    }
    ifd::FileDialog::Instance().Close();
}

void _OpenPatternDialog::show()
{
    ifd::FileDialog::Instance().Open(
        "OpenPatternDialog", "Open pattern", "Pattern file (*.sim){.sim},.*", false);
}
