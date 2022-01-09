#include "OpenSelectionDialog.h"

#include <imgui.h>

#include "EngineInterface/SimulationController.h"
#include "EngineInterface/Serializer.h"
#include "ImFileDialog.h"
#include "StatisticsWindow.h"
#include "Viewport.h"
#include "EditorModel.h"

_OpenSelectionDialog::_OpenSelectionDialog(
    EditorModel const& editorModel,
    SimulationController const& simController,
    Viewport const& viewport)
    : _editorModel(editorModel)
    , _simController(simController)
    , _viewport(viewport)
{}

void _OpenSelectionDialog::process()
{
    if (!ifd::FileDialog::Instance().IsDone("OpenSelectionDialog")) {
        return;
    }
    if (ifd::FileDialog::Instance().HasResult()) {
        const std::vector<std::filesystem::path>& res = ifd::FileDialog::Instance().GetResults();
        auto firstFilename = res.front();

        Serializer serializer = std::make_shared<_Serializer>();

        DataDescription content;
        if (serializer->deserializeContentFromFile(firstFilename.string(), content)) {
            auto center = _viewport->getCenterInWorldPos();
            content.setCenter(center);
            _simController->addAndSelectSimulationData(content);
            _editorModel->update();
        }
    }
    ifd::FileDialog::Instance().Close();
}

void _OpenSelectionDialog::show()
{
    ifd::FileDialog::Instance().Open(
        "OpenSelectionDialog", "Open selection", "Selection file (*.sim){.sim},.*", false);
}
