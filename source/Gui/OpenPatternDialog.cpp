#include "OpenPatternDialog.h"

#include <imgui.h>

#include "EngineInterface/SimulationController.h"
#include "EngineInterface/Serializer.h"
#include "ImFileDialog.h"
#include "StatisticsWindow.h"
#include "Viewport.h"
#include "EditorModel.h"
#include "GlobalSettings.h"

_OpenPatternDialog::_OpenPatternDialog(
    EditorModel const& editorModel,
    SimulationController const& simController,
    Viewport const& viewport)
    : _editorModel(editorModel)
    , _simController(simController)
    , _viewport(viewport)
{
    auto path = std::filesystem::current_path();
    if (path.has_parent_path()) {
        path = path.parent_path();
    }
    _startingPath = GlobalSettings::getInstance().getStringState("dialogs.open pattern.starting path", path.string());
}

_OpenPatternDialog::~_OpenPatternDialog()
{
    GlobalSettings::getInstance().setStringState("dialogs.open pattern.starting path", _startingPath);
}

void _OpenPatternDialog::process()
{
    if (!ifd::FileDialog::Instance().IsDone("OpenPatternDialog")) {
        return;
    }
    if (ifd::FileDialog::Instance().HasResult()) {
        auto firstFilename = ifd::FileDialog::Instance().GetResult();
        auto firstFilenameCopy = firstFilename;
        _startingPath = firstFilenameCopy.remove_filename().string();

        ClusteredDataDescription content;
        if (Serializer::deserializeContentFromFile(firstFilename.string(), content)) {
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
    ifd::FileDialog::Instance().Open("OpenPatternDialog", "Open pattern", "Pattern file (*.sim){.sim},.*", false, _startingPath);
}
