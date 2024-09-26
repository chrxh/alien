#include "AutosaveWindow.h"

#include "Base/GlobalSettings.h"

#include "AlienImGui.h"

namespace
{
    auto constexpr RightColumnWidth = 150.0f;
}

_AutosaveWindow::_AutosaveWindow(SimulationController const& simController)
    : _AlienWindow("Auto save", "windows.auto save", false)
    , _simController(simController)
{
    _origSaveMode = GlobalSettings::getInstance().getInt("windows.auto save.save mode", _origSaveMode);
    _saveMode = _origSaveMode;
    _numberOfFiles = GlobalSettings::getInstance().getInt("windows.auto save.number of files", _origNumberOfFiles);
}

_AutosaveWindow::~_AutosaveWindow()
{
    GlobalSettings::getInstance().setInt("windows.auto save.save mode", _saveMode);
    GlobalSettings::getInstance().setInt("windows.auto save.number of files", _numberOfFiles);
}

void _AutosaveWindow::processIntern()
{
    processHeader();
    processTable();

    validationAndCorrection();
}

void _AutosaveWindow::processHeader()
{
    AlienImGui::Combo(
        AlienImGui::ComboParameters()
            .name("Mode")
            .values({"Circular save files", "Unlimited save files"})
            .textWidth(RightColumnWidth)
            .defaultValue(_origSaveMode),
        _saveMode);
    if (_saveMode == SaveMode_Circular) {
        AlienImGui::InputInt(
            AlienImGui::InputIntParameters().name("Number of files").textWidth(RightColumnWidth).defaultValue(_origNumberOfFiles), _numberOfFiles);
    }
}

void _AutosaveWindow::processTable()
{
}

void _AutosaveWindow::validationAndCorrection()
{
    _numberOfFiles = std::max(0, _numberOfFiles);
}
