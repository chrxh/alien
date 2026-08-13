#include "SavePictureDialog.h"

#include <algorithm>
#include <filesystem>

#include <imgui.h>

#include <Base/AlienExceptions.h>
#include <Base/GlobalSettings.h>

#include "AlienGui.h"
#include "DelayedExecutionController.h"
#include "GenericFileDialog.h"
#include "GenericMessageDialog.h"
#include "OverlayController.h"
#include "SimulationView.h"
#include "Viewport.h"

namespace
{
    auto constexpr TextWidth = 130.0f;
}

SavePictureDialog::SavePictureDialog()
    : AlienDialog("Save picture", {450.0f, 180.0f})
{}

void SavePictureDialog::initIntern()
{
    _width = GlobalSettings::get().getValue("dialogs.save picture.width", _width);
    _height = GlobalSettings::get().getValue("dialogs.save picture.height", _height);
    _referencePath = GlobalSettings::get().getValue("dialogs.save picture.reference path", _referencePath);
}

void SavePictureDialog::shutdownIntern()
{
    GlobalSettings::get().setValue("dialogs.save picture.width", _width);
    GlobalSettings::get().setValue("dialogs.save picture.height", _height);
    GlobalSettings::get().setValue("dialogs.save picture.reference path", _referencePath);
}

void SavePictureDialog::openIntern()
{
    if (_width <= 0 || _height <= 0) {
        auto viewSize = Viewport::get().getViewSize();
        _width = viewSize.x;
        _height = viewSize.y;
    }
}

void SavePictureDialog::processIntern()
{
    AlienGui::InputInt(AlienGui::InputIntParameters().name("Width").textWidth(TextWidth), _width);
    AlienGui::InputInt(AlienGui::InputIntParameters().name("Height").textWidth(TextWidth), _height);
    if (AlienGui::Button("Adopt view size")) {
        auto viewSize = Viewport::get().getViewSize();
        _width = viewSize.x;
        _height = viewSize.y;
    }
    _width = std::max(1, _width);
    _height = std::max(1, _height);

    AlienGui::Separator();

    if (AlienGui::Button("OK")) {
        close();
        delayedExecution([this] { onSavePicture(); });
    }
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienGui::Button("Cancel")) {
        close();
    }
}

void SavePictureDialog::onSavePicture()
{
    GenericFileDialog::get().showSaveFileDialog("Save picture", "Picture (*.png){.png},.*", _referencePath, [this](std::filesystem::path const& path) {
        auto referencePath = path;
        _referencePath = referencePath.remove_filename().string();

        auto filename = path;
        if (filename.extension().empty()) {
            filename.replace_extension(".png");
        }
        try {
            SimulationView::get().savePicture(filename, {_width, _height});
            printOverlayMessage(filename.filename().string());
        } catch (AlienException const& exception) {
            GenericMessageDialog::get().information("Save picture", std::string("The picture could not be saved.\n\n") + exception.what());
        }
    });
}
