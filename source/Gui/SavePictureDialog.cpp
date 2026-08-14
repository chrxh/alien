#include "SavePictureDialog.h"

#include <algorithm>
#include <filesystem>

#include <imgui.h>

#include "Base/GlobalSettings.h"

#include "AlienImGui.h"
#include "DelayedExecutionController.h"
#include "GenericFileDialog.h"
#include "GenericMessageDialog.h"
#include "SimulationView.h"
#include "StyleRepository.h"
#include "Viewport.h"

namespace
{
    auto const ContentTextInputWidth = 130.0f;
}

SavePictureDialog::SavePictureDialog()
    : AlienDialog("Save picture")
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
    AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Width").textWidth(ContentTextInputWidth), _width);
    AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Height").textWidth(ContentTextInputWidth), _height);
    if (AlienImGui::Button("Adopt view size")) {
        auto viewSize = Viewport::get().getViewSize();
        _width = viewSize.x;
        _height = viewSize.y;
    }
    _width = std::max(1, _width);
    _height = std::max(1, _height);

    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
    AlienImGui::Separator();

    if (AlienImGui::Button("OK")) {
        ImGui::CloseCurrentPopup();
        close();
        delayedExecution([this] { onSavePicture(); });
    }
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienImGui::Button("Cancel")) {
        ImGui::CloseCurrentPopup();
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
        } catch (std::exception const& exception) {
            GenericMessageDialog::get().information("Save picture", std::string("The picture could not be saved.\n\n") + exception.what());
        }
    });
}
