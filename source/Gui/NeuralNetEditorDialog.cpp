#include "NeuralNetEditorDialog.h"

#include <imgui.h>

#include "AlienGui.h"
#include "StyleRepository.h"

namespace
{
    auto const DefaultSize = RealVector2D(900.0f, 640.0f);
    auto constexpr ButtonAreaHeight = 50.0f;
}

NeuralNetEditorDialog _NeuralNetEditorDialog::create()
{
    return NeuralNetEditorDialog(new _NeuralNetEditorDialog());
}

_NeuralNetEditorDialog::_NeuralNetEditorDialog()
    : AlienDialog("Neural network", DefaultSize, true)
{}

void _NeuralNetEditorDialog::open()
{
    // The dialog is not registered in the main loop but processed by the owning widget
    openNested();
}

void _NeuralNetEditorDialog::process(std::function<void()> const& contentFunc)
{
    _contentFunc = contentFunc;
    processNested();
    _contentFunc = nullptr;
}

void _NeuralNetEditorDialog::processIntern()
{
    if (ImGui::BeginChild("NeuralNetEditorDialogContent", ImVec2(0, -scale(ButtonAreaHeight)))) {
        if (_contentFunc) {
            _contentFunc();
        }
    }
    ImGui::EndChild();

    AlienGui::Separator();

    if (AlienGui::Button("Close")) {
        ImGui::CloseCurrentPopup();
        close();
    }
}
