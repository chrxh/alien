#include "NeuralNetEditorDialog.h"

#include <imgui.h>

namespace
{
    auto const DefaultSize = RealVector2D(700.0f, 500.0f);
}

NeuralNetEditorDialog _NeuralNetEditorDialog::create()
{
    return NeuralNetEditorDialog(new _NeuralNetEditorDialog());
}

_NeuralNetEditorDialog::_NeuralNetEditorDialog()
    : AlienDialog("Neural network", DefaultSize)
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

// The editor brings its own action row, therefore the dialog adds nothing around the content
void _NeuralNetEditorDialog::processIntern()
{
    if (_contentFunc) {
        _contentFunc();
    }
}
