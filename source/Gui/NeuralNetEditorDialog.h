#pragma once

#include <functional>

#include "AlienDialog.h"
#include "Definitions.h"

// Modal dialog that shows the neural net editor in a larger window. The content is drawn by the owning widget, which
// also provides the buttons that close the dialog.
class _NeuralNetEditorDialog : public AlienDialog
{
public:
    static NeuralNetEditorDialog create();

    void open();
    void process(std::function<void()> const& contentFunc);
    using AlienDialog::close;

private:
    _NeuralNetEditorDialog();

    void processIntern() override;

    std::function<void()> _contentFunc;
};
