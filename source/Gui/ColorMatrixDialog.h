#pragma once

#include <string>

#include <EngineInterface/EngineConstants.h>

#include "AlienGui.h"
#include "Definitions.h"
#include "ModalWindow.h"

// Modal dialog for editing a bool color matrix. The dialog is shared by all color matrix widgets, therefore its owner
// is tracked by the widget id. It is processed by its owner, since it is drawn within the id scope of the widget.
class ColorMatrixDialog
{
public:
    ColorMatrixDialog();

    void open(std::string const& name, unsigned int ownerId);
    void process(AlienGui::BasicInputColorMatrixParameters<bool> const& parameters, bool (&value)[MAX_COLORS][MAX_COLORS], unsigned int ownerId);

private:
    void processContent(AlienGui::BasicInputColorMatrixParameters<bool> const& parameters, bool (&value)[MAX_COLORS][MAX_COLORS]);
    void processToolButtons(AlienGui::BasicInputColorMatrixParameters<bool> const& parameters, bool (&value)[MAX_COLORS][MAX_COLORS]);

    ModalWindow _modalWindow;
    unsigned int _ownerId = 0;
};
