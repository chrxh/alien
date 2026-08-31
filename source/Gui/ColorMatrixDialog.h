#pragma once

#include <string>

#include <EngineInterface/EngineConstants.h>

#include "AlienGui.h"
#include "Definitions.h"
#include "ModalWindow.h"

class ColorMatrixDialog
{
public:
    ColorMatrixDialog();

    void open(std::string const& name);
    void process(AlienGui::BasicInputColorMatrixParameters<bool> const& parameters, bool (&value)[MAX_COLORS][MAX_COLORS]);

private:
    void processContent(AlienGui::BasicInputColorMatrixParameters<bool> const& parameters, bool (&value)[MAX_COLORS][MAX_COLORS]);
    void processToolButtons(AlienGui::BasicInputColorMatrixParameters<bool> const& parameters, bool (&value)[MAX_COLORS][MAX_COLORS]);

    ModalWindow _modalWindow;
};
