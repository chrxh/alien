#pragma once

#include <EngineInterface/EngineConstants.h>

#include "AlienGui.h"
#include "ColorMatrixDialog.h"

class ColorMatrixWidget
{
public:
    void process(AlienGui::CheckboxColorMatrixParameters const& parameters, bool (&value)[MAX_COLORS][MAX_COLORS]);

private:
    ColorMatrixDialog _dialog;
};
