#include "ColorMatrixWidget.h"

void ColorMatrixWidget::process(AlienGui::CheckboxColorMatrixParameters const& parameters, bool (&value)[MAX_COLORS][MAX_COLORS])
{
    AlienGui::CheckboxColorMatrix(parameters, value, _dialog);
}
