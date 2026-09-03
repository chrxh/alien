#pragma once

#include <functional>

#include <EngineInterface/Colors.h>
#include <EngineInterface/EngineConstants.h>

#include "AlienGui.h"
#include "Definitions.h"
#include "ModalWindow.h"

template <typename T>
class ColorMatrixDialog
{
public:
    ColorMatrixDialog();

    void open(
        AlienGui::ExpandedColorMatrixParameters<T> const& parameters,
        T const (&value)[MAX_COLORS][MAX_COLORS],
        std::function<void(ColorMatrix<T> const&)> const& onAdoptCallback);
    void process();

private:
    void processContent();
    void onAdopt();

    ModalWindow _modalWindow;
    AlienGui::ExpandedColorMatrixParameters<T> _parameters;
    ColorMatrix<T> _matrix;
    std::function<void(ColorMatrix<T> const&)> _onAdoptCallback;
};
