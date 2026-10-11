#pragma once

#include <Base/MathTypes.h>

#include "Definitions.h"

class _BrowserLoginHintWidget
{
public:
    static BrowserLoginHintWidget create();

    void process(RealVector2D const& pos, RealVector2D const& size);

private:
    _BrowserLoginHintWidget() = default;

    void processCard();
};
