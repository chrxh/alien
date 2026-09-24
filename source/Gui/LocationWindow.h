#pragma once

#include <EngineInterface/SimulationParametersTypes.h>

#include "LocationWidget.h"

class LocationWindow
{
public:
    void init(LocationWidget const& widgets, RealVector2D const& initialPos);
    void process();

    bool isOn() const;

private:
    LocationWidget _widget;
    int _locationId = 0;
    LocationType _locationType = LocationType::Base;
    int _id = 0;
    RealVector2D _initialPos;
    bool _on = false;
};
