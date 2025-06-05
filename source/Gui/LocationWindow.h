#pragma once

#include "LocationWidget.h"

class LocationWindow
{
public:
    void init(LocationWidget const& widgets, RealVector2D const& initialPos);
    void process();

    bool isOn() const;

    int getOrderNumber() const;
    void setOrderNumber(int orderNumber);

private:
    LocationWidget _widget;
    int _id = 0;
    RealVector2D _initialPos;
    bool _on = false;
};
