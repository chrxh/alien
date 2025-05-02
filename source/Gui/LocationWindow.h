#pragma once

#include "LocationWidgets.h"

class LocationWindow
{
public:
    void init(LocationWidgets const& widgets, RealVector2D const& initialPos);
    void process();

    bool isOn() const;

    int getOrderNumber() const;
    void setOrderNumber(int orderNumber);

private:
    LocationWidgets _widgets;
    int _id = 0;
    RealVector2D _initialPos;
    bool _on = false;
};
