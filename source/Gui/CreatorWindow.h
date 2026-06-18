#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/DescEditService.h>

#include "AlienWindow.h"
#include "Definitions.h"

using CreationMode = int;
enum CreationMode_
{
    CreationMode_CreateObject,
    CreationMode_CreateRectangle,
    CreationMode_CreateHexagon,
    CreationMode_CreateDisc,
    CreationMode_Drawing
};

using CreationMaterial = int;
enum CreationMaterial_
{
    CreationMaterial_Solid,
    CreationMaterial_Fluid,
    CreationMaterial_FreeCell,
    CreationMaterial_EnergyParticle
};

class CreatorWindow : public AlienWindow
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(CreatorWindow);

public:
    void onDrawing();
    void finishDrawing();

private:
    CreatorWindow();

    void initIntern() override;
    void shutdownIntern() override;
    void processIntern() override;
    bool isShown() override;

    void createEntity();
    void createRectangle();
    void createHexagon();
    void createDisc();

    Desc convertToEnergyParticles(Desc const& description) const;

    void validateAndCorrect();
    bool isEnergyMaterial() const;

    ObjectTypeDesc getObjectTypeDesc() const;

    RealVector2D getRandomPos() const;

    float _energy = 100.0f;
    float _stiffness = 1.0f;
    bool _static = false;
    float _objectDistance = 1.0f;
    float _glow = 0.0f;
    bool _makeSticky = false;

    // Rectangle data
    int _rectHorizontalObjects = 10;
    int _rectVerticalObjects = 10;

    // Hexagon data
    int _layers = 10;

    // Disc data
    float _outerRadius = 10.0f;
    float _innerRadius = 5.0f;

    // Drawing data
    CreationMaterial _material = CreationMaterial_Solid;
    Desc _drawingDescription;
    DescEditService::Occupancy _drawingOccupancy;
    RealVector2D _lastDrawPos;

    CreationMode _mode = CreationMode_Drawing;
};
