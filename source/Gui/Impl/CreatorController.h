#pragma once

#include <Base/Singleton.h>

#include <Data/CreatorService.h>
#include <Data/DescEditService.h>
#include <Data/Descs.h>

#include "Definitions.h"
#include "EditInteractionController.h"
#include "EditorModel.h"
#include "MainLoopEntity.h"

struct CreatorParameters
{
    CreationMaterial material = CreationMaterial_Solid;
    float energy = 100.0f;
    float stiffness = 1.0f;
    float glow = 0.0f;
    bool isStatic = false;
    bool sticky = false;
    float objectDistance = 1.0f;
    float pencilWidth = 3.0f;

    int rectHorizontalObjects = 10;
    int rectVerticalObjects = 10;
    int layers = 10;
    float outerRadius = 10.0f;
    float innerRadius = 5.0f;
};

class CreatorController
    : public MainLoopEntity
    , public EditInteractionController
{
    MAKE_SINGLETON(CreatorController);

public:
    CreatorParameters const& getParameters() const;
    void setParameters(CreatorParameters const& parameters);

    bool isFinishingPointsPossible() const;
    void onFinishPoints();
    bool hasPoints() const;
    void onAbortPoints();

    void onLeftMouseButtonPressed(RealVector2D const& viewPos) override;
    void onLeftMouseButtonHold(RealVector2D const& viewPos, RealVector2D const& prevViewPos) override;
    void onLeftMouseButtonReleased(RealVector2D const& viewPos, RealVector2D const& prevViewPos) override;
    void onRightMouseButtonPressed(RealVector2D const& viewPos) override;

    bool isCrosshairCursor() const override;
    void drawCursor(ImDrawList* drawList, ImVec2 const& mousePos) const override;

private:
    void init() override;
    void process() override;
    void shutdown() override;

    void place(RealVector2D const& worldPos);
    void draw(RealVector2D const& worldPos);
    void finishDrawing();

    void processPlacementPreview();
    void processPointPreview(std::vector<RealVector2D> const& path, bool closed) const;
    void processControlPolygonPreview() const;

    ContentDesc createShape(RealVector2D const& center) const;
    void addToSimulation(ContentDesc&& content) const;

    void validateAndCorrect();
    bool isEnergyMaterial() const;
    int getMinNumPoints() const;

    CreatorService::ObjectProperties getObjectProperties() const;

    CreatorParameters _parameters;

    ContentDesc _drawingDescription;
    DescEditService::Occupancy _drawingOccupancy;
    RealVector2D _lastDrawPos;

    std::vector<RealVector2D> _points;
    EditTool _lastTool = EditTool_Select;

    struct PreviewKey
    {
        EditTool tool = EditTool_Select;
        CreationMaterial material = CreationMaterial_Solid;
        int rectHorizontalObjects = 0;
        int rectVerticalObjects = 0;
        int layers = 0;
        float outerRadius = 0;
        float innerRadius = 0;
        float objectDistance = 0;

        bool operator==(PreviewKey const& other) const = default;
    };
    std::optional<PreviewKey> _previewKey;
    std::vector<RealVector2D> _previewPositions;
};
