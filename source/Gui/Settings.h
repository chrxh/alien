#pragma once

#include <QColor>
#include <QFont>
#include <QPalette>

#include "Definitions.h"

namespace Const
{
    auto const ZoomLevelForAutomaticEditorSwitch = 16.0;
    auto const ZoomLevelForAutomaticVectorViewSwitch = 4.0;
    auto const MinZoomLevelForEditor = 4.0;

    //startup
    const QColor StartupTextColor(0x89, 0x94, 0xc4);
    const QColor StartupNewVersionTextColor(0xff, 0x94, 0xc4);

    //visual viewport
	const QColor BackgroundColor(0x00, 0x00, 0x00);
	const QColor UniverseColor(0x00, 0x00, 0x1b);
	const QColor CellColor(0x6F, 0x90, 0xFF, 0xA0);
	const QColor ClusterPenFocusColor(0xFF, 0xFF, 0xFF, 0xFF);
	const QColor LineActiveColor(0xFF, 0xFF, 0xFF, 0xB0);
	const QColor LineInactiveColor(0xB0, 0xB0, 0xB0, 0xB0);
	const QColor TokenColor(0xB0, 0xB0, 0xFF, 0xFF);
	const QColor TokenFocusColor(0xE0, 0xE0, 0xFF, 0xFF);
	const QColor EnergyColor(0xB0, 0x10, 0x0, 0xFF);
	const QColor EnergyPenFocusColor(0xFF, 0xFF, 0xFF, 0xFF);
	const QColor MarkerColor(0x0, 0xC0, 0xFF, 0x30);

	//data editor
	const QColor HexEditColor1(0x80, 0xA0, 0xFF);//(0x30,0x30,0xBB);
	const QColor HexEditColor2(0x80, 0xA0, 0xFF);//(0x30,0x30,0x99);
	const QColor CellEditCaptionColor1(0xD0, 0xD0, 0xD0);
	const QColor CellEditTextColor1(0xD0, 0xD0, 0xD0);
	const QColor CellEditTextColor2(0xB0, 0xB0, 0xB0);
	const QColor CellEditDataColor1(0x80, 0xA0, 0xFF);
	const QColor CellEditDataColor2(0x60, 0x80, 0xC0);
	const QColor CellEditCursorColor(0x6F, 0x90, 0xFF, 0x90);
	const QColor CellEditMetadataColor(0xA0, 0xFF, 0x80);
	const QColor CellEditMetadataCursorColor(0xA0, 0xFF, 0x80, 0x90);
	const QString ResourceInfoOnIcon("://Icons/info_on.png");
	const QString ResourceInfoOffIcon("://Icons/info_off.png");

	const QString ButtonStyleSheet = "background-color: #202020; font-family: Courier New; font-weight: bold; font-size: 12px";
	const QString TableStyleSheet = "background-color: #000000; color: #EEEEEE; gridline-color: #303030; selection-color: #EEEEEE; selection-background-color: #202020; font-family: Courier New; font-weight: bold; font-size: 12px;";
	const QString ScrollbarStyleSheet = "background-color: #303030; color: #B0B0B0; gridline-color: #303030;";
    const QString ToolbarStyleSheet = "background-color: #151540;";
    const QString InfobarStyleSheet = "background-color: #151540; color: #FFF;";
    const QColor ButtonTextColor(0xC2, 0xC2, 0xC2);
	const QColor ButtonTextHighlightColor(0x90, 0x90, 0xFF);
	const QString StandardFont = "Courier New";

	//setting keys and default values
	const std::string MainViewFullScreenKey = "mainView/fullScreen";
	const bool MainViewFullScreenDefault = true;

    const std::string GettingStartedWindowKey = "mainView/gettingStartedWindow";
    const bool GettingStartedWindowKeyDefault = true;

    const std::string ModelComputationTypeKey = "newSim/modelComputationType";
    const ModelComputationType ModelComputationTypeDefault = ModelComputationType::Gpu;

    const std::string GpuUniverseSizeXKey = "newSim/gpu/universeSize/x";
    const int GpuUniverseSizeXDefault = 4000;
    const std::string GpuUniverseSizeYKey = "newSim/gpu/universeSize/y";
    const int GpuUniverseSizeYDefault = 2000;
    const std::string GpuNumBlocksKey = "newSim/gpu/numBlocks";
    const int GpuNumBlocksDefault = 64*8;
    const std::string GpuNumThreadsPerBlockKey = "newSim/gpu/numThreadsPerBlock";
    const int GpuNumThreadsPerBlockDefault = 16;
    const std::string GpuMaxClustersKey = "newSim/gpu/maxClusters";
    const int GpuMaxClustersDefault = 500000;
    const std::string GpuMaxCellsKey = "newSim/gpu/maxCells";
    const int GpuMaxCellsDefault = 2000000;
    const std::string GpuMaxTokensKey = "newSim/gpu/maxTokens";
    const int GpuMaxTokensDefault = 500000;
    const std::string GpuMaxParticlesKey = "newSim/gpu/maxParticles";
    const int GpuMaxParticlesDefault = 2000000;
    const std::string GpuDynamicMemorySizeKey = "newSim/gpu/dynamicMemorySize";
    const int GpuDynamicMemorySizeDefault = 100000000;
    const std::string GpuMetadataDynamicMemorySizeKey = "newSim/gpu/metadataDynamicMemorySize";
    const int GpuMetadataDynamicMemoryDefault = 50000000;

    const std::string InitialEnergyKey = "newSim/initialEnergy";
        const double InitialEnergyDefault = 0.0;

	const std::string GridMulChangeVelXKey = "gridMul/changeVel/x";
	const bool GridMulChangeVelXDefault = false;
	const std::string GridMulChangeVelYKey = "gridMul/changeVel/y";
	const bool GridMulChangeVelYDefault = false;
	const std::string GridMulChangeAngleKey = "gridMul/changeAngle";
	const bool GridMulChangeAngleDefault = false;
	const std::string GridMulChangeAngVelKey = "gridMul/changeAngVel";
	const bool GridMulChangeAngVelDefault = false;

	const std::string GridMulInitialVelXKey = "gridMul/initialVel/x";
	const double GridMulInitialVelXDefault = 0.0;
	const std::string GridMulInitialVelYKey = "gridMul/initialVel/y";
	const double GridMulInitialVelYDefault = 0.0;
	const std::string GridMulInitialAngleKey = "gridMul/initialAngle";
	const double GridMulInitialAngleDefault = 0.0;
	const std::string GridMulInitialAngVelKey = "gridMul/initialAngVel";
	const double GridMulInitialAngVelDefault = 0.0;

	const std::string GridMulHorNumberKey = "gridMul/horNumber";
	const int GridMulHorNumberDefault = 4;
	const std::string GridMulHorIntervalKey = "gridMul/horInterval";
	const double GridMulHorIntervalDefault = 50.0;
	const std::string GridMulHorVelXIncKey = "gridMul/horVelInc/x";
	const double GridMulHorVelXIncDefault = 0.0;
	const std::string GridMulHorVelYIncKey = "gridMul/horVelInc/y";
	const double GridMulHorVelYIncDefault = 0.0;
	const std::string GridMulHorAngleIncKey = "gridMul/horAngleInc";
	const double GridMulHorAngleIncDefault = 0.0;
	const std::string GridMulHorAngVelIncKey = "gridMul/horAngVelInc";
	const double GridMulHorAngVelIncDefault = 0.0;

	const std::string GridMulVerNumberKey = "gridMul/verNumber";
	const int GridMulVerNumberDefault = 4;
	const std::string GridMulVerIntervalKey = "gridMul/verInterval";
	const double GridMulVerIntervalDefault = 50.0;
	const std::string GridMulVerVelXIncKey = "gridMul/verVelInc/x";
	const double GridMulVerVelXIncDefault = 0.0;
	const std::string GridMulVerVelYIncKey = "gridMul/verVelInc/y";
	const double GridMulVerVelYIncDefault = 0.0;
	const std::string GridMulVerAngleIncKey = "gridMul/verAngleInc";
	const double GridMulVerAngleIncDefault = 0.0;
	const std::string GridMulVerAngVelIncKey = "gridMul/verAngVelInc";
	const double GridMulVerAngVelIncDefault = 0.0;

	const std::string RandomMulChangeAngleKey = "randomMul/changeAngle";
	const bool RandomMulChangeAngleDefault = false;
	const std::string RandomMulChangeVelXKey = "randomMul/changeVel/x";
	const bool RandomMulChangeVelXDefault = false;
	const std::string RandomMulChangeVelYKey = "randomMul/changeVel/y";
	const bool RandomMulChangeVelYDefault = false;
	const std::string RandomMulChangeAngVelKey = "randomMul/changeAngVel";
	const bool RandomMulChangeAngVelDefault = false;

	const std::string RandomMulCopiesKey = "randomMul/copies";
	const int RandomMulCopiesDefault = 20;
	const std::string RandomMulMinAngleKey = "randomMul/minAngle";
	const double RandomMulMinAngleDefault = 0.0;
	const std::string RandomMulMaxAngleKey = "randomMul/maxAngle";
	const double RandomMulMaxAngleDefault = 0.0;
	const std::string RandomMulMinVelXKey = "randomMul/minVel/x";
	const double RandomMulMinVelXDefault = 0.0;
	const std::string RandomMulMaxVelXKey = "randomMul/maxVel/x";
	const double RandomMulMaxVelXDefault = 0.0;
	const std::string RandomMulMinVelYKey = "randomMul/minVel/y";
	const double RandomMulMinVelYDefault = 0.0;
	const std::string RandomMulMaxVelYKey = "randomMul/maxVel/y";
	const double RandomMulMaxVelYDefault = 0.0;
	const std::string RandomMulMinAngVelKey = "randomMul/minAngVel";
	const double RandomMulMinAngVelDefault = 0.0;
	const std::string RandomMulMaxAngVelKey = "randomMul/maxAngVel";
	const double RandomMulMaxAngVelDefault = 0.0;

	const std::string NewHexagonLayersKey = "newHaxagon/layers";
	const int NewHexagonLayersDefault = 10;
	const std::string NewHexagonDistanceKey = "newHaxagon/distance";
	const double NewHexagonDistanceDefault = 1.0;
	const std::string NewHexagonCellEnergyKey = "newHaxagon/cellEnergy";
	const double NewHexagonCellEnergyDefault = 100.0;
    const std::string NewHexagonColorCodeKey = "newHaxagon/colorCode";
    const int NewHexagonColorCodeDefault = 0;

	const std::string NewCircleOuterRadiusKey = "newCircle/outerRadius";
    const int NewCircleOuterRadiusDefault = 20;
    const std::string NewCircleInnerRadiusKey = "newCircle/innerRadius";
    const int NewCircleInnerRadiusDefault = 15;
    const std::string NewCircleDistanceKey = "newCircle/distance";
    const double NewCircleDistanceDefault = 1.0;
    const std::string NewCircleCellEnergyKey = "newCircle/cellEnergy";
    const double NewCircleCellEnergyDefault = 100.0;
    const std::string NewCircleColorCodeKey = "newCircle/colorCode";
    const int NewCircleColorCodeDefault = 0;

	const std::string NewParticlesTotalEnergyKey = "newParticles/totalEnergy";
	const double NewParticlesTotalEnergyDefault = 1000000.0;
	const std::string NewParticlesMaxEnergyPerParticleKey = "newParticles/maxEnergyPerParticle";
	const double NewParticlesMaxEnergyPerParticleDefault = 50.0;

	const std::string NewRectangleSizeXKey = "newRectangle/size/x";
	const int NewRectangleSizeXDefault = 10;
	const std::string NewRectangleSizeYKey = "newRectangle/size/y";
	const int NewRectangleSizeYDefault = 10;
	const std::string NewRectangleDistKey = "newRectangle/distance";
	const double NewRectangleDistDefault = 1.0;
	const std::string NewRectangleCellEnergyKey = "newRectangle/cellEnergy";
	const double NewRectangleCellEnergyDefault = 100.0;
    const std::string NewRectangleColorCodeKey = "newRectangle/colorCode";
    const int NewRectangleColorCodeDefault = 0;

    const std::string ExtrapolateContentKey = "computation/extrapolateContent";
    const bool ExtrapolateContentDefault = false;

	const std::string ColorizeColorCodeKey = "colorize/colorCode";
    const int ColorizeColorCodeDefault = 0;

    //messages
    QString const InfoAbout = "Artificial Life Environment, version %1.\nDeveloped by Christian Heinemann.";
    QString const InfoConnectedTo = "You are connected to %1.";

    QString const ErrorCuda = "An error has occurred. Please restart the program and check the CUDA parameters.\n\nError message (press CTRL+C to copy the message):\n\"%1\"";
    QString const ErrorLoadSimulation = "Specified simulation could not be loaded.";
    QString const ErrorLoadCollection = "Specified collection could not be loaded.";
    QString const ErrorSaveCollection = "Collection could not be saved.";
    QString const ErrorPasteFromClipboard = "The clipboard memory does not match the token memory pattern.";
    QString const ErrorInvalidValues = "The values you entered are not valid.";
    QString const ErrorLoadSimulationParameters = "The specified simulation parameter file could not be loaded.";
    QString const ErrorSaveSimulationParameters = "Simulation parameters could not be saved.";
    QString const ErrorLoadSymbolMap = "The specified symbol map could not be loaded.";
    QString const ErrorSaveSymbolMap = "The symbol map could not be saved.";
    QString const ErrorInvalidPassword = "The password you entered is incorrect.";
}

class GuiSettings
{
public:
    static QFont getGlobalFont ();
	static QFont getCellFont();
	static QPalette getPaletteForTabWidget();
	static QPalette getPaletteForTab();

	static int getSettingsValue(std::string const& key, int defaultValue);
    static uint getSettingsValue(std::string const& key, uint defaultValue);
    static double getSettingsValue(std::string const& key, double defaultValue);
	static bool getSettingsValue(std::string const& key, bool defaultValue);

	static void setSettingsValue(std::string const& key, int value);
    static void setSettingsValue(std::string const& key, uint value);
    static void setSettingsValue(std::string const& key, double value);
	static void setSettingsValue(std::string const& key, bool value);
};
