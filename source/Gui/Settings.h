#pragma once

#include <QColor>
#include <QFont>
#include <QPalette>

namespace Const
{
	//visual editor
	const QColor BackgroundColor(0x00, 0x00, 0x00);
	const QColor UniverseColor(0x00, 0x00, 0x1b);
	const QColor CellColor(0x6F, 0x90, 0xFF, 0xA0);
	const QColor ClusterPenFocusColor(0xFF, 0xFF, 0xFF, 0xFF);
	const QColor LineActiveColor(0xFF, 0xFF, 0xFF, 0xB0);
	const QColor LineInactiveColor(0xB0, 0xB0, 0xB0, 0xB0);
	const QColor TokenColor(0xB0, 0xB0, 0xFF, 0xFF);
	const QColor TokenFocusColor(0xE0, 0xE0, 0xFF, 0xFF);
	const QColor EnergyColor(0xB0, 0x10, 0x0, 0xFF);
	const QColor EnergyFocusColor(0xE0, 0x30, 0x20, 0xFF);
	const QColor EnergyPenFocusColor(0xFF, 0xFF, 0xFF, 0xFF);
	const QColor MarkerColor(0x0, 0xC0, 0xFF, 0x30);

	//text editor
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
	const QColor ButtonTextColor(0xC2, 0xC2, 0xC2);
	const QColor ButtonTextHighlightColor(0x90, 0x90, 0xFF);
	const QString StandardFont = "Courier New";

	//setting keys
	const std::string GridSizeXKey = "newSim/gridSize/x";
	const int GridSizeXDefault = 12;
	const std::string GridSizeYKey = "newSim/gridSize/y";
	const int GridSizeYDefault = 6;
	const std::string UnitSizeXKey = "newSim/unitSize/x";
	const int UnitSizeXDefault = 100;
	const std::string UnitSizeYKey = "newSim/unitSize/y";
	const int UnitSizeYDefault = 100;
	const std::string MaxThreadsKey = "newSim/maxThreads";
	const int MaxThreadsDefault = 8;
	const std::string InitialEnergyKey = "newSim/initialEnergy";
	const int InitialEnergyDefault = 0;

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
}

class GuiSettings
{
public:
    static QFont getGlobalFont ();
	static QFont getCellFont();
	static QPalette getPaletteForTabWidget();
	static QPalette getPaletteForTab();

	static int getSettingsValue(std::string const& key, int defaultValue);
	static double getSettingsValue(std::string const& key, double defaultValue);
	static bool getSettingsValue(std::string const& key, bool defaultValue);

	static void setSettingsValue(std::string const& key, int value);
	static void setSettingsValue(std::string const& key, double value);
	static void setSettingsValue(std::string const& key, bool value);
};
