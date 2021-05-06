#pragma once
#include <QLabel>

#include "Definitions.h"

class GeneralInfoController
	: public QObject
{
	Q_OBJECT

public:
	GeneralInfoController(QObject * parent = nullptr);
	~GeneralInfoController() = default;

	void init(QLabel* infoLabel, MainController* mainController);

	void increaseTimestep();
	void setZoomFactor(double factor);

	void setRestrictedTPS(boost::optional<int> tps);

    enum class Rendering
    { OpenGL, Item};
    void setRendering(Rendering value);

private:
	Q_SLOT void oneSecondTimerTimeout();

	void updateInfoLabel();

	QTimer* _oneSecondTimer = nullptr;
	QLabel* _infoLabel = nullptr;
	MainController* _mainController = nullptr;
	int _tpsCounting = 0;
	int _tps = 0;
	double _zoomFactor = 4.0;
    boost::optional<int> _restrictedTPS;
    Rendering _rendering = Rendering::OpenGL;
};
