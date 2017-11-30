#pragma once
#include <QObject>

#include "Model/Api/Definitions.h"

#include "Definitions.h"

class MainController
	: public QObject
{
	Q_OBJECT
public:
	MainController(QObject * parent = nullptr);
	virtual ~MainController();

	void init();

private:
	void restoreLastSession();

	MainView* _view = nullptr;
	MainModel* _model = nullptr;

	SimulationController* _simController = nullptr;
	DataManipulator* _dataManipulator = nullptr;
	Notifier* _notifier = nullptr;
};
