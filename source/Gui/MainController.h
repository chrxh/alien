#pragma once
#include <QObject>

#include "Model/Api/Definitions.h"

#include "Definitions.h"

struct NewSimulationConfig
{
	uint maxThreads;
	IntVector2D gridSize;
	IntVector2D universeSize;
	SymbolTable* symbolTable;
	SimulationParameters* parameters;
};

class MainController
	: public QObject
{
	Q_OBJECT
public:
	MainController(QObject * parent = nullptr);
	virtual ~MainController();

	virtual void init();

	virtual void onRunSimulation(bool run);

private:
	void newSimulation(NewSimulationConfig config);

	MainView* _view = nullptr;
	MainModel* _model = nullptr;

	SimulationController* _simController = nullptr;
	DataManipulator* _dataManipulator = nullptr;
	SimulationAccess* _simAccess = nullptr;
	Notifier* _notifier = nullptr;
};
